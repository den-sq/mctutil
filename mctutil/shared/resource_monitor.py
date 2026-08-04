"""Process-wide resource accounting used by ``ng publish``.

The monitor itself is a spawned process so sampling remains independent of
the publisher's logging and worker-pool implementation. Nothing is activated
unless ``PublishResourceMonitor`` is entered by the publish orchestrator.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import signal
import time

import psutil

from mctutil.shared.log import log, LOG


GIB = 1024 ** 3
_MODE_CODES = {"cgroup-v2": 1, "process-pss": 2, "process-rss": 3}
_COLUMN_LABELS = {
	1: ("CG MEMORY", "CG PEAK"),
	2: ("PSS+SWAP", "PEAK P+S"),
	3: ("RSS+SWAP", "PEAK R+S"),
}
_ISOLATED_CGROUP_MARKERS = (
	"docker",
	"kubepods",
	"containerd",
	"libpod",
	"lxc",
	"machine.slice",
	"mctutil",
	"slurm",
)


@dataclass(frozen=True)
class SystemContext:
	available: int | None
	committed: int | None
	commit_limit: int | None
	overcommit_mode: int | None


@dataclass(frozen=True)
class UsageSample:
	mode: str
	current: int
	component: int
	swap: int
	processes: int
	anon: int | None
	file: int | None
	system: SystemContext


@dataclass(frozen=True)
class StageSummary:
	stage: str
	mode: str
	baseline: int
	peak: int
	peak_exact: bool
	component_at_sampled_peak: int
	anon_at_sampled_peak: int | None
	file_at_sampled_peak: int | None
	peak_swap: int
	peak_processes: int
	min_system_available: int | None
	max_system_committed: int | None
	commit_limit: int | None
	overcommit_mode: int | None
	dataset: str | None = None


@dataclass(frozen=True)
class StagePrediction:
	"""Resource-plan inputs relevant to a running stage."""

	shard_capacity: int | None = None
	downsample_memory: int | None = None
	fixed_reserve: int = 16 * GIB
	capacity_multiplier: int = 3


class _StageAccumulator:
	def __init__(
		self,
		stage: str,
		baseline: UsageSample,
		dataset: str | None = None,
	):
		self.stage = stage
		self.dataset = dataset
		self.baseline = baseline
		self.peak_sample = baseline
		self.peak_processes = baseline.processes
		self.peak_swap = baseline.swap
		self.min_available = baseline.system.available
		self.max_committed = baseline.system.committed
		self.commit_limit = baseline.system.commit_limit
		self.overcommit_mode = baseline.system.overcommit_mode

	def observe(self, sample: UsageSample) -> None:
		if sample.current > self.peak_sample.current:
			self.peak_sample = sample
		self.peak_processes = max(self.peak_processes, sample.processes)
		self.peak_swap = max(self.peak_swap, sample.swap)
		if sample.system.available is not None:
			if self.min_available is None:
				self.min_available = sample.system.available
			else:
				self.min_available = min(
					self.min_available,
					sample.system.available,
				)
		if sample.system.committed is not None:
			if self.max_committed is None:
				self.max_committed = sample.system.committed
			else:
				self.max_committed = max(
					self.max_committed,
					sample.system.committed,
				)
		if sample.system.commit_limit is not None:
			self.commit_limit = sample.system.commit_limit
		if sample.system.overcommit_mode is not None:
			self.overcommit_mode = sample.system.overcommit_mode

	def finish(self, exact_peak: int | None) -> StageSummary:
		peak = self.peak_sample.current
		peak_exact = exact_peak is not None
		if exact_peak is not None:
			peak = max(peak, exact_peak)
		return StageSummary(
			stage=self.stage,
			mode=self.peak_sample.mode,
			baseline=self.baseline.current,
			peak=peak,
			peak_exact=peak_exact,
			component_at_sampled_peak=self.peak_sample.component,
			anon_at_sampled_peak=self.peak_sample.anon,
			file_at_sampled_peak=self.peak_sample.file,
			peak_swap=self.peak_swap,
			peak_processes=self.peak_processes,
			min_system_available=self.min_available,
			max_system_committed=self.max_committed,
			commit_limit=self.commit_limit,
			overcommit_mode=self.overcommit_mode,
			dataset=self.dataset,
		)


def _read_int(path: Path) -> int | None:
	try:
		value = path.read_text(encoding="utf-8").strip()
		return int(value)
	except (FileNotFoundError, OSError, ValueError):
		return None


def _read_key_values(path: Path) -> dict[str, int]:
	values = {}
	try:
		lines = path.read_text(encoding="utf-8").splitlines()
	except (FileNotFoundError, OSError):
		return values
	for line in lines:
		parts = line.split()
		if len(parts) < 2:
			continue
		try:
			values[parts[0].rstrip(":")] = int(parts[1])
		except ValueError:
			continue
	return values


def _system_context(proc_root: Path = Path("/proc")) -> SystemContext:
	values = _read_key_values(proc_root / "meminfo")
	overcommit_mode = _read_int(proc_root / "sys/vm/overcommit_memory")
	return SystemContext(
		available=(
			values["MemAvailable"] * 1024
			if "MemAvailable" in values
			else None
		),
		committed=(
			values["Committed_AS"] * 1024
			if "Committed_AS" in values
			else None
		),
		commit_limit=(
			values["CommitLimit"] * 1024
			if "CommitLimit" in values
			else None
		),
		overcommit_mode=overcommit_mode,
	)


def _cgroup_directory(
	pid: int,
	proc_root: Path = Path("/proc"),
	cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> Path | None:
	"""Return an isolated cgroup-v2 directory, or ``None`` for host scopes."""
	try:
		lines = (proc_root / str(pid) / "cgroup").read_text(
			encoding="utf-8"
		).splitlines()
	except (FileNotFoundError, OSError):
		return None
	entry = next((line for line in lines if line.startswith("0::")), None)
	if entry is None:
		return None
	relative = entry[3:].lstrip("/")
	directory = cgroup_root / relative
	if not (directory / "memory.current").is_file():
		return None
	path_text = f"/{relative}".lower()
	marked_isolated = any(
		marker in path_text
		for marker in _ISOLATED_CGROUP_MARKERS
	)
	limit_text = None
	try:
		limit_text = (directory / "memory.max").read_text(
			encoding="utf-8"
		).strip()
	except (FileNotFoundError, OSError):
		pass
	finite_root_limit = relative == "" and (
		limit_text is not None
		and limit_text != "max"
		and limit_text.isdigit()
	)
	containerized_root = relative == "" and (
		Path("/.dockerenv").exists()
		or Path("/run/.containerenv").exists()
		or bool(os.environ.get("container"))
	)
	return (
		directory
		if marked_isolated or finite_root_limit or containerized_root
		else None
	)


class _CgroupSampler:
	mode = "cgroup-v2"

	def __init__(self, directory: Path, proc_root: Path = Path("/proc")):
		self.directory = directory
		self.proc_root = proc_root
		self._peak_handle = None

	def sample(self) -> UsageSample:
		current = _read_int(self.directory / "memory.current")
		if current is None:
			raise RuntimeError("cgroup memory.current became unavailable")
		stats = _read_key_values(self.directory / "memory.stat")
		return UsageSample(
			mode=self.mode,
			current=current,
			component=current,
			swap=_read_int(self.directory / "memory.swap.current") or 0,
			processes=_read_int(self.directory / "pids.current") or 0,
			anon=stats.get("anon"),
			file=stats.get("file"),
			system=_system_context(self.proc_root),
		)

	def reset_peak(self) -> bool:
		self.close_peak()
		handle = None
		try:
			handle = (self.directory / "memory.peak").open(
				"r+",
				encoding="utf-8",
			)
			handle.write("0")
			handle.flush()
			handle.seek(0)
			int(handle.read().strip())
		except (FileNotFoundError, OSError, ValueError):
			if handle is not None:
				try:
					handle.close()
				except OSError:
					pass
			return False
		self._peak_handle = handle
		return True

	def exact_peak(self) -> int | None:
		if self._peak_handle is None:
			return None
		try:
			self._peak_handle.seek(0)
			return int(self._peak_handle.read().strip())
		except (OSError, ValueError):
			return None

	def close_peak(self) -> None:
		if self._peak_handle is not None:
			try:
				self._peak_handle.close()
			except OSError:
				pass
			self._peak_handle = None


def _smaps_usage(path: Path) -> tuple[int, int] | None:
	values = _read_key_values(path)
	if "Pss" not in values:
		return None
	return (
		values["Pss"] * 1024,
		values.get("SwapPss", values.get("Swap", 0)) * 1024,
	)


def _status_swap(path: Path) -> int:
	return _read_key_values(path).get("VmSwap", 0) * 1024


class _ProcessTreeSampler:
	def __init__(self, root_pid: int, proc_root: Path = Path("/proc")):
		self.root_pid = root_pid
		self.proc_root = proc_root
		self.monitor_pid = os.getpid()
		self.mode = (
			"process-pss"
			if _smaps_usage(proc_root / str(root_pid) / "smaps_rollup")
			is not None
			else "process-rss"
		)

	def _pids(self) -> tuple[int, ...]:
		try:
			root = psutil.Process(self.root_pid)
		except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
			return ()
		try:
			children = root.children(recursive=True)
		except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
			children = []
		return tuple(
			dict.fromkeys(
				process.pid
				for process in (root, *children)
				if process.pid != self.monitor_pid
			)
		)

	def sample(self) -> UsageSample:
		component = 0
		swap = 0
		count = 0
		for pid in self._pids():
			if self.mode == "process-pss":
				usage = _smaps_usage(
					self.proc_root / str(pid) / "smaps_rollup"
				)
				if usage is None:
					continue
				resident, process_swap = usage
			else:
				try:
					resident = int(psutil.Process(pid).memory_info().rss)
				except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
					continue
				process_swap = _status_swap(
					self.proc_root / str(pid) / "status"
				)
			component += resident
			swap += process_swap
			count += 1
		return UsageSample(
			mode=self.mode,
			current=component + swap,
			component=component,
			swap=swap,
			processes=count,
			anon=None,
			file=None,
			system=_system_context(self.proc_root),
		)

	def reset_peak(self) -> bool:
		return False

	def exact_peak(self) -> int | None:
		return None

	def close_peak(self) -> None:
		pass


def _make_sampler(root_pid: int):
	cgroup = _cgroup_directory(root_pid)
	if cgroup is not None:
		return _CgroupSampler(cgroup)
	return _ProcessTreeSampler(root_pid)


def _update_columns(shared, sample: UsageSample, peak: int) -> None:
	# One writer updates naturally aligned counters. Avoid a process-shared lock
	# so a killed monitor can never block the publisher's logging path.
	shared[0] = max(0, sample.current)
	shared[1] = max(0, peak)
	shared[2] = _MODE_CODES[sample.mode]


def _handle_monitor_command(
	connection,
	shared,
	sampler,
	accumulator,
	command: str,
	value,
):
	if command == "start":
		sampler.reset_peak()
		sample = sampler.sample()
		dataset, stage = value
		accumulator = _StageAccumulator(stage, sample, dataset)
		_update_columns(shared, sample, sample.current)
		connection.send(("started", sample.mode))
		return accumulator, False
	if command == "stop":
		if accumulator is None:
			connection.send(("error", "no resource stage is active"))
			return accumulator, False
		sample = sampler.sample()
		accumulator.observe(sample)
		summary = accumulator.finish(sampler.exact_peak())
		_update_columns(shared, sample, summary.peak)
		sampler.close_peak()
		connection.send(("summary", summary))
		return None, False
	if command == "shutdown":
		connection.send(("stopped", None))
		return accumulator, True
	connection.send(("error", f"unknown resource command: {command}"))
	return accumulator, False


def _poll_timeout(accumulator, next_sample: float, interval: float) -> float:
	if accumulator is None:
		return interval
	return max(0.0, min(interval, next_sample - time.monotonic()))


def _sample_stage(shared, sampler, accumulator) -> None:
	sample = sampler.sample()
	accumulator.observe(sample)
	peak = accumulator.peak_sample.current
	exact_peak = sampler.exact_peak()
	if exact_peak is not None:
		peak = max(peak, exact_peak)
	_update_columns(shared, sample, peak)


def _run_monitor_loop(connection, shared, sampler, interval: float) -> None:
	accumulator = None
	next_sample = time.monotonic()
	initial = sampler.sample()
	_update_columns(shared, initial, initial.current)
	connection.send(("ready", initial.mode))
	while True:
		if not connection.poll(_poll_timeout(accumulator, next_sample, interval)):
			if accumulator is not None:
				_sample_stage(shared, sampler, accumulator)
				next_sample = time.monotonic() + interval
			continue
		command, value = connection.recv()
		accumulator, should_stop = _handle_monitor_command(
			connection,
			shared,
			sampler,
			accumulator,
			command,
			value,
		)
		if should_stop:
			return
		if command == "start":
			next_sample = time.monotonic() + interval


def _report_monitor_failure(connection, exc: BaseException) -> None:
	try:
		connection.send(("fatal", f"{type(exc).__name__}: {exc}"))
	except (BrokenPipeError, EOFError, OSError):
		pass


def _monitor_main(connection, shared, root_pid: int, interval: float) -> None:
	"""Sampling process entry point. All messages are request/reply pairs."""
	sampler = None
	try:
		signal.signal(signal.SIGINT, signal.SIG_IGN)
		sampler = _make_sampler(root_pid)
		_run_monitor_loop(connection, shared, sampler, interval)
	except EOFError:
		pass
	except BaseException as exc:
		_report_monitor_failure(connection, exc)
	finally:
		if sampler is not None:
			sampler.close_peak()
		connection.close()


_active_monitor = None


def record_active_workers(count: int) -> None:
	"""Record an effective worker count when publish accounting is active."""
	if _active_monitor is not None:
		_active_monitor.observe_workers(count)


def _format_size(value: int | None) -> str:
	if value is None:
		return "unavailable"
	for suffix, unit in (
		("TiB", 1024 ** 4),
		("GiB", GIB),
		("MiB", 1024 ** 2),
		("KiB", 1024),
	):
		if value >= unit:
			amount = value / unit
			return f"{amount:.2f} {suffix}"
	return f"{value} B"


def format_stage_summary(
	summary: StageSummary,
	active_workers: int,
	prediction: StagePrediction | None,
) -> str:
	peak_kind = "exact cgroup peak" if summary.peak_exact else "sampled peak"
	parts = []
	if summary.dataset is not None:
		parts.append(f"dataset={summary.dataset}")
	parts.extend([
		f"stage={summary.stage}",
		f"accounting={summary.mode}",
		f"baseline={_format_size(summary.baseline)}",
		f"total peak={_format_size(summary.peak)} ({peak_kind})",
		f"delta={_format_size(max(0, summary.peak - summary.baseline))}",
		f"effective workers={active_workers}",
		f"peak processes={summary.peak_processes}",
		f"peak swap={_format_size(summary.peak_swap)}",
	])
	if summary.mode == "cgroup-v2":
		parts.append(
			"near sampled peak: "
			f"anon={_format_size(summary.anon_at_sampled_peak)}, "
			f"file={_format_size(summary.file_at_sampled_peak)}"
		)
	else:
		metric = "PSS" if summary.mode == "process-pss" else "RSS"
		parts.append(
			f"{metric} at sampled peak="
			f"{_format_size(summary.component_at_sampled_peak)}"
		)
	parts.append(
		"system-wide: "
		f"min available={_format_size(summary.min_system_available)}, "
		f"max commit={_format_size(summary.max_system_committed)}/"
		f"{_format_size(summary.commit_limit)}, "
		f"overcommit={summary.overcommit_mode}"
	)
	if prediction is None or prediction.shard_capacity is None:
		parts.append("ResourcePlan prediction=n/a")
	else:
		worker_bytes = (
			prediction.capacity_multiplier
			* prediction.shard_capacity
			* active_workers
		)
		combined = prediction.fixed_reserve + worker_bytes
		parts.append(
			"ResourcePlan max shard capacity="
			f"{_format_size(prediction.shard_capacity)}, prediction="
			f"{_format_size(prediction.fixed_reserve)} + "
			f"{active_workers} x {prediction.capacity_multiplier} x "
			f"{_format_size(prediction.shard_capacity)} = "
			f"{_format_size(combined)}"
		)
	if prediction is not None and prediction.downsample_memory is not None:
		parts.append(
			"Igneous downsample target="
			f"{_format_size(prediction.downsample_memory)}"
		)
	return "; ".join(parts) + "."


class PublishResourceMonitor:
	"""Manage one sampler process and per-stage summaries for a publish run."""

	def __init__(self, root_pid: int | None = None, interval: float = 1.0):
		self.root_pid = os.getpid() if root_pid is None else root_pid
		self.interval = interval
		self.connection = None
		self.process = None
		self.shared = None
		self.mode = None
		self.active_workers = 1
		self._previous_monitor = None
		self._reported_dead = False

	@property
	def enabled(self) -> bool:
		return self.process is not None and self.process.is_alive()

	def __enter__(self):
		global _active_monitor
		context = multiprocessing.get_context("spawn")
		parent_connection, child_connection = context.Pipe()
		shared = context.Array("Q", (0, 0, 0), lock=False)
		process = context.Process(
			target=_monitor_main,
			args=(child_connection, shared, self.root_pid, self.interval),
			name="mctutil-resource-monitor",
			daemon=True,
		)
		try:
			process.start()
			child_connection.close()
			if not parent_connection.poll(10.0):
				raise RuntimeError("resource monitor did not start")
			kind, value = parent_connection.recv()
			if kind != "ready":
				raise RuntimeError(str(value))
		except Exception as exc:
			parent_connection.close()
			try:
				child_connection.close()
			except OSError:
				pass
			if process.is_alive():
				process.terminate()
				process.join(timeout=2.0)
			log.write(
				"Resources",
				f"Resource accounting unavailable; continuing without it: {exc}",
				log_level=LOG.WARN,
			)
			return self
		self.connection = parent_connection
		self.process = process
		self.shared = shared
		self.mode = value
		self._previous_monitor = _active_monitor
		_active_monitor = self
		return self

	def announce(self) -> None:
		if not self.enabled:
			return
		if self.mode == "cgroup-v2":
			detail = "isolated cgroup-v2 workload"
		else:
			detail = "recursive publish process tree; monitor PID excluded"
		log.write(
			"Resources",
			(
				f"Accounting mode: {self.mode} ({detail}); sampled every "
				f"{self.interval:g}s. System memory is labeled context only."
			),
			log_level=LOG.STATUS,
		)

	def columns(self):
		if not self.enabled or self.shared is None:
			process = psutil.Process(self.root_pid)
			return (
				"PARENT RSS",
				process.memory_info().rss,
				"SYS AVAIL",
				psutil.virtual_memory().available,
			)
		current, peak, mode = self.shared[:]
		labels = _COLUMN_LABELS.get(mode, ("WORKLOAD", "WL PEAK"))
		return labels[0], current, labels[1], peak

	def observe_workers(self, count: int) -> None:
		if count > 0:
			self.active_workers = max(self.active_workers, int(count))

	def _request(self, command: str, value=None):
		if not self.enabled or self.connection is None:
			if command != "shutdown" and not self._reported_dead:
				log.write(
					"Resources",
					"Resource accounting process stopped; work will continue.",
					log_level=LOG.WARN,
				)
				self._reported_dead = True
			return None
		try:
			self.connection.send((command, value))
			if not self.connection.poll(10.0):
				raise RuntimeError(
					f"resource monitor did not answer {command}"
				)
			kind, response = self.connection.recv()
		except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
			log.write(
				"Resources",
				f"Resource accounting stopped; work will continue: {exc}",
				log_level=LOG.WARN,
			)
			return None
		if kind in {"error", "fatal"}:
			log.write(
				"Resources",
				f"Resource accounting skipped: {response}",
				log_level=LOG.WARN,
			)
			return None
		return response

	@contextmanager
	def stage(
		self,
		name: str,
		prediction: StagePrediction | None = None,
		dataset: str | None = None,
	):
		self.active_workers = 1
		started = self._request("start", (dataset, name)) is not None
		try:
			yield
		finally:
			if started:
				summary = self._request("stop")
				if isinstance(summary, StageSummary):
					log.write(
						"Resources",
						format_stage_summary(
							summary,
							self.active_workers,
							prediction,
						),
						log_level=LOG.STATUS,
					)

	def close(self) -> None:
		global _active_monitor
		if _active_monitor is self:
			_active_monitor = self._previous_monitor
		if self.process is None:
			return
		if self.process.is_alive():
			self._request("shutdown")
			self.process.join(timeout=3.0)
		if self.process.is_alive():
			self.process.terminate()
			self.process.join(timeout=2.0)
		if self.connection is not None:
			self.connection.close()

	def __exit__(self, _exc_type, _exc_value, _traceback):
		self.close()
		return False
