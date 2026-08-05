"""Durable local task-queue execution for resumable Igneous stages."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib
from io import StringIO
import json
import multiprocessing
from pathlib import Path
from queue import Empty as EventQueueEmpty
import threading
import time
import traceback

from mctutil.shared.igneous_output import igneous_output_session
from mctutil.shared.deps import require
from mctutil.shared.log import log, LOG
from mctutil.shared.resource_monitor import record_active_workers


class QueueDrainError(RuntimeError):
	"""A worker-process failure with its buffered output events."""

	def __init__(self, message: str, events: list[dict]):
		super().__init__(message)
		self.events = events


class _ListEventQueue:
	def __init__(self, events: list[dict]):
		self.events = events

	def put(self, event: dict) -> None:
		self.events.append(event)


def stable_fingerprint(value) -> str:
	"""Return a stable short identifier for JSON-compatible configuration."""
	encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()[:16]


def read_state(path: Path, default=None):
	if not path.exists():
		return default
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def write_state(path: Path, value) -> None:
	"""Atomically replace a small JSON state file."""
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f".{path.name}.tmp")
	with temporary.open("w", encoding="utf-8") as handle:
		json.dump(value, handle, indent=2, sort_keys=True)
		handle.write("\n")
	temporary.replace(path)


def file_queue_url(path: Path) -> str:
	return f"fq://{path.resolve()}"


def _require_taskqueue():
	taskqueue = require(
		"taskqueue",
		"mesh",
		purpose="persistent task execution requires task-queue",
	)
	return taskqueue.QueueEmptyError, taskqueue.TaskQueue


def _renew_lease(queue, task, lease_seconds: int, stopped: threading.Event) -> None:
	interval = max(1.0, lease_seconds / 2)
	while not stopped.wait(interval):
		queue.renew(task, lease_seconds)


def _drain_worker(
	queue_url: str,
	lease_seconds: int,
	task_modules: tuple[str, ...],
	event_queue,
	poll=None,
) -> None:
	"""Lease, execute, and acknowledge tasks until the durable queue is empty."""
	for module_name in task_modules:
		importlib.import_module(module_name)
	QueueEmptyError, TaskQueue = _require_taskqueue()
	queue = TaskQueue(queue_url, progress=False)
	while True:
		try:
			task = queue.lease(seconds=lease_seconds)
		except QueueEmptyError:
			if queue.is_empty():
				return
			time.sleep(0.1)
			continue

		stopped = threading.Event()
		renewer = threading.Thread(
			target=_renew_lease,
			args=(queue, task, lease_seconds, stopped),
			daemon=True,
		)
		renewer.start()
		stdout_buffer = StringIO()
		stderr_buffer = StringIO()
		try:
			with (
				redirect_stdout(stdout_buffer),
				redirect_stderr(stderr_buffer),
			):
				task.execute()
			stopped.set()
			renewer.join()
			queue.delete(task, tally=True)
			if poll is not None:
				poll()
		except Exception:
			stopped.set()
			renewer.join()
			queue.cancel(task)
			event_queue.put(
				{
					"status": "failed",
					"stdout": stdout_buffer.getvalue(),
					"stderr": stderr_buffer.getvalue(),
					"traceback": traceback.format_exc(),
				}
			)
			raise
		else:
			stdout_value = stdout_buffer.getvalue()
			stderr_value = stderr_buffer.getvalue()
			if stdout_value or stderr_value:
				event_queue.put(
					{
						"status": "complete",
						"stdout": stdout_value,
						"stderr": stderr_value,
					}
				)


def _drain_worker_entry(
	queue_url: str,
	lease_seconds: int,
	task_modules: tuple[str, ...],
	event_queue,
) -> None:
	try:
		_drain_worker(
			queue_url,
			lease_seconds,
			task_modules,
			event_queue,
		)
	except Exception:
		event_queue.put(
			{
				"status": "failed",
				"stdout": "",
				"stderr": "",
				"traceback": traceback.format_exc(),
			}
		)
		raise SystemExit(1)


def _collect_worker_events(event_queue, events: list[dict]) -> None:
	while True:
		try:
			events.append(event_queue.get_nowait())
		except EventQueueEmpty:
			return


def _parallel_worker_events(
	queue_url: str,
	parallel: int,
	lease_seconds: int,
	task_modules: tuple[str, ...],
	poll,
	poll_interval: float,
):
	context = multiprocessing.get_context("spawn")
	event_queue = context.Queue()
	events = []
	processes = [
		context.Process(
			target=_drain_worker_entry,
			args=(queue_url, lease_seconds, task_modules, event_queue),
		)
		for _index in range(parallel)
	]
	for process in processes:
		process.start()
	while any(process.is_alive() for process in processes):
		if poll is not None:
			poll()
		_collect_worker_events(event_queue, events)
		live_process = next(
			(
				process
				for process in processes
				if process.is_alive()
			),
			None,
		)
		if live_process is not None:
			live_process.join(timeout=poll_interval)
	for process in processes:
		process.join()
	if poll is not None:
		poll()
	time.sleep(min(poll_interval, 0.05))
	_collect_worker_events(event_queue, events)
	event_queue.close()
	event_queue.join_thread()
	return processes, events


def _raise_worker_failures(processes, events: list[dict]) -> None:
	failed = [process.exitcode for process in processes if process.exitcode]
	if not failed:
		return
	failure_events = [
		event
		for event in events
		if event["status"] == "failed"
	]
	detail = ""
	if failure_events:
		detail = failure_events[-1]["traceback"].strip().splitlines()[-1]
	suffix = f": {detail}" if detail else ""
	raise QueueDrainError(
		f"{len(failed)} task worker(s) failed with exit codes {failed}{suffix}",
		events,
	)


def drain_file_queue(
	queue_url: str,
	parallel: int,
	lease_seconds: int,
	task_modules: tuple[str, ...] = ("igneous.task_creation",),
	poll=None,
	poll_interval: float = 0.1,
) -> list[dict]:
	"""Execute a file queue in one or more independent worker processes."""
	if parallel < 1:
		raise ValueError("parallel must be at least 1")

	events = []
	if parallel == 1:
		try:
			_drain_worker(
				queue_url,
				lease_seconds,
				task_modules,
				_ListEventQueue(events),
				poll=poll,
			)
		except Exception as exc:
			raise QueueDrainError(str(exc), events) from exc
		return events

	processes, events = _parallel_worker_events(
		queue_url,
		parallel,
		lease_seconds,
		task_modules,
		poll,
		poll_interval,
	)
	_raise_worker_failures(processes, events)
	return events


def _announce_queue_start(
	queue_path: Path,
	state: dict | None,
	expected_existing: bool,
) -> None:
	log.write(
		"Queue",
		f"Path: {queue_path}",
		log_level=LOG.DEBUG,
	)
	if state is not None:
		log.write(
			"Queue",
			f"Resuming persistent queue (status={state.get('status', 'unknown')}).",
			log_level=LOG.STATUS,
		)
	elif expected_existing:
		log.write(
			"Queue",
			"Persistent queue is missing; regenerating the full task set.",
			log_level=LOG.WARN,
		)
	else:
		log.write(
			"Queue",
			"Creating a fresh persistent queue.",
			log_level=LOG.STATUS,
		)


def _finish_insertion(queue, state: dict, state_path: Path, tasks_factory) -> None:
	if state["status"] != "inserting":
		return
	if queue.inserted == 0:
		# FileQueue commits its insertion counter only after the full insert
		# returns. If a process dies mid-insert, retrying the complete task
		# set can duplicate idempotent tasks but cannot accept a partial set.
		state["inserted"] = queue.insert(tasks_factory())
	else:
		state["inserted"] = queue.inserted
	state["status"] = "enqueued"
	write_state(state_path, state)
	log.write(
		"Queue",
		f"Tasks: inserted={state['inserted']}; remaining={queue.enqueued}.",
		log_level=LOG.STATUS,
	)


def _release_resume_leases(queue, state: dict, release_leases: bool) -> None:
	if state["status"] not in {"enqueued", "executing"}:
		return
	leased = queue.leased
	if release_leases:
		queue.release_all()
		statement = f"Released {leased} existing task lease(s) before resume."
	else:
		statement = f"Preserved {leased} existing task lease(s) before resume."
	log.write("Queue", statement, log_level=LOG.STATUS)
	log.write(
		"Queue",
		(
			f"Tasks: inserted={state.get('inserted', queue.inserted)}; "
			f"remaining={queue.enqueued}."
		),
		log_level=LOG.STATUS,
	)


class QueueCompletionMonitor:
	"""Translate a durable completion tally into bounded progress updates."""

	def __init__(self, queue, total: int, progress):
		self.queue = queue
		self.total = total
		self.progress = progress
		self.highest_tally = int(queue.completed or 0)

	@property
	def display_position(self) -> int:
		return min(self.total, self.highest_tally)

	def poll(self) -> None:
		self.highest_tally = max(
			self.highest_tally,
			int(self.queue.completed or 0),
		)
		delta = self.display_position - self.progress.position
		if delta > 0:
			self.progress.update(delta)

	def reconcile_empty(self) -> None:
		"""Finish the display from the authoritative empty-queue state."""
		self.poll()
		if self.progress.position < self.total:
			log.write(
				"Queue",
				(
					"Completion tally trailed the empty queue; "
					f"advancing {self.progress.position}/{self.total} to complete."
				),
				log_level=LOG.WARN,
			)
			self.progress.update(self.total - self.progress.position)

	def report_overrun(self) -> None:
		if self.highest_tally > self.total:
			log.write(
				"Queue",
				(
					f"Completion tally exceeded inserted total "
					f"({self.highest_tally}>{self.total}); display was clamped."
				),
				log_level=LOG.DEBUG,
			)


def _emit_worker_events(events: list[dict]) -> None:
	if not events:
		return
	with igneous_output_session() as normalizer:
		for event in events:
			failed = event["status"] == "failed"
			normalizer.emit(
				event.get("stdout", ""),
				unexpected_level=LOG.WARN if failed else LOG.DEBUG,
			)
			normalizer.emit(
				event.get("stderr", ""),
				stderr=True,
				unexpected_level=LOG.WARN if failed else LOG.DEBUG,
			)
			if failed and event.get("traceback"):
				log.write(
					"Queue Worker",
					event["traceback"],
					log_level=LOG.DEBUG,
				)


def _drain_with_progress(
	queue,
	queue_path: Path,
	total: int,
	parallel: int,
	lease_seconds: int,
	progress_label: str,
) -> QueueCompletionMonitor:
	initial = min(total, int(queue.completed or 0))
	active_parallel = min(parallel, max(1, total - initial))
	record_active_workers(active_parallel)
	worker_events = []
	monitor = None
	try:
		with log.progress(
			progress_label,
			length=total,
			initial=initial,
			start_message=(
				f"Executing queue with {active_parallel} worker(s): "
				f"completed={initial}; total={total}."
			),
			final_message=lambda handle: (
				f"Queue execution complete: completed="
				f"{handle.position}/{total}."
			),
		) as progress:
			monitor = QueueCompletionMonitor(queue, total, progress)
			try:
				worker_events = drain_file_queue(
					file_queue_url(queue_path),
					active_parallel,
					lease_seconds,
					poll=monitor.poll,
				)
			except QueueDrainError as exc:
				worker_events = exc.events
				raise
			if not queue.is_empty():
				raise RuntimeError(
					f"persistent queue did not drain: {queue_path}"
				)
			monitor.reconcile_empty()
	except Exception:
		_emit_worker_events(worker_events)
		raise
	_emit_worker_events(worker_events)
	return monitor


def run_persistent_tasks(
	queue_path: Path,
	task_fingerprint: str,
	tasks_factory,
	parallel: int,
	lease_seconds: int = 3600,
	release_leases: bool = True,
	expected_existing: bool = False,
	progress_label: str = "Queue Tasks",
) -> dict:
	"""Insert a task set once and resume its durable queue until completion."""
	_QueueEmptyError, TaskQueue = _require_taskqueue()
	queue_path = queue_path.resolve()
	state_path = queue_path / "mctutil-state.json"
	state = read_state(state_path)
	resumed = state is not None
	if state is not None and state.get("fingerprint") != task_fingerprint:
		raise RuntimeError(f"persistent queue fingerprint mismatch: {queue_path}")

	_announce_queue_start(queue_path, state, expected_existing)
	queue = TaskQueue(file_queue_url(queue_path), progress=False)
	if state is None:
		state = {
			"fingerprint": task_fingerprint,
			"status": "inserting",
			"inserted": 0,
		}
		write_state(state_path, state)

	_finish_insertion(queue, state, state_path, tasks_factory)

	if state["status"] == "complete":
		log.write(
			"Queue",
			(
				f"Already complete: inserted="
				f"{state.get('inserted', queue.inserted)}; remaining=0."
			),
			log_level=LOG.STATUS,
		)
		return state

	if resumed:
		_release_resume_leases(queue, state, release_leases)

	if queue.is_empty():
		state["status"] = "complete"
		write_state(state_path, state)
		log.write(
			"Queue",
			(
				f"Queue is empty: inserted="
				f"{state.get('inserted', queue.inserted)}; remaining=0."
			),
			log_level=LOG.STATUS,
		)
		return state

	state["status"] = "executing"
	write_state(state_path, state)
	total = int(state.get("inserted", queue.inserted))
	monitor = _drain_with_progress(
		queue,
		queue_path,
		total,
		parallel,
		lease_seconds,
		progress_label,
	)
	monitor.report_overrun()
	state["status"] = "complete"
	write_state(state_path, state)
	return state
