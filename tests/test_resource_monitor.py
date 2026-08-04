from __future__ import annotations

from io import StringIO
import os
from pathlib import Path
import time
import types

from mctutil.shared.log import Logger, LOG, LOG_MASK_DEFAULT
from mctutil.shared import resource_monitor as module


KIB = 1024
MIB = 1024 ** 2
GIB = 1024 ** 3


def write(path: Path, value: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(value, encoding="utf-8")


def system_files(proc_root: Path) -> None:
	write(
		proc_root / "meminfo",
		(
			"MemAvailable: 1048576 kB\n"
			"Committed_AS: 2097152 kB\n"
			"CommitLimit: 4194304 kB\n"
		),
	)
	write(proc_root / "sys/vm/overcommit_memory", "1\n")


def usage(current, *, component=None, swap=0, processes=1):
	return module.UsageSample(
		mode="process-pss",
		current=current,
		component=current - swap if component is None else component,
		swap=swap,
		processes=processes,
		anon=None,
		file=None,
		system=module.SystemContext(
			available=8 * GIB - current,
			committed=4 * GIB + current,
			commit_limit=16 * GIB,
			overcommit_mode=1,
		),
	)


def report_spawned_monitor_state(connection):
	from mctutil.shared import resource_monitor

	connection.send(resource_monitor._active_monitor is None)
	connection.close()


def test_cgroup_selection_requires_an_isolated_workload(tmp_path):
	proc_root = tmp_path / "proc"
	cgroup_root = tmp_path / "cgroup"
	pid = 123

	write(proc_root / str(pid) / "cgroup", "0::/user.slice/session.scope\n")
	write(cgroup_root / "user.slice/session.scope/memory.current", "100\n")
	write(cgroup_root / "user.slice/session.scope/memory.max", "max\n")
	assert module._cgroup_directory(pid, proc_root, cgroup_root) is None

	write(proc_root / str(pid) / "cgroup", "0::/slurm/job_42/step_0\n")
	isolated = cgroup_root / "slurm/job_42/step_0"
	write(isolated / "memory.current", "100\n")
	write(isolated / "memory.max", "max\n")
	assert module._cgroup_directory(pid, proc_root, cgroup_root) == isolated

	write(proc_root / str(pid) / "cgroup", "0::/\n")
	write(cgroup_root / "memory.current", "100\n")
	write(cgroup_root / "memory.max", str(32 * GIB))
	assert module._cgroup_directory(pid, proc_root, cgroup_root) == cgroup_root


def test_cgroup_selection_rejects_shared_slurm_and_machine_slices(tmp_path):
	proc_root = tmp_path / "proc"
	cgroup_root = tmp_path / "cgroup"
	pid = 123

	for relative in (
		"slurm",
		"system.slice/slurmctld.service",
		"machine.slice",
	):
		write(proc_root / str(pid) / "cgroup", f"0::/{relative}\n")
		directory = cgroup_root / relative
		write(directory / "memory.current", "100\n")
		write(directory / "memory.max", "max\n")
		assert module._cgroup_directory(pid, proc_root, cgroup_root) is None


def test_cgroup_selection_accepts_specific_slurm_job_and_machine_scope(tmp_path):
	proc_root = tmp_path / "proc"
	cgroup_root = tmp_path / "cgroup"
	pid = 123

	for relative in (
		"slurm/uid_1000/job_42/step_0",
		"machine.slice/machine-worker.scope",
	):
		write(proc_root / str(pid) / "cgroup", f"0::/{relative}\n")
		directory = cgroup_root / relative
		write(directory / "memory.current", "100\n")
		write(directory / "memory.max", "max\n")
		assert module._cgroup_directory(pid, proc_root, cgroup_root) == directory


def test_cgroup_sampler_reports_memory_categories_and_system_context(tmp_path):
	proc_root = tmp_path / "proc"
	cgroup = tmp_path / "cgroup"
	system_files(proc_root)
	write(cgroup / "memory.current", str(3 * GIB))
	write(cgroup / "memory.swap.current", str(2 * MIB))
	write(cgroup / "pids.current", "7\n")
	write(cgroup / "memory.stat", f"anon {2 * GIB}\nfile {GIB}\n")

	sample = module._CgroupSampler(cgroup, proc_root).sample()

	assert sample.mode == "cgroup-v2"
	assert sample.current == 3 * GIB
	assert sample.swap == 2 * MIB
	assert sample.processes == 7
	assert sample.anon == 2 * GIB
	assert sample.file == GIB
	assert sample.system == module.SystemContext(
		available=GIB,
		committed=2 * GIB,
		commit_limit=4 * GIB,
		overcommit_mode=1,
	)


def test_process_tree_sampler_sums_pss_and_swap_from_rollups(tmp_path):
	proc_root = tmp_path / "proc"
	system_files(proc_root)
	write(
		proc_root / "101/smaps_rollup",
		"Rss: 150 kB\nPss: 100 kB\nSwapPss: 20 kB\n",
	)
	write(
		proc_root / "202/smaps_rollup",
		"Rss: 80 kB\nPss: 50 kB\nSwapPss: 3 kB\n",
	)
	sampler = module._ProcessTreeSampler(101, proc_root)
	sampler._pids = lambda: (101, 202, 303)

	sample = sampler.sample()

	assert sample.mode == "process-pss"
	assert sample.component == 150 * KIB
	assert sample.swap == 23 * KIB
	assert sample.current == 173 * KIB
	assert sample.processes == 2


def test_process_tree_sampler_labels_rss_fallback(tmp_path, monkeypatch):
	proc_root = tmp_path / "proc"
	system_files(proc_root)
	write(proc_root / "101/status", "VmSwap: 7 kB\n")
	write(proc_root / "202/status", "VmSwap: 2 kB\n")
	rss = {101: 10 * MIB, 202: 5 * MIB}

	def process(pid):
		if pid not in rss:
			raise module.psutil.NoSuchProcess(pid)
		return types.SimpleNamespace(
			memory_info=lambda: types.SimpleNamespace(rss=rss[pid])
		)

	monkeypatch.setattr(
		module.psutil,
		"Process",
		process,
	)
	sampler = module._ProcessTreeSampler(101, proc_root)
	sampler._pids = lambda: (101, 202, 303)

	sample = sampler.sample()

	assert sample.mode == "process-rss"
	assert sample.component == 15 * MIB
	assert sample.swap == 9 * KIB
	assert sample.current == 15 * MIB + 9 * KIB
	assert sample.processes == 2


def test_process_tree_walk_is_recursive_and_excludes_monitor(monkeypatch, tmp_path):
	proc_root = tmp_path / "proc"
	write(proc_root / "101/smaps_rollup", "Pss: 1 kB\n")
	monitor_pid = os.getpid()
	root = types.SimpleNamespace(pid=101)
	children = [
		types.SimpleNamespace(pid=202),
		types.SimpleNamespace(pid=monitor_pid),
	]
	root.children = lambda recursive: children if recursive else []
	monkeypatch.setattr(module.psutil, "Process", lambda _pid: root)

	sampler = module._ProcessTreeSampler(101, proc_root)

	assert sampler._pids() == (101, 202)


def test_process_tree_walk_keeps_root_when_children_are_inaccessible(
	monkeypatch,
	tmp_path,
):
	proc_root = tmp_path / "proc"
	write(proc_root / "101/smaps_rollup", "Pss: 1 kB\n")
	root = types.SimpleNamespace(pid=101)
	root.children = lambda recursive: (_ for _ in ()).throw(
		module.psutil.AccessDenied(101)
	)
	monkeypatch.setattr(module.psutil, "Process", lambda _pid: root)

	assert module._ProcessTreeSampler(101, proc_root)._pids() == (101,)


def test_stage_peak_is_maximum_of_whole_tree_samples():
	accumulator = module._StageAccumulator("shard", usage(100, processes=2))
	accumulator.observe(usage(180, processes=3))
	accumulator.observe(usage(140, processes=4))

	summary = accumulator.finish(None)

	assert summary.baseline == 100
	assert summary.peak == 180
	assert summary.peak_processes == 4
	assert summary.min_system_available == 8 * GIB - 180
	assert summary.max_system_committed == 4 * GIB + 180
	assert summary.peak_exact is False


def test_missing_cgroup_peak_reset_uses_sampled_peak(tmp_path):
	cgroup = tmp_path / "cgroup"
	write(cgroup / "memory.current", "100\n")
	sampler = module._CgroupSampler(cgroup, tmp_path / "proc")

	assert sampler.reset_peak() is False
	assert sampler.exact_peak() is None

	accumulator = module._StageAccumulator("shard", sampler.sample())
	write(cgroup / "memory.current", "200\n")
	accumulator.observe(sampler.sample())
	summary = accumulator.finish(sampler.exact_peak())
	assert summary.peak == 200
	assert summary.peak_exact is False


def test_writable_cgroup_peak_is_read_as_an_exact_counter(tmp_path):
	cgroup = tmp_path / "cgroup"
	write(cgroup / "memory.peak", "0\n")
	sampler = module._CgroupSampler(cgroup, tmp_path / "proc")

	assert sampler.reset_peak() is True
	write(cgroup / "memory.peak", "450\n")
	assert sampler.exact_peak() == 450
	sampler.close_peak()


def test_stage_summary_includes_plan_and_downsample_target():
	accumulator = module._StageAccumulator("downsample", usage(2 * GIB))
	accumulator.observe(usage(5 * GIB, swap=GIB, processes=5))
	summary = accumulator.finish(None)

	message = module.format_stage_summary(
		summary,
		active_workers=4,
		prediction=module.StagePrediction(
			shard_capacity=2 * GIB,
			downsample_memory=10_000_000_000,
		),
	)

	assert "accounting=process-pss" in message
	assert "total peak=5.00 GiB (sampled peak)" in message
	assert "effective workers=4" in message
	assert "max shard capacity=2.00 GiB" in message
	assert "16.00 GiB + 4 x 3 x 2.00 GiB = 40.00 GiB" in message
	assert "Igneous downsample target=9.31 GiB" in message
	assert "system-wide:" in message


def test_cgroup_summary_labels_exact_total_and_sampled_categories():
	summary = module.StageSummary(
		stage="shard",
		mode="cgroup-v2",
		baseline=GIB,
		peak=3 * GIB,
		peak_exact=True,
		component_at_sampled_peak=2 * GIB,
		anon_at_sampled_peak=2 * GIB,
		file_at_sampled_peak=512 * MIB,
		peak_swap=0,
		peak_processes=6,
		min_system_available=8 * GIB,
		max_system_committed=5 * GIB,
		commit_limit=16 * GIB,
		overcommit_mode=2,
	)

	message = module.format_stage_summary(summary, 2, None)

	assert "total peak=3.00 GiB (exact cgroup peak)" in message
	assert "near sampled peak: anon=2.00 GiB, file=512.00 MiB" in message


def test_logger_can_use_workload_current_and_peak_columns():
	logger = Logger(
		log_screen={"stdout": LOG_MASK_DEFAULT, "stderr": LOG.ERROR}
	)
	header = StringIO()
	line = StringIO()

	with logger.resource_columns(
		lambda: ("PSS+SWAP", 2 * MIB, "PEAK P+S", 5 * MIB)
	):
		logger.header(out=header)
		logger.write("Work", "sample", log_level=LOG.STATUS, out=line)

	assert "|PSS+SWAP |PEAK P+S |" in header.getvalue()
	assert "|000002.00MB|000005.00MB|" in line.getvalue()


def test_spawned_monitor_flushes_a_stage_summary(capsys):
	monitor = module.PublishResourceMonitor(interval=0.02)
	with monitor:
		assert monitor.enabled
		with module.log.resource_columns(monitor.columns):
			monitor.announce()
			with monitor.stage("precompute"):
				module.record_active_workers(3)
				allocation = bytearray(2 * MIB)
				for index in range(0, len(allocation), 4096):
					allocation[index] = 1
				time.sleep(0.06)
	process = monitor.process

	output = capsys.readouterr().out
	assert process is not None and not process.is_alive()
	assert "Accounting mode:" in output
	assert "stage=precompute" in output
	assert "effective workers=3" in output
	assert "ResourcePlan prediction=n/a" in output


def test_spawned_monitor_flushes_when_stage_raises(capsys):
	monitor = module.PublishResourceMonitor(interval=0.02)
	try:
		with monitor:
			with monitor.stage("shard", dataset="broken"):
				raise KeyboardInterrupt
	except KeyboardInterrupt:
		pass

	assert monitor.process is not None and not monitor.process.is_alive()
	output = capsys.readouterr().out
	assert "dataset=broken; stage=shard" in output


def test_spawned_worker_does_not_inherit_active_monitor():
	context = module.multiprocessing.get_context("spawn")
	parent, child = context.Pipe(duplex=False)
	previous = module._active_monitor
	module._active_monitor = object()
	process = context.Process(target=report_spawned_monitor_state, args=(child,))
	try:
		process.start()
		child.close()
		assert parent.poll(5.0)
		assert parent.recv() is True
		process.join(timeout=5.0)
		assert process.exitcode == 0
	finally:
		module._active_monitor = previous
		if process.is_alive():
			process.terminate()
			process.join(timeout=2.0)
		parent.close()
