from __future__ import annotations

import importlib
from io import StringIO

import pytest

from mctutil.shared.log import log, LOG_MASK_ALL, LOG_MASK_DEFAULT


class TtyBuffer(StringIO):
	def isatty(self):
		return True


class RecordingProgress:
	def __init__(self, **configuration):
		self.configuration = configuration
		self.position = configuration["initial"]
		self.updates = []

	def __enter__(self):
		return self

	def update(self, count):
		self.position += count
		self.updates.append(count)

	def __exit__(self, *_args):
		return False


def test_parallel_poll_tolerates_last_worker_exiting_during_selection(
	monkeypatch,
):
	module = importlib.import_module("mctutil.shared.persistent_queue")

	class FakeEventQueue:
		def get_nowait(self):
			raise module.EventQueueEmpty

		def close(self):
			pass

		def join_thread(self):
			pass

	class ExitingProcess:
		exitcode = 0

		def __init__(self):
			self.alive_checks = 0
			self.joins = []

		def start(self):
			pass

		def is_alive(self):
			self.alive_checks += 1
			return self.alive_checks == 1

		def join(self, timeout=None):
			self.joins.append(timeout)

	process = ExitingProcess()

	class FakeContext:
		def Queue(self):
			return FakeEventQueue()

		def Process(self, **_kwargs):
			return process

	monkeypatch.setattr(
		module.multiprocessing,
		"get_context",
		lambda _method: FakeContext(),
	)

	processes, events = module._parallel_worker_events(
		"fq://unused",
		parallel=1,
		lease_seconds=60,
		task_modules=(),
		poll=None,
		poll_interval=0,
	)

	assert processes == [process]
	assert events == []
	assert process.alive_checks >= 2
	assert process.joins == [None]


def test_progress_renders_resume_position_on_first_frame():
	terminal = TtyBuffer()

	with log.progress(
		"Resume",
		length=3,
		initial=2,
		out=terminal,
	) as progress:
		progress.update(1)

	output = terminal.getvalue()
	assert "2/3" in output
	assert "0/3" not in output
	assert "3/3" in output


def test_resumed_queue_progress_starts_at_durable_completion(
	tmp_path,
	monkeypatch,
):
	taskqueue = pytest.importorskip("taskqueue")
	module = importlib.import_module("mctutil.shared.persistent_queue")
	queue_path = tmp_path / "resume-progress"
	queue = taskqueue.TaskQueue(module.file_queue_url(queue_path), progress=False)
	queue.insert(
		[
			taskqueue.PrintTask("already-complete"),
			taskqueue.PrintTask("remaining-a"),
			taskqueue.PrintTask("remaining-b"),
		]
	)
	completed_task = queue.lease(seconds=60)
	queue.delete(completed_task, tally=True)
	module.write_state(
		queue_path / "mctutil-state.json",
		{
			"fingerprint": "resume-progress",
			"status": "executing",
			"inserted": 3,
		},
	)
	recorded = {}

	def progress_factory(_label, **configuration):
		progress = RecordingProgress(**configuration)
		recorded["progress"] = progress
		recorded["label"] = _label
		return progress

	def drain(queue_url, _parallel, _lease_seconds, poll):
		resumed = taskqueue.TaskQueue(queue_url, progress=False)
		while not resumed.is_empty():
			task = resumed.lease(seconds=60)
			resumed.delete(task, tally=True)
			poll()
		return []

	monkeypatch.setattr(module.log, "progress", progress_factory)
	monkeypatch.setattr(module, "drain_file_queue", drain)

	state = module.run_persistent_tasks(
		queue_path,
		"resume-progress",
		lambda: (_ for _ in ()).throw(AssertionError("tasks were reinserted")),
		parallel=2,
		progress_label="Downsample extend 2",
	)

	progress = recorded["progress"]
	assert state["status"] == "complete"
	assert recorded["label"] == "Downsample extend 2"
	assert progress.configuration["length"] == 3
	assert progress.configuration["initial"] == 1
	assert progress.updates == [1, 1]
	assert progress.position == 3


@pytest.mark.parametrize(
	("total", "completed", "requested", "expected"),
	[
		(1, 0, 20, 1),
		(5, 3, 20, 2),
		(25, 3, 20, 20),
	],
)
def test_queue_workers_are_capped_by_unfinished_tasks(
	monkeypatch,
	tmp_path,
	total,
	completed,
	requested,
	expected,
):
	module = importlib.import_module("mctutil.shared.persistent_queue")
	recorded = {}

	class FakeQueue:
		def __init__(self):
			self.completed = completed

		def is_empty(self):
			return True

	queue = FakeQueue()

	def progress_factory(_label, **configuration):
		progress = RecordingProgress(**configuration)
		recorded["progress"] = progress
		return progress

	def drain(_queue_url, parallel, _lease_seconds, poll):
		recorded["parallel"] = parallel
		queue.completed = total
		poll()
		return []

	monkeypatch.setattr(module.log, "progress", progress_factory)
	monkeypatch.setattr(module, "drain_file_queue", drain)

	module._drain_with_progress(
		queue,
		tmp_path / "queue",
		total=total,
		parallel=requested,
		lease_seconds=60,
		progress_label="Queue Tasks",
	)

	assert recorded["parallel"] == expected
	assert (
		f"Executing queue with {expected} worker(s)"
		in recorded["progress"].configuration["start_message"]
	)


def test_duplicate_insert_completion_overrun_is_clamped_not_failed(
	tmp_path,
	capsys,
):
	taskqueue = pytest.importorskip("taskqueue")
	pytest.importorskip("igneous.task_creation")
	module = importlib.import_module("mctutil.shared.persistent_queue")
	queue_path = tmp_path / "duplicate-insert"
	queue = taskqueue.TaskQueue(module.file_queue_url(queue_path), progress=False)
	queue.insert(
		[taskqueue.PrintTask("partial")],
		skip_insert_counter=True,
	)
	log.set_threshold(LOG_MASK_ALL)
	try:
		state = module.run_persistent_tasks(
			queue_path,
			"duplicate-insert",
			lambda: [
				taskqueue.PrintTask("full-a"),
				taskqueue.PrintTask("full-b"),
			],
			parallel=1,
		)
	finally:
		log.set_threshold(LOG_MASK_DEFAULT)

	assert state["status"] == "complete"
	assert queue.completed == 3
	output = capsys.readouterr().out
	assert "completed=2/2" in output
	assert "Completion tally exceeded inserted total (3>2)" in output
	assert "display was clamped" in output


def test_parallel_worker_output_is_emitted_after_parent_progress(
	tmp_path,
	capsys,
):
	taskqueue = pytest.importorskip("taskqueue")
	pytest.importorskip("igneous.task_creation")
	module = importlib.import_module("mctutil.shared.persistent_queue")
	log.set_threshold(LOG_MASK_ALL)
	try:
		state = module.run_persistent_tasks(
			tmp_path / "parallel-output",
			"parallel-output",
			lambda: [
				taskqueue.PrintTask("buffered-worker-a"),
				taskqueue.PrintTask("buffered-worker-b"),
				taskqueue.PrintTask("buffered-worker-c"),
				taskqueue.PrintTask("buffered-worker-d"),
			],
			parallel=2,
		)
	finally:
		log.set_threshold(LOG_MASK_DEFAULT)

	assert state["status"] == "complete"
	output = capsys.readouterr().out
	final_position = output.index("Queue execution complete: completed=4/4")
	assert final_position < output.index("buffered-worker-a")
	assert final_position < output.index("buffered-worker-b")


def test_worker_provenance_warnings_are_deduplicated_by_parent(
	capsys,
	monkeypatch,
):
	module = importlib.import_module("mctutil.shared.persistent_queue")
	monkeypatch.setenv("USER", "worker-user")
	upstream_warning = (
		'Unable to determine provenance contact email. Set "git config '
		'user.email". Using unix $USER instead.'
	)
	events = [
		{
			"status": "complete",
			"stdout": f"{upstream_warning}\n{upstream_warning}\n",
			"stderr": "",
		},
		{
			"status": "complete",
			"stdout": upstream_warning,
			"stderr": "",
		},
	]

	log.set_threshold(LOG_MASK_ALL)
	try:
		module._emit_worker_events(events)
	finally:
		log.set_threshold(LOG_MASK_DEFAULT)

	output = capsys.readouterr().out
	assert output.count("WARN  |Igneous") == 1
	assert output.count("DEBUG |Igneous") == 2
	assert output.count("using Unix user 'worker-user'") == 3
	assert "Unable to determine provenance contact email" not in output
	assert "$USER" not in output


def test_worker_failure_closes_progress_before_diagnostics(
	tmp_path,
	monkeypatch,
	capsys,
):
	taskqueue = pytest.importorskip("taskqueue")
	module = importlib.import_module("mctutil.shared.persistent_queue")
	queue_path = tmp_path / "failed-worker"
	queue = taskqueue.TaskQueue(module.file_queue_url(queue_path), progress=False)
	queue.insert([taskqueue.PrintTask("never-run")])
	module.write_state(
		queue_path / "mctutil-state.json",
		{
			"fingerprint": "failed-worker",
			"status": "executing",
			"inserted": 1,
		},
	)

	def fail(*_args, **_kwargs):
		raise module.QueueDrainError(
			"worker exploded",
			[
				{
					"status": "failed",
					"stdout": "actionable worker output",
					"stderr": "",
					"traceback": "traceback detail",
				}
			],
		)

	monkeypatch.setattr(module, "drain_file_queue", fail)

	with pytest.raises(module.QueueDrainError, match="worker exploded"):
		module.run_persistent_tasks(
			queue_path,
			"failed-worker",
			lambda: (_ for _ in ()).throw(AssertionError("tasks were reinserted")),
			parallel=1,
		)

	output = capsys.readouterr().out
	assert "actionable worker output" in output
	assert "Queue execution complete" not in output
	state = module.read_state(queue_path / "mctutil-state.json")
	assert state["status"] == "executing"
