from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
from io import StringIO
from pathlib import Path
import types

from mctutil.ng import precompute as precompute_module
from mctutil.shared.igneous_output import (
	capture_igneous_call,
	igneous_output_session,
)
from mctutil.shared.log import (
	Logger,
	LOG,
	LOG_MASK_DEFAULT,
	LOG_MASK_QUIET,
	LOG_MASK_VERBOSE,
)


class TtyBuffer(StringIO):
	def isatty(self):
		return True


def test_progress_handle_supports_manual_updates_and_durable_records(capsys):
	logger = Logger(
		log_screen={
			"stdout": LOG_MASK_DEFAULT,
			"stderr": LOG.ERROR,
		}
	)
	terminal = TtyBuffer()

	with logger.progress(
		"Work",
		length=3,
		out=terminal,
		start_message="Starting three units.",
		final_message=lambda handle: f"Finished {handle.position} units.",
	) as progress:
		progress.update(1)
		progress.update(2)

	assert progress.position == 3
	assert "3/3" in terminal.getvalue()
	durable = capsys.readouterr().out
	assert "Starting three units." in durable
	assert "Finished 3 units." in durable


def test_progress_handle_uses_durable_records_for_redirected_output(capsys):
	logger = Logger(
		log_screen={
			"stdout": LOG_MASK_DEFAULT,
			"stderr": LOG.ERROR,
		}
	)
	redirected = StringIO()

	with logger.progress("Work", length=2, out=redirected) as progress:
		progress.update(2)

	assert redirected.getvalue() == ""
	durable = capsys.readouterr().out
	assert "Started; total=2." in durable
	assert "Completed 2/2." in durable


def test_progress_handle_respects_quiet_threshold(capsys):
	logger = Logger(
		log_screen={
			"stdout": LOG_MASK_QUIET,
			"stderr": LOG.ERROR,
		}
	)
	terminal = TtyBuffer()

	with logger.progress("Work", length=1, out=terminal) as progress:
		progress.update(1)

	assert terminal.getvalue() == ""
	assert capsys.readouterr().out == ""


def test_progress_handle_preserves_iterable_api(capsys):
	logger = Logger(
		log_screen={
			"stdout": LOG_MASK_DEFAULT,
			"stderr": LOG.ERROR,
		}
	)

	with logger.progress("Items", items=[1, 2, 3], out=StringIO()) as progress:
		assert list(progress) == [1, 2, 3]

	assert progress.position == 3
	assert "Completed 3/3." in capsys.readouterr().out


def test_igneous_task_creation_output_is_classified_and_deduplicated(
	capsys,
):
	from mctutil.shared.log import log

	log.set_threshold(LOG_MASK_VERBOSE)

	def noisy_factory():
		print("Volume Bounds:  Bbox([0, 0, 0],[4, 4, 4])")
		print("Selected ROI:   Bbox([0, 0, 0],[4, 4, 4])")
		print(
			'Unable to determine provenance contact email. Set "git config '
			'user.email". Using unix $USER instead.'
		)
		print("WARNING: No scales generated.")
		return ["task"]

	try:
		with igneous_output_session():
			assert capture_igneous_call(noisy_factory) == ["task"]
			assert capture_igneous_call(noisy_factory) == ["task"]
	finally:
		log.set_threshold(LOG_MASK_DEFAULT)

	output = capsys.readouterr().out
	assert output.count("Volume Bounds") == 1
	assert output.count("Selected ROI") == 1
	assert output.count("provenance contact email") == 1
	assert output.count("No additional scales generated.") == 1
	assert "WARN  |Igneous" in output
	assert "INFO  |Igneous" in output


def test_igneous_unexpected_output_remains_visible(capsys):
	def noisy_factory():
		print("unexpected upstream diagnostic")

	capture_igneous_call(noisy_factory)

	output = capsys.readouterr().out
	assert "WARN  |Igneous" in output
	assert "unexpected upstream diagnostic" in output


def test_z_plane_progress_spans_reduced_worker_retry(
	tmp_path,
	monkeypatch,
):
	updates = []
	calls = []

	class RecordingProgress:
		def __init__(self, length):
			self.length = length
			self.position = 0
			self.enters = 0

		def __enter__(self):
			self.enters += 1
			return self

		def update(self, count):
			self.position += count
			updates.append(count)

		def __exit__(self, *_args):
			return False

	progress = RecordingProgress(4)
	monkeypatch.setattr(
		precompute_module.log,
		"progress",
		lambda _step, **kwargs: progress,
	)

	def execute(
		_cloudpath,
		_input_spec,
		_plan,
		z_indices,
		workers,
		progress,
	):
		calls.append((tuple(z_indices), workers))
		if len(calls) == 1:
			progress.update(2)
			return precompute_module.WorkerBatchResult(
				frozenset({0, 1}),
				BrokenProcessPool("failed"),
			)
		progress.update(2)
		return precompute_module.WorkerBatchResult(
			frozenset({2, 3}),
			None,
		)

	monkeypatch.setattr(precompute_module, "_execute_slices", execute)
	input_spec = types.SimpleNamespace(shape=(4, 2, 2))

	written = precompute_module.write_all_slices(
		Path(tmp_path / "output"),
		input_spec,
		types.SimpleNamespace(),
		workers=4,
	)

	assert written == 4
	assert calls == [((0, 1, 2, 3), 4), ((2, 3), 2)]
	assert updates == [2, 2]
	assert progress.position == progress.length == 4
	assert progress.enters == 1
