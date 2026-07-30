from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
import importlib
import json
from pathlib import Path
import types

from click.testing import CliRunner
import numpy as np
import pytest

import mctutil.ng.completeness as completeness_module
import mctutil.ng.precompute as precompute_module
from mctutil.ng.completeness import check_mip0_completeness


def write_info(
	root: Path,
	*,
	encoding: str,
	size=(4, 3, 2),
	chunk_size=(2, 2, 1),
	dtype="uint16",
) -> Path:
	scale_path = root / "700_700_700"
	scale_path.mkdir(parents=True)
	(root / "info").write_text(
		json.dumps(
			{
				"type": "image" if encoding == "raw" else "segmentation",
				"data_type": dtype,
				"num_channels": 1,
				"scales": [
					{
						"key": scale_path.name,
						"encoding": encoding,
						"size": list(size),
						"chunk_sizes": [list(chunk_size)],
					}
				],
			}
		),
		encoding="utf-8",
	)
	return scale_path


def test_raw_completeness_uses_one_enumeration_and_exact_byte_total(
	tmp_path,
	monkeypatch,
):
	scale_path = write_info(tmp_path / "layer", encoding="raw")
	(scale_path / "chunk-a").write_bytes(b"a" * 24)
	(scale_path / "chunk-b").write_bytes(b"b" * 24)
	real_scandir = completeness_module.os.scandir
	scans = []

	def counted_scandir(path):
		scans.append(Path(path))
		return real_scandir(path)

	monkeypatch.setattr(completeness_module.os, "scandir", counted_scandir)

	result = check_mip0_completeness(tmp_path / "layer")

	assert result.complete is True
	assert result.metric == "bytes"
	assert result.expected == 48
	assert result.actual == 48
	assert scans == [scale_path]

	(scale_path / "chunk-b").write_bytes(b"truncated")
	result = check_mip0_completeness(tmp_path / "layer")
	assert result.complete is False
	assert result.actual == 24 + len(b"truncated")


def test_segmentation_completeness_compares_expected_chunk_count(tmp_path):
	scale_path = write_info(
		tmp_path / "labels",
		encoding="compressed_segmentation",
		size=(5, 4, 3),
		chunk_size=(2, 3, 2),
		dtype="uint32",
	)
	for index in range(12):
		(scale_path / f"chunk-{index}").write_bytes(b"variable")

	result = check_mip0_completeness(tmp_path / "labels")
	assert result.complete is True
	assert result.metric == "chunks"
	assert result.expected == 12
	assert result.actual == 12

	(scale_path / "chunk-11").unlink()
	assert check_mip0_completeness(tmp_path / "labels").complete is False


def test_precompute_retries_only_in_process_incomplete_planes(monkeypatch, tmp_path):
	spec = precompute_module.InputSpec(
		mode="directory",
		source=(),
		shape=(4051, 2, 2),
		dtype=np.dtype("uint16"),
	)
	plan = precompute_module.VolumePlan(
		layer_type="image",
		encoding="raw",
		dtype=np.dtype("uint16"),
		resolution=(700, 700, 700),
		voxel_offset=(0, 0, 0),
		chunk_size=(2, 2, 1),
		segmentation_block=(8, 8, 8),
	)
	failure = BrokenProcessPool("worker died")
	calls = []

	def execute(_cloudpath, _spec, _plan, z_indices, workers):
		calls.append((tuple(z_indices), workers))
		if len(calls) == 1:
			return precompute_module.WorkerBatchResult(
				frozenset(z_indices[:101]),
				failure,
			)
		return precompute_module.WorkerBatchResult(frozenset(z_indices), None)

	monkeypatch.setattr(precompute_module, "_execute_slices", execute)

	written = precompute_module.write_all_slices(tmp_path / "output", spec, plan, 24)

	assert written == 4051
	assert calls[0][1] == 24
	assert calls[1][1] == 12
	assert calls[0][0] == tuple(range(4051))
	assert calls[1][0] == tuple(range(101, 4051))
	assert not hasattr(precompute_module, "missing_slices")
	assert not hasattr(precompute_module, "slice_complete")


def test_downsample_rejects_incomplete_mip0_unless_forced(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/ng/downsample_pyramid.py")

	class FakeVolume:
		def __init__(self, *_args, **_kwargs):
			self.info = {
				"type": "image",
				"scales": [{"encoding": "raw"}],
			}

	incomplete = types.SimpleNamespace(
		complete=False,
		verifiable=True,
		summary=lambda: "bytes: expected=48, actual=24",
	)
	monkeypatch.setattr(
		module,
		"_require_dependencies",
		lambda: (FakeVolume, types.SimpleNamespace()),
	)
	monkeypatch.setattr(module, "check_mip0_completeness", lambda _path: incomplete)
	calls = []
	monkeypatch.setattr(
		module,
		"downsample_volume",
		lambda *args, **kwargs: calls.append((args, kwargs)),
	)
	layer = tmp_path / "layer"
	layer.mkdir()
	base_arguments = [
		str(layer),
		"--queue", str(tmp_path / "queue"),
		"--max-extend-passes", "0",
	]

	refused = CliRunner().invoke(module.downsample_pyramid, base_arguments)
	assert refused.exit_code != 0
	assert "source MIP 0 is incomplete" in refused.output
	assert not calls

	forced = CliRunner().invoke(
		module.downsample_pyramid,
		[*base_arguments, "--force"],
	)
	assert forced.exit_code == 0, forced.output
	assert "forcing downsample" in forced.output
	assert len(calls) == 1


def test_downsample_reports_nonlocal_mip0_as_unverifiable(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/ng/downsample_pyramid.py")

	class FakeVolume:
		def __init__(self, *_args, **_kwargs):
			self.info = {
				"type": "image",
				"scales": [{"encoding": "raw"}],
			}

	unverifiable = types.SimpleNamespace(
		complete=False,
		verifiable=False,
		summary=lambda: "completeness checks require a local file:// layer",
	)
	monkeypatch.setattr(
		module,
		"_require_dependencies",
		lambda: (FakeVolume, types.SimpleNamespace()),
	)
	monkeypatch.setattr(module, "check_mip0_completeness", lambda _path: unverifiable)

	result = CliRunner().invoke(
		module.downsample_pyramid,
		[
			"s3://bucket/layer",
			"--queue", str(tmp_path / "queue"),
			"--max-extend-passes", "0",
		],
	)

	assert result.exit_code != 0
	assert "source MIP 0 completeness could not be verified" in result.output
	assert "source MIP 0 is incomplete" not in result.output


def test_publish_precompute_artifact_uses_completeness_predicate(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/ng/publish.py")
	plan = types.SimpleNamespace(precomputed=tmp_path / "precomputed")
	result = types.SimpleNamespace(complete=False)
	monkeypatch.setattr(module, "check_mip0_completeness", lambda _path: result)

	assert module.stage_artifact_valid("precompute", plan) is False
	result.complete = True
	assert module.stage_artifact_valid("precompute", plan) is True


def test_persistent_queue_resume_releases_existing_leases(tmp_path, capsys):
	taskqueue = pytest.importorskip("taskqueue")
	pytest.importorskip("igneous.task_creation")
	module = importlib.import_module("mctutil.shared.persistent_queue")
	queue_path = tmp_path / "resume"
	queue = taskqueue.TaskQueue(module.file_queue_url(queue_path), progress=False)
	queue.insert([taskqueue.PrintTask("resumed")])
	assert queue.lease(seconds=3600) is not None
	assert queue.leased == 1
	module.write_state(
		queue_path / "mctutil-state.json",
		{
			"fingerprint": "resume-release",
			"status": "executing",
			"inserted": 1,
		},
	)

	state = module.run_persistent_tasks(
		queue_path,
		"resume-release",
		lambda: (_ for _ in ()).throw(AssertionError("tasks were reinserted")),
		parallel=1,
		lease_seconds=60,
		release_leases=True,
	)

	assert state["status"] == "complete"
	assert queue.is_empty() is True
	assert "Released 1 existing task lease(s)" in capsys.readouterr().out


def test_persistent_queue_resume_can_preserve_existing_leases(
	tmp_path,
	monkeypatch,
	capsys,
):
	taskqueue = pytest.importorskip("taskqueue")
	module = importlib.import_module("mctutil.shared.persistent_queue")
	queue_path = tmp_path / "preserve"
	queue = taskqueue.TaskQueue(module.file_queue_url(queue_path), progress=False)
	queue.insert([taskqueue.PrintTask("preserved")])
	leased_task = queue.lease(seconds=3600)
	assert leased_task is not None
	module.write_state(
		queue_path / "mctutil-state.json",
		{
			"fingerprint": "resume-preserve",
			"status": "executing",
			"inserted": 1,
		},
	)
	observed = {}

	def drain(queue_url, _parallel, _lease_seconds):
		resumed = taskqueue.TaskQueue(queue_url, progress=False)
		observed["leased"] = resumed.leased
		resumed.delete(leased_task, tally=True)

	monkeypatch.setattr(module, "drain_file_queue", drain)

	state = module.run_persistent_tasks(
		queue_path,
		"resume-preserve",
		lambda: (_ for _ in ()).throw(AssertionError("tasks were reinserted")),
		parallel=1,
		lease_seconds=60,
		release_leases=False,
	)

	assert state["status"] == "complete"
	assert observed["leased"] == 1
	assert "Preserved 1 existing task lease(s)" in capsys.readouterr().out


def test_missing_expected_queue_announces_full_reinsert(tmp_path, capsys):
	taskqueue = pytest.importorskip("taskqueue")
	pytest.importorskip("igneous.task_creation")
	module = importlib.import_module("mctutil.shared.persistent_queue")

	state = module.run_persistent_tasks(
		tmp_path / "lost",
		"lost-queue",
		lambda: [taskqueue.PrintTask("replacement")],
		parallel=1,
		lease_seconds=60,
		expected_existing=True,
	)

	assert state["status"] == "complete"
	assert "missing; regenerating the full task set" in capsys.readouterr().out


def test_queue_fingerprint_mismatch_fails_before_queue_mutation(
	tmp_path,
	monkeypatch,
):
	module = importlib.import_module("mctutil.shared.persistent_queue")
	queue_path = tmp_path / "mismatch"
	module.write_state(
		queue_path / "mctutil-state.json",
		{
			"fingerprint": "old",
			"status": "executing",
			"inserted": 1,
		},
	)
	constructed = []

	class FakeTaskQueue:
		def __init__(self, *_args, **_kwargs):
			constructed.append(True)

	monkeypatch.setattr(
		module,
		"_require_taskqueue",
		lambda: (RuntimeError, FakeTaskQueue),
	)

	with pytest.raises(RuntimeError, match="fingerprint mismatch"):
		module.run_persistent_tasks(
			queue_path,
			"new",
			lambda: (),
			parallel=1,
		)

	assert not constructed
