from __future__ import annotations

import importlib
import json
import types

from click.testing import CliRunner
import numpy as np
import pytest


class FakeVolume:
	max_mip = 0

	def __init__(self, *_args, **_kwargs):
		self.info = {
			"type": "image",
			"scales": [
				{"encoding": "raw"}
				for _index in range(self.max_mip + 1)
			],
		}


def test_downsample_dry_run_reports_two_pass_plan(load_module, tmp_path, monkeypatch):
	module = load_module("mctutil/ng/downsample_pyramid.py")
	dependencies = (FakeVolume, types.SimpleNamespace())
	monkeypatch.setattr(module, "_require_dependencies", lambda: dependencies)
	layer = tmp_path / "layer"
	layer.mkdir()

	result = CliRunner().invoke(
		module.downsample_pyramid,
		[str(layer), "--dry-run"],
	)

	assert result.exit_code == 0, result.output
	assert "(64, 64, 64)" in result.output
	assert "(16, 16, 16)" in result.output
	assert "factor=(2, 2, 2)" in result.output
	assert not (layer / ".mctutil-queues").exists()


def test_downsample_uses_persistent_pass_state(load_module, tmp_path, monkeypatch):
	module = load_module("mctutil/ng/downsample_pyramid.py")
	FakeVolume.max_mip = 0
	task_calls = []
	queue_calls = []

	def create_tasks(*_args, **kwargs):
		task_calls.append(kwargs)
		return ["task"]

	task_creation = types.SimpleNamespace(create_downsampling_tasks=create_tasks)
	monkeypatch.setattr(module, "_require_dependencies", lambda: (FakeVolume, task_creation))

	def run_tasks(queue_path, fingerprint, tasks_factory, parallel, lease_seconds):
		queue_calls.append((queue_path, fingerprint, parallel, lease_seconds))
		tasks_factory()
		FakeVolume.max_mip += 1
		return {"status": "complete"}

	monkeypatch.setattr(module, "run_persistent_tasks", run_tasks)
	layer = tmp_path / "layer"
	layer.mkdir()
	queue = tmp_path / "queue"

	result = CliRunner().invoke(
		module.downsample_pyramid,
		[
			str(layer),
			"--queue", str(queue),
			"--max-extend-passes", "2",
			"--initial-parallel", "3",
			"--extend-parallel", "2",
		],
	)

	assert result.exit_code == 0, result.output
	assert len(task_calls) == 3
	assert task_calls[0]["mip"] == 0
	assert task_calls[0]["chunk_size"] == (64, 64, 64)
	assert task_calls[0]["factor"] == (2, 2, 2)
	assert task_calls[1]["mip"] == 1
	assert task_calls[1]["chunk_size"] == (16, 16, 16)
	assert [call[2] for call in queue_calls] == [3, 2, 2]

	state_files = list(queue.rglob("pipeline.json"))
	assert len(state_files) == 1
	state = json.loads(state_files[0].read_text(encoding="utf-8"))
	assert state["complete"] is True
	assert len(state["extensions"]) == 2


def test_downsample_restart_drains_recorded_source_mip(load_module, tmp_path, monkeypatch):
	module = load_module("mctutil/ng/downsample_pyramid.py")
	FakeVolume.max_mip = 4
	dependencies = (FakeVolume, types.SimpleNamespace())
	monkeypatch.setattr(module, "_require_dependencies", lambda: dependencies)
	queue = tmp_path / "queue"
	config = {
		"layer_path": (tmp_path / "layer").resolve().as_uri(),
		"initial_chunk": (64, 64, 64),
		"extend_chunk": (16, 16, 16),
		"max_extend_passes": 1,
		"memory": 10_000_000_000,
		"encoding": "raw",
	}
	state_path = module._pipeline_state_path(queue, config)
	module.write_state(
		state_path,
		{
			"configuration": config,
			"initial_complete": True,
			"extensions": [{"source_mip": 2, "complete": False}],
			"complete": False,
		},
	)
	recorded = []

	def fake_run_pass(_layer, _root, _name, source_mip, *_args):
		recorded.append(source_mip)

	monkeypatch.setattr(module, "run_pass", fake_run_pass)
	module.downsample_volume(
		str(tmp_path / "layer"),
		queue,
		(64, 64, 64),
		(16, 16, 16),
		1,
		16,
		16,
		10_000_000_000,
		"raw",
		3600,
	)

	assert recorded == [2]


def test_partial_file_queue_insertion_retries_the_complete_task_set(tmp_path):
	taskqueue = pytest.importorskip("taskqueue")
	pytest.importorskip("igneous.task_creation")
	module = importlib.import_module("mctutil.shared.persistent_queue")
	queue_path = tmp_path / "partial-insert"
	queue = taskqueue.TaskQueue(
		module.file_queue_url(queue_path),
		progress=False,
	)
	queue.insert(
		[taskqueue.PrintTask("partial")],
		skip_insert_counter=True,
	)

	assert queue.inserted == 0
	assert queue.is_empty() is False

	state = module.run_persistent_tasks(
		queue_path,
		"partial-insert-regression",
		lambda: [
			taskqueue.PrintTask("full-a"),
			taskqueue.PrintTask("full-b"),
		],
		parallel=1,
		lease_seconds=60,
	)

	assert state["status"] == "complete"
	assert state["inserted"] == 2
	assert queue.inserted == 2
	assert queue.completed == 3
	assert queue.is_empty() is True


def test_downsample_executes_real_igneous_file_queue(tmp_path):
	pytest.importorskip("igneous.task_creation")
	pytest.importorskip("taskqueue")
	CloudVolume = pytest.importorskip("cloudvolume").CloudVolume
	module = importlib.import_module("mctutil.ng.downsample_pyramid")
	layer = tmp_path / "layer"
	info = CloudVolume.create_new_info(
		num_channels=1,
		layer_type="image",
		data_type="uint8",
		encoding="raw",
		resolution=[700, 700, 700],
		voxel_offset=[0, 0, 0],
		chunk_size=[4, 4, 4],
		volume_size=[8, 8, 8],
	)
	volume = CloudVolume(
		layer.resolve().as_uri(),
		info=info,
		parallel=False,
		compress=False,
	)
	volume.commit_info()
	volume.commit_provenance()
	volume[:] = np.arange(8 ** 3, dtype=np.uint8).reshape((8, 8, 8, 1))

	module.run_pass(
		str(layer),
		tmp_path / "queue",
		"real",
		0,
		(4, 4, 4),
		"raw",
		100_000_000,
		1,
		60,
	)

	result = CloudVolume(layer.resolve().as_uri(), mip=1, parallel=False)
	assert len(result.info["scales"]) > 1
	assert np.asarray(result[:]).size > 0
