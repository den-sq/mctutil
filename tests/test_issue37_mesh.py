from __future__ import annotations

import types

from click.testing import CliRunner


class FakeTaskQueue:
	def __init__(self, parallel):
		self.parallel = parallel
		self.events = []

	def insert(self, tasks):
		self.events.append(("insert", tasks))

	def execute(self):
		self.events.append(("execute",))


def test_build_mesh_runs_forge_then_multires_merge(load_module, monkeypatch):
	module = load_module("mctutil/shared/mesh.py")
	calls = []
	queues = []

	def create_meshing_tasks(layer_path, mip, **kwargs):
		calls.append(("forge", layer_path, mip, kwargs))
		return ["forge-task"]

	def create_unsharded_multires_mesh_tasks(layer_path, **kwargs):
		calls.append(("merge", layer_path, kwargs))
		return ["merge-task"]

	class RecordingTaskQueue(FakeTaskQueue):
		def __init__(self, parallel):
			super().__init__(parallel)
			queues.append(self)

	task_creation = types.SimpleNamespace(
		create_meshing_tasks=create_meshing_tasks,
		create_unsharded_multires_mesh_tasks=create_unsharded_multires_mesh_tasks,
	)
	monkeypatch.setattr(module, "_require_mesh_dependencies", lambda: (RecordingTaskQueue, task_creation))
	monkeypatch.setattr(module, "configure_aws_profile", lambda _profile, _bucket: "chenglab")
	monkeypatch.setattr(module, "preflight_s3_info", lambda _path, _profile: {"scales": [{}]})
	monkeypatch.setattr(module.log, "write", lambda *_args, **_kwargs: None)

	module.build_mesh(
		"precomputed://s3://bucket/layer",
		mip=2,
		num_lod=3,
		parallel=6,
		shape=(64, 32, 16),
		simplification=False,
		max_simplification_error=12,
		mesh_dir="custom-mesh",
		cdn_cache=True,
		dust_threshold=25,
		object_ids=(4, 9),
		fill_missing=True,
		encoding="draco",
		spatial_index=False,
		magnitude=2,
		vertex_quantization_bits=10,
		min_chunk_size=(32, 32, 16),
	)

	assert len(queues) == 1
	assert queues[0].parallel == 6
	assert queues[0].events == [
		("insert", ["forge-task"]),
		("execute",),
		("insert", ["merge-task"]),
		("execute",),
	]

	forge = calls[0]
	assert forge[:3] == ("forge", "precomputed://s3://bucket/layer", 2)
	assert forge[3]["shape"] == (64, 32, 16)
	assert forge[3]["simplification"] is False
	assert forge[3]["max_simplification_error"] == 12
	assert forge[3]["mesh_dir"] == "custom-mesh"
	assert forge[3]["cdn_cache"] is True
	assert forge[3]["dust_threshold"] == 25
	assert forge[3]["object_ids"] == [4, 9]
	assert forge[3]["fill_missing"] is True
	assert forge[3]["encoding"] == "draco"
	assert forge[3]["spatial_index"] is False
	assert forge[3]["sharded"] is False

	merge = calls[1]
	assert merge[:2] == ("merge", "precomputed://s3://bucket/layer")
	assert merge[2] == {
		"num_lod": 3,
		"magnitude": 2,
		"mesh_dir": "custom-mesh",
		"vertex_quantization_bits": 10,
		"min_chunk_size": (32, 32, 16),
	}


def test_build_mesh_dry_run_does_not_load_or_call_igneous(load_module, monkeypatch):
	module = load_module("mctutil/shared/mesh.py")
	monkeypatch.setattr(
		module,
		"_require_mesh_dependencies",
		lambda: (_ for _ in ()).throw(AssertionError("dependencies loaded during dry-run")),
	)
	monkeypatch.setattr(module.log, "write", lambda *_args, **_kwargs: None)

	module.build_mesh("file:///tmp/layer", parallel=2, execute=False)


def test_build_mesh_uses_durable_queues_when_requested(
	load_module,
	monkeypatch,
	tmp_path,
):
	module = load_module("mctutil/shared/mesh.py")
	calls = []
	task_creation = types.SimpleNamespace(
		create_meshing_tasks=lambda *_args, **_kwargs: ["forge"],
		create_unsharded_multires_mesh_tasks=lambda *_args, **_kwargs: ["merge"],
	)
	monkeypatch.setattr(
		module,
		"_require_mesh_dependencies",
		lambda: (FakeTaskQueue, task_creation),
	)

	def run_tasks(
		queue,
		fingerprint,
		tasks_factory,
		parallel,
		lease_seconds,
		**kwargs,
	):
		calls.append(
			(
				queue,
				fingerprint,
				list(tasks_factory()),
				parallel,
				lease_seconds,
				kwargs,
			)
		)

	monkeypatch.setattr(module, "run_persistent_tasks", run_tasks)
	monkeypatch.setattr(module.log, "write", lambda *_args, **_kwargs: None)

	module.build_mesh(
		"file:///tmp/sharded-layer",
		parallel=3,
		queue_dir=tmp_path / "queue",
		lease_seconds=120,
	)

	assert [call[2] for call in calls] == [["forge"], ["merge"]]
	assert [call[3:5] for call in calls] == [(3, 120), (3, 120)]
	assert [call[5]["progress_label"] for call in calls] == [
		"Mesh Forge",
		"Mesh Merge",
	]
	assert calls[0][0].parts[-2] == "forge"
	assert calls[1][0].parts[-2] == "merge"


def test_mesh_command_forwards_configured_options(load_module, monkeypatch):
	module = load_module("mctutil/mesh/build.py")
	recorded = {}
	monkeypatch.setattr(module, "build_mesh", lambda layer_path, **kwargs: recorded.update(
		layer_path=layer_path,
		**kwargs,
	))

	result = CliRunner().invoke(
		module.mesh,
		[
			"--parallel", "3",
			"--mip", "2",
			"--num-lod", "1",
			"--shape", "64,32,16",
			"--skip-simplify",
			"--max-error", "8",
			"--mesh-dir", "mesh-custom",
			"--cdn-cache",
			"--dust-threshold", "20",
			"--object-id", "4",
			"--object-id", "9",
			"--fill-missing",
			"--encoding", "draco",
			"--skip-spatial-index",
			"--magnitude", "2",
			"--vertex-quantization-bits", "10",
			"--min-chunk-size", "32,32,16",
			"--dry-run",
			"file:///tmp/layer",
		],
	)

	assert result.exit_code == 0, result.output
	assert recorded == {
		"layer_path": "file:///tmp/layer",
		"mip": 2,
		"num_lod": 1,
		"parallel": 3,
		"shape": (64, 32, 16),
		"simplification": False,
		"max_simplification_error": 8,
		"mesh_dir": "mesh-custom",
		"cdn_cache": True,
		"dust_threshold": 20,
		"object_ids": (4, 9),
		"fill_missing": True,
		"encoding": "draco",
		"spatial_index": False,
		"magnitude": 2,
		"vertex_quantization_bits": 10,
		"min_chunk_size": (32, 32, 16),
		"execute": False,
		"aws_profile": None,
	}


def test_s3upload_routes_meshing_through_shared_helper(load_module, monkeypatch, tmp_path):
	module = load_module("mctutil/transport/s3upload.py")
	source_dir = tmp_path / "input"
	source_dir.mkdir()
	recorded = {}

	monkeypatch.setattr(module, "upload_folder_to_s3_parallel", lambda *_args, **_kwargs: None)
	monkeypatch.setattr(module, "build_mesh", lambda layer_path, **kwargs: recorded.update(
		layer_path=layer_path,
		**kwargs,
	))

	result = CliRunner().invoke(
		module.s3upload,
		[
			"--bucket-prefix", "prefix",
			"--bucket-name", "bucket",
			"--aws-profile", "test-profile",
			"--process-count", "3",
			"--mesh",
			"--dry-run",
			str(source_dir),
			"target",
		],
	)

	assert result.exit_code == 0, result.output
	assert recorded == {
		"layer_path": "precomputed://s3://bucket/prefix/target",
		"mip": 0,
		"num_lod": 4,
		"parallel": 1,
		"execute": False,
		"aws_profile": "test-profile",
	}
