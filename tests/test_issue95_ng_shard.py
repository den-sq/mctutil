from __future__ import annotations

import importlib
import json
import types

from click.testing import CliRunner
import numpy as np
import pytest

from mctutil.ng.resource_planning import plan_resources


def fixed_resource_plan(info, mips, requested, **kwargs):
	return plan_resources(
		info,
		mips,
		requested,
		capacity_override=kwargs.get("capacity_override"),
		memory_capacity=128 * 1024 ** 3,
		cpu_limit=requested,
		low_chunk=kwargs.get("low_chunk", (96, 96, 96)),
		mid_chunk=kwargs.get("mid_chunk", (64, 64, 64)),
		high_chunk=kwargs.get("high_chunk", (16, 16, 16)),
	)


class FakeVolume:
	def __init__(self, *_args, **_kwargs):
		self.info = {
			"type": "image",
			"data_type": "uint16",
			"num_channels": 1,
			"scales": [
				{
					"key": f"{700 * 2 ** mip}_{700 * 2 ** mip}_{700 * 2 ** mip}",
					"encoding": "raw",
					"size": [64, 64, 64],
				}
				for mip in range(6)
			],
		}


def test_shard_dry_run_reports_per_mip_mapping(
	load_module,
	tmp_path,
	monkeypatch,
	verbose_logging,
):
	module = load_module("mctutil/ng/shard.py")
	monkeypatch.setattr(
		module,
		"_require_dependencies",
		lambda: (FakeVolume, types.SimpleNamespace()),
	)
	monkeypatch.setattr(module, "plan_resources", fixed_resource_plan)
	source = tmp_path / "source"
	source.mkdir()
	destination = tmp_path / "staged"

	result = CliRunner().invoke(
		module.shard,
		[str(source), str(destination), "--dry-run"],
	)

	assert result.exit_code == 0, result.output
	assert "Shard targets:" in result.output
	assert "mips 0-2: 1.6875 GiB at (96, 96, 96)" in result.output
	assert "mips 3-4: 2 GiB at (64, 64, 64)" in result.output
	assert "mips 5: 2 GiB at (16, 16, 16)" in result.output
	assert "MIP 0=" in result.output
	assert "shards=2 GiB ceiling" in result.output
	assert "workers=8/8 requested" in result.output
	assert not destination.exists()


def test_shard_excludes_mip0_and_uses_one_durable_queue(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/ng/shard.py")
	task_calls = []
	queue_calls = []

	def create_sharded(*_args, **kwargs):
		task_calls.append(kwargs)
		return [f"mip-{kwargs['mip']}"]

	task_creation = types.SimpleNamespace(
		create_image_shard_transfer_tasks=create_sharded,
	)
	monkeypatch.setattr(
		module,
		"_require_dependencies",
		lambda: (FakeVolume, task_creation),
	)
	monkeypatch.setattr(module, "plan_resources", fixed_resource_plan)

	def run_tasks(
		queue_path,
		fingerprint,
		tasks_factory,
		parallel,
		lease_seconds,
		**kwargs,
	):
		queue_calls.append(
			(queue_path, fingerprint, parallel, lease_seconds, kwargs)
		)
		assert list(tasks_factory()) == ["mip-1", "mip-2"]
		return {"status": "complete"}

	monkeypatch.setattr(module, "run_persistent_tasks", run_tasks)
	source = tmp_path / "source"
	source.mkdir()
	destination = tmp_path / "staged"
	queue = tmp_path / "queue"

	result = CliRunner().invoke(
		module.shard,
		[
			str(source),
			str(destination),
			"--mips", "0,1,2",
			"--exclude-mip0",
			"--queue", str(queue),
			"--parallel", "3",
			"--preserve-leases",
		],
	)

	assert result.exit_code == 0, result.output
	assert [call["mip"] for call in task_calls] == [1, 2]
	assert all(call["fill_missing"] is True for call in task_calls)
	assert all(call["compress"] == "gzip" for call in task_calls)
	assert len(queue_calls) == 1
	assert queue_calls[0][2] == 3
	assert queue_calls[0][4]["release_leases"] is False
	assert queue_calls[0][4]["progress_label"] == "Shard Tasks"
	state_files = list(queue.rglob("pipeline.json"))
	assert len(state_files) == 1
	state = json.loads(state_files[0].read_text(encoding="utf-8"))
	assert state["completed_mips"] == [1, 2]
	assert state["complete"] is True


def test_shard_legacy_transfer_uses_gzip(load_module, monkeypatch):
	module = load_module("mctutil/ng/shard.py")
	task_calls = []

	def create_transfer(
		_source,
		_destination,
		*,
		mip,
		chunk_size,
		fill_missing,
		encoding,
		memory_target,
		compress,
		sharded,
	):
		task_calls.append(
			{
				"mip": mip,
				"chunk_size": chunk_size,
				"fill_missing": fill_missing,
				"encoding": encoding,
				"memory_target": memory_target,
				"compress": compress,
				"sharded": sharded,
			}
		)
		return ["legacy-task"]

	task_creation = types.SimpleNamespace(create_transfer_tasks=create_transfer)
	monkeypatch.setattr(
		module,
		"_require_dependencies",
		lambda: (FakeVolume, task_creation),
	)

	tasks = module.create_shard_tasks(
		"source",
		"destination",
		2,
		(32, 32, 32),
		"raw",
		100_000_000,
	)

	assert tasks == ["legacy-task"]
	assert task_calls == [
		{
			"mip": 2,
			"chunk_size": (32, 32, 32),
			"fill_missing": True,
			"encoding": "raw",
			"memory_target": 100_000_000,
			"compress": "gzip",
			"sharded": True,
		}
	]


def test_detect_mips_matches_source_key_and_directory_rules(load_module, tmp_path):
	module = load_module("mctutil/ng/shard.py")
	source = tmp_path / "source"
	source.mkdir()
	(source / "700_700_700").mkdir()
	(source / "4").mkdir()
	info = {
		"scales": [
			{"key": "700_700_700"},
			{"key": "3"},
		],
	}

	assert module.detect_mips(str(source), info) == (0, 3, 4)


def test_destination_completion_requires_sharding_and_files(load_module, tmp_path):
	module = load_module("mctutil/ng/shard.py")
	destination = tmp_path / "staged"
	scale = destination / "700_700_700"
	scale.mkdir(parents=True)
	(destination / "info").write_text(
		json.dumps(
			{
				"scales": [
					{
						"key": "700_700_700",
						"sharding": {"@type": "neuroglancer_uint64_sharded_v1"},
					}
				]
			}
		),
		encoding="utf-8",
	)
	assert module.destination_scale_complete(str(destination), 0) is False
	(scale / "0.shard").write_bytes(b"data")
	assert module.destination_scale_complete(str(destination), 0) is True


def test_shard_executes_real_igneous_transfer(tmp_path, monkeypatch):
	pytest.importorskip("igneous.task_creation")
	pytest.importorskip("taskqueue")
	CloudVolume = pytest.importorskip("cloudvolume").CloudVolume
	cloudfiles_module = pytest.importorskip("cloudfiles.cloudfiles")
	locks = tmp_path / "cloudfiles-locks"
	locks.mkdir()
	monkeypatch.setattr(
		cloudfiles_module,
		"CLOUD_FILES_LOCK_DIR",
		str(locks),
	)
	module = importlib.import_module("mctutil.ng.shard")
	source = tmp_path / "source"
	destination = tmp_path / "staged"
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
	source_volume = CloudVolume(
		source.resolve().as_uri(),
		info=info,
		parallel=False,
		compress=False,
	)
	source_volume.commit_info()
	source_volume.commit_provenance()
	data = np.arange(8 ** 3, dtype=np.uint8).reshape((8, 8, 8, 1))
	source_volume[:] = data
	resources = module.plan_resources(
		source_volume.info,
		(0,),
		1,
		capacity_override=100_000_000,
		memory_capacity=128 * 1024 ** 3,
		cpu_limit=1,
		low_chunk=(4, 4, 4),
		mid_chunk=(4, 4, 4),
		high_chunk=(4, 4, 4),
	)

	module.shard_volume(
		str(source),
		str(destination),
		tmp_path / "queue",
		resources,
		"raw",
		1,
		60,
	)

	staged = CloudVolume(destination.resolve().as_uri(), parallel=False)
	assert staged.scale["sharding"]["@type"] == "neuroglancer_uint64_sharded_v1"
	assert staged.scale["sharding"]["data_encoding"] == "gzip"
	assert np.array_equal(np.asarray(staged[:]), data)
	assert module.destination_scale_complete(str(destination), 0) is True


def test_shard_preserves_absolute_positions_for_noncontiguous_mips(
	tmp_path,
	monkeypatch,
):
	pytest.importorskip("igneous.task_creation")
	pytest.importorskip("taskqueue")
	CloudVolume = pytest.importorskip("cloudvolume").CloudVolume
	cloudfiles_module = pytest.importorskip("cloudfiles.cloudfiles")
	locks = tmp_path / "cloudfiles-locks"
	locks.mkdir()
	monkeypatch.setattr(
		cloudfiles_module,
		"CLOUD_FILES_LOCK_DIR",
		str(locks),
	)
	module = importlib.import_module("mctutil.ng.shard")
	source = tmp_path / "source"
	destination = tmp_path / "staged"
	info = CloudVolume.create_new_info(
		num_channels=1,
		layer_type="image",
		data_type="uint8",
		encoding="raw",
		resolution=[700, 700, 700],
		voxel_offset=[0, 0, 0],
		chunk_size=[4, 4, 4],
		volume_size=[16, 16, 16],
	)
	source_volume = CloudVolume(
		source.resolve().as_uri(),
		info=info,
		parallel=False,
		compress=False,
	)
	source_volume.commit_info()
	source_volume.commit_provenance()
	data = np.arange(16 ** 3, dtype=np.uint8).reshape((16, 16, 16, 1))
	source_volume[:] = data
	for mip, factor in enumerate((2, 4, 8), start=1):
		chunk = min(4, 16 // factor)
		source_volume.add_scale(
			(factor, factor, factor),
			encoding="raw",
			chunk_size=(chunk, chunk, chunk),
		)
		source_volume.commit_info()
		mip_volume = CloudVolume(
			source.resolve().as_uri(),
			mip=mip,
			parallel=False,
			compress=False,
		)
		mip_volume[:] = data[::factor, ::factor, ::factor, :]
	resources = module.plan_resources(
		source_volume.info,
		(1, 3),
		1,
		capacity_override=100_000_000,
		memory_capacity=128 * 1024 ** 3,
		cpu_limit=1,
		low_chunk=(4, 4, 4),
		mid_chunk=(2, 2, 2),
		high_chunk=(2, 2, 2),
	)

	module.shard_volume(
		str(source),
		str(destination),
		tmp_path / "queue",
		resources,
		"raw",
		1,
		60,
	)

	staged = CloudVolume(destination.resolve().as_uri(), parallel=False)
	assert len(staged.info["scales"]) == 4
	assert staged.info["scales"][0].get("sharding") is None
	assert staged.info["scales"][2].get("sharding") is None
	for mip, factor in ((1, 2), (3, 8)):
		scale = staged.info["scales"][mip]
		assert scale["sharding"]["@type"] == "neuroglancer_uint64_sharded_v1"
		mip_volume = CloudVolume(
			destination.resolve().as_uri(),
			mip=mip,
			parallel=False,
		)
		assert np.array_equal(
			np.asarray(mip_volume[:]),
			data[::factor, ::factor, ::factor, :],
		)
		assert module.destination_scale_complete(str(destination), mip) is True
