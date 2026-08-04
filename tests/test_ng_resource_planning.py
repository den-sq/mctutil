from __future__ import annotations

import json
from pathlib import Path
import types

import pytest

from mctutil.ng import resource_planning


GIB = 1024 ** 3
TIB = 1024 ** 4


def volume_info(logical_bytes: int, scale_count: int = 6) -> dict:
	return {
		"type": "image",
		"data_type": "uint16",
		"num_channels": 1,
		"scales": [
			{
				"key": str(mip),
				"size": [logical_bytes // 2, 1, 1],
				"encoding": "raw",
			}
			for mip in range(scale_count)
		],
	}


def resources(
	logical_bytes: int,
	capacity: int | None = None,
	memory_capacity: int = 126_000_000_000,
	cpu_limit: int = 64,
):
	return resource_planning.plan_resources(
		volume_info(logical_bytes),
		(0, 3, 5),
		32,
		capacity_override=capacity,
		memory_capacity=memory_capacity,
		cpu_limit=cpu_limit,
	)


@pytest.mark.parametrize(
	("logical_bytes", "capacity"),
	(
		(512 * GIB, 2 * GIB),
		(512 * GIB + 1, 4 * GIB),
		(TIB, 4 * GIB),
		(TIB + 1, 8 * GIB),
	),
)
def test_shard_capacity_tiers_include_their_upper_bound(
	logical_bytes,
	capacity,
):
	info = {
		"data_type": "uint8",
		"num_channels": 1,
		"scales": [{"size": [logical_bytes, 1, 1]}],
	}
	plan = resource_planning.plan_resources(
		info,
		(0, 3, 5),
		32,
		memory_capacity=126_000_000_000,
		cpu_limit=64,
	)

	assert plan.shard_ceiling == capacity


@pytest.mark.parametrize(
	("ceiling", "expected"),
	(
		(2 * GIB, (int(1.6875 * GIB), 2 * GIB, 2 * GIB)),
		(4 * GIB, (int(3.375 * GIB), 4 * GIB, 4 * GIB)),
		(8 * GIB, (int(6.75 * GIB), 8 * GIB, 8 * GIB)),
	),
)
def test_shard_targets_are_largest_power_of_two_chunk_payloads(
	ceiling,
	expected,
):
	plan = resources(GIB, capacity=ceiling)

	assert tuple(shard[3] for shard in plan.shards) == expected
	for _mip, _chunk, count, capacity in plan.shards:
		assert count & (count - 1) == 0
		assert capacity <= ceiling
		assert 2 * capacity > ceiling


def test_logical_size_accounts_for_dtype_and_channels():
	info = {
		"data_type": "uint32",
		"num_channels": 3,
		"scales": [{"size": [10, 20, 30]}],
	}
	plan = resource_planning.plan_resources(
		info,
		(0,),
		1,
		memory_capacity=64 * GIB,
		cpu_limit=1,
	)

	assert plan.logical_bytes == 10 * 20 * 30 * 4 * 3


@pytest.mark.parametrize(
	("value", "expected"),
	(
		("2147483648", 2 * GIB),
		("2GiB", 2 * GIB),
		("3.5 GiB", int(3.5 * GIB)),
	),
)
def test_binary_size_override_accepts_bytes_and_units(value, expected):
	assert resource_planning.parse_size(value) == expected


@pytest.mark.parametrize(
	("capacity", "downsample_workers", "shard_workers"),
	(
		(2 * GIB, 32, 23),
		(4 * GIB, 23, 11),
		(8 * GIB, 11, 5),
	),
)
def test_stage_worker_limits_scale_with_shard_capacity(
	capacity,
	downsample_workers,
	shard_workers,
):
	plan = resources(GIB, capacity=capacity)

	assert plan.downsample_workers == downsample_workers
	assert plan.shard_workers == shard_workers


def test_worker_limit_honors_cpu_user_and_low_memory_bounds():
	user_limited = resource_planning.plan_resources(
		volume_info(GIB),
		(0, 3, 5),
		3,
		memory_capacity=256 * GIB,
		cpu_limit=64,
	)
	cpu_limited = resources(GIB, memory_capacity=256 * GIB, cpu_limit=6)
	low_memory = resources(GIB, memory_capacity=16 * GIB)

	assert user_limited.downsample_workers == 3
	assert user_limited.shard_workers == 3
	assert cpu_limited.downsample_workers == 6
	assert cpu_limited.shard_workers == 6
	assert low_memory.downsample_workers == 6
	assert low_memory.shard_workers == 3
	assert low_memory.warning is None


@pytest.mark.parametrize(
	("memory_capacity", "reserve"),
	(
		(16 * GIB, 4 * GIB),
		(32 * GIB, 8 * GIB),
		(96 * GIB, 24 * GIB),
		(128 * GIB, 24 * GIB),
	),
)
def test_memory_reserve_is_quarter_capacity_capped_at_24_gib(
	memory_capacity,
	reserve,
):
	assert resource_planning.calculate_memory_reserve(memory_capacity) == reserve


@pytest.mark.parametrize(
	("memory_capacity", "downsample_workers", "shard_workers"),
	(
		(16 * GIB, 6, 3),
		(32 * GIB, 12, 6),
	),
)
def test_low_memory_worker_counts(
	memory_capacity,
	downsample_workers,
	shard_workers,
):
	plan = resources(GIB, memory_capacity=memory_capacity)

	assert plan.downsample_workers == downsample_workers
	assert plan.shard_workers == shard_workers


def test_large_machine_uses_cpu_for_downsample_and_memory_for_shard():
	plan = resources(
		GIB,
		capacity=2 * GIB,
		memory_capacity=142 * GIB,
		cpu_limit=32,
	)

	assert plan.memory_reserve == 24 * GIB
	assert plan.downsample_workers == 32
	assert plan.shard_workers == 29


def test_system_resources_uses_capacity_not_current_usage(
	tmp_path,
	monkeypatch,
):
	(tmp_path / "memory.max").write_text(str(64 * GIB), encoding="utf-8")
	(tmp_path / "memory.current").write_text(str(63 * GIB), encoding="utf-8")
	monkeypatch.setattr(
		resource_planning.psutil,
		"virtual_memory",
		lambda: types.SimpleNamespace(total=128 * GIB, available=1 * GIB),
	)
	monkeypatch.setattr(
		resource_planning.os,
		"sched_getaffinity",
		lambda _pid: set(range(8)),
	)

	assert resource_planning.system_resources(tmp_path) == (64 * GIB, 8)


def write_info(path: Path, logical_bytes: int) -> None:
	path.mkdir()
	(path / "info").write_text(
		json.dumps(volume_info(logical_bytes)),
		encoding="utf-8",
	)


def publish_plan(tmp_path, logical_bytes=2 * TIB):
	dataset = tmp_path / "sample"
	dataset.mkdir()
	precomputed = tmp_path / "sample_precomputed"
	write_info(precomputed, logical_bytes)
	return types.SimpleNamespace(
		dataset=dataset,
		layer_type="image",
		prep_input=None,
		prep_output=None,
		precompute_input=dataset,
		precomputed=precomputed,
		staged=tmp_path / "sample_precomputed_sharded_local",
	)


def publish_options(module, **updates):
	options = {
		"selected_stages": module.STAGES,
		"effective_stages": module.STAGES,
		"workers": 32,
		"downsample_memory": 10_000_000_000,
		"shard_capacity": None,
		"memory_capacity": 126_000_000_000,
		"cpu_count": 64,
		"release_queue_leases": True,
		"segmentation_encoding": "compressed_segmentation",
		"voxel_resolution": (700, 700, 700),
		"voxel_offset": (0, 0, 0),
		"stage_include_mip0": True,
		"upload_include_mip0": True,
		"upload_jobs": 6,
		"mesh_at": "local",
		"mesh_mip": 0,
		"mesh_num_lod": 4,
		"mesh_parallel": 16,
		"overwrite_prep": False,
		"no_upload": True,
		"s3_prefix": None,
		"aws_profile": None,
	}
	options.update(updates)
	return options


def test_publish_keeps_full_mip0_workers_and_caps_later_stages(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/ng/publish.py")
	plan = publish_plan(tmp_path)
	options = publish_options(module)
	calls = {}

	def callback_for(stage):
		def callback(**kwargs):
			calls[stage] = kwargs

		return callback

	modules = {
		"mctutil.ng.precompute": types.SimpleNamespace(
			precompute=types.SimpleNamespace(callback=callback_for("precompute")),
		),
		"mctutil.ng.downsample_pyramid": types.SimpleNamespace(
			downsample_pyramid=types.SimpleNamespace(
				callback=callback_for("downsample")
			),
		),
		"mctutil.ng.shard": types.SimpleNamespace(
			shard=types.SimpleNamespace(callback=callback_for("shard")),
		),
	}
	monkeypatch.setattr(module.importlib, "import_module", lambda name: modules[name])

	for stage in ("precompute", "downsample", "shard"):
		module.run_stage(stage, plan, options)

	assert calls["precompute"]["workers"] == 32
	assert calls["downsample"]["initial_parallel"] == 11
	assert calls["downsample"]["extend_parallel"] == 11
	assert calls["downsample"]["memory"] == 10_000_000_000
	assert calls["downsample"]["capacity_override"] == 8 * GIB
	assert calls["shard"]["parallel"] == 5
	assert calls["shard"]["capacity_override"] == 8 * GIB


def test_stage_prediction_uses_actual_shard_payload_and_separate_target(
	load_module,
	tmp_path,
):
	module = load_module("mctutil/ng/publish.py")
	plan = publish_plan(tmp_path, logical_bytes=256 * GIB)
	options = publish_options(module, shard_capacity=2 * GIB)

	downsample = module.stage_resource_prediction("downsample", plan, options)
	shard = module.stage_resource_prediction("shard", plan, options)

	resources = module.dataset_resources(plan, options, mips=(0, 3, 5))
	assert resources is not None
	assert downsample.shard_capacity == max(
		entry[3] for entry in resources.shards
	)
	assert downsample.reserve == 24 * GIB
	assert downsample.capacity_multiplier == 1
	assert downsample.downsample_memory == 10_000_000_000
	assert shard.shard_capacity is not None
	assert shard.reserve == 24 * GIB
	assert shard.capacity_multiplier == 2
	assert shard.downsample_memory is None


def test_shard_resume_fingerprint_ignores_workers_but_tracks_capacity(
	load_module,
	tmp_path,
):
	module = load_module("mctutil/ng/publish.py")
	plan = publish_plan(tmp_path, logical_bytes=256 * GIB)
	options = publish_options(module)

	baseline = module.stage_configuration("shard", plan, options)
	fewer_workers = module.stage_configuration(
		"shard",
		plan,
		{
			**options,
			"workers": 2,
			"memory_capacity": 32 * GIB,
			"cpu_count": 2,
		},
	)
	different_capacity = module.stage_configuration(
		"shard",
		plan,
		{**options, "shard_capacity": 4 * GIB},
	)

	assert fewer_workers == baseline
	assert different_capacity != baseline
