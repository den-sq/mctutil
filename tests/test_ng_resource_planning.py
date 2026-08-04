from __future__ import annotations

import json
from pathlib import Path
import types

import pytest

from mctutil.ng import resource_planning


GIB = 1024 ** 3
TIB = 1024 ** 4


def volume_info(logical_bytes: int, scale_count: int = 6) -> dict:
	voxel_count = logical_bytes // 2
	return {
		"type": "image",
		"data_type": "uint16",
		"num_channels": 1,
		"scales": [
			{
				"key": str(mip),
				"size": [voxel_count, 1, 1],
				"encoding": "raw",
			}
			for mip in range(scale_count)
		],
	}


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
	assert resource_planning.select_shard_capacity(logical_bytes) == capacity


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
	plan = resource_planning.plan_shard_capacities(
		volume_info(1 * GIB),
		(0, 3, 5),
		capacity_override=ceiling,
	)

	assert tuple(scale.capacity for scale in plan.scales) == expected
	for scale in plan.scales:
		assert scale.chunks_per_shard & (scale.chunks_per_shard - 1) == 0
		assert scale.capacity <= ceiling
		assert 2 * scale.capacity > ceiling


def test_logical_size_accounts_for_dtype_and_channels():
	info = {
		"data_type": "uint32",
		"num_channels": 3,
		"scales": [{"size": [10, 20, 30]}],
	}

	assert resource_planning.logical_mip0_bytes(info) == 10 * 20 * 30 * 4 * 3


@pytest.mark.parametrize(
	("value", "expected"),
	(
		("2147483648", 2 * GIB),
		("2GiB", 2 * GIB),
		("3.5 GiB", int(3.5 * GIB)),
	),
)
def test_binary_size_override_accepts_bytes_and_units(value, expected):
	assert resource_planning.parse_binary_size(value) == expected


@pytest.mark.parametrize(
	("capacity", "workers"),
	(
		(2 * GIB, 16),
		(4 * GIB, 8),
		(8 * GIB, 4),
	),
)
def test_worker_limit_scales_conservatively_with_shard_capacity(
	capacity,
	workers,
):
	plan = resource_planning.plan_worker_limit(
		requested_limit=32,
		capacity_budget=capacity,
		available_ram=126_000_000_000,
		cpu_limit=64,
	)

	assert plan.workers == workers


def test_worker_limit_honors_cpu_user_and_low_memory_bounds():
	assert resource_planning.plan_worker_limit(
		3,
		2 * GIB,
		available_ram=256 * GIB,
		cpu_limit=64,
	).workers == 3
	assert resource_planning.plan_worker_limit(
		32,
		2 * GIB,
		available_ram=256 * GIB,
		cpu_limit=6,
	).workers == 6

	low_memory = resource_planning.plan_worker_limit(
		32,
		2 * GIB,
		available_ram=16 * GIB,
		cpu_limit=64,
	)
	assert low_memory.workers == 1
	assert low_memory.warning is not None


def test_cgroup_available_memory_is_limit_minus_current(tmp_path):
	(tmp_path / "memory.max").write_text(str(64 * GIB), encoding="utf-8")
	(tmp_path / "memory.current").write_text(str(16 * GIB), encoding="utf-8")

	assert resource_planning.cgroup_available_ram(tmp_path) == 48 * GIB


def test_effective_available_memory_uses_lower_cgroup_value(monkeypatch):
	monkeypatch.setattr(
		resource_planning.psutil,
		"virtual_memory",
		lambda: types.SimpleNamespace(available=128 * GIB),
	)
	monkeypatch.setattr(
		resource_planning,
		"cgroup_available_ram",
		lambda: 48 * GIB,
	)

	assert resource_planning.system_available_ram() == 48 * GIB


def write_info(path: Path, logical_bytes: int) -> None:
	path.mkdir()
	(path / "info").write_text(
		json.dumps(volume_info(logical_bytes)),
		encoding="utf-8",
	)


def publish_plan(module, tmp_path, logical_bytes=2 * TIB):
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
		"available_ram": 126_000_000_000,
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
	plan = publish_plan(module, tmp_path)
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
	assert calls["downsample"]["initial_parallel"] == 4
	assert calls["downsample"]["extend_parallel"] == 4
	assert calls["downsample"]["memory"] == 10_000_000_000
	assert calls["downsample"]["capacity_override"] == 8 * GIB
	assert calls["shard"]["parallel"] == 4
	assert calls["shard"]["capacity_override"] == 8 * GIB


def test_shard_resume_fingerprint_ignores_workers_but_tracks_capacity(
	load_module,
	tmp_path,
):
	module = load_module("mctutil/ng/publish.py")
	plan = publish_plan(module, tmp_path, logical_bytes=256 * GIB)
	options = publish_options(module)

	baseline = module.stage_configuration("shard", plan, options)
	fewer_workers = module.stage_configuration(
		"shard",
		plan,
		{
			**options,
			"workers": 2,
			"available_ram": 32 * GIB,
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
