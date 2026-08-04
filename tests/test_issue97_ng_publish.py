from __future__ import annotations

import importlib
import json
from pathlib import Path
import types

from click.testing import CliRunner
import numpy as np
import pytest


def make_dataset(root, name="sample"):
	dataset = root / name
	dataset.mkdir()
	(dataset / "slice_1.tif").write_bytes(b"one")
	(dataset / "slice_2.tif").write_bytes(b"two")
	return dataset


def test_stage_aware_extra_resolution(load_module):
	module = load_module("mctutil/ng/publish.py")

	assert module.required_extras(
		module.resolve_stage_range("prep", "precompute")
	) == ("ng",)
	assert module.required_extras(
		module.resolve_stage_range("upload", None)
	) == ("aws",)
	assert module.required_extras(
		module.effective_stages(
			module.resolve_stage_range("prep", None),
			True,
		)
	) == ("ng", "mesh")
	assert module.required_extras(
		module.resolve_stage_range("prep", None)
	) == ("ng", "mesh", "aws")


def test_mesh_preflight_checks_cloudvolume(load_module, monkeypatch):
	module = load_module("mctutil/ng/publish.py")
	monkeypatch.setattr(
		module,
		"module_available",
		lambda name: name != "cloudvolume",
	)

	assert module.missing_dependencies(("mesh",)) == {
		"mesh": ("cloudvolume",),
	}


def test_mesh_extra_directly_declares_cloudvolume():
	pyproject = Path("pyproject.toml").read_text(encoding="utf-8").lower()
	mesh_extra = pyproject.split("mesh = [", 1)[1].split("\n]\n", 1)[0]

	assert '"cloud-volume"' in mesh_extra


def test_run_stage_dispatches_sibling_commands_by_keyword(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/ng/publish.py")
	calls = {}

	def callback_for(name):
		def callback(*args, **kwargs):
			calls[name] = (args, kwargs)

		return callback

	modules = {
		"mctutil.transform.memmap_prep": types.SimpleNamespace(
			memmap_prep=types.SimpleNamespace(callback=callback_for("prep")),
		),
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
		"mctutil.transport.s3upload": types.SimpleNamespace(
			upload_sharded_tree=callback_for("upload"),
		),
		"mctutil.shared.mesh": types.SimpleNamespace(
			build_mesh=callback_for("mesh"),
		),
	}
	monkeypatch.setattr(
		module.importlib,
		"import_module",
		lambda name: modules[name],
	)
	monkeypatch.setattr(
		module,
		"dataset_resources",
		lambda *_args, **_kwargs: types.SimpleNamespace(
			shard_ceiling=2 * 1024 ** 3,
			downsample_workers=2,
			shard_workers=2,
		),
	)
	dataset = tmp_path / "cell_labels"
	dataset.mkdir()
	plan = types.SimpleNamespace(
		dataset=dataset,
		layer_type="segmentation",
		prep_input=dataset / "input.tif",
		prep_output=dataset / "memmap.tif",
		precompute_input=dataset / "memmap.tif",
		precomputed=tmp_path / "precomputed",
		staged=tmp_path / "staged",
	)
	options = {
		"effective_stages": module.STAGES,
		"selected_stages": module.STAGES,
		"aws_profile": "test-profile",
		"workers": 2,
		"downsample_memory": 123,
		"shard_capacity": None,
		"memory_capacity": 128 * 1024 ** 3,
		"cpu_count": 32,
		"release_queue_leases": True,
		"segmentation_encoding": "compressed_segmentation",
		"voxel_resolution": (700, 800, 900),
		"voxel_offset": (10, 20, 30),
		"stage_include_mip0": True,
		"upload_include_mip0": True,
		"upload_jobs": 3,
		"mesh_at": "s3",
		"mesh_mip": 0,
		"mesh_num_lod": 4,
		"mesh_parallel": 5,
		"overwrite_prep": False,
		"s3_prefix": "s3://bucket/prefix",
	}

	for stage in module.STAGES:
		module.run_stage(stage, plan, options)

	assert set(calls) == set(module.STAGES)
	for stage, (args, kwargs) in calls.items():
		assert args == (), stage
		assert kwargs["execute"] is True
	assert calls["prep"][1]["input_tif"] == plan.prep_input
	assert calls["precompute"][1]["voxel_offset"] == (10, 20, 30)
	assert calls["downsample"][1]["layer_path"] == str(plan.precomputed)
	assert calls["downsample"][1]["force"] is False
	assert calls["downsample"][1]["release_leases"] is True
	assert calls["downsample"][1]["initial_parallel"] == 2
	assert calls["downsample"][1]["memory"] == 123
	assert calls["downsample"][1]["capacity_override"] == 2 * 1024 ** 3
	assert calls["shard"][1]["destination"] == str(plan.staged)
	assert calls["shard"][1]["release_leases"] is True
	assert calls["shard"][1]["capacity_override"] == 2 * 1024 ** 3
	assert calls["shard"][1]["parallel"] == 2
	assert calls["upload"][1]["aws_profile"] == "test-profile"
	assert calls["mesh"][1]["aws_profile"] == "test-profile"


def test_publish_dry_run_reports_metadata_and_never_writes(
	load_module,
	tmp_path,
	monkeypatch,
	verbose_logging,
):
	module = load_module("mctutil/ng/publish.py")
	root = tmp_path / "root"
	root.mkdir()
	dataset = make_dataset(root)
	monkeypatch.setattr(module, "module_available", lambda _name: False)

	result = CliRunner().invoke(
		module.publish,
		[
			str(root),
			"--s3-prefix", "s3://bucket/prefix",
			"--dry-run",
		],
	)

	assert result.exit_code == 0, result.output
	assert "Required extras: [ng], [mesh], [aws]" in result.output
	assert "Missing [ng]" in result.output
	assert "Voxel resolution (nm): (700, 700, 700)" in result.output
	assert "Voxel offset: (0, 0, 0)" in result.output
	assert "sample (image)" in result.output
	assert "prep: omitted" in result.output
	assert "mesh: omitted" in result.output
	assert not (dataset / ".mctutil_ng_publish.json").exists()


@pytest.mark.parametrize(
	"mesh_arguments",
	(
		("--mesh-at", "local"),
		("--upload-exclude-mip0",),
	),
)
def test_publish_warns_when_upload_precedes_local_mesh(
	load_module,
	tmp_path,
	monkeypatch,
	mesh_arguments,
):
	module = load_module("mctutil/ng/publish.py")
	root = tmp_path / "root"
	root.mkdir()
	make_dataset(root, "cell_labels")
	monkeypatch.setattr(module, "module_available", lambda _name: True)

	result = CliRunner().invoke(
		module.publish,
		[
			str(root),
			"--s3-prefix", "s3://bucket/prefix",
			"--dry-run",
			*mesh_arguments,
		],
	)

	assert result.exit_code == 0, result.output
	assert "Warning: local mesh for cell_labels runs after upload" in result.output
	assert "will not be present in S3" in result.output


def test_missing_dependencies_abort_before_state_write(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/ng/publish.py")
	root = tmp_path / "root"
	root.mkdir()
	dataset = make_dataset(root)
	monkeypatch.setattr(module, "module_available", lambda _name: False)

	result = CliRunner().invoke(
		module.publish,
		[
			str(root),
			"--s3-prefix", "s3://bucket/prefix",
		],
	)

	assert result.exit_code != 0
	assert "pip install -e '.[ng,mesh,aws]'" in result.output
	assert not (dataset / ".mctutil_ng_publish.json").exists()


def test_publish_continues_when_resource_accounting_is_unavailable(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/ng/publish.py")
	root = tmp_path / "root"
	root.mkdir()
	make_dataset(root)
	calls = []

	class UnavailableMonitor:
		enabled = False

		def __enter__(self):
			return self

		def __exit__(self, *_args):
			return False

	monkeypatch.setattr(module, "module_available", lambda _name: True)
	monkeypatch.setattr(module, "PublishResourceMonitor", UnavailableMonitor)
	monkeypatch.setattr(
		module,
		"run_stage",
		lambda stage, _plan, _options: calls.append(stage),
	)

	result = CliRunner().invoke(
		module.publish,
		[str(root), "--stop-after", "precompute"],
	)

	assert result.exit_code == 0, result.output
	assert calls == ["precompute"]


def test_no_upload_records_omitted_separately_from_complete(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/ng/publish.py")
	root = tmp_path / "root"
	root.mkdir()
	dataset = make_dataset(root, "cell_labels")
	monkeypatch.setattr(module, "module_available", lambda _name: True)
	calls = []
	monkeypatch.setattr(
		module,
		"run_stage",
		lambda stage, _plan, _options: calls.append(stage),
	)

	result = CliRunner().invoke(
		module.publish,
		[str(root), "--no-upload"],
	)

	assert result.exit_code == 0, result.output
	assert calls == ["precompute", "downsample", "shard", "mesh"]
	state = json.loads(
		(dataset / ".mctutil_ng_publish.json").read_text(encoding="utf-8")
	)
	assert state["stages"]["prep"]["status"] == "omitted"
	assert state["stages"]["upload"]["status"] == "omitted"
	assert state["stages"]["mesh"]["status"] == "complete"
	assert "upload" in state["requested_stages"]


def test_start_at_upload_is_aws_only_and_resumable(
	load_module,
	tmp_path,
	monkeypatch,
	verbose_logging,
):
	module = load_module("mctutil/ng/publish.py")
	root = tmp_path / "root"
	root.mkdir()
	dataset = root / "sample"
	dataset.mkdir()
	staged = root / "sample_precomputed_sharded_local"
	staged.mkdir()
	(staged / "info").write_text('{"scales":[{}]}', encoding="utf-8")
	monkeypatch.setattr(module, "module_available", lambda name: name == "boto3")
	monkeypatch.setattr(
		module,
		"system_resources",
		lambda: (32 * 1024 ** 3, 6),
	)
	calls = []
	monkeypatch.setattr(
		module,
		"run_stage",
		lambda stage, _plan, options: calls.append((stage, options["workers"])),
	)

	result = CliRunner().invoke(
		module.publish,
		[
			str(root),
			"--start-at", "upload",
			"--s3-prefix", "s3://bucket/prefix",
		],
	)

	assert result.exit_code == 0, result.output
	assert "Selected stages: upload" in result.output
	assert "Required extras: [aws]" in result.output
	assert calls == [("upload", 6)]


def test_publish_real_prep_precompute_and_resume(tmp_path, verbose_logging):
	pytest.importorskip("zarr")
	tifffile = pytest.importorskip("tifffile")
	CloudVolume = pytest.importorskip("cloudvolume").CloudVolume
	module = importlib.import_module("mctutil.ng.publish")
	root = tmp_path / "root"
	dataset = root / "sample"
	dataset.mkdir(parents=True)
	tifffile.imwrite(
		dataset / "volume.tif",
		np.arange(2 * 4 * 4, dtype=np.uint16).reshape((2, 4, 4)),
		photometric="minisblack",
	)
	arguments = [
		str(root),
		"--stop-after", "precompute",
		"--workers", "1",
		"--voxel-resolution", "700,800,900",
		"--voxel-offset", "10,20,30",
	]

	result = CliRunner().invoke(module.publish, arguments)

	assert result.exit_code == 0, result.output
	volume = CloudVolume((root / "sample_precomputed").resolve().as_uri())
	assert volume.resolution.tolist() == [700, 800, 900]
	assert volume.voxel_offset.tolist() == [10, 20, 30]
	state = json.loads(
		(dataset / ".mctutil_ng_publish.json").read_text(encoding="utf-8")
	)
	assert state["stages"]["prep"]["status"] == "complete"
	assert state["stages"]["precompute"]["status"] == "complete"

	resume = CliRunner().invoke(module.publish, arguments)
	assert resume.exit_code == 0, resume.output
	assert "prep: complete" in resume.output
	assert "precompute: complete" in resume.output
	assert "Running prep" not in resume.output
	assert "Running precompute" not in resume.output


def test_changed_input_fingerprint_stops_before_stage_execution(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/ng/publish.py")
	root = tmp_path / "root"
	root.mkdir()
	dataset = make_dataset(root)
	plan = module.build_dataset_plan(dataset, True)
	module.write_state(
		plan.state_path,
		{
			"version": 1,
			"input_fingerprint": "stale",
			"stages": {},
		},
	)
	monkeypatch.setattr(module, "module_available", lambda _name: True)
	monkeypatch.setattr(
		module,
		"run_stage",
		lambda *_args: (_ for _ in ()).throw(AssertionError("stage ran")),
	)

	result = CliRunner().invoke(
		module.publish,
		[
			str(root),
			"--stop-after", "precompute",
		],
	)

	assert result.exit_code != 0
	assert "input changed since publish state was recorded" in result.output


def test_publish_rejects_contradictory_stage_controls(load_module, tmp_path):
	module = load_module("mctutil/ng/publish.py")
	root = tmp_path / "root"
	root.mkdir()
	make_dataset(root)

	result = CliRunner().invoke(
		module.publish,
		[
			str(root),
			"--start-at", "upload",
			"--no-upload",
		],
	)

	assert result.exit_code != 0
	assert "contradicts --no-upload" in result.output
