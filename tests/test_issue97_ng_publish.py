from __future__ import annotations

import importlib
import json

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


def test_publish_dry_run_reports_metadata_and_never_writes(
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
	calls = []
	monkeypatch.setattr(
		module,
		"run_stage",
		lambda stage, _plan, _options: calls.append(stage),
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
	assert calls == ["upload"]


def test_publish_real_prep_precompute_and_resume(tmp_path):
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
