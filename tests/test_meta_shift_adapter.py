"""Regression tests for the meta_shift adapter seam.

Exercises that the generic engine in parsing/meta_shift.py routes through the
chenglab adapter for path computation, status mapping, discovery, and sheet
row layout.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest


def _write_sample_conf(path: Path, energy: str = "35", inner: str = "left", project: str = "AAA590"):
	path.write_text(
		"storage:\n"
		f"  project: {project}\n"
		f"  inner_comment: {inner}\n"
		"  outer_comment: ''\n"
		"  has_scan: false\n"
		"  trip_dir: trip-001\n"
		"sample:\n"
		"  id: SAMPLE_001\n"
		"scan:\n"
		f"  energy: {energy}\n"
		"  id: SAMPLE_001_X\n"
	)


def test_load_adapter_returns_chenglab_by_default(load_module):
	engine = load_module("parsing/meta_shift.py")
	adapter = engine.load_adapter("chenglab")
	assert type(adapter).__name__ == "ChenglabMicroCTAdapter"
	assert adapter.default_spreadsheet
	assert adapter.default_sheet


def test_load_adapter_rejects_unknown_schema(load_module):
	engine = load_module("parsing/meta_shift.py")
	with pytest.raises(click.BadParameter):
		engine.load_adapter("does-not-exist")


def test_chenglab_adapter_computes_step_paths(load_module):
	chenglab = load_module("chenglab/meta_shift.py")
	adapter = chenglab.ChenglabMicroCTAdapter()
	conf = {
		"storage": {"project": "ProjA", "inner_comment": "left", "outer_comment": "", "has_scan": False},
		"sample": {"id": "S001"},
		"scan": {"energy": "35", "id": "S001_X"},
	}
	old, new = adapter.compute_paths(conf, "job42")
	assert old["flats"] == Path("ProjA/S001/data/35kV_left_flats_pjob42")
	assert new["flats"] == Path("ProjA/S001/data/35kV_S001_X_flats_pjob42")
	assert old["script"].parts[-1] == "job42"
	assert new["script"].parts[-2] == "35kV_S001_X"


def test_chenglab_adapter_status_pipeline_orders_steps(load_module):
	chenglab = load_module("chenglab/meta_shift.py")
	adapter = chenglab.ChenglabMicroCTAdapter()
	assert adapter.empty_status() is chenglab.STATUS.EMPTY
	assert adapter.status_for_step("flats") is chenglab.STATUS.FLATS_GENERATED_UC
	assert adapter.status_for_step("flats", uncorrected=False) is chenglab.STATUS.FLATS_GENERATED
	assert adapter.status_for_step("recon") is chenglab.STATUS.RECONSTRUCTED_UC
	assert adapter.status_for_step("unknown") is chenglab.STATUS.EMPTY


def test_chenglab_adapter_discovers_v_yaml_only(load_module, tmp_path):
	chenglab = load_module("chenglab/meta_shift.py")
	adapter = chenglab.ChenglabMicroCTAdapter()
	(tmp_path / "samples").mkdir()
	(tmp_path / "samples" / "35V.yaml").write_text("placeholder")
	(tmp_path / "samples" / "notes.yaml").write_text("placeholder")
	(tmp_path / "samples" / "70V.yaml").write_text("placeholder")
	found = adapter.discover_sample_configs(tmp_path)
	names = sorted(p.name for p in found)
	assert names == ["35V.yaml", "70V.yaml"]


def test_chenglab_adapter_builds_sheet_row(load_module, tmp_path):
	chenglab = load_module("chenglab/meta_shift.py")
	adapter = chenglab.ChenglabMicroCTAdapter()
	conf = {
		"storage": {"project": "ProjA", "trip_dir": "trip-7"},
		"sample": {"id": "S001"},
		"scan": {"id": "S001_X"},
	}
	run_params = {"slice_range": "0-100", "phase_alpha": "0.03"}
	move_pairs = {
		"flats": {"to": tmp_path / "ProjA/S001/data/35kV_S001_X_flats_pjob42"},
	}
	row = adapter.build_sheet_row(conf, run_params, tmp_path, move_pairs, "scan-3", chenglab.STATUS.FLATS_GENERATED_UC)
	# Layout sanity: scan_num, scan_id, project surface in fixed columns; status string is present.
	assert "S001" in row
	assert "trip-7" in row
	assert "scan-3" in row
	assert "S001_X" in row
	assert "ProjA" in row
	assert "Flats Generated Uc" in row


def test_engine_shift_old_new_honors_dry_run_default(load_module, tmp_path):
	engine = load_module("parsing/meta_shift.py")
	chenglab = load_module("chenglab/meta_shift.py")
	adapter = chenglab.ChenglabMicroCTAdapter()
	drive = tmp_path
	(drive / "ProjA/S001/data").mkdir(parents=True)
	old_paths = {"flats": Path("ProjA/S001/data/old"), "script": Path("history/ProjA/S001/job")}
	new_paths = {"flats": Path("ProjA/S001/data/new"), "script": Path("history/ProjA/S001/job")}
	(drive / old_paths["flats"]).mkdir(parents=True)
	(drive / old_paths["script"]).mkdir(parents=True)
	conf_dict = {"sample": {"id": "S001"}, "scan": {"id": "S001_X"}}

	status, can_move, move_pairs, keep_pairs = engine.shift_old_new(
		adapter, drive, conf_dict, old_paths, new_paths, {}, "scan-1", execute=False
	)
	assert (drive / old_paths["flats"]).exists()
	assert not (drive / new_paths["flats"]).exists()
	assert can_move is True
	assert "flats" in move_pairs
