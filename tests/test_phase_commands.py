from __future__ import annotations

import json
import subprocess
import sys

from click.testing import CliRunner
import numpy as np
import pytest

from mctutil.cli import main


def test_phase_help_and_dry_run_do_not_import_scientific_dependencies(tmp_path):
	config = tmp_path / "plan.json"
	config.write_text("not read during dry-run", encoding="utf-8")
	output = tmp_path / "result.npz"
	check = subprocess.run([
		sys.executable, "-c", """
import sys
from click.testing import CliRunner
from mctutil.cli import main

commands = ("harvest", "estimate-line-slope", "refine", "make-material-table", "export")
result = CliRunner().invoke(main, ["phase", "--help"])
assert result.exit_code == 0, result.output
for command in commands:
	assert command in result.output
	help_result = CliRunner().invoke(main, ["phase", command, "--help"])
	assert help_result.exit_code == 0, help_result.output
	assert "--dry-run" in help_result.output
	planned = CliRunner().invoke(main, ["phase", command, sys.argv[1], sys.argv[2], "--dry-run"])
	assert planned.exit_code == 0, planned.output
	assert sys.argv[1] in planned.output and sys.argv[2] in planned.output
for module in ("scipy", "xraylib", "mctutil.phase.phase_tomo", "mctutil.phase.materials"):
	assert module not in sys.modules, module
""", str(config), str(output),
	], capture_output=True, text=True)
	assert check.returncode == 0, check.stdout + check.stderr
	assert not output.exists()


def test_phase_scan_commands_execute_and_preserve_provenance(phase_scan, tmp_path, monkeypatch):
	data, common, slope, shape = phase_scan
	# Configuration-relative array paths must work from an unrelated directory.
	monkeypatch.chdir(tmp_path)
	runner = CliRunner()
	for command, producer, additional in [
		("harvest", "harvest_material_table", {
			"angles_deg": "angles.npy", "n_materials": 1, "rows": [1], "slope": slope,
		}),
		("estimate-line-slope", "estimate_line_slope", {"angle_step": 4}),
		("refine", "gauss_newton_refine", {
			"B_init": "B.npy", "phi_init": "phi.npy", "n_newton": 1, "n_cg": 4, "slope": slope,
		}),
	]:
		parameters = common | additional
		config = data / f"{command}.json"
		config.write_text(json.dumps(parameters), encoding="utf-8")
		output = tmp_path / "results" / (f"{command}.npz" if command == "refine" else f"{command}.json")
		result = runner.invoke(main, ["phase", command, str(config), str(output)])
		assert result.exit_code == 0, result.output
		if command == "refine":
			with np.load(output, allow_pickle=False) as refined:
				assert set(refined.files) == {"B", "phi", "residuals", "metadata"}
				assert refined["B"].shape == refined["phi"].shape == shape
				assert np.isfinite(refined["B"]).all() and np.isfinite(refined["phi"]).all()
				assert refined["residuals"].mean(axis=0)[-1] < refined["residuals"].mean(axis=0)[0]
				assert refined["metadata"].shape == ()
				record = json.loads(refined["metadata"].item())
		else:
			record = json.loads(output.read_text(encoding="utf-8"))
			if command == "harvest":
				assert len(record["result"]["mu"]) == len(record["result"]["delta"]) == 1
				assert np.isfinite(record["result"]["cov"]).all()
				assert record["result"]["slope_used"] == slope
				assert record["result"]["delta_prior_informed"] is True
			else:
				assert record["result"]["slope"] > 0
				assert record["result"]["info"]["n_bins"] > 0
				assert len(record["result"]["info"]["s_grid"]) == len(record["result"]["info"]["residual"])
		assert record["schema_version"] == 1
		assert record["producer"] == producer
		assert record["parameters"] == parameters
		assert record["config_path"] == str(config.resolve())


def test_phase_material_table_uses_compositions(tmp_path):
	pytest.importorskip("scipy")
	pytest.importorskip("xraylib")
	parameters = {"materials": {"water": ["H2O", 1.0], "PMMA": ["C5H8O2", 1.18]}, "energy_kev": 20.0}
	config = tmp_path / "materials.json"
	config.write_text(json.dumps(parameters), encoding="utf-8")
	output = tmp_path / "table.json"
	result = CliRunner().invoke(main, ["phase", "make-material-table", str(config), str(output)])
	assert result.exit_code == 0, result.output
	record = json.loads(output.read_text(encoding="utf-8"))
	assert record["producer"] == "build_table"
	assert record["parameters"] == parameters
	assert record["result"]["names"] == ["water", "PMMA"]
	assert np.all(np.asarray(record["result"]["mu"]) > 0)
	assert np.all(np.asarray(record["result"]["delta"]) > 0)
	assert len(record["result"]["fit"]["eps"]) == 2
	assert np.isfinite(record["result"]["plateau_dB_per_m"]).all()


def test_phase_missing_input_reports_click_error_without_overwriting_output(tmp_path):
	pytest.importorskip("scipy")
	config = tmp_path / "harvest.json"
	config.write_text(json.dumps({"projections": "missing.npy"}), encoding="utf-8")
	output = tmp_path / "table.json"
	output.write_text("existing result", encoding="utf-8")
	result = CliRunner().invoke(main, ["phase", "harvest", str(config), str(output)])
	assert result.exit_code == 1
	assert "Error:" in result.output and "missing.npy" in result.output
	assert output.read_text(encoding="utf-8") == "existing result"
