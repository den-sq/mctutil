"""Boundary and provenance checks for the material-table export contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from click.testing import CliRunner
import numpy as np
import pytest

from mctutil.phase.export import export, material_table_export
from mctutil.phase.io import load_parameters, result_json


@pytest.fixture
def composition_record():
	pytest.importorskip("scipy")
	pytest.importorskip("xraylib")
	from mctutil.phase.materials import build_table

	parameters = {
		"materials": {"water": ["H2O", 1.0], "aluminium": ["Al", 2.7], "PMMA": ["C5H8O2", 1.19]},
		"energy_kev": 20.0,
		"weights": [1.0, 2.0, 3.0],
		"through_origin": False,
	}
	return json.loads(result_json("build_table", parameters, Path("composition.json"), build_table(**parameters)))


def _configuration(energy=20.0):
	return {
		"effective_energy_kev": energy,
		"geometry": {"kind": "parallel", "pixel_m": 1e-6, "distance_m": 0.1},
		"envelope_m": 1e-3,
	}


@pytest.mark.parametrize("through_origin", [False, True, None])
def test_composition_export_rebuilds_target_points_across_absorption_edge(
	composition_record, monkeypatch, through_origin,
):
	import xraylib
	from mctutil.phase import materials, phase_tomo

	parameters = composition_record["parameters"]
	parameters["materials"] = {"PMMA": ["C5H8O2", 1.19], "AgCl": ["AgCl", 5.56], "aluminium": ["Al", 2.7]}
	if through_origin is None:
		parameters.pop("through_origin")
	else:
		parameters["through_origin"] = through_origin
	composition_record = json.loads(result_json(
		"build_table", parameters, Path("composition.json"), materials.build_table(**parameters),
	))
	parameters = composition_record["parameters"]
	unchanged = deepcopy(composition_record)
	original = composition_record["result"]
	expected = materials.build_table(**(parameters | {"energy_kev": 30.0}))
	assert 20.0 < xraylib.EdgeEnergy(xraylib.SymbolToAtomicNumber("Ag"), xraylib.K_SHELL) < 30.0

	build_calls, transfer_calls, fit_calls = [], [], []
	build = materials.build_table
	transfer, fit = phase_tomo.transfer_table, phase_tomo.fit_material_line

	def record_build(*args, **kwargs):
		result = build(*args, **kwargs)
		build_calls.append((args, kwargs, result))
		return result

	def record_transfer(*args, **kwargs):
		result = transfer(*args, **kwargs)
		transfer_calls.append((args, kwargs, result))
		return result

	def record_fit(*args, **kwargs):
		result = fit(*args, **kwargs)
		fit_calls.append((args, kwargs, result))
		return result

	monkeypatch.setattr(materials, "build_table", record_build)
	monkeypatch.setattr(phase_tomo, "transfer_table", record_transfer)
	monkeypatch.setattr(phase_tomo, "fit_material_line", record_fit)
	same = material_table_export(composition_record, **_configuration())
	assert same["material_table"]["line_fit"] is original["fit"]
	assert same["transfer"] is None
	assert same["status"] == "model_derived"
	assert same["calibration"]["uncertainty"]["statistical"] is None
	assert not {"mu", "delta", "fit", "line_fit", "plateau_dB_per_m"} & same["calibration"].keys()
	assert same["provenance"]["target_points"] == "calibration_points"
	assert same["target_uncertainty"] == "see_calibration_budget"
	assert build_calls == transfer_calls == fit_calls == []

	changed = material_table_export(composition_record, **_configuration(30.0))
	table = changed["material_table"]
	np.testing.assert_array_equal(table["mu"], expected["mu"])
	np.testing.assert_array_equal(table["delta"], expected["delta"])
	for name, value in expected["fit"].items():
		np.testing.assert_array_equal(table["line_fit"][name], value)
	assert table["names"] == ["PMMA", "AgCl", "aluminium"]
	assert len(build_calls) == 1
	assert build_calls[0][0] == (parameters["materials"], 30.0)
	assert build_calls[0][1] == {
		"weights": parameters["weights"], "through_origin": True if through_origin is None else through_origin,
	}
	for name in ("names", "mu", "delta"):
		assert table[name] is build_calls[0][2][name]
	assert table["line_fit"] is build_calls[0][2]["fit"]
	assert transfer_calls == fit_calls == []
	assert changed["target_uncertainty"] == "not_propagated"
	assert changed["transfer"] is None
	assert changed["provenance"]["target_points"] == "recomputed_from_composition"
	assert composition_record == unchanged


def test_harvest_export_preserves_source_budget_and_prior_labels(phase_scan, monkeypatch):
	pytest.importorskip("xraylib")
	from mctutil.phase import phase_tomo

	data, common, slope, _ = phase_scan
	config = data / "harvest.json"
	config.write_text(json.dumps(common | {
		"angles_deg": "angles.npy", "n_materials": 1, "rows": [1], "slope": slope,
		"slope_sensitivity": 0.1, "n_folds": 2,
	}))
	parameters, arguments = load_parameters(config)
	record = json.loads(result_json(
		"harvest_material_table", parameters, config, phase_tomo.harvest_material_table(**arguments),
	))
	original = deepcopy(record)
	result = record["result"]
	transfer_calls, fit_calls = [], []
	transfer, fit = phase_tomo.transfer_table, phase_tomo.fit_material_line

	def record_transfer(*args, **kwargs):
		result = transfer(*args, **kwargs)
		transfer_calls.append((args, kwargs, result))
		return result

	def record_fit(*args, **kwargs):
		result = fit(*args, **kwargs)
		fit_calls.append((args, kwargs, result))
		return result

	monkeypatch.setattr(phase_tomo, "transfer_table", record_transfer)
	monkeypatch.setattr(phase_tomo, "fit_material_line", record_fit)
	same = material_table_export(record, **_configuration())
	assert same["material_table"]["line_fit"] is result["line_fit"]
	assert same["transfer"] is None
	assert transfer_calls == fit_calls == []
	changed = material_table_export(record, **_configuration(24.0))
	assert len(transfer_calls) == len(fit_calls) == 1
	transfer_args, transfer_kwargs, transferred = transfer_calls[0]
	np.testing.assert_array_equal(transfer_args[0], result["mu"])
	np.testing.assert_array_equal(transfer_args[1], result["delta"])
	assert transfer_args[2:] == (20.0, 24.0)
	assert transfer_kwargs == {"basis": (("H2O", 1.0), ("Al", 2.7))}
	assert changed["material_table"]["mu"] is transferred[0]
	assert changed["material_table"]["delta"] is transferred[1]
	assert fit_calls[0][0][0] is transferred[0]
	assert fit_calls[0][0][1] is transferred[1]
	assert "weights" not in record["parameters"]
	assert fit_calls[0][1] == {"weights": None, "through_origin": True}
	assert changed["material_table"]["line_fit"] is fit_calls[0][2]
	assert changed["transfer"]["basis"] == transfer_kwargs["basis"]
	assert changed["transfer"]["weights"] is transferred[2]["weights"]
	assert changed["transfer"]["basis_condition"] == transferred[2]["basis_condition"]
	assert changed["provenance"]["target_points"] == "energy_transfer_model"
	assert record == original
	calibration = changed["calibration"]
	assert not {"energy_kev", "distances_m", "rows", "air_correct", "slope", "n_folds"} & calibration["parameters"].keys()
	assert "energy_kev" not in calibration["provenance"]
	assert "delta_prior_informed" not in calibration["observations"]
	assert calibration["provenance"]["distances_m"] == common["distances_m"]
	assert calibration["provenance"]["rows"] == [1]
	assert calibration["provenance"]["air_correct"] is True
	budget = changed["calibration"]["uncertainty"]
	assert budget["scope"] == "calibration_energy"
	assert budget["statistical"] is result["uncertainty"]
	assert budget["voxel_scatter_cov"] is result["cov"]
	assert budget["delta_slope_sensitivity"] is result["delta_slope_sensitivity"]
	assert budget["common_mode_bias"] == "not_quantified"
	assert changed["calibration"]["energy_kev"] == 20.0
	assert changed["effective_energy_kev"] == 24.0
	assert changed["calibration"]["observations"]["collinearity_residual"] is None
	assert changed["calibration"]["observations"]["slope_used"] == slope
	assert budget["statistical"]["n_folds_used"] == 2
	assert len(budget["delta_slope_sensitivity"]) == len(changed["material_table"]["mu"]) == 1
	assert changed["material_table"]["names"] is None
	assert changed["material_table"]["line_fit"]["alpha"] == 0
	assert changed["status"] == "experimental"
	assert changed["provenance"]["calibration_mu_bias"] == "fbp_discretization_bias_not_removed"
	assert changed["provenance"]["calibration_delta"] == "slope_prior_informed"
	assert changed["provenance"]["eps"] == "prior_informed_not_independently_measured"
	assert changed["target_uncertainty"] == "not_propagated"
	result["uncertainty"] = None
	record["parameters"]["n_folds"] = 0
	unquantified = material_table_export(record, **_configuration())["calibration"]
	assert unquantified["uncertainty"]["statistical"] is None
	assert unquantified["parameters"]["n_folds"] == 0
	result["provenance"]["energy_kev"] = 21.0
	with pytest.raises(ValueError, match="energy_kev must match"):
		material_table_export(record, **_configuration())
	result.pop("delta_prior_informed")
	with pytest.raises(ValueError, match="prior flag"):
		material_table_export(record, **_configuration())


def test_export_cli_reads_relative_record_and_keeps_native_cone_geometry(composition_record, tmp_path):
	(tmp_path / "calibration.json").write_text(json.dumps(composition_record))
	configuration = {
		**_configuration(24.0), "calibration": "calibration.json",
		"geometry": {
			"kind": "cone", "source_to_sample_m": 1.0, "sample_to_detector_m": 0.1, "detector_pixel_m": 1e-6,
		},
	}
	config = tmp_path / "export.json"
	config.write_text(json.dumps(configuration))
	output = tmp_path / "result" / "table.json"
	run = CliRunner().invoke(export, [str(config), str(output)])
	assert run.exit_code == 0, run.output
	artifact = json.loads(output.read_text())
	assert artifact["schema"] == "mctutil.phase.material_table"
	assert artifact["schema_version"] == 1
	assert artifact["geometry"] == configuration["geometry"]
	assert artifact["material_table"]["names"] == ["water", "aluminium", "PMMA"]
	assert artifact["transfer"] is None
	assert artifact["provenance"]["target_points"] == "recomputed_from_composition"
	assert artifact["target_uncertainty"] == "not_propagated"

	configuration["envelope_m"] = None
	config.write_text(json.dumps(configuration))
	previous = output.read_bytes()
	run = CliRunner().invoke(export, [str(config), str(output)])
	assert run.exit_code != 0
	assert "nonzero material-line alpha requires envelope_m" in run.output
	assert output.read_bytes() == previous


@pytest.mark.parametrize(
	"failure", ["version", "producer", "table_length", "fit_length", "nonfinite", "mixed_geometry"],
)
def test_export_rejects_records_that_cannot_define_the_contract(composition_record, failure):
	configuration = _configuration()
	if failure == "version":
		composition_record["schema_version"] = 2
	elif failure == "producer":
		composition_record["producer"] = "gauss_newton_refine"
	elif failure == "table_length":
		composition_record["result"]["delta"].pop()
	elif failure == "fit_length":
		composition_record["result"]["fit"]["eps"].pop()
	elif failure == "nonfinite":
		composition_record["result"]["mu"][0] = float("nan")
	else:
		configuration["geometry"]["source_to_sample_m"] = 1.0
	with pytest.raises(ValueError):
		material_table_export(composition_record, **configuration)
