"""Export the phase producers' results as a versioned, plain-data contract."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from mctutil.phase.commands import _command_errors
from mctutil.phase.io import json_text
from mctutil.shared.deps import require
from mctutil.shared.log import log


def _positive(value, name):
	value = float(value)
	if not np.isfinite(value) or value <= 0:
		raise ValueError(f"{name} must be finite and positive")
	return value


def _geometry(geometry):
	fields = {
		"parallel": ("pixel_m", "distance_m"),
		"cone": ("source_to_sample_m", "sample_to_detector_m", "detector_pixel_m"),
	}
	kind = geometry.get("kind")
	if kind not in fields:
		raise ValueError("geometry.kind must be 'parallel' or 'cone'")
	if set(geometry) != {"kind", *fields[kind]}:
		raise ValueError(f"{kind} geometry requires only kind and {', '.join(fields[kind])}")
	return {"kind": kind, **{name: _positive(geometry[name], name) for name in fields[kind]}}


def _source(record):
	if record.get("schema_version") != 1:
		raise ValueError("calibration must be a phase producer record with schema_version 1")
	producer = record["producer"]
	if producer not in ("harvest_material_table", "build_table"):
		raise ValueError("calibration producer must be harvest_material_table or build_table")
	parameters, result = record["parameters"], record["result"]
	energy = _positive(parameters["energy_kev"], "calibration energy_kev")
	if producer == "harvest_material_table":
		_harvest_evidence(result, energy)
	mu, delta = np.asarray(result["mu"], float), np.asarray(result["delta"], float)
	if mu.ndim != 1 or not mu.size or mu.shape != delta.shape:
		raise ValueError("calibration mu and delta must be nonempty matching one-dimensional arrays")
	if not np.isfinite(mu).all() or not np.isfinite(delta).all():
		raise ValueError("calibration mu and delta must be finite")
	return producer, parameters, result, energy, mu, delta


def _harvest_evidence(result, energy):
	required = {"provenance", "delta_prior_informed", "uncertainty", "delta_slope_sensitivity", "cov"}
	if not required <= result.keys():
		raise ValueError("harvest calibration must retain its provenance, prior flag and uncertainty fields")
	if not isinstance(result["provenance"], dict) or result["delta_prior_informed"] is not True:
		raise ValueError("harvest calibration must retain provenance and delta_prior_informed=true")
	if result["provenance"].get("energy_kev") != energy:
		raise ValueError("harvest provenance energy_kev must match the producer parameters")


def _target_table(producer, parameters, result, mu, delta, source_energy, target_energy, basis):
	fit = result["line_fit" if producer == "harvest_material_table" else "fit"]
	if np.asarray(fit["eps"]).shape != mu.shape:
		raise ValueError("calibration line-fit eps must match the material table length")
	transfer = None
	if target_energy != source_energy:
		require(("scipy", "xraylib"), "phase", purpose="Phase material energy transfer")
		from mctutil.phase.phase_tomo import fit_material_line, transfer_table

		mu, delta, info = transfer_table(mu, delta, source_energy, target_energy, basis=basis)
		fit = fit_material_line(
			mu, delta, weights=parameters.get("weights"),
			through_origin=True if producer == "harvest_material_table" else parameters.get("through_origin", True),
		)
		transfer = {"basis": basis, **info}
	names = result.get("names")
	if names is not None and len(names) != len(mu):
		raise ValueError("calibration names must match the material table length")
	return {"names": names, "mu": mu, "delta": delta, "line_fit": fit}, transfer


def _calibration(record, producer, parameters, result, energy):
	observation_keys = ("fractions", "air_point", "slope_used", "collinearity_residual")
	parameters = {name: value for name, value in parameters.items() if name != "energy_kev"}
	provenance = result.get("provenance")
	if producer == "harvest_material_table":
		provenance = {name: value for name, value in provenance.items() if name != "energy_kev"}
		for name in ("distances_m", "rows", "air_correct", "slope"):
			parameters.pop(name, None)
		if isinstance(result["uncertainty"], dict):
			parameters.pop("n_folds", None)
	return {
		"producer": producer,
		"energy_kev": energy,
		"config_path": record["config_path"],
		"parameters": parameters,
		"provenance": provenance,
		"observations": {name: result[name] for name in observation_keys if name in result},
		"uncertainty": {
			"scope": "calibration_energy",
			"statistical": result.get("uncertainty"),
			"delta_slope_sensitivity": result.get("delta_slope_sensitivity"),
			"voxel_scatter_cov": result.get("cov"),
			"common_mode_bias": "not_quantified",
		},
	}


def material_table_export(record, *, effective_energy_kev, geometry, envelope_m=None, transfer_basis=None):
	"""Adapt a producer record once; retain source uncertainty without propagation."""

	producer, parameters, result, source_energy, mu, delta = _source(record)
	target_energy = _positive(effective_energy_kev, "effective_energy_kev")
	basis = (("H2O", 1.00), ("Al", 2.70)) if transfer_basis is None else tuple(map(tuple, transfer_basis))
	table, transfer = _target_table(producer, parameters, result, mu, delta, source_energy, target_energy, basis)
	if table["line_fit"]["alpha"] != 0 and envelope_m is None:
		raise ValueError("nonzero material-line alpha requires envelope_m (spatial thickness in metres)")
	envelope = None if envelope_m is None else np.asarray(envelope_m, float)
	harvested = producer == "harvest_material_table"
	return {
		"schema": "mctutil.phase.material_table",
		"schema_version": 1,
		"effective_energy_kev": target_energy,
		"geometry": _geometry(geometry),
		"envelope_m": envelope,
		"material_table": table,
		"calibration": _calibration(record, producer, parameters, result, source_energy),
		"transfer": transfer,
		"status": "experimental" if harvested else "model_derived",
		"provenance": {
			"calibration_mu": "measured_absorption" if harvested else "xraylib_from_supplied_composition_density",
			"calibration_mu_bias": "fbp_discretization_bias_not_removed" if harvested else "not_applicable",
			"calibration_delta": "slope_prior_informed" if harvested else "xraylib_from_supplied_composition_density",
			"target_points": "energy_transfer_model" if transfer else "calibration_points",
			"line_fit": "derived_from_target_points",
			"eps": "prior_informed_not_independently_measured" if harvested else "derived_from_model_points",
		},
		"target_uncertainty": "not_propagated" if transfer else "see_calibration_budget",
	}


@click.command("export")
@click.argument("config_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Plan the output without reading calibration data or computing results.")
def export(config_json: Path, output: Path, dry_run: bool):
	"""Export a calibration producer record using CONFIG_JSON into OUTPUT JSON."""

	if dry_run:
		log.write("Dry Run", f"Would export a material table from {config_json} into {output}")
		return
	with _command_errors():
		parameters = json.loads(config_json.read_text(encoding="utf-8"))
		calibration_path = config_json.parent / parameters.pop("calibration")
		record = json.loads(calibration_path.read_text(encoding="utf-8"))
		payload = json_text(material_table_export(record, **parameters))
		output.parent.mkdir(parents=True, exist_ok=True)
		output.write_text(payload, encoding="utf-8")
		log.write("Phase Export", f"Wrote {output}")
