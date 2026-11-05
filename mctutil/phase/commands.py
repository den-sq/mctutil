"""Thin Click adapters for the supplied phase and material functions."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import click
import numpy as np

from mctutil.phase.io import load_parameters, result_json
from mctutil.shared.deps import require
from mctutil.shared.log import log


@contextmanager
def _command_errors():
	try:
		yield
	except click.ClickException:
		raise
	except Exception as error:
		raise click.ClickException(str(error)) from error


def _write_json(output: Path, producer: str, parameters: dict, config_path: Path, result):
	payload = result_json(producer, parameters, config_path, result)
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(payload, encoding="utf-8")
	log.write("Phase Result", f"Wrote {output}")


@click.command()
@click.argument("config_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Plan the output without reading arrays or computing results.")
def harvest(config_json: Path, output: Path, dry_run: bool):
	"""Harvest material parameters from CONFIG_JSON into OUTPUT JSON."""

	if dry_run:
		log.write("Dry Run", f"Would harvest from {config_json} into {output}")
		return
	with _command_errors():
		require("scipy", "phase", purpose="Phase material harvesting", error_type=click.ClickException)
		from mctutil.phase.phase_tomo import harvest_material_table

		parameters, arguments = load_parameters(config_json)
		result = harvest_material_table(**arguments)
		_write_json(output, harvest_material_table.__name__, parameters, config_json, result)


@click.command("estimate-line-slope")
@click.argument("config_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Plan the output without reading arrays or computing results.")
def estimate_line_slope(config_json: Path, output: Path, dry_run: bool):
	"""Estimate the material-line slope from CONFIG_JSON into OUTPUT JSON."""

	if dry_run:
		log.write("Dry Run", f"Would estimate a line slope from {config_json} into {output}")
		return
	with _command_errors():
		require("scipy", "phase", purpose="Phase line-slope estimation", error_type=click.ClickException)
		from mctutil.phase.phase_tomo import estimate_line_slope as estimate

		parameters, arguments = load_parameters(config_json)
		slope, info = estimate(**arguments)
		_write_json(output, estimate.__name__, parameters, config_json, {"slope": slope, "info": info})


@click.command()
@click.argument("config_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Plan the output without reading arrays or computing results.")
def refine(config_json: Path, output: Path, dry_run: bool):
	"""Refine B and phi from CONFIG_JSON into OUTPUT NPZ with JSON metadata."""

	if dry_run:
		log.write("Dry Run", f"Would refine from {config_json} into {output}")
		return
	with _command_errors():
		require("scipy", "phase", purpose="Phase refinement", error_type=click.ClickException)
		from mctutil.phase.phase_tomo import gauss_newton_refine

		parameters, arguments = load_parameters(config_json)
		result = gauss_newton_refine(**arguments)
		arrays = {name: np.asarray(result[name]) for name in ("B", "phi", "residuals")}
		metadata = result_json(gauss_newton_refine.__name__, parameters, config_json, {
			name: value for name, value in result.items() if name not in arrays
		})
		output.parent.mkdir(parents=True, exist_ok=True)
		with output.open("wb") as stream:
			np.savez(stream, **arrays, metadata=np.asarray(metadata))
		log.write("Phase Result", f"Wrote {output}")


@click.command("make-material-table")
@click.argument("config_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Plan the output without reading arrays or computing results.")
def make_material_table(config_json: Path, output: Path, dry_run: bool):
	"""Build a composition-based material table from CONFIG_JSON into OUTPUT JSON."""

	if dry_run:
		log.write("Dry Run", f"Would build a material table from {config_json} into {output}")
		return
	with _command_errors():
		require(("scipy", "xraylib"), "phase", purpose="Phase material tables", error_type=click.ClickException)
		from mctutil.phase.materials import build_table

		parameters, arguments = load_parameters(config_json)
		result = build_table(**arguments)
		_write_json(output, build_table.__name__, parameters, config_json, result)
