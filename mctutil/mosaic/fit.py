"""Joint least-squares fit of group offsets and the cross-group depth relation.

Unknowns: the depth offset of every reference-group scan (global row G = row + offset)
and of every fixed-group scan (group frame Z = row + offset), each group anchored at
its first scan, plus K, gy, gx, gz of G - Z = K + gy * y + gx * x + gz * Z.
Observations, all weighted equally:

- a same-group depth link (moving row = fixed row + T): offset[fixed] - offset[moving] = T;
- a cross-group point (mosaic tilt-points): G - Z = reference offset - fixed offset + T.

Points whose residual exceeds 3 x the median absolute residual + 1 row are dropped and
the system is solved again. The lateral relation d(Z) and phi are carried over from the
tilt-points file.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import read_json, write_json
from mctutil.shared.log import log


def read_links(paths) -> list[dict]:
	return [json.loads(line) for path in paths for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


class Unknowns:
	"""Column layout: free (non-anchor) offsets of each group, then K, gy, gx, gz."""

	def __init__(self, reference: list[str], fixed: list[str]):
		self.reference, self.fixed = reference, fixed
		free = [("reference", name) for name in reference[1:]] + [("fixed", name) for name in fixed[1:]]
		self.column = {key: index for index, key in enumerate(free)}
		self.depth = len(free)
		self.size = self.depth + 4

	def offset_row(self, group: str, name: str, sign: float, row: np.ndarray) -> None:
		if (group, name) in self.column:
			row[self.column[(group, name)]] += sign

	def solution(self, values) -> dict:
		offsets = {group: {names[0]: 0.0} for group, names in (("reference", self.reference), ("fixed", self.fixed))}
		for (group, name), index in self.column.items():
			offsets[group][name] = float(values[index])
		return offsets


def link_equations(unknowns: Unknowns, links, group: str) -> tuple[list, list]:
	names = unknowns.reference if group == "reference" else unknowns.fixed
	rows, values = [], []
	for link in links:
		if link["fixed"] in names and link["moving"] in names:
			row = np.zeros(unknowns.size)
			unknowns.offset_row(group, link["fixed"], 1.0, row)
			unknowns.offset_row(group, link["moving"], -1.0, row)
			rows.append(row)
			values.append(link["T_median"])
	return rows, values


def point_equations(unknowns: Unknowns, points) -> tuple[list, list]:
	rows, values = [], []
	for point in points:
		row = np.zeros(unknowns.size)
		unknowns.offset_row("reference", point["moving"], 1.0, row)
		unknowns.offset_row("fixed", point["link"], -1.0, row)
		row[unknowns.depth:] = (-1.0, -point["y"], -point["x"], -point["Z"])
		rows.append(row)
		values.append(-point["T"])
	return rows, values


def solve(link_rows, link_values, point_rows, point_values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Least squares with one round of point outlier rejection; returns solution, residuals, kept mask."""
	design = np.array(link_rows + point_rows)
	target = np.array(link_values + point_values, dtype=np.float64)
	solution, *_ = np.linalg.lstsq(design, target, rcond=None)
	residual = target - design @ solution
	points = np.arange(len(target)) >= len(link_rows)
	keep = ~points | (np.abs(residual) < 3 * np.median(np.abs(residual[points])) + 1)
	solution, *_ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
	return solution, target - design @ solution, keep


@click.command()
@click.argument("tilt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--links", "link_paths", type=click.Path(exists=True, dir_okay=False, path_type=Path),
				multiple=True, required=True, help="JSON-lines file(s) from mosaic link-depth.")
@click.option("--reference", "reference_names", required=True,
				help="Comma-separated reference-group scans; the first is anchored at offset 0.")
@click.option("--fixed", "fixed_names", required=True,
				help="Comma-separated fixed-group scans; the first is anchored at offset 0.")
@click.option("--min-peak", type=float, default=0.003, show_default=True, help="Minimum tilt-point peak.")
def fit(tilt, output, link_paths, reference_names, fixed_names, min_peak):
	"""Solve group offsets and the depth relation from TILT points and same-group LINKS."""
	log.start()
	tilt_data = read_json(tilt)
	unknowns = Unknowns(reference_names.split(","), fixed_names.split(","))
	links = read_links(link_paths)
	points = [point for point in tilt_data["points"] if point["peak"] > min_peak]
	if not points:
		raise click.ClickException(
			f"no cross-group tilt points exceed --min-peak {min_peak}; the depth relation is undetermined")
	reference_rows, reference_values = link_equations(unknowns, links, "reference")
	fixed_rows, fixed_values = link_equations(unknowns, links, "fixed")
	point_rows, point_values = point_equations(unknowns, points)
	solution, residual, keep = solve(
		reference_rows + fixed_rows, reference_values + fixed_values, point_rows, point_values,
	)
	link_count = len(reference_rows) + len(fixed_rows)
	k, gy, gx, gz = (float(value) for value in solution[unknowns.depth:])
	offsets = unknowns.solution(solution)
	result = {
		"phi": tilt_data["phi"], "K": k, "gy": gy, "gx": gx, "gz": gz,
		"dy": tilt_data["dy"], "dx": tilt_data["dx"],
		"reference": offsets["reference"], "fixed": offsets["fixed"],
		"link_residual_rms": float(np.sqrt(np.mean(residual[:link_count] ** 2))) if link_count else 0.0,
		"point_residual_rms": float(residual[link_count:][keep[link_count:]].std()),
		"points_used": int(keep[link_count:].sum()), "points_total": len(points),
	}
	write_json(output, result)
	log.write("Joint Fit", json.dumps({key: value for key, value in result.items() if key not in ("reference", "fixed")}))
