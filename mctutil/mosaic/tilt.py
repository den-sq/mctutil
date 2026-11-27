"""Cross-group depth points from 3-D sub-block correlation of in-plane-registered blocks.

Model between a fixed group (frame Z = fixed row + fixed offset) and the reference group
(global row G = moving row + moving offset): in-plane, the moving slice is rotated by phi
about the slice centre and shifted by d(Z) = d0 + dd * Z; in depth,
G - Z = K + gy * y + gx * x + gz * Z. Each sub-block point records the measured row
offset T (moving row = fixed row + T) at its lateral position; ``mosaic fit`` solves the model.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import RowStack, band_pass, phase_correlation, read_json, rotation, warp, write_json
from mctutil.mosaic.depth import block_centres, windowed_block
from mctutil.shared.log import log


def lateral_fit(links, fixed_offsets, min_peak: float) -> tuple[list[float], list[float]]:
	"""Linear d(Z) for dy and dx, as [slope, intercept], from the links' refine tables."""
	shifts = np.array([
		(record["row"] + fixed_offsets[link["fixed"]], record["dy"], record["dx"])
		for link in links for record in read_json(link["refine"]) if record["peak"] > min_peak
	])
	if len(shifts) < 2:
		raise click.ClickException("fewer than two refine rows pass the peak threshold")
	return np.polyfit(shifts[:, 0], shifts[:, 1], 1).tolist(), np.polyfit(shifts[:, 0], shifts[:, 2], 1).tolist()


def block_points(fixed, moving, centres, *, block: int, search: int) -> list[tuple]:
	"""(yc, xc, z shift, peak) for each sub-block centre."""
	depth = fixed.shape[0]
	window = (np.hanning(depth)[:, None, None] * np.outer(np.hanning(block), np.hanning(block))[None]).astype(np.float32)
	shape = (depth + 2 * search + depth, block, block)
	points = []
	for yc, xc in centres:
		spectrum = np.fft.rfftn(windowed_block(fixed, yc, xc, block) * window, shape, axes=(0, 1, 2))
		peak, shift = phase_correlation(spectrum, windowed_block(moving, yc, xc, block), shape)
		points.append((yc, xc, shift[0], peak))
	return points


def link_points(link, config, lateral, settings) -> list[dict]:
	fixed_stack, moving_stack = RowStack(link["fixed_dir"]), RowStack(link["moving_dir"])
	n, depth, search = fixed_stack.size, settings["depth"], settings["search"]
	centre = (n - 1) / 2
	fixed_offset = config["fixed_offsets"][link["fixed"]]
	moving_offset = config["moving_offsets"][link["moving"]]
	points = []
	for fixed_row in link["rows"]:
		z = fixed_row + depth / 2 + fixed_offset
		dy, dx = np.polyval(lateral[0], z), np.polyval(lateral[1], z)
		guess = int(round(config["depth_guess"] + fixed_offset - moving_offset))
		matrix = rotation(n, config["phi"], 1.0, dy, dx)
		a = np.stack([band_pass(fixed_stack.read(fixed_row + k), 1.5, 12) for k in range(depth)])
		b = np.stack([
			band_pass(warp(moving_stack.read(fixed_row + guess + k), matrix, (n, n)), 1.5, 12)
			for k in range(-search, depth + search)
		])
		centres = [
			(yc, xc) for yc, xc in block_centres(n, settings["block"], settings["step"], settings["radius"], settings["grid"])
			if np.hypot(yc - centre - dy, xc - centre - dx) <= settings["radius"]
		]
		for yc, xc, shift, peak in block_points(a, b, centres, block=settings["block"], search=search):
			t = guess - shift - search
			points.append({
				"link": link["fixed"], "moving": link["moving"], "row": fixed_row + depth / 2, "T": t,
				"Z": float(z), "y": yc, "x": xc, "G_minus_Z": float(moving_offset - fixed_offset + t), "peak": peak,
			})
		log.write("Tilt Block", f"{link['fixed']} row {fixed_row}: {len(points)} point(s)")
	return points


@click.command(name="tilt-points")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--radius", type=float, required=True,
				help="Sub-block centres stay within this many px of both scans' axes.")
@click.option("--depth", type=click.IntRange(min=2), default=48, show_default=True, help="Rows per block.")
@click.option("--search", type=click.IntRange(min=0), default=24, show_default=True,
				help="Extra moving rows on each side of the guess.")
@click.option("--block", type=click.IntRange(min=8), default=320, show_default=True, help="Sub-block width (px).")
@click.option("--step", type=click.IntRange(min=1), default=160, show_default=True, help="Sub-block spacing (px).")
@click.option("--grid", type=(int, int, int), help="Explicit centre range START STOP STEP for both axes.")
@click.option("--min-refine-peak", type=float, default=0.01, show_default=True,
				help="Refine rows below this peak are left out of the lateral fit.")
def tilt_points(config, output, radius, depth, search, block, step, grid, min_refine_peak):
	"""Measure cross-group depth points for the links in CONFIG and write them to OUTPUT.

	CONFIG is JSON: {"phi", "depth_guess" (approximate K), "fixed_offsets", "moving_offsets"
	(approximate group-frame offsets by scan name), "links": [{"fixed", "fixed_dir", "moving",
	"moving_dir", "rows": [first fixed row of each block], "refine": path to a mosaic refine table}]}.
	"""
	log.start()
	setup = read_json(config)
	lateral = lateral_fit(setup["links"], setup["fixed_offsets"], min_refine_peak)
	settings = {"depth": depth, "search": search, "block": block, "step": step, "radius": radius, "grid": grid}
	points = [point for link in setup["links"] for point in link_points(link, setup, lateral, settings)]
	write_json(output, {
		"phi": setup["phi"], "dy": lateral[0], "dx": lateral[1],
		"fixed_offsets": setup["fixed_offsets"], "moving_offsets": setup["moving_offsets"], "points": points,
	})
	log.write("Tilt Points", f"{len(points)} point(s) -> {output}")
