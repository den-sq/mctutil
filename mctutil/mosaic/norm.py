"""Per-scan linear intensity maps (a, b) for stitching, by moment matching in overlaps.

For each (moving, reference) pair, up to six canvas slices inside both scans' row ranges are
rendered per scan alone (a = 1, b = 0), smoothed (sigma 3), and compared where both FOV
disks are well inside (non-zero after an erosion, 61 px by default). a and b match the moving scan's
median and 5-95 % range to the reference's; maps chain from the reference scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import cv2, ndimage, read_json, write_json
from mctutil.mosaic.stitch import ScanSampler, canvas_grid, fuse, slice_centre
from mctutil.shared.log import log


def canvas_rows(scan: dict, canvas, depth: int, margin: int = 100) -> set[int]:
	"""Every 4th canvas slice whose scan row at the canvas centre lies in [margin, nrow - margin)."""
	a = np.array(scan["A"])
	height, width = canvas

	def row(g):
		return a[0, 0] * g + a[0, 1] * height / 2 + a[0, 2] * width / 2 + a[0, 3]

	return {g for g in range(0, depth, 4) if margin <= row(g) < scan["nrow"] - margin}


class SingleScanRenderer:
	"""Renders canvas slices of one scan through the stitcher with unit intensity map."""

	def __init__(self, document: dict, options: dict):
		self.shape = tuple(document["canvas"])
		self.grid, self.centre = canvas_grid(document), slice_centre(document)
		self.options = options
		self.cache = {}

	def render(self, scan: dict, rows) -> dict:
		if scan["name"] not in self.cache:
			sampler = ScanSampler(dict(scan, a=1.0, b=0.0), self.grid, self.centre, self.options)
			self.cache[scan["name"]] = (sampler, {})
		sampler, images = self.cache[scan["name"]]
		for g in rows:
			if g not in images:
				images[g] = fuse([sampler], g, self.shape, self.options).result()[0]
		return {g: images[g] for g in rows}


def overlap_values(moving_images, reference_images, rows, erode: int) -> tuple[np.ndarray, np.ndarray]:
	opencv, gaussian_filter = cv2(), ndimage().gaussian_filter
	moving_values, reference_values = [], []
	for g in rows:
		a, b = moving_images[g], reference_images[g]
		inside = opencv.erode(((a != 0) & (b != 0)).astype(np.uint8), np.ones((erode, erode), np.uint8)) > 0
		moving_values.append(gaussian_filter(a, 3)[inside][::7])
		reference_values.append(gaussian_filter(b, 3)[inside][::7])
	return np.concatenate(moving_values), np.concatenate(reference_values)


def moment_match(moving, reference) -> tuple[float, float]:
	"""(scale, offset) mapping the moving median and 5-95 % range onto the reference's."""
	pm, pr = np.percentile(moving, [5, 50, 95]), np.percentile(reference, [5, 50, 95])
	scale = (pr[2] - pr[0]) / (pm[2] - pm[0])
	return scale, pr[1] - scale * pm[1]


@click.command(name="norm-fit")
@click.argument("layout_path", metavar="LAYOUT", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--reference", "reference_name", required=True, help="Scan kept at a = 1, b = 0.")
@click.option("--pair", "pairs", multiple=True, required=True, metavar="MOVING:REFERENCE",
				help="Fit MOVING against REFERENCE; pairs chain from --reference, in the given order.")
@click.option("--fov-feather", type=(float, float), required=True, metavar="R_IN R_OUT",
				help="FOV weight ramp used when rendering (as for mosaic stitch).")
@click.option("--erode", type=click.IntRange(min=1), default=61, show_default=True,
				help="Square erosion (px) of the joint coverage before comparing.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path),
				help="Write here instead of updating LAYOUT.")
def norm_fit(layout_path, reference_name, pairs, fov_feather, erode, output):
	"""Fill each scan's a, b in LAYOUT from overlap moment matching."""
	log.start()
	document = read_json(layout_path)
	scans = {scan["name"]: scan for scan in document["scans"]}
	options = {"fov_feather": fov_feather, "row_feather": 48.0, "hp_inner": None, "lp_inner": None,
				"flat_cupping": False, "sigma_lp": 24.0}
	renderer = SingleScanRenderer(document, options)
	maps = {reference_name: (1.0, 0.0)}
	for pair in pairs:
		moving, reference = pair.split(":")
		common = sorted(canvas_rows(scans[moving], document["canvas"], document["nz"])
						& canvas_rows(scans[reference], document["canvas"], document["nz"]))
		rows = common[::max(1, len(common) // 6)][:6]
		values = overlap_values(renderer.render(scans[moving], rows), renderer.render(scans[reference], rows), rows, erode)
		scale, offset = moment_match(*values)
		a_ref, b_ref = maps[reference]
		maps[moving] = (a_ref * scale, a_ref * offset + b_ref)
		log.write("Intensity Map", f"{moving} -> {reference}: slices {rows}, scale {scale:.4f}, offset {offset:.3g}, "
					f"corr {np.corrcoef(*values)[0, 1]:.3f}")
	for scan in document["scans"]:
		scan["a"], scan["b"] = maps.get(scan["name"], (1.0, 0.0))
	write_json(output or layout_path, document)
	log.write("Intensity Maps", json.dumps({name: [round(a, 5), float(b)] for name, (a, b) in maps.items()}))
