"""Stitch layout (canvas -> scan affine maps) from a mosaic fit.

The canvas is the reference-group frame shifted by (oy, ox): canvas (g, y, x) is global
row g at reference slice position (y - oy, x - ox). Reference scans map by a pure integer
depth translation. A fixed-group point (Z, p) with p = R(phi)(p_ref - c) + c + d0 + dd * Z
(OpenCV (x, y) convention) sits at g = Z (1 + gz) + K + gy * y + gx * x, which is affine in
(g, y, x); each fixed scan's 3x4 map is evaluated from that inverse.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import RowStack, cv2, read_json, write_json
from mctutil.shared.log import log


class GroupRelation:
	"""The fitted fixed-group -> reference-frame relation of a mosaic fit."""

	def __init__(self, fit: dict, centre: float):
		self.centre = centre
		self.k, self.gy, self.gx, self.gz = fit["K"], fit["gy"], fit["gx"], fit["gz"]
		self.d0 = np.array([fit["dx"][1], fit["dy"][1]])
		self.dd = np.array([fit["dx"][0], fit["dy"][0]])
		self.rotation = cv2().getRotationMatrix2D((0, 0), fit["phi"], 1.0)[:, :2]

	def reference_point(self, z: float, x: float, y: float) -> np.ndarray:
		"""Reference-slice (x, y) of fixed-group position (x, y) at depth Z."""
		return np.linalg.solve(self.rotation, np.array([x, y]) - self.centre - self.d0 - self.dd * z) + self.centre

	def fixed_point(self, g: float, x: float, y: float) -> tuple[float, float, float]:
		"""(Z, x, y) in the fixed group of global row g at reference-slice (x, y)."""
		q = self.rotation @ (np.array([x, y]) - self.centre) + self.centre + self.d0
		z = (g - self.k - self.gy * q[1] - self.gx * q[0]) / (1 + self.gz + self.gy * self.dd[1] + self.gx * self.dd[0])
		p = q + self.dd * z
		return z, p[0], p[1]


def canvas_bounds(relation: GroupRelation, radius: float, depths, margin: int = 8) -> tuple[int, int, int, int]:
	"""(oy, ox, H, W) covering the reference FOV disks and the fixed disks at the sampled depths."""
	centre = relation.centre
	points = []
	for angle in np.linspace(0, 2 * np.pi, 72, endpoint=False):
		x, y = centre + radius * np.cos(angle), centre + radius * np.sin(angle)
		points.append((x, y))
		points.extend(tuple(relation.reference_point(z, x, y)) for z in depths)
	points = np.array(points)
	ox = int(np.ceil(-points[:, 0].min())) + margin
	oy = int(np.ceil(-points[:, 1].min())) + margin
	width = int(np.ceil(points[:, 0].max())) + ox + margin
	height = int(np.ceil(points[:, 1].max())) + oy + margin
	return oy, ox, height, width


def fixed_map(relation: GroupRelation, offset: float, oy: int, ox: int) -> list[list[float]]:
	"""3x4 canvas (g, y, x, 1) -> fixed scan (row, y, x) map by finite evaluation of the affine inverse."""
	def scan_point(g, y, x):
		z, xs, ys = relation.fixed_point(g, x - ox, y - oy)
		return np.array([z - offset, ys, xs])

	origin = scan_point(0, 0, 0)
	columns = [scan_point(1, 0, 0) - origin, scan_point(0, 1, 0) - origin, scan_point(0, 0, 1) - origin, origin]
	return np.stack(columns, axis=1).tolist()


def scan_entry(scan: dict, stack: RowStack, matrix, lowpass_weight: float, **extra) -> dict:
	entry = {"name": scan["name"], "dir": str(scan["dir"]), "nrow": stack.rows, "A": matrix,
				"lowpass_weight": lowpass_weight, **extra}
	if "group" in scan:
		entry["group"] = scan["group"]
	return entry


@click.command()
@click.argument("fit_path", metavar="FIT", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--fov-radius", type=float, required=True, help="Reconstructed field-of-view radius (px).")
@click.option("--depth-sample", "depths", type=float, multiple=True,
				help="Fixed-group depths sampled for the canvas bounds (default: first, middle, and last row).")
@click.option("--reference-lowpass", type=float, default=1.0, show_default=True,
				help="Low-pass fusion weight of reference scans.")
@click.option("--fixed-lowpass", type=float, default=1.0, show_default=True,
				help="Low-pass fusion weight of fixed-group scans.")
def layout(fit_path, manifest, output, fov_radius, depths, reference_lowpass, fixed_lowpass):
	"""Write the stitch layout for the scans in MANIFEST ({"scans": [{"name", "dir", "group"?}]}) from FIT."""
	log.start()
	fit_data = read_json(fit_path)
	scans = {scan["name"]: scan for scan in read_json(manifest)["scans"]}
	stacks = {name: RowStack(scans[name]["dir"]) for name in [*fit_data["reference"], *fit_data["fixed"]]}
	size = next(iter(stacks.values())).size
	relation = GroupRelation(fit_data, (size - 1) / 2)
	if not depths:
		last = max(offset + stacks[name].rows - 1 for name, offset in fit_data["fixed"].items())
		depths = (0.0, last / 2, last)
	oy, ox, height, width = canvas_bounds(relation, fov_radius, depths)
	entries = []
	for name, offset in fit_data["reference"].items():
		matrix = [[1, 0, 0, -int(round(offset))], [0, 1, 0, -oy], [0, 0, 1, -ox]]
		entries.append(scan_entry(scans[name], stacks[name], matrix, reference_lowpass))
	for name, offset in fit_data["fixed"].items():
		matrix = fixed_map(relation, offset, oy, ox)
		entries.append(scan_entry(scans[name], stacks[name], matrix, fixed_lowpass, off=offset))
	depth = max(int(round(offset)) + stacks[name].rows for name, offset in fit_data["reference"].items())
	write_json(output, {
		"canvas": [height, width], "nz": depth, "offset_yx": [oy, ox], "slice_size": size,
		"fov_radius": fov_radius, "fit": fit_data, "scans": entries,
	})
	log.write("Layout", f"canvas {height} x {width}, nz {depth}, offset ({oy}, {ox}), {len(entries)} scan(s)")
