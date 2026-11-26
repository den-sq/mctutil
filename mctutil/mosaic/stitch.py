"""Stitch row-stack reconstructions onto one canvas through a mosaic layout.

Each scan contributes through its 3x4 canvas -> scan map. Scans whose map keeps the row
independent of (y, x) and integer use a 2-D warp of one row; for tilted maps every scan
row is warped once onto the canvas (cached) and each canvas pixel interpolates linearly
between the two rows bracketing its fractional scan row.

Weights are an FOV-disk feather times a feather over the scan's first and last rows.
Fusion is frequency-split: out = sum(w HP) / sum(w) + sum(u LP) / sum(u), with LP the
weight-normalised Gaussian of each scan and u = w * lowpass_weight. Scans with an ``off``
field (written by ``mosaic layout``) form the fixed group; the rest are the reference group.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import RowStack, cv2, read_json, tifffile
from mctutil.shared.log import log


class ScanSampler:
	"""One layout scan resampled onto the canvas grid."""

	def __init__(self, scan: dict, grid, centre: float, options: dict):
		yy, xx = grid
		self.scan, self.stack = scan, RowStack(scan["dir"])
		self.fixed = "off" in scan
		self.nrow = scan["nrow"]
		self.a, self.b = scan.get("a", 1.0), scan.get("b", 0.0)
		self.lowpass_weight = scan.get("lowpass_weight", 1.0)
		self.noise = scan.get("noise", 0.0)
		self.row_feather = options["row_feather"]
		a = self.A = np.array(scan["A"], np.float64)
		self.planar = abs(a[0, 1]) < 1e-9 and abs(a[0, 2]) < 1e-9
		sy = (a[1, 1] * yy + a[1, 2] * xx + a[1, 3]).astype(np.float32)
		sx = (a[2, 1] * yy + a[2, 2] * xx + a[2, 3]).astype(np.float32)
		radius = np.hypot(sy - centre, sx - centre)
		inner, outer = options["fov_feather"]
		self.disk = np.clip((outer - radius) / (outer - inner), 0, 1).astype(np.float32)
		self.zpl = (a[0, 1] * yy + a[0, 2] * xx).astype(np.float32)
		self._bounds(self.disk > 0)
		self.rc = self._radial_correction(radius, options["flat_cupping"])
		self.hp_in = self._inner_ramp(radius, options["hp_inner"]) if self.fixed else None
		self.lp_in = None if self.fixed else self._inner_ramp(radius, options["lp_inner"])
		self.rows, self.warped_rows = OrderedDict(), OrderedDict()

	def _bounds(self, inside) -> None:
		ys, xs = np.nonzero(inside)
		self.bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1) if inside.any() else (0, 1, 0, 1)
		y0, y1, x0, x1 = self.bbox
		# sample() reads every row spanned by the bounding box, not just the disk.
		zbb = self.zpl[y0:y1, x0:x1]
		self.wmax = int(np.ceil(zbb.max() - zbb.min())) + 6
		self.zlo = float(self.zpl[inside].min()) if inside.any() else 0.0
		self.zhi = float(self.zpl[inside].max()) if inside.any() else 0.0

	def _radial_correction(self, radius, flat_cupping: bool):
		if not self.scan.get("radcorr"):
			return None
		profile = np.load(self.scan["radcorr"])
		if flat_cupping:
			return (np.interp(radius, profile[0], profile[2]) - profile[2][0]).astype(np.float32)
		return np.interp(radius, profile[0], profile[1]).astype(np.float32)

	@staticmethod
	def _inner_ramp(radius, ramp):
		if not ramp:
			return None
		r1, r2 = ramp
		return np.clip((r2 - radius) / (r2 - r1), 0, 1).astype(np.float32)

	def row(self, z: int) -> np.ndarray:
		if z not in self.rows:
			self.rows[z] = self.stack.read(z)
			while len(self.rows) > 4:
				self.rows.popitem(last=False)
		return self.rows[z]

	def warped(self, z: int) -> np.ndarray:
		"""Scan row z mapped onto the scan's canvas bounding box (2-D affine at that row's depth).

		The canvas depth is solved at the bounding-box centre; the in-plane map varies with g
		only by A[1:, 0] per row, far below a pixel across the box.
		"""
		if z not in self.warped_rows:
			opencv, a = cv2(), self.A
			y0, y1, x0, x1 = self.bbox
			yc, xc = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
			g = (z - a[0, 1] * yc - a[0, 2] * xc - a[0, 3]) / a[0, 0]
			matrix = np.array([[a[2, 2], a[2, 1], a[2, 0] * g + a[2, 1] * y0 + a[2, 2] * x0 + a[2, 3]],
								[a[1, 2], a[1, 1], a[1, 0] * g + a[1, 1] * y0 + a[1, 2] * x0 + a[1, 3]]], np.float64)
			self.warped_rows[z] = opencv.warpAffine(
				self.row(z), matrix, (x1 - x0, y1 - y0), flags=opencv.INTER_LINEAR | opencv.WARP_INVERSE_MAP)
			while len(self.warped_rows) > self.wmax:
				self.warped_rows.popitem(last=False)
		return self.warped_rows[z]

	def _planar_values(self, z: int, shape):
		opencv, a = cv2(), self.A
		height, width = shape
		matrix = np.array([[a[2, 2], a[2, 1], a[2, 3]], [a[1, 2], a[1, 1], a[1, 3]]], np.float64)
		values = opencv.warpAffine(self.row(z), matrix, (width, height), flags=opencv.INTER_LINEAR | opencv.WARP_INVERSE_MAP)
		return values, np.full(shape, float(z), np.float32)

	def _tilted_values(self, zc: float, shape):
		y0, y1, x0, x1 = self.bbox
		zz = (zc + self.zpl).astype(np.float32)
		zb = np.clip(zz[y0:y1, x0:x1], 0, self.nrow - 1)
		lo, hi = int(np.floor(zb.min())), min(int(np.floor(zb.max())) + 1, self.nrow - 1)
		stack = np.stack([self.warped(z) for z in range(lo, hi + 1)])
		i0 = np.clip(np.floor(zb).astype(np.int32) - lo, 0, hi - lo)
		i1 = np.minimum(i0 + 1, hi - lo)
		t = zb - np.floor(zb)
		iy, ix = np.indices(zb.shape)
		values = np.zeros(shape, np.float32)
		values[y0:y1, x0:x1] = (1 - t) * stack[i0, iy, ix] + t * stack[i1, iy, ix]
		return values, zz

	def sample(self, g: int, shape):
		"""(values, weight) for canvas slice g, or None if the scan misses it."""
		zc = self.A[0, 0] * g + self.A[0, 3]
		z0, z1 = int(np.floor(zc + self.zlo)), int(np.ceil(zc + self.zhi))
		if z1 < 0 or z0 > self.nrow - 1:
			return None
		if self.planar and abs(zc - round(zc)) < 1e-6:
			z = int(round(zc))
			if not 0 <= z < self.nrow:
				return None
			values, zz = self._planar_values(z, shape)
		else:
			values, zz = self._tilted_values(zc, shape)
		fz = np.clip((np.minimum(zz, self.nrow - 1 - zz) + 1) / self.row_feather, 0, 1)
		fz[(zz < -0.5) | (zz > self.nrow - 0.5)] = 0
		if self.rc is not None:
			values = values - self.rc
		return self.a * values + self.b, self.disk * fz


def coverage(samples, fixed: bool, shape) -> np.ndarray:
	return np.clip(sum((weight for sampler, (_, weight) in samples if sampler.fixed == fixed),
						np.zeros(shape, np.float32)), 0, 1)


def lowpass_weights(samples, lp_inner: bool, shape) -> dict:
	"""Per-scan LP weight u. With an LP handover the reference LP fades toward its FOV edge where the
	fixed group covers, and the fixed LP weight rises to fill the gap."""
	if not lp_inner:
		return {id(sampler): weight * sampler.lowpass_weight for sampler, (_, weight) in samples}
	fixed_cover = coverage(samples, True, shape)
	reference = {
		id(sampler): weight * sampler.lowpass_weight * (1 - fixed_cover * (1 - sampler.lp_in))
		for sampler, (_, weight) in samples if sampler.lp_in is not None
	}
	reference_total = np.clip(sum(reference.values(), np.zeros(shape, np.float32)), 0, 1)
	weights = {}
	for sampler, (_, weight) in samples:
		if id(sampler) in reference:
			weights[id(sampler)] = reference[id(sampler)]
		elif sampler.fixed:
			weights[id(sampler)] = weight * np.maximum(sampler.lowpass_weight, 1 - reference_total)
		else:
			weights[id(sampler)] = weight * sampler.lowpass_weight
	return weights


class Fusion:
	"""Accumulators for one canvas slice."""

	def __init__(self, shape):
		self.hp, self.lp, self.ws, self.us, self.wt = (np.zeros(shape, np.float32) for _ in range(5))
		self.noise = np.zeros((2, len(range(0, shape[0], 8)), len(range(0, shape[1], 8))), np.float32)

	def result(self) -> tuple[np.ndarray, np.ndarray]:
		covered = (self.wt > 1e-3) & (self.ws > 1e-6)
		values = np.where(covered, self.hp / np.maximum(self.ws, 1e-6) + self.lp / np.maximum(self.us, 1e-9), 0)
		return values.astype(np.float32), covered

	def noise_map(self) -> np.ndarray:
		"""HP-band noise variance of the weighted mean, sum(w_i^2 N_i) / (sum w)^2, split by group."""
		ws8 = self.ws[::8, ::8]
		return (self.noise / np.maximum(ws8, 1e-3) ** 2 * (ws8 > 1e-3)).astype(np.float16)


def fuse(samplers, g: int, shape, options: dict) -> Fusion:
	opencv = cv2()
	sigma = options["sigma_lp"]
	kernel = int(4 * sigma) | 1
	samples = [(sampler, sampler.sample(g, shape)) for sampler in samplers]
	samples = [(sampler, value) for sampler, value in samples if value is not None]
	reference_cover = coverage(samples, False, shape)
	lowpass = lowpass_weights(samples, options["lp_inner"] is not None, shape)
	fusion = Fusion(shape)
	for sampler, (values, weight) in samples:
		num = opencv.GaussianBlur(values * weight, (kernel, kernel), sigma)
		den = opencv.GaussianBlur(weight, (kernel, kernel), sigma)
		low = num / np.maximum(den, 1e-6)
		high_weight = weight * (1 - reference_cover * (1 - sampler.hp_in)) if sampler.hp_in is not None else weight
		fusion.hp += high_weight * (values - low)
		fusion.ws += high_weight
		fusion.wt += weight
		w8 = high_weight[::8, ::8]
		fusion.noise[1 if sampler.fixed else 0] += w8 * w8 * sampler.noise
		fusion.lp += lowpass[id(sampler)] * low
		fusion.us += lowpass[id(sampler)]
	return fusion


def open_noise_map(path: Path, depth: int, shape):
	shape = (depth, 2, len(range(0, shape[0], 8)), len(range(0, shape[1], 8)))
	if path.exists():
		return np.lib.format.open_memmap(path, mode="r+")
	return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)


def slice_centre(document: dict) -> float:
	size = document.get("slice_size") or RowStack(document["scans"][0]["dir"]).size
	return (size - 1) / 2


def canvas_grid(document: dict) -> np.ndarray:
	height, width = document["canvas"]
	return np.mgrid[0:height, 0:width].astype(np.float32)


def samplers_for(document: dict, options: dict) -> list[ScanSampler]:
	grid, centre = canvas_grid(document), slice_centre(document)
	return [ScanSampler(scan, grid, centre, options) for scan in document["scans"]]


@click.command()
@click.argument("layout_path", metavar="LAYOUT", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(file_okay=False, path_type=Path))
@click.option("--fov-feather", type=(float, float), required=True, metavar="R_IN R_OUT",
				help="FOV weight ramps from 1 at R_IN to 0 at R_OUT (scan radius, px).")
@click.option("--g0", type=click.IntRange(min=0), default=0, show_default=True, help="First canvas slice.")
@click.option("--g1", type=click.IntRange(min=0), help="End canvas slice (default: layout nz).")
@click.option("--sigma-lp", type=float, default=24.0, show_default=True, help="Low/high-pass split (px).")
@click.option("--row-feather", type=float, default=48.0, show_default=True,
				help="Weight ramp over each scan's first and last rows.")
@click.option("--hp-inner", type=(float, float), metavar="R1 R2",
				help="Fixed-group HP weight ramps to 0 between R1 and R2 (scan radius) where the reference covers.")
@click.option("--lp-inner", type=(float, float), metavar="R1 R2",
				help="Reference LP weight ramps to 0 between R1 and R2 where the fixed group covers.")
@click.option("--flat-cupping", is_flag=True,
				help="Subtract each scan's whole mean radial profile p(r) - p(0) instead of its edge excess.")
@click.option("--noise-out", type=click.Path(dir_okay=False, path_type=Path),
				help="Write per-slice HP noise maps (reference, fixed) at every 8th px, float16 .npy memmap.")
def stitch(layout_path, output, fov_feather, g0, g1, sigma_lp, row_feather, hp_inner, lp_inner, flat_cupping,
			noise_out):
	"""Write the stitched float32 canvas slices g0..g1 of LAYOUT to OUTPUT (resumable)."""
	log.start()
	document = read_json(layout_path)
	options = {"fov_feather": fov_feather, "row_feather": row_feather, "hp_inner": hp_inner, "lp_inner": lp_inner,
				"flat_cupping": flat_cupping, "sigma_lp": sigma_lp}
	samplers = samplers_for(document, options)
	shape = tuple(document["canvas"])
	output.mkdir(parents=True, exist_ok=True)
	noise = open_noise_map(noise_out, document["nz"], shape) if noise_out else None
	tiff = tifffile()
	for g in range(g0, document["nz"] if g1 is None else g1):
		target = output / f"{g:05d}.tif"
		if target.exists():
			continue
		fusion = fuse(samplers, g, shape, options)
		values, _ = fusion.result()
		if noise is not None:
			noise[g] = fusion.noise_map()
		tiff.imwrite(target, values)
		if g % 250 == 0:
			log.write("Stitched", f"slice {g}")
