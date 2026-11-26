"""Cross-scan rigid + scale refinement with multi-slice correlation averaging.

Convention: fixed(p) ~ moving(A^-1 (p - c - d) + c), A = scale * R(phi) in OpenCV's
rotation sign, c the slice centre and d = (dy, dx).
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import RowStack, band_pass, cv2, fov_disk, rotation, subpixel_peak, write_json
from mctutil.shared.log import log


class StackCorrelator:
	"""Normalised cross-power averaged over ``slices`` fixed rows spaced two apart."""

	def __init__(self, fixed: RowStack, moving: RowStack, *, slices: int, disk_fraction: float, disk_taper: float):
		self.fixed, self.moving, self.slices = fixed, moving, slices
		self.n = fixed.size
		self.disk = fov_disk(self.n, disk_fraction, disk_taper)
		self.cache = {}

	def filtered(self, stack: RowStack, row: int) -> np.ndarray:
		return band_pass(stack.read(row), 2.0, 16.0) * self.disk

	def fixed_rows(self, row: int) -> list[np.ndarray]:
		return [self.filtered(self.fixed, row + 2 * k) for k in range(self.slices)]

	def moving_row(self, row: int) -> np.ndarray:
		if row not in self.cache:
			self.cache[row] = self.filtered(self.moving, row)
		return self.cache[row]

	def correlate(self, fixed_rows, row, params, binning) -> tuple[float, list[float]]:
		"""Mean peak and sub-pixel shift (full-resolution px) for one parameter set."""
		opencv, n = cv2(), self.n
		matrix = rotation(n, params["phi"], params["s"], params["dy"], params["dx"])
		mask = opencv.warpAffine(self.disk, matrix, (n, n))
		total = None
		for k in range(self.slices):
			z = row + params["dz"] + 2 * k
			if not 0 <= z < self.moving.rows:
				continue
			a = fixed_rows[k] * mask
			b = opencv.warpAffine(self.moving_row(z), matrix, (n, n), flags=opencv.INTER_LINEAR)
			if binning > 1:
				a = opencv.resize(a, (n // binning, n // binning), interpolation=opencv.INTER_AREA)
				b = opencv.resize(b, (n // binning, n // binning), interpolation=opencv.INTER_AREA)
			cross = np.fft.rfft2(a) * np.conj(np.fft.rfft2(b))
			cross /= np.maximum(np.abs(cross), 1e-12)
			total = cross if total is None else total + cross
		if total is None:
			raise click.ClickException(f"no moving rows overlap fixed row {row} at dz {params['dz']}")
		peak, shift = subpixel_peak(np.fft.irfft2(total, (n // binning, n // binning)))
		return peak / self.slices, [value * binning for value in shift]


def descend(correlator: StackCorrelator, fixed_rows, row: int, params: dict) -> dict:
	"""Two rounds of coordinate descent over phi, scale, dz at half resolution.

	Every candidate is evaluated with the shift its own correlation measures, and each
	round's candidate values are relative to the parameters at the start of that round.
	"""
	def evaluate(**change):
		trial = {**params, **change}
		peak, (sy, sx) = correlator.correlate(fixed_rows, row, trial, 2)
		return peak, {**trial, "dy": trial["dy"] + sy, "dx": trial["dx"] + sx}

	for round_index in range(2):
		span = 0.6 if round_index == 0 else 0.15
		candidates = (
			("phi", params["phi"] + np.linspace(-span, span, 7)),
			("s", params["s"] + np.linspace(-0.006, 0.006, 7) / (round_index + 1)),
			("dz", params["dz"] + np.arange(-4, 5) // (round_index + 1)),
		)
		for key, values in candidates:
			params = max((evaluate(**{key: value}) for value in values), key=lambda item: item[0])[1]
	return params


def refine_row(correlator: StackCorrelator, row: int, start: dict) -> dict:
	fixed_rows = correlator.fixed_rows(row)
	params = descend(correlator, fixed_rows, row, dict(start))
	peak, (sy, sx) = correlator.correlate(fixed_rows, row, params, 1)
	params["dy"] += sy
	params["dx"] += sx
	neighbour = max(
		correlator.correlate(fixed_rows, row, {**params, "dz": params["dz"] + dd}, 1)[0] for dd in (-2, 2)
	)
	correlator.cache.clear()
	return {
		"row": row, **{key: round(float(value), 4) for key, value in params.items()},
		"peak": round(peak, 4), "peak_dz2": round(neighbour, 4),
	}


@click.command()
@click.argument("fixed_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("moving_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.argument("rows", type=int, nargs=-1, required=True)
@click.option("--dz", type=int, required=True, help="Starting depth offset: moving row = fixed row + dz.")
@click.option("--phi", type=float, required=True, help="Starting rotation (degrees).")
@click.option("--dy", type=float, required=True, help="Starting row shift (px).")
@click.option("--dx", type=float, required=True, help="Starting column shift (px).")
@click.option("--slices", type=click.IntRange(min=1), default=12, show_default=True,
				help="Fixed rows (spaced two apart) averaged per correlation.")
@click.option("--disk-fraction", type=float, default=0.86, show_default=True,
				help="FOV mask radius as a fraction of the slice half-width.")
@click.option("--disk-taper", type=float, default=60.0, show_default=True, help="FOV mask edge ramp (px).")
def refine(fixed_dir, moving_dir, output, rows, dz, phi, dy, dx, slices, disk_fraction, disk_taper):
	"""Refine a coarse rigid link at each sampled fixed ROW and write the per-row table to OUTPUT."""
	log.start()
	correlator = StackCorrelator(
		RowStack(fixed_dir), RowStack(moving_dir),
		slices=slices, disk_fraction=disk_fraction, disk_taper=disk_taper,
	)
	start = {"dz": dz, "phi": phi, "s": 1.0, "dy": dy, "dx": dx}
	results = []
	for row in rows:
		results.append(refine_row(correlator, row, start))
		log.write("Refined Row", str(results[-1]))
		write_json(output, results)
