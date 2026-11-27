"""Mean radial profile of a reconstruction and its excess near the FOV edge."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import ndimage, tifffile
from mctutil.shared.log import log


def radial_profile(paths) -> np.ndarray:
	"""Mean value per integer radius about the slice centre, averaged over the given slices."""
	tiff = tifffile()
	first = tiff.imread(paths[0])
	n = first.shape[0]
	centre = (n - 1) / 2
	radius = np.hypot(*(np.mgrid[:n, :n] - centre)).astype(np.int32).ravel()
	counts = np.bincount(radius)
	total = np.zeros(counts.size)
	for path in paths:
		total += np.bincount(radius, tiff.imread(path).astype(np.float64).ravel(), minlength=counts.size)
	return total / np.maximum(counts, 1) / len(paths)


def edge_excess(profile: np.ndarray, r_lo: float, r_ref: float, ramp: float, smooth: float) -> np.ndarray:
	"""Profile minus a quadratic fitted over [r_lo, r_ref), ramped in beyond r_ref; zero inside."""
	radius = np.arange(profile.size)
	window = slice(int(r_lo), int(r_ref))
	trend = np.polyval(np.polyfit(radius[window], profile[window], 2), radius)
	weight = np.clip((radius - r_ref) / ramp, 0, 1)
	excess = ndimage().gaussian_filter1d(weight * (profile - trend), smooth)
	excess[radius < r_ref - 4] = 0.0
	return excess


@click.command()
@click.argument("recon_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--r-lo", type=float, required=True, help="Inner radius of the quadratic trend fit (px).")
@click.option("--r-ref", type=float, required=True, help="Radius beyond which the excess is kept (px).")
@click.option("--step", type=click.IntRange(min=1), default=16, show_default=True, help="Average every n-th slice.")
@click.option("--ramp", type=float, default=20.0, show_default=True, help="Ramp width beyond r_ref (px).")
@click.option("--smooth", type=float, default=1.0, show_default=True, help="Gaussian smoothing of the excess (px).")
def profile(recon_dir, output, r_lo, r_ref, step, ramp, smooth):
	"""Write (radius, edge excess, mean profile) as a (3, R) .npy for the stitcher's radial correction.

	The profile rises smoothly with radius (interior-tomography cupping); the trend is kept and
	only the excess beyond r_ref is removed, unless the stitcher is asked to remove the whole profile.
	"""
	log.start()
	paths = sorted(path for path in recon_dir.glob("*.tif") if path.stem.isdigit())[::step]
	if not paths:
		raise click.ClickException(f"no NNNNN.tif rows in {recon_dir}")
	mean = radial_profile(paths)
	excess = edge_excess(mean, r_lo, r_ref, ramp, smooth)
	np.save(output, np.stack([np.arange(mean.size).astype(np.float64), excess, mean]))
	log.write("Radial Profile", f"{len(paths)} slice(s) -> {output}")
