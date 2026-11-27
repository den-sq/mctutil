"""Coarse attenuation reconstruction of a whole DXchange scan for registration."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import conda_module
from mctutil.shared.deps import require
from mctutil.shared.log import log


def binned_attenuation(projections, dark, gain, binning: int) -> np.ndarray:
	"""Flat-corrected ``-log`` of (theta, rows, columns) counts, mean-binned over rows and columns."""
	attenuation = -np.log(np.clip((projections - dark) / gain, 1e-6, None))
	rows = attenuation.shape[1] // binning
	columns = attenuation.shape[2] // binning
	attenuation = attenuation[:, :rows * binning, :columns * binning]
	return attenuation.reshape(attenuation.shape[0], rows, binning, columns, binning).mean(axis=(2, 4))


def chunk_starts(rows: int, chunk_rows: int, binning: int) -> range:
	"""Chunk starts on a multiple of the binning over the complete bins only.

	Every bin keeps its physical rows; the trailing rows % binning rows form no bin.
	"""
	return range(0, rows - rows % binning, max(binning, chunk_rows - chunk_rows % binning))


def coarse_volume(scan, center: float, binning: int, theta_step: int, chunk_rows: int, ncore):
	"""Gridrec of binned attenuation; no phase retrieval or stripe removal."""
	tomopy = conda_module("tomopy", "TomoPy is required for coarse reconstruction")
	data = scan["exchange/data"]
	theta = np.deg2rad(scan["exchange/theta"][()])[::theta_step]
	white = np.median(scan["exchange/data_white"][()], axis=0).astype(np.float32)
	dark = np.median(scan["exchange/data_dark"][()], axis=0).astype(np.float32)
	gain = np.maximum(white - dark, 1.0)
	coarse_center = (center + 0.5) / binning - 0.5
	pieces = []
	starts = chunk_starts(data.shape[1], chunk_rows, binning)
	for y0 in starts:
		y1 = min(y0 + starts.step, starts.stop)
		block = data[::theta_step, y0:y1, :].astype(np.float32)
		attenuation = binned_attenuation(block, dark[y0:y1], gain[y0:y1], binning)
		recon = tomopy.recon(
			np.ascontiguousarray(attenuation), theta, center=coarse_center,
			algorithm="gridrec", filter_name="shepp", ncore=ncore,
		)
		pieces.append(recon.astype(np.float32))
		log.write("Rows", f"{y0}-{y1}")
	return np.concatenate(pieces)


@click.command()
@click.argument("scan", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--center", type=float, required=True, help="Rotation centre in full-resolution detector columns.")
@click.option("--bin", "binning", type=click.IntRange(min=1), default=4, show_default=True,
				help="Row and column binning factor.")
@click.option("--theta-step", type=click.IntRange(min=1), default=2, show_default=True,
				help="Use every n-th projection.")
@click.option("--chunk-rows", type=click.IntRange(min=1), default=128, show_default=True,
				help="Detector rows reconstructed per chunk; rounded down to a multiple of --bin.")
@click.option("--ncore", type=click.IntRange(min=1), help="TomoPy worker threads.")
def coarse(scan, output, center, binning, theta_step, chunk_rows, ncore):
	"""Write a binned (Z, N, N) float32 .npy reconstruction of SCAN for coarse registration."""
	h5py = require("h5py", "mosaic", purpose="h5py is required to read DXchange scans",
					error_type=click.ClickException)
	log.start()
	with h5py.File(scan, "r") as source:
		volume = coarse_volume(source, center, binning, theta_step, chunk_rows, ncore)
	np.save(output, volume)
	log.write("Coarse Volume", f"{output}: {volume.shape}")
