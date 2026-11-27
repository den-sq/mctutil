"""Signal and noise spectra of a reconstruction block from a simulated noise realisation.

``FULL`` and ``NOISY`` are (Z, N, N) .npy reconstructions of the same block, the second
with one extra noise realisation added upstream of reconstruction, so their difference has
the noise spectrum of ``FULL``. Over a central box: the XY radial power spectra of signal
and noise, and the z spectra (kz at kx, ky ~ 0, from 1-D spectra of z columns).
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import read_json, write_json
from mctutil.shared.log import log


def noise_total(scan: dict) -> float:
	"""A scan's noise power in stitched units: a^2 * sum(nps) of its ``snr`` spectra."""
	if not scan.get("snr"):
		raise click.ClickException(f"scan {scan.get('name')} has no snr spectra (mosaic spectra)")
	total = float((scan.get("a", 1.0) ** 2 * np.array(read_json(scan["snr"])["nps"])).sum())
	if not total > 0:
		raise click.ClickException(f"scan {scan.get('name')} has no positive spectral noise")
	return total


def radial(power: np.ndarray) -> np.ndarray:
	"""Mean of a centred 2-D power spectrum per integer radius, up to half the size."""
	n = power.shape[-1]
	y, x = np.indices(power.shape) - n // 2
	radius = np.hypot(y, x).astype(int)
	totals = np.bincount(radius.ravel(), power.ravel(), minlength=n)
	return (totals / np.maximum(np.bincount(radius.ravel(), minlength=n), 1))[: n // 2]


def xy_spectrum(volume: np.ndarray, box: int) -> np.ndarray:
	window = np.outer(np.hanning(box), np.hanning(box))
	total = np.zeros(box // 2)
	for image in volume:
		total += radial(np.abs(np.fft.fftshift(np.fft.fft2((image - image.mean()) * window))) ** 2)
	return total


def z_spectrum(volume: np.ndarray) -> np.ndarray:
	window = np.hanning(volume.shape[0])[:, None, None]
	return (np.abs(np.fft.rfft((volume - volume.mean(0)) * window, axis=0)) ** 2).mean(axis=(1, 2))


def block_spectra(full, noisy, box: int, centre) -> dict:
	cy, cx = centre
	rows, columns = slice(cy - box // 2, cy + box // 2), slice(cx - box // 2, cx + box // 2)
	signal = np.asarray(full[:, rows, columns], np.float64)
	noise = np.asarray(noisy[:, rows, columns], np.float64) - signal
	ps, nps = xy_spectrum(signal, box), xy_spectrum(noise, box)
	return {
		"freq": (np.arange(box // 2) / box).tolist(), "snr": ((ps - nps) / nps).tolist(),
		"nps": nps.tolist(), "ps": ps.tolist(), "fz": np.fft.rfftfreq(signal.shape[0]).tolist(),
		"pz_full": z_spectrum(signal).tolist(), "pz_noise": z_spectrum(noise).tolist(),
	}


@click.command()
@click.argument("full", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("noisy", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--box", type=click.IntRange(min=8), default=768, show_default=True, help="Central box width (px).")
@click.option("--centre", type=(int, int), help="Box centre (y, x); default: the slice centre.")
def spectra(full, noisy, output, box, centre):
	"""Write the XY and z signal/noise spectra of FULL against NOISY to OUTPUT for mosaic wiener."""
	log.start()
	full_volume = np.load(full, mmap_mode="r")
	noisy_volume = np.load(noisy, mmap_mode="r")
	n = full_volume.shape[-1]
	result = block_spectra(full_volume, noisy_volume, box, centre or (n // 2, n // 2))
	write_json(output, result)
	log.write("Spectra", f"{full_volume.shape[0]} slice(s), box {box} -> {output}")
