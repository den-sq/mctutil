"""Position-dependent separable Wiener filter for a stitched (unfiltered) canvas.

Per canvas pixel the HP-band noise is N(f) = alpha * n_ref(f) + beta * n_fixed(f), with alpha
and beta the reference and fixed-group noise maps written by ``mosaic stitch --noise-out``.
Spectral shapes come from each scan's ``snr`` file (``mosaic spectra``): n_group is the
group median of a^2 nps_i / s_i (s_i = sum a^2 nps_i) and the signal S_group the median of
a^2 (ps_i - nps_i). The signal at a pixel mixes the two groups by the fixed-group weight
fraction q implied by r = beta / (alpha + beta). Gains G = S / (S + N) are tabulated on a
(log nu, r) grid, nu = alpha + beta, and every pixel bilinearly blends the filtered grid
outputs: 2-D FFT gains in XY, then a symmetric FIR along z. Uncovered pixels are mean-filled
before filtering and zeroed after. Output is uint16 with a fixed window (0 = no data).
"""

from __future__ import annotations

import collections
from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import cv2, read_json, tifffile, write_json
from mctutil.mosaic.spectra import noise_total
from mctutil.mosaic.window import require_window, to_u16
from mctutil.shared.deps import require
from mctutil.shared.log import log


def scan_spectrum(scan: dict) -> dict:
	data = read_json(scan["snr"])
	gain = scan.get("a", 1.0) ** 2
	nps, ps = gain * np.array(data["nps"]), gain * np.array(data["ps"])
	pz_noise, pz_full = gain * np.array(data["pz_noise"]), gain * np.array(data["pz_full"])
	total = noise_total(scan)
	return {"f": np.array(data["freq"]), "fz": np.array(data["fz"]), "n": nps / total, "S": np.clip(ps - nps, 0, None),
			"nz": pz_noise / total, "Sz": np.clip(pz_full - pz_noise, 0, None), "s": total}


def group_spectra(scans) -> tuple[dict, np.ndarray, np.ndarray]:
	"""Median spectra per group ("reference", "fixed") and the reference frequency axes."""
	spectra = {"reference": [], "fixed": []}
	for scan in scans:
		spectra["fixed" if "off" in scan else "reference"].append(scan_spectrum(scan))
	if not spectra["reference"] or not spectra["fixed"]:
		raise click.ClickException("the Wiener filter needs spectra for both a reference and a fixed group")
	medians = {group: {key: np.median([item[key] for item in items], axis=0) for key in ("n", "S", "nz", "Sz", "s")}
				for group, items in spectra.items()}
	return medians, spectra["reference"][0]["f"], spectra["reference"][0]["fz"]


class GainBank:
	"""XY gains and z FIR taps on the (nu, r) grid."""

	R_LEVELS = np.array([0.0, 0.5, 1.0])

	def __init__(self, medians, f, fz, nu_levels, shape, taps: int):
		self.medians, self.f, self.nu = medians, f, nu_levels
		self.s_ref, self.s_fixed = medians["reference"]["s"], medians["fixed"]["s"]
		height, width = shape
		radius = np.hypot(require("scipy.fft", "mosaic").fftfreq(height)[:, None],
							require("scipy.fft", "mosaic").rfftfreq(width)[None, :])
		length = 2 * (len(fz) - 1)
		middle = length // 2
		window = np.hanning(2 * taps + 3)[1:-1]
		self.xy, firs = [], []
		for nu in nu_levels:
			for r in self.R_LEVELS:
				signal, noise = self.mix(nu, r, "S", "n")
				gain = signal / (signal + noise)
				gain[0] = 1
				self.xy.append(np.interp(radius, f, gain, right=0).astype(np.float32))
				signal_z, noise_z = self.mix(nu, r, "Sz", "nz")
				gain_z = signal_z / (signal_z + noise_z)
				gain_z[0] = 1
				fir = np.fft.fftshift(np.fft.irfft(gain_z, length))[middle - taps:middle + taps + 1] * window
				firs.append(fir / fir.sum())
		self.z = np.array(firs, np.float32)

	def fixed_fraction(self, r: float) -> float:
		"""Fixed-group weight fraction q with r = q^2 s_fixed / ((1 - q)^2 s_ref + q^2 s_fixed)."""
		if r <= 0:
			return 0.0
		if r >= 1:
			return 1.0
		k = np.sqrt(r * self.s_ref / ((1 - r) * self.s_fixed))
		return k / (1 + k)

	def mix(self, nu: float, r: float, signal_key: str, noise_key: str):
		q = self.fixed_fraction(r)
		ref, fixed = self.medians["reference"], self.medians["fixed"]
		signal = (1 - q) * ref[signal_key] + q * fixed[signal_key]
		noise = nu * ((1 - r) * ref[noise_key] + r * fixed[noise_key])
		return signal, noise

	def summary(self, taps: int) -> dict:
		def gain_at(frequency, nu, r):
			signal, noise = self.mix(nu, r, "S", "n")
			return float(np.interp(frequency, self.f, signal / (signal + noise)))

		return {"nu_levels": self.nu.tolist(), "r_levels": self.R_LEVELS.tolist(), "s_reference": float(self.s_ref),
				"s_fixed": float(self.s_fixed), "taps": taps,
				"gxy_at_f": {str(f): [gain_at(f, nu, r) for nu in self.nu for r in self.R_LEVELS]
								for f in (0.05, 0.1, 0.2, 0.3)}}


def positive_noise(noise, rows, scale: float) -> np.ndarray:
	samples = []
	for g in rows:
		maps = noise[g].astype(np.float32) * scale
		total = maps[0] + maps[1]
		samples.append(total[total > 0])
	return np.concatenate(samples) if samples else np.zeros(0, np.float32)


def noise_levels(noise, depth: int, span, scale: float, count: int) -> tuple[np.ndarray, float] | None:
	"""Log-spaced total-noise levels between the 0.5 and 99.5 percentiles, and the low percentile.

	Levels come from every 37th canvas slice, so separate slab runs share them. If those slices
	carry no noise (a map written for slabs only), every slice of the requested span is used
	instead. A single level when the range collapses; None when the requested span has no noise.
	"""
	samples = positive_noise(noise, range(0, depth, 37), scale)
	if samples.size == 0:
		samples = positive_noise(noise, range(*span), scale)
	if samples.size == 0:
		return None
	low, high = np.percentile(samples, 0.5), np.percentile(samples, 99.5)
	if not high > low:
		return np.array([low]), low
	return np.exp(np.linspace(np.log(low), np.log(high), count)), low


def grid_weights(noise_slice, scale: float, nu, nu_low: float, r_count: int, shape) -> dict:
	"""Bilinear (nu, r) grid weights at full resolution: grid index -> (H, W) weight.

	Computed on the 1/8-resolution noise maps (uncovered cells take their covered neighbours'
	values through a normalised blur) and upsampled; the weights are smooth, so this matches
	full-resolution evaluation.
	"""
	opencv = cv2()
	maps = noise_slice.astype(np.float32) * scale
	total = maps[0] + maps[1]
	covered = (total > 0).astype(np.float32)
	if not covered.any():
		return {}
	density = opencv.GaussianBlur(covered, (0, 0), 3) + 1e-9
	alpha = np.where(covered > 0, maps[0], opencv.GaussianBlur(maps[0] * covered, (0, 0), 3) / density)
	beta = np.where(covered > 0, maps[1], opencv.GaussianBlur(maps[1] * covered, (0, 0), 3) / density)
	total = np.maximum(alpha + beta, nu_low * 1e-3)
	r = beta / total
	if len(nu) > 1:
		u = np.clip((np.log(np.maximum(total, nu[0])) - np.log(nu[0])) / (np.log(nu[-1]) - np.log(nu[0])) * (len(nu) - 1),
					0, len(nu) - 1 - 1e-6)
	else:
		u = np.zeros_like(total)
	v = np.clip(r * (r_count - 1), 0, r_count - 1 - 1e-6)
	iu, iv = u.astype(np.int32), v.astype(np.int32)
	fu, fv = u - iu, v - iv
	low = {}
	for du, wu in ((0, 1 - fu), (1, fu))[:2 if len(nu) > 1 else 1]:
		for dv, wv in ((0, 1 - fv), (1, fv)):
			index = (iu + du) * r_count + (iv + dv)
			weight = wu * wv
			for k in np.unique(index):
				low[k] = low.get(k, 0) + np.where(index == k, weight, 0)
	height, width = shape
	return {k: opencv.resize(value.astype(np.float32), (width, height), interpolation=opencv.INTER_LINEAR)
			for k, value in low.items() if value.max() > 1e-4}


class Filter:
	"""Streams canvas slices through the XY blend and the z FIR."""

	def __init__(self, source: Path, noise, bank: GainBank, nu_low: float, options: dict):
		self.source, self.noise, self.bank, self.nu_low = source, noise, bank, nu_low
		self.scale, self.depth, self.shape = options["noise_scale"], options["depth"], options["shape"]
		self.workers = options["fft_workers"]
		self.fft = require("scipy.fft", "mosaic")
		self.tiff = tifffile()

	def weights(self, g: int) -> dict:
		return grid_weights(self.noise[g], self.scale, self.bank.nu, self.nu_low, len(GainBank.R_LEVELS), self.shape)

	def xy_blend(self, g: int) -> tuple[np.ndarray, np.ndarray]:
		g = min(max(g, 0), self.depth - 1)
		image = self.tiff.imread(self.source / f"{g:05d}.tif").astype(np.float32)
		covered = image != 0
		weights = self.weights(g)
		if not covered.any():
			return np.zeros(self.shape, np.float32), covered
		if not weights:
			raise click.ClickException(
				f"slice {g} is covered but its noise map is empty; the z filter reads slices g0 - taps to "
				"g1 + taps, so stitch --noise-out must cover them")
		mean = image[covered].mean()
		image = np.where(covered, image, mean)
		spectrum = self.fft.rfft2(image, workers=self.workers)
		blended = np.zeros(self.shape, np.float32)
		total = np.zeros(self.shape, np.float32)
		for k, weight in weights.items():
			blended += weight * self.fft.irfft2(spectrum * self.bank.xy[k], self.shape, workers=self.workers).astype(np.float32)
			total += weight
		blended /= np.maximum(total, 1e-6)
		return np.where(covered, blended, mean).astype(np.float32), covered

	def z_blend(self, g: int, buffer, taps: int):
		"""Filtered slice g and its coverage, or (None, coverage) when nothing is covered."""
		covered = buffer[g][1]
		weights = self.weights(g) if covered.any() else {}
		if not weights:
			return None, covered
		levels = sorted(weights)
		height, width = self.shape
		stack = np.stack([buffer[z][0] for z in range(g - taps, g + taps + 1)]).reshape(2 * taps + 1, -1)
		filtered = (self.bank.z[levels] @ stack).reshape(len(levels), height, width)
		result = np.zeros(self.shape, np.float32)
		total = np.zeros(self.shape, np.float32)
		for i, k in enumerate(levels):
			result += weights[k] * filtered[i]
			total += weights[k]
		return result / np.maximum(total, 1e-6), covered


def write_identity(source: Path, target: Path, window, span) -> None:
	tiff = tifffile()
	for g in range(*span):
		output = target / f"{g:05d}.tif"
		if output.exists():
			continue
		image = tiff.imread(source / f"{g:05d}.tif").astype(np.float32)
		tiff.imwrite(output, to_u16(image, *window, image != 0))


def run_filter(flow: Filter, target: Path, window, span, taps: int) -> None:
	low, high = window
	buffer = collections.OrderedDict()
	for g in range(*span):
		output = target / f"{g:05d}.tif"
		if output.exists():
			continue
		for z in range(g - taps, g + taps + 1):
			if z not in buffer:
				buffer[z] = flow.xy_blend(z)
		for z in list(buffer):
			if z < g - taps:
				del buffer[z]
		values, covered = flow.z_blend(g, buffer, taps)
		if values is None:
			flow.tiff.imwrite(output, np.zeros(flow.shape, np.uint16))
			continue
		flow.tiff.imwrite(output, to_u16(values, low, high, covered))
		if g % 250 == 0:
			log.write("Filtered", f"slice {g}")


@click.command()
@click.argument("layout_path", metavar="LAYOUT", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("source", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("noise_path", metavar="NOISE", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(file_okay=False, path_type=Path))
@click.option("--window", type=(float, float), required=True, metavar="LO HI",
				help="uint16 window: clip((v - LO) / (HI - LO)) * 65534 + 1; 0 = no data.")
@click.option("--g0", type=click.IntRange(min=0), default=0, show_default=True, help="First canvas slice.")
@click.option("--g1", type=click.IntRange(min=0), help="End canvas slice (default: layout nz).")
@click.option("--taps", type=click.IntRange(min=1), default=12, show_default=True, help="z FIR half-length.")
@click.option("--nu-levels", type=click.IntRange(min=2), default=4, show_default=True, help="Noise-level grid size.")
@click.option("--fft-workers", type=click.IntRange(min=1), default=4, show_default=True, help="scipy.fft workers.")
@click.option("--noise-scale", type=float, default=1.0, show_default=True, help="Multiply the noise maps.")
def wiener(layout_path, source, noise_path, output, window, g0, g1, taps, nu_levels, fft_workers, noise_scale):
	"""Wiener-filter the stitched float32 SOURCE of LAYOUT with NOISE maps into uint16 OUTPUT (resumable)."""
	require_window(window)
	log.start()
	document = read_json(layout_path)
	shape, depth = tuple(document["canvas"]), document["nz"]
	medians, f, fz = group_spectra(document["scans"])
	noise = np.load(noise_path, mmap_mode="r")
	span = (g0, depth if g1 is None else g1)
	levels = noise_levels(noise, depth, span, noise_scale, nu_levels)
	output.mkdir(parents=True, exist_ok=True)
	if levels is None:
		# No noise anywhere in the requested span: the Wiener gain is 1, so the output is the windowed source.
		write_json(output.parent / f"{output.name}_wiener_params.json", {"identity": True})
		write_identity(source, output, window, span)
		return
	nu, nu_low = levels
	bank = GainBank(medians, f, fz, nu, shape, taps)
	write_json(output.parent / f"{output.name}_wiener_params.json", bank.summary(taps))
	options = {"noise_scale": noise_scale, "depth": depth, "shape": shape, "fft_workers": fft_workers}
	run_filter(Filter(source, noise, bank, nu_low, options), output, window, span, taps)
