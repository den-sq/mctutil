"""Pairwise coarse rigid registration of coarse reconstructions (rotation about z + 3-D shift)."""

from __future__ import annotations

import itertools
from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import cv2, ndimage, phase_correlation, read_json, write_json
from mctutil.shared.log import log


def prepare(volume, binning: int) -> np.ndarray:
	"""Bin, band-pass, clip, and mask a coarse volume to its inner FOV disk with a z taper."""
	gaussian_filter = ndimage().gaussian_filter
	volume = np.asarray(volume, np.float32)
	z, y, x = (size // binning * binning for size in volume.shape)
	volume = volume[:z, :y, :x].reshape(
		z // binning, binning, y // binning, binning, x // binning, binning).mean(axis=(1, 3, 5))
	volume = gaussian_filter(volume, 0.7) - gaussian_filter(volume, 6)
	n = volume.shape[-1]
	yy, xx = np.indices(volume.shape[1:]) - (n - 1) / 2
	radius = np.hypot(yy, xx) / (n / 2)
	mask = np.clip((0.82 - radius) / 0.1, 0, 1).astype(np.float32)
	volume = np.clip(volume, *np.percentile(volume, (0.5, 99.5)))
	volume = (volume - volume.mean()) * mask
	taper = np.hanning(volume.shape[0] + 2)[1:-1, None, None].astype(np.float32)
	return volume * np.sqrt(taper)


def rotate_slices(volume, phi: float) -> np.ndarray:
	opencv = cv2()
	n = volume.shape[-1]
	matrix = opencv.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), phi, 1.0)
	return np.stack([opencv.warpAffine(item, matrix, (n, n), flags=opencv.INTER_LINEAR) for item in volume])


def sweep(fixed, moving, phis) -> list[tuple]:
	"""(peak, phi, dz, dy, dx) per angle: moving rotated by phi then shifted by d matches fixed."""
	shape = (fixed.shape[0] + moving.shape[0],) + fixed.shape[1:]
	spectrum = np.fft.rfftn(fixed, shape, axes=(0, 1, 2))
	results = []
	for phi in phis:
		peak, shift = phase_correlation(spectrum, rotate_slices(moving, phi), shape)
		results.append((peak, float(phi), *shift))
	return results


def scan_pairs(scans) -> list[tuple[dict, dict]]:
	"""All within-group pairs, then every cross pair with the earlier group's scan fixed."""
	groups = list(dict.fromkeys(scan["group"] for scan in scans))
	members = {group: [scan for scan in scans if scan["group"] == group] for group in groups}
	pairs = [pair for group in groups for pair in itertools.combinations(members[group], 2)]
	for first, second in itertools.combinations(groups, 2):
		pairs.extend(itertools.product(members[first], members[second]))
	return pairs


def register_pair(fixed, moving, *, phi0: float, span: float, step: float, nulls, scale: int) -> dict:
	results = sweep(fixed, moving, np.arange(phi0 - span, phi0 + span + 0.01, step))
	null = sweep(fixed, moving, [phi0 + offset for offset in nulls])
	best = max(results)
	significance = best[0] / max(np.mean([item[0] for item in null]), 1e-9)
	return {
		"peak": round(best[0], 4), "sig": round(float(significance), 1), "phi": best[1],
		"dz": best[2] * scale, "dy": best[3] * scale, "dx": best[4] * scale,
	}


@click.command()
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--bin", "binning", type=click.IntRange(min=1), default=2, show_default=True,
				help="Additional binning of the coarse volumes before correlation.")
@click.option("--scale", type=click.IntRange(min=1), required=True,
				help="Full-resolution pixels per correlated voxel (coarse binning x --bin).")
@click.option("--within-phi", type=float, default=0.0, show_default=True, help="Angle centre for same-group pairs.")
@click.option("--within-span", type=float, default=2.0, show_default=True, help="Half-range for same-group pairs.")
@click.option("--cross-phi", type=float, default=0.0, show_default=True, help="Angle centre for cross-group pairs.")
@click.option("--cross-span", type=float, default=180.0, show_default=True, help="Half-range for cross-group pairs.")
@click.option("--phi-step", type=float, default=1.0, show_default=True, help="Angle step in degrees.")
@click.option("--null-offset", "nulls", type=float, multiple=True, default=(90.0, 135.0), show_default=True,
				help="Far-off angles whose peaks estimate chance correlation.")
@click.option("--shard", type=(int, int), default=(0, 1), show_default=True,
				help="Process pairs index::count of the full pair list.")
def register(manifest, output, binning, scale, within_phi, within_span, cross_phi, cross_span, phi_step, nulls, shard):
	"""Coarse-register every scan pair in MANIFEST and write the links to OUTPUT.

	MANIFEST is JSON: {"scans": [{"name", "coarse": path to a coarse .npy, "group"}]}.
	Each link gives phi and (dz, dy, dx) in full-resolution pixels, moving -> fixed,
	and its significance: best peak over the mean far-off-angle peak.
	"""
	log.start()
	scans = read_json(manifest)["scans"]
	volumes = {scan["name"]: prepare(np.load(scan["coarse"], mmap_mode="r"), binning) for scan in scans}
	index, count = shard
	links = []
	for fixed, moving in scan_pairs(scans)[index::count]:
		same = fixed["group"] == moving["group"]
		link = register_pair(
			volumes[fixed["name"]], volumes[moving["name"]],
			phi0=within_phi if same else cross_phi, span=within_span if same else cross_span,
			step=phi_step, nulls=nulls, scale=scale,
		)
		links.append({"fixed": fixed["name"], "moving": moving["name"], **link})
		log.write("Link", f"{fixed['name']} <- {moving['name']}: {link}")
		write_json(output, links)
