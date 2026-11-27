"""Shared filters, correlation, and row-stack access for the mosaic commands.

Reconstructions are read as row stacks: one TIFF slice per detector row in
unambiguous natural order, square slices centred on the rotation axis. Rotations
and shifts follow OpenCV's ``getRotationMatrix2D`` convention about the slice centre
``((n - 1) / 2, (n - 1) / 2)``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import click
import numpy as np

from mctutil.shared.deps import require
from mctutil.shared.stack_apply import discover_slices


def ndimage():
	return require(
		"scipy.ndimage",
		"mosaic",
		purpose="scipy is required for mosaic filtering",
		error_type=click.ClickException,
	)


def tifffile():
	return require(
		"tifffile",
		"mosaic",
		purpose="tifffile is required for mosaic row stacks",
		error_type=click.ClickException,
	)


def conda_module(name: str, purpose: str):
	"""Import a module provided only by the conda runtime (OpenCV, TomoPy)."""
	try:
		return importlib.import_module(name)
	except ImportError as exc:
		raise click.ClickException(f"{purpose}; use the environment.yml runtime") from exc


def cv2():
	return conda_module("cv2", "OpenCV is required for mosaic warps")


def band_pass(image, inner: float, outer: float) -> np.ndarray:
	"""Difference of Gaussians in float32."""
	gaussian_filter = ndimage().gaussian_filter
	image = np.asarray(image, np.float32)
	return gaussian_filter(image, inner) - gaussian_filter(image, outer)


def fov_disk(n: int, fraction: float, taper: float) -> np.ndarray:
	"""Weight 1 inside ``fraction`` of the slice half-width, ramping to 0 over ``taper`` px."""
	y, x = np.indices((n, n)) - (n - 1) / 2
	return np.clip((fraction * n / 2 - np.hypot(y, x)) / taper, 0, 1).astype(np.float32)


def rotation(n: int, phi: float, scale: float = 1.0, dy: float = 0.0, dx: float = 0.0) -> np.ndarray:
	"""2x3 warp: rotation by ``phi`` degrees and scale about the slice centre, then (dy, dx)."""
	matrix = cv2().getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), phi, scale)
	matrix[0, 2] += dx
	matrix[1, 2] += dy
	return matrix


def warp(image, matrix, shape=None) -> np.ndarray:
	opencv = cv2()
	height, width = image.shape if shape is None else shape
	return opencv.warpAffine(image, matrix, (width, height), flags=opencv.INTER_LINEAR)


def wrapped(index: int, size: int) -> int:
	"""Shortest signed displacement of a circular index; ties at size / 2 go negative."""
	return index if index < (size + 1) // 2 else index - size


def phase_correlation(fixed_spectrum, moving, shape) -> tuple[float, list[int]]:
	"""Peak and integer shift of the normalised cross-power between a spectrum and an array."""
	cross = fixed_spectrum * np.conj(np.fft.rfftn(moving, shape, axes=tuple(range(len(shape)))))
	surface = np.fft.irfftn(cross / np.maximum(np.abs(cross), 1e-12), shape, axes=tuple(range(len(shape))))
	peak = np.unravel_index(np.argmax(surface), shape)
	return float(surface[peak]), [wrapped(int(k), s) for k, s in zip(peak, shape)]


def subpixel_peak(surface: np.ndarray) -> tuple[float, list[float]]:
	"""Peak value and parabola-refined, wrapped 2-D peak position of a correlation surface."""
	peak = np.unravel_index(np.argmax(surface), surface.shape)
	position = []
	for axis in (0, 1):
		below, above = list(peak), list(peak)
		below[axis] -= 1
		above[axis] = (above[axis] + 1) % surface.shape[axis]
		centre, low, high = surface[peak], surface[tuple(below)], surface[tuple(above)]
		refined = peak[axis] + 0.5 * (low - high) / (low - 2 * centre + high)
		position.append(refined if refined < surface.shape[axis] / 2 else refined - surface.shape[axis])
	return float(surface[peak]), position


def stack_paths(source: Path) -> tuple[Path, ...]:
	"""Discover naturally ordered TIFF rows, reporting invalid stacks through Click."""
	try:
		return discover_slices(source)
	except ValueError as exc:
		raise click.ClickException(str(exc)) from exc


class RowStack:
	"""Read-only access by position in a naturally ordered TIFF stack."""

	def __init__(self, path):
		self.path = Path(path)
		self.paths = stack_paths(self.path)
		self._first = None

	@property
	def rows(self) -> int:
		return len(self.paths)

	def read(self, row: int) -> np.ndarray:
		if not 0 <= row < self.rows:
			raise click.ClickException(f"row {row} is outside the stack of {self.rows} slices in {self.path}")
		return np.asarray(tifffile().imread(self.paths[row]), np.float32)

	@property
	def size(self) -> int:
		if self._first is None:
			self._first = self.read(0)
		if self._first.shape[0] != self._first.shape[1]:
			raise click.ClickException(f"{self.path} rows are not square slices")
		return self._first.shape[0]


def read_json(path):
	return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value) -> None:
	Path(path).write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
