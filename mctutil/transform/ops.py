"""Pure volume operations restored for the fused transform pipeline."""

from __future__ import annotations

import numpy as np


def _axis_index(axis: int, ndim: int) -> int:
	axis = int(axis)
	if axis < 0:
		axis += ndim
	if not 0 <= axis < ndim:
		raise ValueError(f"axis {axis} is out of bounds for a {ndim}D array")
	return axis


def maximum_intensity_projection(
	volume: np.ndarray,
	width: int,
	axis: int = 0,
) -> np.ndarray:
	"""Return trailing-window maximum-intensity projections along one axis."""
	array = np.asarray(volume)
	axis = _axis_index(axis, array.ndim)
	if width < 1:
		raise ValueError("MIP width must be positive")
	if width > array.shape[axis]:
		raise ValueError(
			f"MIP width {width} exceeds axis {axis} length {array.shape[axis]}"
		)
	if width == 1:
		return np.array(array, copy=True)
	windows = np.lib.stride_tricks.sliding_window_view(
		array,
		window_shape=width,
		axis=axis,
	)
	return np.max(windows, axis=-1)


def circular_mask(
	volume: np.ndarray,
	ratio: float,
	*,
	axis: int = 0,
	value=0,
) -> np.ndarray:
	"""Mask outside a centered circle on planes perpendicular to ``axis``."""
	array = np.asarray(volume)
	if array.ndim != 3:
		raise ValueError("circular masking requires a three-dimensional volume")
	axis = _axis_index(axis, array.ndim)
	if not 0.0 < ratio <= 1.0:
		raise ValueError("circular mask ratio must be greater than 0 and at most 1")

	plane_axes = tuple(index for index in range(array.ndim) if index != axis)
	first_size, second_size = (array.shape[index] for index in plane_axes)
	first = np.arange(first_size, dtype=np.float64) - (first_size - 1) / 2
	second = np.arange(second_size, dtype=np.float64) - (second_size - 1) / 2
	radius = ratio * min(first_size, second_size) / 2
	plane_mask = (
		first[:, np.newaxis] ** 2 + second[np.newaxis, :] ** 2
		<= radius ** 2
	)
	mask_shape = [1] * array.ndim
	mask_shape[plane_axes[0]] = first_size
	mask_shape[plane_axes[1]] = second_size
	return np.where(plane_mask.reshape(mask_shape), array, value)


def spatial_bin(volume: np.ndarray, power: int) -> np.ndarray:
	"""Average non-overlapping XY blocks of size ``2**power``.

	Incomplete blocks at the bottom and right edges are dropped. Integer inputs
	are rounded back to their input dtype, matching area-resize dtype flow.
	"""
	array = np.asarray(volume)
	if array.ndim < 2:
		raise ValueError("spatial binning requires at least two dimensions")
	if power < 0:
		raise ValueError("bin power cannot be negative")
	if power == 0:
		return np.array(array, copy=True)

	factor = 2 ** power
	height, width = array.shape[-2:]
	binned_height = height // factor
	binned_width = width // factor
	if binned_height == 0 or binned_width == 0:
		raise ValueError(
			f"bin factor {factor} exceeds XY shape {(height, width)}"
		)
	trimmed = array[
		...,
		:binned_height * factor,
		:binned_width * factor,
	]
	blocks = trimmed.reshape(
		trimmed.shape[:-2]
		+ (binned_height, factor, binned_width, factor)
	)
	binned = blocks.mean(axis=(-3, -1))
	if np.issubdtype(array.dtype, np.integer):
		return np.floor(binned + 0.5).astype(array.dtype)
	if np.issubdtype(array.dtype, np.bool_):
		return (binned >= 0.5).astype(array.dtype)
	return binned.astype(array.dtype, copy=False)
