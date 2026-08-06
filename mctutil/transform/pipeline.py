"""Single-read, multi-operation TIFF volume transform pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from multiprocessing import shared_memory
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Iterator

import click
import numpy as np
import psutil
import tifffile as tf

from mctutil.shared import cli
from mctutil.shared.log import LOG, log
from mctutil.shared.stack_apply import (
	StackMapItem,
	plan_stack_map,
	require_tiff_paths,
	run_parallel,
)
from mctutil.shared.tiff_stack_writer import compression_for, write_tiff_stack
from mctutil.transform.convert import converted_image
from mctutil.transform.flip import flipped_volume
from mctutil.transform.normalize import normalization_bounds, normalized_image
from mctutil.transform.ops import (
	circular_mask,
	maximum_intensity_projection,
	spatial_bin,
)
from mctutil.transform.simple_noise import denoised_volume
from mctutil.transform.trim import cropped_image


PIPELINE_ORDER = (
	"normalize",
	"trim",
	"mip",
	"circular-mask",
	"denoise",
	"flip",
	"dtype-convert",
	"spatial-bin",
	"compress/write",
)
MIP_AXES = {"z": 0, "y": 1, "x": 2, "0": 0, "1": 1, "2": 2}
FLIP_AXES = MIP_AXES


def mip_axis_index(value: str | int) -> int:
	"""Resolve named and legacy numeric MIP-axis values."""
	try:
		return MIP_AXES[str(value).lower()]
	except KeyError as error:
		raise ValueError(f"unsupported MIP axis: {value}") from error


def flip_axis_index(value: str | int) -> int:
	"""Resolve named and legacy numeric flip-axis values."""
	try:
		return FLIP_AXES[str(value).lower()]
	except KeyError as error:
		raise ValueError(f"unsupported flip axis: {value}") from error


def _validate_normalize_range(
	normalize_range: tuple[float, float] | None,
) -> None:
	if normalize_range is None:
		return
	bottom, top = normalize_range
	if not 0 <= bottom < top <= 100:
		raise ValueError(
			"normalization percentiles must satisfy 0 <= bottom < top <= 100"
		)


def _validate_denoise_config(mode, threshold) -> None:
	if mode is None and threshold is None:
		return
	if mode is None or threshold is None:
		raise ValueError(
			"denoise mode and threshold must be supplied together"
		)
	if mode == "threshold" and not 0 <= threshold <= 1:
		raise ValueError(
			"threshold denoise requires a fraction between 0 and 1"
		)


def _normalized_volume(
	volume: np.ndarray,
	normalize_range: tuple[float, float] | None,
) -> np.ndarray:
	_validate_normalize_range(normalize_range)
	if normalize_range is None:
		return volume
	floor, ceiling = normalization_bounds(volume, *normalize_range)
	if ceiling <= floor:
		raise ValueError(
			f"normalization bounds must differ; both resolved to {floor:.4g}"
		)
	return normalized_image(volume, floor, ceiling)


def _trimmed_volume(
	volume: np.ndarray,
	vertical_trim,
	horizontal_trim,
	z_trim,
) -> np.ndarray:
	result = volume[cli.crop_val(z_trim, volume.shape[0])]
	result = cropped_image(result, vertical_trim, horizontal_trim)
	if any(size == 0 for size in result.shape):
		raise ValueError(f"trim settings produce an empty volume: {result.shape}")
	return result


def _volume_axis(axis: int, ndim: int) -> int:
	if axis < 0:
		axis += ndim
	if not 0 <= axis < ndim:
		raise ValueError(
			f"MIP axis {axis} is out of bounds for a {ndim}D volume"
		)
	return axis


def _optionally_denoised(volume, mode, threshold):
	_validate_denoise_config(mode, threshold)
	if mode is None:
		return volume
	return denoised_volume(
		volume,
		mode,
		threshold,
		boundary="preserve",
	)


def _optionally_flipped(volume, flip_axis):
	if flip_axis is None:
		return volume
	return flipped_volume(volume, flip_axis)


def apply_transform_pipeline(
	volume: np.ndarray,
	*,
	normalize_range: tuple[float, float] | None = None,
	vertical_trim=(0.0, 0.0),
	horizontal_trim=(0.0, 0.0),
	z_trim=(0.0, 0.0),
	mip_width: int = 0,
	mip_axis: int = 0,
	circ_mask_ratio: float | None = None,
	denoise_mode: str | None = None,
	denoise_threshold: float | None = None,
	flip_axis: int | None = None,
	out_dtype: np.dtype | type | None = np.uint8,
	bin_power: int = 0,
) -> np.ndarray:
	"""Apply the fused operation chain to an already-loaded ZYX volume.

	The order is fixed by ``PIPELINE_ORDER``. Dtype conversion is performed per
	output Z image, as it is in the standalone ``transform convert`` core, and
	spatial binning preserves that converted dtype while changing only Y and X.
	"""
	result = np.asarray(volume)
	if result.ndim != 3:
		raise ValueError(
			f"transform pipeline requires a three-dimensional ZYX volume; got {result.shape}"
		)
	if mip_width < 0:
		raise ValueError("MIP width cannot be negative")
	result = _normalized_volume(result, normalize_range)
	result = _trimmed_volume(
		result,
		vertical_trim,
		horizontal_trim,
		z_trim,
	)
	mip_axis = _volume_axis(mip_axis, result.ndim)
	if mip_width > 1:
		result = maximum_intensity_projection(result, mip_width, mip_axis)

	if circ_mask_ratio is not None:
		mask_value = 0 if normalize_range is not None else np.min(result)
		result = circular_mask(
			result,
			circ_mask_ratio,
			axis=0,
			value=mask_value,
		)

	result = _optionally_denoised(
		result,
		denoise_mode,
		denoise_threshold,
	)
	result = _optionally_flipped(result, flip_axis)

	if out_dtype is not None:
		result = np.stack(
			tuple(converted_image(image, out_dtype) for image in result),
			axis=0,
		)

	if bin_power:
		result = spatial_bin(result, bin_power)
	return result


def _read_tiff_image(
	volume: np.ndarray,
	index: int,
	path: Path,
	expected_shape: tuple[int, int],
	expected_dtype: np.dtype,
) -> None:
	image = np.asarray(tf.imread(path))
	if image.shape != expected_shape:
		raise ValueError(
			f"TIFF shape mismatch for {path}: {image.shape} != {expected_shape}"
		)
	if image.dtype != expected_dtype:
		raise ValueError(
			f"TIFF dtype mismatch for {path}: {image.dtype} != {expected_dtype}"
		)
	volume[index] = image


@contextmanager
def shared_tiff_volume(
	paths: tuple[Path, ...],
	workers: int,
) -> Iterator[np.ndarray]:
	"""Read each single-image TIFF once into one shared-memory ZYX volume."""
	if not paths:
		raise ValueError("cannot load an empty TIFF stack")
	with tf.TiffFile(paths[0]) as tif:
		if len(tif.pages) != 1 or len(tif.pages[0].shape) != 2:
			raise ValueError(
				"transform pipeline expects one two-dimensional image per TIFF file"
			)
		image_shape = tuple(tif.pages[0].shape)
		dtype = np.dtype(tif.pages[0].dtype)
	volume_shape = (len(paths),) + image_shape
	byte_count = int(np.prod(volume_shape, dtype=np.int64)) * dtype.itemsize
	segment = shared_memory.SharedMemory(create=True, size=byte_count)
	volume = np.ndarray(volume_shape, dtype=dtype, buffer=segment.buf)
	try:
		active_workers = min(workers, len(paths))
		run_parallel(
			_read_tiff_image,
			(
				(volume, index, path, image_shape, dtype)
				for index, path in enumerate(paths)
			),
			active_workers,
			pool_factory=ThreadPool,
		)
		yield volume
	finally:
		del volume
		segment.close()
		segment.unlink()


def plan_pipeline_outputs(
	inputs: tuple[Path, ...],
	output_dir: str | Path,
	*,
	z_trim=(0.0, 0.0),
	mip_width: int = 0,
	mip_axis: int = 0,
	flip_axis: int | None = None,
) -> tuple[StackMapItem, ...]:
	"""Map transformed Z planes to trailing inputs, including Z reversal."""
	selected = inputs[cli.crop_val(z_trim, len(inputs))]
	if mip_width > 1 and mip_axis == 0:
		selected = selected[mip_width - 1:]
	if not selected:
		raise ValueError("trim and MIP settings produce no output images")
	sources = tuple(reversed(selected)) if flip_axis == 0 else selected
	return plan_stack_map(
		sources,
		output_dir,
		target_names=(path.name for path in selected),
	)


def _write_pipeline_image(
	item: StackMapItem,
	image: np.ndarray,
	compression: str | None,
) -> None:
	write_tiff_stack(
		lambda _index: image,
		1,
		item.target,
		mode="image",
		compression=compression,
	)
	log.write("File Written", str(item.target), log_level=LOG.INFO)


def write_pipeline_outputs(
	volume: np.ndarray,
	items: tuple[StackMapItem, ...],
	*,
	compression: str | None,
	workers: int,
) -> tuple[Path, ...]:
	"""Write each transformed Z plane once through the canonical TIFF writer."""
	if len(volume) != len(items):
		raise ValueError(
			f"output plane count {len(volume)} does not match path count {len(items)}"
		)
	active_workers = min(workers, len(items))
	run_parallel(
		_write_pipeline_image,
		(
			(item, volume[index], compression)
			for index, item in enumerate(items)
		),
		active_workers,
		pool_factory=ThreadPool,
	)
	return tuple(item.target for item in items)


@click.command()
@click.option(
	"-n",
	"--normalize-over",
	type=cli.FRANGE,
	help="Optional bottom,top percentile range to normalize before other operations.",
)
@click.option(
	"-d",
	"--data-dir",
	type=click.Path(exists=True, file_okay=False, path_type=Path),
	required=True,
	help="Input directory containing one 2D image per TIFF file.",
)
@click.option(
	"-o",
	"--output-dir",
	type=click.Path(file_okay=False, path_type=Path),
	required=True,
	help="Output directory for the transformed TIFF stack.",
)
@click.option(
	"-v",
	"--vertical-trim",
	type=cli.CROP_NUMBER,
	default="0.0",
	show_default=True,
	help="Y trim as one or two absolute counts or fractional values.",
)
@click.option(
	"-h",
	"--horizontal-trim",
	type=cli.CROP_NUMBER,
	default="0.0",
	show_default=True,
	help="X trim as one or two absolute counts or fractional values.",
)
@click.option(
	"-z",
	"--z-trim",
	type=cli.CROP_NUMBER,
	default="0.0",
	show_default=True,
	help="Z trim as one or two absolute counts or fractional values.",
)
@click.option(
	"-m",
	"--mips",
	type=click.IntRange(min=0),
	default=0,
	show_default=True,
	help="Trailing-window MIP width; 0 or 1 disables projection.",
)
@click.option(
	"--mips-axis",
	"--mips-index",
	type=click.Choice(tuple(MIP_AXES)),
	default="z",
	show_default=True,
	help="MIP axis (z/y/x or legacy 0/1/2).",
)
@click.option(
	"-c",
	"--circ-mask-ratio",
	type=click.FloatRange(min=0.0, max=1.0, min_open=True),
	help="Optional centered XY circular-mask diameter ratio.",
)
@click.option(
	"--denoise-mode",
	type=click.Choice(("threshold", "flat")),
	help="Optional neighboring-Z denoise operation.",
)
@click.option(
	"--denoise-threshold",
	type=click.FLOAT,
	help="Fractional threshold for threshold mode; absolute value for flat mode.",
)
@click.option(
	"--flip-axis",
	type=click.Choice(tuple(FLIP_AXES)),
	help="Optional volume flip axis (z/y/x or legacy 0/1/2).",
)
@click.option(
	"-t",
	"--out-dtype",
	type=cli.NUMPYTYPE,
	default="uint8",
	show_default=True,
	help="Output NumPy dtype.",
)
@click.option(
	"-b",
	"--bin-power",
	type=click.IntRange(min=0),
	default=0,
	show_default=True,
	help="Average XY blocks by a factor of 2**power.",
)
@click.option(
	"-p",
	"--processes",
	type=click.IntRange(min=1),
	default=psutil.cpu_count() or 1,
	show_default=True,
	help="Worker threads used for TIFF reads and writes.",
)
@click.option(
	"--compressed/--uncompressed",
	default=False,
	help="Write zlib-compressed or uncompressed TIFFs.",
)
@click.option(
	"--execute/--dry-run",
	default=True,
	help="Write outputs or only report the planned output paths.",
)
def pipeline(
	normalize_over,
	data_dir,
	output_dir,
	vertical_trim,
	horizontal_trim,
	z_trim,
	mips,
	mips_axis,
	circ_mask_ratio,
	denoise_mode,
	denoise_threshold,
	flip_axis,
	out_dtype,
	bin_power,
	processes,
	compressed,
	execute,
):
	"""Apply an ordered operation chain with one TIFF read and write per plane.

	Order: normalize, trim, MIP, circular mask, denoise, flip, dtype conversion,
	spatial binning, then compression/write.
	"""
	log.start()
	inputs = require_tiff_paths(data_dir)
	axis = mip_axis_index(mips_axis)
	resolved_flip_axis = (
		None
		if flip_axis is None
		else flip_axis_index(flip_axis)
	)
	normalize_range = (
		None
		if normalize_over is None
		else (normalize_over.start, normalize_over.stop)
	)
	_validate_normalize_range(normalize_range)
	try:
		_validate_denoise_config(denoise_mode, denoise_threshold)
	except ValueError as error:
		raise click.UsageError(str(error)) from error
	items = plan_pipeline_outputs(
		inputs,
		output_dir,
		z_trim=z_trim,
		mip_width=mips,
		mip_axis=axis,
		flip_axis=resolved_flip_axis,
	)
	log.write("Pipeline Order", " -> ".join(PIPELINE_ORDER), log_level=LOG.INFO)
	if not execute:
		for item in items:
			log.write(
				"Dry Run",
				f"Would write {item.target} from fused input operations",
				log_level=LOG.INFO,
			)
		return

	with shared_tiff_volume(inputs, processes) as volume:
		log.write(
			"Image Load",
			f"Read {len(inputs)} images once into shared memory {volume.shape}",
			log_level=LOG.INFO,
		)
		transformed = apply_transform_pipeline(
			volume,
			normalize_range=normalize_range,
			vertical_trim=vertical_trim,
			horizontal_trim=horizontal_trim,
			z_trim=z_trim,
			mip_width=mips,
			mip_axis=axis,
			circ_mask_ratio=circ_mask_ratio,
			denoise_mode=denoise_mode,
			denoise_threshold=denoise_threshold,
			flip_axis=resolved_flip_axis,
			out_dtype=out_dtype.nptype,
			bin_power=bin_power,
		)
		write_pipeline_outputs(
			transformed,
			items,
			compression=compression_for(compressed),
			workers=processes,
		)
	log.write("Complete", f"Wrote {len(items)} transformed images")


if __name__ == "__main__":
	pipeline()
