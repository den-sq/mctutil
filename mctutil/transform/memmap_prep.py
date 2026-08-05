"""Normalize a three-dimensional TIFF into a memmappable TIFF stack."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import click
import numpy as np
import tifffile

from mctutil.shared.deps import require

OUTPUT_DTYPES = ("original", "uint16", "uint32", "uint64")
NORMALIZE_MODES = ("none", "minmax", "percentile", "manual")


def parse_output_dtypes(value: str) -> tuple[str, ...]:
	"""Parse and de-duplicate a comma-separated output dtype list."""
	parts = [part.strip().lower() for part in value.split(",") if part.strip()]
	if not parts:
		parts = ["original"]

	invalid = [part for part in parts if part not in OUTPUT_DTYPES]
	if invalid:
		raise ValueError(
			f"invalid output dtype(s): {', '.join(invalid)}; "
			f"choose from {', '.join(OUTPUT_DTYPES)}"
		)

	return tuple(dict.fromkeys(parts))


def estimate_bytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
	"""Return the uncompressed byte size using Python integers."""
	return math.prod(int(length) for length in shape) * int(np.dtype(dtype).itemsize)


def choose_bigtiff(mode: str, total_bytes: int) -> bool:
	"""Resolve the requested BigTIFF mode."""
	if mode == "yes":
		return True
	if mode == "no":
		return False
	return total_bytes >= 4 * 1024**3


def resolve_output_paths(
	input_path: Path,
	output: Path | None,
	output_dir: Path | None,
	output_dtypes: tuple[str, ...],
) -> dict[str, Path]:
	"""Resolve one output path per requested dtype."""
	if output is not None and output_dir is not None:
		raise ValueError("use either OUTPUT or --output-dir, not both")
	if output is None and output_dir is None:
		raise ValueError("provide OUTPUT or --output-dir")

	directory = output_dir
	if directory is None and output is not None and output.exists() and output.is_dir():
		directory = output

	if directory is not None:
		stem = input_path.stem
		return {
			dtype_name: directory / f"{stem}_MEMMAP_{dtype_name}.tif"
			for dtype_name in output_dtypes
		}

	assert output is not None
	suffix = output.suffix if output.suffix.lower() in {".tif", ".tiff"} else ".tif"
	root = output.with_suffix("") if output.suffix.lower() in {".tif", ".tiff"} else output
	if len(output_dtypes) == 1:
		return {output_dtypes[0]: root.with_suffix(suffix)}
	return {
		dtype_name: root.with_name(f"{root.name}_{dtype_name}").with_suffix(suffix)
		for dtype_name in output_dtypes
	}


def output_dtype(dtype_name: str, source_dtype: np.dtype, normalize_mode: str) -> np.dtype:
	"""Return the on-disk dtype for one requested output."""
	if dtype_name == "original":
		return source_dtype if normalize_mode == "none" else np.dtype(np.float32)
	return np.dtype(dtype_name)


def compute_global_minmax(array, z_count: int) -> tuple[float, float]:
	"""Compute a global range one Z plane at a time."""
	minimum = np.inf
	maximum = -np.inf
	with click.progressbar(range(z_count), label="Computing global min/max") as indices:
		for z_index in indices:
			image = np.asarray(array[z_index, :, :], dtype=np.float32)
			minimum = min(minimum, float(np.min(image)))
			maximum = max(maximum, float(np.max(image)))
	return float(minimum), float(maximum)


def compute_sampled_percentiles(
	array,
	z_count: int,
	percentile_low: float,
	percentile_high: float,
	sample_slices: int,
	sample_pixels: int,
	rng_seed: int,
) -> tuple[float, float]:
	"""Estimate a robust range without loading the full volume."""
	rng = np.random.default_rng(rng_seed)
	slice_count = max(1, min(sample_slices, z_count))
	indices = np.linspace(0, z_count - 1, slice_count).astype(int)
	samples: list[np.ndarray] = []

	with click.progressbar(indices, label="Sampling percentile range") as sampled_indices:
		for z_index in sampled_indices:
			image = np.asarray(array[z_index, :, :], dtype=np.float32).reshape(-1)
			pixel_count = min(sample_pixels, image.size)
			if pixel_count:
				selected = rng.choice(image.size, size=pixel_count, replace=False)
				samples.append(image[selected])

	if not samples:
		raise ValueError("percentile sampling collected no pixels")

	values = np.concatenate(samples)
	return (
		float(np.percentile(values, percentile_low)),
		float(np.percentile(values, percentile_high)),
	)


def normalize_and_cast(
	image,
	dtype_name: str,
	source_dtype: np.dtype,
	normalize_mode: str,
	normalize_min: float | None,
	normalize_max: float | None,
) -> np.ndarray:
	"""Normalize and cast one image plane."""
	if normalize_mode == "none":
		target_dtype = source_dtype if dtype_name == "original" else np.dtype(dtype_name)
		return np.asarray(image, dtype=target_dtype)

	if (
		normalize_min is None
		or normalize_max is None
		or not np.isfinite(normalize_min)
		or not np.isfinite(normalize_max)
		or normalize_max <= normalize_min
	):
		raise ValueError(
			f"invalid normalization range: minimum={normalize_min}, maximum={normalize_max}"
		)

	values = np.asarray(image, dtype=np.float32)
	values = np.clip((values - normalize_min) / (normalize_max - normalize_min), 0.0, 1.0)
	if dtype_name == "original":
		return values.astype(np.float32)

	target_dtype = np.dtype(dtype_name)
	values = np.round(values * np.iinfo(target_dtype).max)
	return values.astype(target_dtype)


def verify_memmap(path: Path, shape: tuple[int, int, int], dtype: np.dtype) -> None:
	"""Verify that an output reopens with the expected memory layout."""
	mapped = tifffile.memmap(path)
	try:
		if tuple(mapped.shape) != shape:
			raise ValueError(f"{path} has shape {mapped.shape}, expected {shape}")
		if np.dtype(mapped.dtype) != np.dtype(dtype):
			raise ValueError(f"{path} has dtype {mapped.dtype}, expected {dtype}")
	finally:
		del mapped


def _normalization_range(
	array,
	z_count: int,
	normalize_mode: str,
	normalize_min: float | None,
	normalize_max: float | None,
	percentile_low: float,
	percentile_high: float,
	sample_slices: int,
	sample_pixels: int,
	rng_seed: int,
) -> tuple[float | None, float | None]:
	if normalize_mode == "none":
		return None, None
	if normalize_mode == "manual":
		if normalize_min is None or normalize_max is None:
			raise ValueError("--normalize manual requires --norm-min and --norm-max")
		return normalize_min, normalize_max
	if normalize_mode == "minmax":
		return compute_global_minmax(array, z_count)
	return compute_sampled_percentiles(
		array,
		z_count,
		percentile_low,
		percentile_high,
		sample_slices,
		sample_pixels,
		rng_seed,
	)


@contextmanager
def open_tiff_zarr(tif):
	"""Open tifffile's streaming store with its source-compatible Zarr API."""
	zarr = require(
		"zarr",
		"transform",
		purpose="memmap preparation requires zarr",
	)

	store = tif.aszarr()
	try:
		yield zarr.open(store, mode="r")
	finally:
		close_store = getattr(store, "close", None)
		if close_store is not None:
			close_store()


def _describe_plans(
	input_path: Path,
	shape: tuple[int, int, int],
	source_dtype: np.dtype,
	normalize_mode: str,
	plans: dict[str, tuple[Path, np.dtype]],
	bigtiff_mode: str,
) -> None:
	click.echo(f"Input: {input_path}")
	click.echo(f"Shape: {shape}; source dtype: {source_dtype}")
	click.echo(f"Normalization: {normalize_mode}")
	for dtype_name, (path, dtype) in plans.items():
		total_bytes = estimate_bytes(shape, dtype)
		click.echo(
			f"Output [{dtype_name}]: {path} "
			f"(dtype={dtype}, size={total_bytes / 1024**3:.2f} GiB, "
			f"BigTIFF={choose_bigtiff(bigtiff_mode, total_bytes)})"
		)


def _check_output_paths(plans: dict[str, tuple[Path, np.dtype]], overwrite: bool) -> None:
	existing = [path for path, _dtype in plans.values() if path.exists()]
	if existing and not overwrite:
		raise FileExistsError(
			"output exists; use --overwrite: " + ", ".join(str(path) for path in existing)
		)
	for path, _dtype in plans.values():
		path.parent.mkdir(parents=True, exist_ok=True)


def _write_planes(
	array,
	shape: tuple[int, int, int],
	source_dtype: np.dtype,
	output_dtypes: tuple[str, ...],
	plans: dict[str, tuple[Path, np.dtype]],
	normalize_mode: str,
	range_min: float | None,
	range_max: float | None,
	bigtiff_mode: str,
	contiguous: bool,
) -> None:
	writers = {
		dtype_name: tifffile.TiffWriter(
			path,
			bigtiff=choose_bigtiff(bigtiff_mode, estimate_bytes(shape, dtype)),
		)
		for dtype_name, (path, dtype) in plans.items()
	}
	try:
		with click.progressbar(range(shape[0]), label="Writing Z planes") as indices:
			for z_index in indices:
				image = array[z_index, :, :]
				for dtype_name in output_dtypes:
					writers[dtype_name].write(
						normalize_and_cast(
							image,
							dtype_name,
							source_dtype,
							normalize_mode,
							range_min,
							range_max,
						),
						compression=None,
						contiguous=contiguous,
						photometric="minisblack",
						metadata=None,
					)
	finally:
		for writer in writers.values():
			writer.close()


def prepare_memmappable(
	input_path: Path,
	output_paths: dict[str, Path],
	output_dtypes: tuple[str, ...],
	normalize_mode: str,
	normalize_min: float | None,
	normalize_max: float | None,
	percentile_low: float,
	percentile_high: float,
	sample_slices: int,
	sample_pixels: int,
	rng_seed: int,
	bigtiff_mode: str,
	contiguous: bool,
	overwrite: bool,
	verify: bool,
	execute: bool,
) -> None:
	"""Run the streaming TIFF normalization workflow."""
	with tifffile.TiffFile(input_path) as tif:
		series = tif.series[0]
		if len(series.shape) != 3:
			raise ValueError(f"expected a three-dimensional TIFF series, got {series.shape}")

		shape = tuple(int(length) for length in series.shape)
		source_dtype = np.dtype(series.dtype)
		plans = {
			dtype_name: (
				output_paths[dtype_name],
				output_dtype(dtype_name, source_dtype, normalize_mode),
			)
			for dtype_name in output_dtypes
		}

		_describe_plans(input_path, shape, source_dtype, normalize_mode, plans, bigtiff_mode)

		if not execute:
			return

		_check_output_paths(plans, overwrite)
		with open_tiff_zarr(tif) as array:
			range_min, range_max = _normalization_range(
				array,
				shape[0],
				normalize_mode,
				normalize_min,
				normalize_max,
				percentile_low,
				percentile_high,
				sample_slices,
				sample_pixels,
				rng_seed,
			)
			if range_min is not None:
				click.echo(f"Normalization range: {range_min} to {range_max}")
			_write_planes(
				array,
				shape,
				source_dtype,
				output_dtypes,
				plans,
				normalize_mode,
				range_min,
				range_max,
				bigtiff_mode,
				contiguous,
			)

	for dtype_name, (path, dtype) in plans.items():
		if verify:
			verify_memmap(path, shape, dtype)
			click.echo(f"Verified [{dtype_name}]: {path}")


@click.command("memmap-prep")
@click.argument(
	"input_tif",
	type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument("output", required=False, type=click.Path(path_type=Path))
@click.option(
	"--output-dir",
	type=click.Path(file_okay=False, path_type=Path),
	help="Write dtype-named outputs into this directory instead of OUTPUT.",
)
@click.option(
	"--out-dtypes",
	default="original",
	show_default=True,
	help="Comma-separated outputs: original,uint16,uint32,uint64.",
)
@click.option(
	"--normalize",
	"normalize_mode",
	type=click.Choice(NORMALIZE_MODES),
	default="none",
	show_default=True,
)
@click.option("--norm-min", type=float, help="Manual normalization minimum.")
@click.option("--norm-max", type=float, help="Manual normalization maximum.")
@click.option("--pct-low", type=click.FloatRange(0.0, 100.0), default=0.1, show_default=True)
@click.option("--pct-high", type=click.FloatRange(0.0, 100.0), default=99.9, show_default=True)
@click.option("--sample-slices", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--sample-pixels", type=click.IntRange(min=1), default=200_000, show_default=True)
@click.option("--rng-seed", type=int, default=0, show_default=True)
@click.option(
	"--bigtiff",
	"bigtiff_mode",
	type=click.Choice(("auto", "yes", "no")),
	default="auto",
	show_default=True,
)
@click.option("--contiguous/--no-contiguous", default=True, show_default=True)
@click.option("--overwrite", is_flag=True, help="Replace existing output files.")
@click.option("--verify", is_flag=True, help="Reopen outputs with tifffile.memmap after writing.")
@click.option("--execute/--dry-run", default=True, show_default=True)
def memmap_prep(
	input_tif: Path,
	output: Path | None,
	output_dir: Path | None,
	out_dtypes: str,
	normalize_mode: str,
	norm_min: float | None,
	norm_max: float | None,
	pct_low: float,
	pct_high: float,
	sample_slices: int,
	sample_pixels: int,
	rng_seed: int,
	bigtiff_mode: str,
	contiguous: bool,
	overwrite: bool,
	verify: bool,
	execute: bool,
) -> None:
	"""Create an uncompressed TIFF that tifffile.memmap can reopen."""
	try:
		parsed_dtypes = parse_output_dtypes(out_dtypes)
		output_paths = resolve_output_paths(input_tif, output, output_dir, parsed_dtypes)
		if pct_low >= pct_high:
			raise ValueError("--pct-low must be less than --pct-high")
		if normalize_mode == "manual" and (norm_min is None or norm_max is None):
			raise ValueError("--normalize manual requires --norm-min and --norm-max")
		prepare_memmappable(
			input_tif,
			output_paths,
			parsed_dtypes,
			normalize_mode,
			norm_min,
			norm_max,
			pct_low,
			pct_high,
			sample_slices,
			sample_pixels,
			rng_seed,
			bigtiff_mode,
			contiguous,
			overwrite,
			verify,
			execute,
		)
	except (FileExistsError, RuntimeError, ValueError) as exc:
		raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
	memmap_prep()
