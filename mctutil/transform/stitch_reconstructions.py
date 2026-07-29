"""Join two reconstructed TIFF slice stacks at explicit Z boundaries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4

import click
import numpy as np
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}
SUPPORTED_DTYPE_KINDS = {"b", "u", "i", "f"}
MANIFEST_NAME = "stitch-reconstructions.json"


@dataclass(frozen=True)
class SliceSpec:
	"""Validated source metadata and deterministic destination mapping."""

	source: Path
	stack: str
	source_index: int
	output_index: int
	output_name: str
	shape: tuple[int, ...]
	axes: str
	dtype: np.dtype
	photometric: str


@dataclass(frozen=True)
class StitchPlan:
	"""Complete, validated plan for a manual two-stack concatenation."""

	stack_a: Path
	stack_b: Path
	output_dir: Path
	a_total: int
	b_total: int
	a_stop: int
	b_start: int
	slices: tuple[SliceSpec, ...]
	shape: tuple[int, ...]
	axes: str
	source_dtypes: tuple[np.dtype, ...]
	output_dtype: np.dtype
	workers: int
	overwrite: bool
	filename_width: int


def natural_sort_key(path: Path) -> tuple[str | int, ...]:
	"""Return a case-insensitive key where digit runs compare numerically."""
	return tuple(
		int(part) if part.isdigit() else part.lower()
		for part in re.split(r"([0-9]+)", path.stem)
	)


def discover_slices(directory: Path) -> tuple[Path, ...]:
	"""Discover immediate TIFF children in unambiguous natural order."""
	directory = directory.resolve()
	if not directory.is_dir():
		raise ValueError(f"input is not a directory: {directory}")

	paths = [
		entry.resolve()
		for entry in directory.iterdir()
		if entry.is_file() and entry.suffix.lower() in TIFF_SUFFIXES
	]
	if not paths:
		raise ValueError(f"no TIFF slices found in {directory}")

	by_key: dict[tuple[str | int, ...], Path] = {}
	for path in paths:
		key = natural_sort_key(path)
		if key in by_key:
			raise ValueError(
				"ambiguous natural ordering: "
				f"{by_key[key].name!r} and {path.name!r} in {directory}"
			)
		by_key[key] = path
	return tuple(sorted(paths, key=natural_sort_key))


def _enum_name(value) -> str:
	name = getattr(value, "name", None)
	return str(name if name is not None else value).lower()


def inspect_slice(path: Path) -> tuple[tuple[int, ...], str, np.dtype, str]:
	"""Read TIFF metadata without materializing the pixel array."""
	try:
		with tifffile.TiffFile(path) as source:
			if len(source.series) != 1:
				raise ValueError(
					f"expected one image series, found {len(source.series)}"
				)
			series = source.series[0]
			shape = tuple(int(length) for length in series.shape)
			axes = str(series.axes)
			dtype = np.dtype(series.dtype)
			photometric = _enum_name(source.pages[0].photometric)
	except ValueError:
		raise
	except Exception as exc:
		raise ValueError(f"cannot inspect TIFF metadata: {path}: {exc}") from exc

	if dtype.kind not in SUPPORTED_DTYPE_KINDS:
		raise ValueError(f"unsupported dtype {dtype} in {path}")
	if axes == "YX" and len(shape) == 2:
		if photometric not in {"minisblack", "miniswhite"}:
			raise ValueError(
				f"unsupported grayscale photometric {photometric!r} in {path}"
			)
	elif axes == "YXS" and len(shape) == 3 and shape[-1] in {3, 4}:
		if photometric != "rgb":
			raise ValueError(
				f"channel-last data must use RGB photometric in {path}"
			)
	else:
		raise ValueError(
			f"unsupported slice layout shape={shape}, axes={axes!r} in {path}; "
			"expected grayscale YX or channel-last RGB/RGBA YXS"
		)
	return shape, axes, dtype, photometric


def _validate_cut(name: str, value: int, length: int) -> None:
	if value > length:
		raise ValueError(f"{name}={value} is outside the valid range 0..{length}")


def _resolve_output_dtype(
	metadata: list[tuple[tuple[int, ...], str, np.dtype, str]],
	dtype_override: str | None,
) -> tuple[tuple[np.dtype, ...], np.dtype]:
	source_dtypes = tuple(dict.fromkeys(entry[2] for entry in metadata))
	if dtype_override is None:
		if len(source_dtypes) != 1:
			names = ", ".join(dtype.name for dtype in source_dtypes)
			raise ValueError(
				f"retained slices have mixed dtypes ({names}); select an integer --dtype"
			)
		return source_dtypes, source_dtypes[0]

	try:
		output_dtype = np.dtype(dtype_override)
	except TypeError as exc:
		raise ValueError(f"invalid NumPy dtype: {dtype_override!r}") from exc
	if output_dtype.kind not in {"u", "i"}:
		raise ValueError(
			f"--dtype must select an integer dtype, got {output_dtype.name}"
		)
	return source_dtypes, output_dtype


def _validate_output(
	stack_a: Path,
	stack_b: Path,
	output_dir: Path,
	overwrite: bool,
) -> Path:
	if output_dir.is_symlink():
		raise ValueError(f"output directory may not be a symbolic link: {output_dir}")
	output_dir = output_dir.resolve()
	for source in (stack_a, stack_b):
		if output_dir == source or source in output_dir.parents:
			raise ValueError(f"output directory may not be inside an input stack: {source}")
	if output_dir.exists():
		if not output_dir.is_dir():
			raise ValueError(f"output path exists and is not a directory: {output_dir}")
		if not overwrite:
			raise ValueError(
				f"output directory already exists: {output_dir}; pass --overwrite to replace it"
			)
	return output_dir


def build_plan(
	stack_a: Path,
	stack_b: Path,
	output_dir: Path,
	a_stop: int,
	b_start: int,
	dtype_override: str | None,
	workers: int,
	overwrite: bool,
) -> StitchPlan:
	"""Discover, range-select, and validate every retained TIFF slice."""
	stack_a = stack_a.resolve()
	stack_b = stack_b.resolve()
	output_dir = _validate_output(stack_a, stack_b, output_dir, overwrite)
	a_paths = discover_slices(stack_a)
	b_paths = discover_slices(stack_b)
	_validate_cut("a_stop", a_stop, len(a_paths))
	_validate_cut("b_start", b_start, len(b_paths))

	selected = (
		[("A", index, path) for index, path in enumerate(a_paths[:a_stop])]
		+ [("B", index, path) for index, path in enumerate(b_paths[b_start:], b_start)]
	)
	if not selected:
		raise ValueError("the selected ranges produce an empty output stack")

	metadata = [inspect_slice(path) for _stack, _index, path in selected]
	reference_shape, reference_axes, _dtype, reference_photometric = metadata[0]
	for (_stack, _index, path), (shape, axes, _dtype, photometric) in zip(
		selected,
		metadata,
	):
		if (shape, axes, photometric) != (
			reference_shape,
			reference_axes,
			reference_photometric,
		):
			raise ValueError(
				f"incompatible slice layout in {path}: "
				f"shape={shape}, axes={axes!r}, photometric={photometric!r}; "
				f"expected shape={reference_shape}, axes={reference_axes!r}, "
				f"photometric={reference_photometric!r}"
			)

	source_dtypes, output_dtype = _resolve_output_dtype(metadata, dtype_override)
	filename_width = max(5, len(str(len(selected) - 1)))
	slices = tuple(
		SliceSpec(
			source=path,
			stack=stack,
			source_index=source_index,
			output_index=output_index,
			output_name=f"slice_{output_index:0{filename_width}d}.tif",
			shape=shape,
			axes=axes,
			dtype=dtype,
			photometric=photometric,
		)
		for output_index, (
			(stack, source_index, path),
			(shape, axes, dtype, photometric),
		) in enumerate(zip(selected, metadata))
	)
	return StitchPlan(
		stack_a=stack_a,
		stack_b=stack_b,
		output_dir=output_dir,
		a_total=len(a_paths),
		b_total=len(b_paths),
		a_stop=a_stop,
		b_start=b_start,
		slices=slices,
		shape=reference_shape,
		axes=reference_axes,
		source_dtypes=source_dtypes,
		output_dtype=output_dtype,
		workers=workers,
		overwrite=overwrite,
		filename_width=filename_width,
	)


def conversion_description(plan: StitchPlan) -> str:
	if len(plan.source_dtypes) == 1 and plan.source_dtypes[0] == plan.output_dtype:
		return "preserve source values and dtype"
	return (
		f"clip to {plan.output_dtype.name} range, then cast; "
		"floating-point values truncate toward zero; no rescaling"
	)


def describe_plan(plan: StitchPlan) -> None:
	"""Report the complete validated plan for execution or dry-run."""
	click.echo(f"Stack A: {plan.stack_a} ({plan.a_total} slices)")
	click.echo(f"Retain A: [0:{plan.a_stop}) ({plan.a_stop} slices)")
	click.echo(f"Stack B: {plan.stack_b} ({plan.b_total} slices)")
	click.echo(
		f"Retain B: [{plan.b_start}:{plan.b_total}) "
		f"({plan.b_total - plan.b_start} slices)"
	)
	click.echo(f"Output: {plan.output_dir}")
	click.echo(
		f"Output stack: {len(plan.slices)} slices; shape={plan.shape}; "
		f"axes={plan.axes}; dtype={plan.output_dtype.name}"
	)
	click.echo(
		f"Names: slice_{0:0{plan.filename_width}d}.tif .. "
		f"slice_{len(plan.slices) - 1:0{plan.filename_width}d}.tif"
	)
	click.echo(f"Conversion: {conversion_description(plan)}")
	click.echo(f"Workers: {plan.workers}")
	if plan.output_dir.exists():
		click.echo("Overwrite: replace the existing output only after all new slices succeed")


def _convert_image(image: np.ndarray, output_dtype: np.dtype, source: Path) -> np.ndarray:
	if image.dtype == output_dtype:
		return image
	if image.dtype.kind == "f" and not np.isfinite(image).all():
		raise ValueError(f"cannot convert non-finite floating-point values in {source}")
	limits = np.iinfo(output_dtype)
	converted = np.empty(image.shape, dtype=output_dtype)
	at_minimum = image <= limits.min
	at_maximum = image >= limits.max
	between = ~(at_minimum | at_maximum)
	converted[at_minimum] = limits.min
	converted[at_maximum] = limits.max
	converted[between] = image[between].astype(output_dtype)
	return converted


def write_planned_slice(spec: SliceSpec, staging_dir: Path, output_dtype: np.dtype) -> int:
	"""Read, verify, convert, and write exactly one retained slice."""
	image = np.asarray(tifffile.imread(spec.source))
	if image.shape != spec.shape or image.dtype != spec.dtype:
		raise ValueError(
			f"source changed after validation: {spec.source}; "
			f"got shape={image.shape}, dtype={image.dtype}; "
			f"expected shape={spec.shape}, dtype={spec.dtype}"
		)
	image = _convert_image(image, output_dtype, spec.source)
	tifffile.imwrite(
		staging_dir / spec.output_name,
		image,
		photometric=spec.photometric,
	)
	return spec.output_index


def _write_slices(plan: StitchPlan, staging_dir: Path) -> None:
	if plan.workers == 1:
		for spec in plan.slices:
			write_planned_slice(spec, staging_dir, plan.output_dtype)
		return

	with ThreadPoolExecutor(max_workers=min(plan.workers, len(plan.slices))) as pool:
		futures = [
			pool.submit(write_planned_slice, spec, staging_dir, plan.output_dtype)
			for spec in plan.slices
		]
		with click.progressbar(
			as_completed(futures),
			length=len(futures),
			label="Writing slices",
		) as completed:
			for future in completed:
				future.result()


def manifest_for(plan: StitchPlan) -> dict:
	"""Build deterministic provenance written only after all slices succeed."""
	return {
		"command": "mctutil transform stitch-reconstructions",
		"version": 1,
		"ordering": (
			"immediate .tif/.tiff children in case-insensitive natural order; "
			"numeric runs compare as integers"
		),
		"inputs": [
			{
				"stack": "A",
				"path": str(plan.stack_a),
				"total_slices": plan.a_total,
				"retained": {"start": 0, "stop": plan.a_stop},
			},
			{
				"stack": "B",
				"path": str(plan.stack_b),
				"total_slices": plan.b_total,
				"retained": {"start": plan.b_start, "stop": plan.b_total},
			},
		],
		"output": {
			"count": len(plan.slices),
			"shape": list(plan.shape),
			"axes": plan.axes,
			"dtype": plan.output_dtype.name,
			"filename_pattern": f"slice_{{index:0{plan.filename_width}d}}.tif",
		},
		"source_dtypes": [dtype.name for dtype in plan.source_dtypes],
		"conversion": conversion_description(plan),
	}


def _promote_staging(plan: StitchPlan, staging_dir: Path) -> None:
	output_dir = plan.output_dir
	if not output_dir.exists():
		staging_dir.replace(output_dir)
		return
	if not plan.overwrite:
		raise ValueError(
			f"output directory appeared during execution: {output_dir}; "
			"rerun with --overwrite to replace it"
		)
	if not output_dir.is_dir() or output_dir.is_symlink():
		raise ValueError(f"cannot replace non-directory output: {output_dir}")

	backup = output_dir.with_name(f".{output_dir.name}.backup-{uuid4().hex}")
	output_dir.replace(backup)
	try:
		staging_dir.replace(output_dir)
	except Exception:
		backup.replace(output_dir)
		raise
	try:
		shutil.rmtree(backup)
	except OSError as exc:
		click.echo(f"Warning: could not remove replaced output backup {backup}: {exc}", err=True)


def execute_plan(plan: StitchPlan) -> None:
	"""Write into a sibling staging directory and expose it only when complete."""
	plan.output_dir.parent.mkdir(parents=True, exist_ok=True)
	staging_dir = Path(
		tempfile.mkdtemp(
			prefix=f".{plan.output_dir.name}.stitching-",
			dir=plan.output_dir.parent,
		)
	)
	try:
		_write_slices(plan, staging_dir)
		(staging_dir / MANIFEST_NAME).write_text(
			json.dumps(manifest_for(plan), indent=2, sort_keys=True) + "\n",
			encoding="utf-8",
		)
		_promote_staging(plan, staging_dir)
	finally:
		if staging_dir.exists():
			shutil.rmtree(staging_dir)


@click.command("stitch-reconstructions")
@click.argument(
	"stack_a",
	type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
	"stack_b",
	type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
	"--a-stop",
	type=click.IntRange(min=0),
	required=True,
	help="Exclusive stop index for stack A: retain A[:a_stop].",
)
@click.option(
	"--b-start",
	type=click.IntRange(min=0),
	required=True,
	help="Inclusive start index for stack B: retain B[b_start:].",
)
@click.option(
	"--dtype",
	"dtype_override",
	help=(
		"Integer output dtype. Values are clipped then cast without rescaling; "
		"omit to preserve the common source dtype."
	),
)
@click.option(
	"--workers",
	type=click.IntRange(min=1),
	default=8,
	show_default=True,
	help="Number of local per-slice writer threads.",
)
@click.option(
	"--overwrite",
	is_flag=True,
	help="Transactionally replace an existing output directory after successful writing.",
)
@click.option("--execute/--dry-run", default=True, show_default=True)
def stitch_reconstructions(
	stack_a: Path,
	stack_b: Path,
	output_dir: Path,
	a_stop: int,
	b_start: int,
	dtype_override: str | None,
	workers: int,
	overwrite: bool,
	execute: bool,
) -> None:
	"""Concatenate reconstructed TIFF stacks using manual half-open Z cuts.

	This command performs no registration, overlap detection, or blending. Use
	``transform stitch`` for projection stitching.
	"""
	try:
		plan = build_plan(
			stack_a,
			stack_b,
			output_dir,
			a_stop,
			b_start,
			dtype_override,
			workers,
			overwrite,
		)
		describe_plan(plan)
		if not execute:
			return
		execute_plan(plan)
		click.echo(f"Stitch complete: {len(plan.slices)} slices in {plan.output_dir}")
	except click.ClickException:
		raise
	except Exception as exc:
		raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
	stitch_reconstructions()
