"""CloudVolume-backed TIFF to Neuroglancer precomputed conversion."""

from __future__ import annotations

from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import click
import numpy as np
import tifffile

from mctutil.shared.cli import XYZ
from mctutil.shared.cloudfiles_monitoring import patch_cloudfiles_monitoring
from mctutil.shared.log import log, LOG
from mctutil.ng.completeness import check_mip0_completeness


LAYER_TYPES = ("auto", "image", "segmentation")
SEGMENTATION_ENCODINGS = ("compressed_segmentation", "compresso")
SEGMENTATION_NAME_HINTS = ("segmentation", "labels")

_WORKER_VOLUME = None
_WORKER_SOURCE = None
_WORKER_MEMMAP = False
_WORKER_DTYPE = None
_WORKER_OFFSET_Z = 0


@dataclass(frozen=True)
class InputSpec:
	"""Resolved input layout shared by planning and workers."""

	mode: str
	source: str | tuple[str, ...]
	shape: tuple[int, int, int]
	dtype: np.dtype


@dataclass(frozen=True)
class VolumePlan:
	"""CloudVolume metadata derived from the CLI and input."""

	layer_type: str
	encoding: str
	dtype: np.dtype
	resolution: tuple[int, int, int]
	voxel_offset: tuple[int, int, int]
	chunk_size: tuple[int, int, int]
	segmentation_block: tuple[int, int, int]


@dataclass(frozen=True)
class WorkerBatchResult:
	"""Completed planes and an optional process-pool failure."""

	completed: frozenset[int]
	failure: BrokenProcessPool | None


def _require_cloudvolume():
	try:
		from cloudvolume import CloudVolume
	except ImportError as exc:
		raise RuntimeError(
			"ng precompute requires CloudVolume; install with pip install -e '.[ng]'"
		) from exc
	return CloudVolume


def natural_sort_key(path: Path) -> tuple:
	"""Sort slice_2 before slice_10."""
	return tuple(
		int(part) if part.isdigit() else part.lower()
		for part in re.split(r"([0-9]+)", path.name)
	)


def discover_input(path: Path) -> InputSpec:
	"""Inspect a memmappable TIFF or a directory of TIFF planes."""
	path = path.resolve()
	if path.is_file():
		try:
			mapped = tifffile.memmap(path)
		except Exception as exc:
			raise ValueError(
				f"single TIFF input is not memmappable; run transform memmap-prep first: {path}"
			) from exc
		try:
			if mapped.ndim != 3:
				raise ValueError(f"expected a 3-D memmappable TIFF, got shape {mapped.shape}")
			return InputSpec(
				mode="memmap",
				source=str(path),
				shape=tuple(int(length) for length in mapped.shape),
				dtype=np.dtype(mapped.dtype),
			)
		finally:
			del mapped

	paths = sorted(
		(
			entry.resolve()
			for entry in path.iterdir()
			if entry.is_file() and entry.suffix.lower() in {".tif", ".tiff"}
		),
		key=natural_sort_key,
	)
	if not paths:
		raise ValueError(f"no TIFF slices found in {path}")

	first = tifffile.imread(paths[0])
	if first.ndim != 2:
		raise ValueError(f"directory slices must be 2-D, got {first.shape} in {paths[0]}")
	return InputSpec(
		mode="directory",
		source=tuple(str(entry) for entry in paths),
		shape=(len(paths), int(first.shape[0]), int(first.shape[1])),
		dtype=np.dtype(first.dtype),
	)


def guess_layer_type(path: Path) -> str:
	name = path.resolve().name.lower()
	if any(hint in name for hint in SEGMENTATION_NAME_HINTS) or name.endswith("_seg"):
		return "segmentation"
	return "image"


def coerce_segmentation_dtype(
	source_dtype: np.dtype,
	encoding: str,
	dtype_override: str | None,
) -> np.dtype:
	"""Choose a legal unsigned label dtype for the selected encoding."""
	dtype = np.dtype(dtype_override) if dtype_override else np.dtype(source_dtype)
	if dtype.kind not in {"u", "i"}:
		dtype = np.dtype("uint32")

	if encoding == "compressed_segmentation":
		if dtype.kind == "u" and dtype.itemsize in {4, 8}:
			return dtype
		return np.dtype("uint32" if dtype.itemsize <= 4 else "uint64")

	if dtype.kind == "i":
		return np.dtype(f"uint{dtype.itemsize * 8}")
	if dtype.kind != "u":
		return np.dtype("uint32")
	return dtype


def build_plan(
	input_path: Path,
	input_spec: InputSpec,
	layer_type: str,
	segmentation_encoding: str,
	dtype_override: str | None,
	chunk_size: tuple[int, int, int] | None,
	voxel_resolution: tuple[int, int, int],
	voxel_offset: tuple[int, int, int],
	segmentation_block: tuple[int, int, int],
) -> VolumePlan:
	"""Resolve dtype, encoding, and chunk defaults."""
	resolved_layer = guess_layer_type(input_path) if layer_type == "auto" else layer_type
	if resolved_layer == "segmentation":
		dtype = coerce_segmentation_dtype(
			input_spec.dtype,
			segmentation_encoding,
			dtype_override,
		)
		encoding = segmentation_encoding
	else:
		dtype = np.dtype(dtype_override) if dtype_override else input_spec.dtype
		encoding = "raw"

	z_count, y_size, x_size = input_spec.shape
	del z_count
	if chunk_size is None:
		chunk_size = (512, 512, 1) if resolved_layer == "image" else (x_size, y_size, 1)

	for name, values in {
		"voxel resolution": voxel_resolution,
		"chunk size": chunk_size,
		"segmentation block": segmentation_block,
	}.items():
		if any(value <= 0 for value in values):
			raise ValueError(f"{name} entries must be positive: {values}")

	return VolumePlan(
		layer_type=resolved_layer,
		encoding=encoding,
		dtype=np.dtype(dtype),
		resolution=voxel_resolution,
		voxel_offset=voxel_offset,
		chunk_size=chunk_size,
		segmentation_block=segmentation_block,
	)


def default_output_path(input_path: Path) -> Path:
	if input_path.is_dir():
		return input_path.with_name(f"{input_path.name}_precomputed")
	return input_path.with_name(f"{input_path.stem}_precomputed")


def cloudpath_for(output_path: Path) -> str:
	return output_path.resolve().as_uri()


def create_volume_info(plan: VolumePlan, input_spec: InputSpec) -> dict:
	CloudVolume = _require_cloudvolume()
	z_count, y_size, x_size = input_spec.shape
	options = dict(
		num_channels=1,
		layer_type=plan.layer_type,
		data_type=plan.dtype.name,
		encoding=plan.encoding,
		resolution=list(plan.resolution),
		voxel_offset=list(plan.voxel_offset),
		chunk_size=list(plan.chunk_size),
		volume_size=[x_size, y_size, z_count],
	)
	if plan.layer_type == "segmentation" and plan.encoding == "compressed_segmentation":
		options["compressed_segmentation_block_size"] = list(plan.segmentation_block)
	info = CloudVolume.create_new_info(**options)
	if plan.layer_type == "segmentation":
		info["mesh"] = "mesh"
	return info


def _expected_scale(plan: VolumePlan, input_spec: InputSpec) -> dict:
	z_count, y_size, x_size = input_spec.shape
	expected = {
		"type": plan.layer_type,
		"data_type": plan.dtype.name,
		"encoding": plan.encoding,
		"resolution": list(plan.resolution),
		"voxel_offset": list(plan.voxel_offset),
		"chunk_sizes": [list(plan.chunk_size)],
		"size": [x_size, y_size, z_count],
	}
	if plan.layer_type == "segmentation" and plan.encoding == "compressed_segmentation":
		expected["compressed_segmentation_block_size"] = list(plan.segmentation_block)
	return expected


def validate_existing_info(info: dict, plan: VolumePlan, input_spec: InputSpec) -> None:
	"""Refuse to resume into a volume with incompatible metadata."""
	expected = _expected_scale(plan, input_spec)
	scales = info.get("scales", [])
	if not scales:
		raise ValueError("existing precomputed info has no scales")
	scale = scales[0]
	actual = {
		"type": info.get("type"),
		"data_type": info.get("data_type"),
		"encoding": scale.get("encoding"),
		"resolution": scale.get("resolution"),
		"voxel_offset": scale.get("voxel_offset"),
		"chunk_sizes": scale.get("chunk_sizes"),
		"size": scale.get("size"),
	}
	if "compressed_segmentation_block_size" in expected:
		actual["compressed_segmentation_block_size"] = scale.get(
			"compressed_segmentation_block_size"
		)
	differences = [
		f"{key}: existing={actual[key]!r}, requested={value!r}"
		for key, value in expected.items()
		if actual[key] != value
	]
	if differences:
		raise ValueError("cannot resume incompatible volume; " + "; ".join(differences))


def open_or_create_volume(output_path: Path, plan: VolumePlan, input_spec: InputSpec):
	"""Create MIP 0 metadata once, or validate it for a resume."""
	CloudVolume = _require_cloudvolume()
	patch_cloudfiles_monitoring()
	output_path = output_path.resolve()
	info_path = output_path / "info"
	if info_path.exists():
		volume = CloudVolume(
			cloudpath_for(output_path),
			parallel=False,
			bounded=True,
			cache=False,
			compress=False,
		)
		validate_existing_info(volume.info, plan, input_spec)
		return volume, False

	if output_path.exists() and any(output_path.iterdir()):
		raise ValueError(f"output exists without precomputed info: {output_path}")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	volume = CloudVolume(
		cloudpath_for(output_path),
		info=create_volume_info(plan, input_spec),
		parallel=False,
		bounded=True,
		cache=False,
		compress=False,
		compress_cache=False,
	)
	volume.commit_info()
	return volume, True


def _init_worker(
	cloudpath: str,
	dtype_name: str,
	source,
	memmap_source: bool,
	offset_z: int,
) -> None:
	global _WORKER_VOLUME, _WORKER_SOURCE, _WORKER_MEMMAP, _WORKER_DTYPE, _WORKER_OFFSET_Z
	CloudVolume = _require_cloudvolume()
	patch_cloudfiles_monitoring()
	_WORKER_VOLUME = CloudVolume(
		cloudpath,
		parallel=False,
		bounded=True,
		cache=False,
		compress=False,
	)
	_WORKER_DTYPE = np.dtype(dtype_name)
	_WORKER_MEMMAP = memmap_source
	_WORKER_OFFSET_Z = offset_z
	_WORKER_SOURCE = tifffile.memmap(source) if memmap_source else source


def _write_slice(z_index: int) -> int:
	if _WORKER_MEMMAP:
		image = _WORKER_SOURCE[z_index, :, :]
	else:
		image = tifffile.imread(_WORKER_SOURCE[z_index])
	if image.ndim != 2:
		raise ValueError(f"Z={z_index} is not a 2-D image: {image.shape}")
	image = np.asarray(image, dtype=_WORKER_DTYPE)
	image = np.ascontiguousarray(image.T[:, :, None, None])
	global_z = _WORKER_OFFSET_Z + z_index
	_WORKER_VOLUME[:, :, global_z:global_z + 1, 0:1] = image
	return z_index


def _harvest_completed_futures(futures, completed: set[int]) -> None:
	for future in futures:
		if not future.done() or future.cancelled():
			continue
		try:
			completed.add(future.result())
		except (BrokenProcessPool, CancelledError):
			continue


def _execute_slices(
	cloudpath: str,
	input_spec: InputSpec,
	plan: VolumePlan,
	z_indices: list[int],
	workers: int,
	progress=None,
) -> WorkerBatchResult:
	pool_options = {}
	if sys.version_info >= (3, 11):
		pool_options["max_tasks_per_child"] = 500

	pool = ProcessPoolExecutor(
		max_workers=workers,
		initializer=_init_worker,
		initargs=(
			cloudpath,
			plan.dtype.name,
			input_spec.source,
			input_spec.mode == "memmap",
			plan.voxel_offset[2],
		),
		**pool_options,
	)
	futures = []
	completed = set()
	failure = None
	try:
		for z_index in z_indices:
			futures.append(pool.submit(_write_slice, z_index))
		for future in as_completed(futures):
			completed.add(future.result())
			if progress is not None:
				progress.update(1)
	except BrokenProcessPool as exc:
		failure = exc
	finally:
		pool.shutdown(wait=True, cancel_futures=failure is not None)

	if failure is not None:
		known_completed = set(completed)
		_harvest_completed_futures(futures, completed)
		if progress is not None:
			progress.update(len(completed - known_completed))
	return WorkerBatchResult(frozenset(completed), failure)


def write_all_slices(
	output_path: Path,
	input_spec: InputSpec,
	plan: VolumePlan,
	workers: int,
) -> int:
	"""Write every plane, retrying incomplete work from a broken worker pool."""
	remaining = set(range(input_spec.shape[0]))
	initial_count = len(remaining)
	active_workers = min(workers, len(remaining))
	with log.progress(
		"Z Planes",
		length=initial_count,
		start_message=(
			f"Writing {initial_count} Z plane(s) with {active_workers} worker(s)."
		),
		final_message=lambda handle: (
			f"Wrote {handle.position} Z plane(s)."
		),
	) as progress:
		while remaining:
			result = _execute_slices(
				cloudpath_for(output_path),
				input_spec,
				plan,
				sorted(remaining),
				active_workers,
				progress=progress,
			)
			remaining.difference_update(result.completed)
			if result.failure is not None:
				if active_workers == 1:
					raise result.failure
				active_workers = max(1, active_workers // 2)
				log.write(
					"Z Planes",
					(
						f"Worker pool failed after {len(result.completed)} "
						f"plane(s); retrying {len(remaining)} plane(s) with "
						f"{active_workers} workers."
					),
					log_level=LOG.WARN,
				)
				continue
			if remaining:
				raise RuntimeError(
					f"worker pool exited without completing "
					f"{len(remaining)} Z plane(s)"
				)
	return initial_count


def describe_plan(
	input_path: Path,
	output_path: Path,
	input_spec: InputSpec,
	plan: VolumePlan,
	workers: int,
) -> None:
	statements = (
		f"Input: {input_path.resolve()} ({input_spec.mode})",
		f"Output: {output_path.resolve()}",
		f"Shape (Z,Y,X): {input_spec.shape}; source dtype: {input_spec.dtype}",
		(
			f"Layer: {plan.layer_type}; encoding: {plan.encoding}; "
			f"output dtype: {plan.dtype}"
		),
		f"Voxel resolution (nm): {plan.resolution}",
		f"Voxel offset: {plan.voxel_offset}",
		f"Chunk size: {plan.chunk_size}; workers: {workers}",
	)
	for statement in statements:
		log.write("Precompute", statement, log_level=LOG.INFO)


@click.command("precompute")
@click.argument(
	"input_path",
	type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
)
@click.argument("output_path", required=False, type=click.Path(path_type=Path))
@click.option(
	"--workers",
	type=click.IntRange(min=1),
	default=8,
	show_default=True,
	help="Number of process-parallel Z writers.",
)
@click.option("--layer-type", type=click.Choice(LAYER_TYPES), default="auto", show_default=True)
@click.option(
	"--seg-encoding",
	"segmentation_encoding",
	type=click.Choice(SEGMENTATION_ENCODINGS),
	default="compressed_segmentation",
	show_default=True,
)
@click.option("--dtype", "dtype_override", help="Override the source/output NumPy dtype.")
@click.option("--chunk-size", type=XYZ, help="CloudVolume chunk size as X,Y,Z.")
@click.option(
	"--seg-block",
	"segmentation_block",
	type=XYZ,
	default="8,8,8",
	show_default=True,
	help="Compressed-segmentation block size as X,Y,Z.",
)
@click.option(
	"--voxel-resolution",
	type=XYZ,
	default="700,700,700",
	show_default=True,
	help="Voxel resolution in nanometers as X,Y,Z.",
)
@click.option(
	"--voxel-offset",
	type=XYZ,
	default="0,0,0",
	show_default=True,
	help="Voxel-coordinate offset as X,Y,Z.",
)
@click.option("--execute/--dry-run", default=True, show_default=True)
def precompute(
	input_path: Path,
	output_path: Path | None,
	workers: int,
	layer_type: str,
	segmentation_encoding: str,
	dtype_override: str | None,
	chunk_size: tuple[int, int, int] | None,
	segmentation_block: tuple[int, int, int],
	voxel_resolution: tuple[int, int, int],
	voxel_offset: tuple[int, int, int],
	execute: bool,
) -> None:
	"""Write an unsharded Neuroglancer precomputed volume at MIP 0."""
	try:
		input_spec = discover_input(input_path)
		output_path = output_path or default_output_path(input_path)
		plan = build_plan(
			input_path,
			input_spec,
			layer_type,
			segmentation_encoding,
			dtype_override,
			chunk_size,
			voxel_resolution,
			voxel_offset,
			segmentation_block,
		)
		describe_plan(input_path, output_path, input_spec, plan, workers)
		if not execute:
			return

		volume, created = open_or_create_volume(output_path, plan, input_spec)
		log.write(
			"Precompute",
			"Created MIP 0 metadata."
			if created
			else "Using compatible existing MIP 0 metadata; rewriting all Z planes.",
			log_level=LOG.STATUS,
		)
		written = write_all_slices(output_path, input_spec, plan, workers)
		completeness = check_mip0_completeness(output_path, volume.info)
		if not completeness.complete:
			raise RuntimeError(
				f"MIP 0 completeness check failed: {completeness.summary()}"
			)
		log.write(
			"Precompute",
			f"MIP 0 completeness check passed: {completeness.summary()}",
			log_level=LOG.INFO,
		)
		log.write(
			"Precompute",
			f"Precompute complete; wrote {written} Z plane(s).",
			log_level=LOG.STATUS,
		)
	except click.ClickException:
		raise
	except Exception as exc:
		raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
	precompute()
