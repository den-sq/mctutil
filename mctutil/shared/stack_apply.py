"""Shared listing, planning, and parallel-map scaffolding for TIFF stacks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from mctutil.shared.deps import require
from mctutil.shared.log import LOG, log
from mctutil.shared.tiff_stack_writer import write_tiff_stack


TIFF_SUFFIXES = frozenset({".tif", ".tiff"})


@dataclass(frozen=True)
class StackMapItem:
	"""One input TIFF and its mapped output path."""

	source: Path
	target: Path


def tiff_paths(input_dir: str | Path) -> tuple[Path, ...]:
	"""Return the canonical sorted, non-recursive TIFF listing."""
	return tuple(
		sorted(
			path
			for path in Path(input_dir).iterdir()
			if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
		)
	)


def require_tiff_paths(
	input_dir: str | Path,
	message: str | None = None,
) -> tuple[Path, ...]:
	"""List TIFFs or raise the shared empty-stack error."""
	paths = tiff_paths(input_dir)
	if not paths:
		raise ValueError(message or f"No TIFF files found in {input_dir}.")
	return paths


def batched(values: Iterable[Any], size: int) -> tuple[tuple[Any, ...], ...]:
	"""Partition a finite stack listing while retaining a partial final batch."""
	if size < 1:
		raise ValueError("batch size must be positive")
	values = tuple(values)
	return tuple(
		values[start:start + size]
		for start in range(0, len(values), size)
	)


def plan_stack_map(
	sources: Iterable[Path],
	output_dir: str | Path,
	*,
	target_names: Iterable[str] | None = None,
) -> tuple[StackMapItem, ...]:
	"""Pair source paths with output names without touching the filesystem."""
	sources = tuple(Path(path) for path in sources)
	names = (
		tuple(path.name for path in sources)
		if target_names is None
		else tuple(target_names)
	)
	if len(sources) != len(names):
		raise ValueError("source and target-name counts differ")
	output_dir = Path(output_dir)
	return tuple(
		StackMapItem(source, output_dir / name)
		for source, name in zip(sources, names)
	)


def run_parallel(
	worker: Callable[..., Any],
	arguments: Iterable[tuple],
	workers: int,
	*,
	pool_factory=Pool,
) -> list[Any]:
	"""Apply a picklable worker over prepared arguments with serial parity."""
	arguments = tuple(arguments)
	if workers <= 1:
		return [worker(*args) for args in arguments]
	with pool_factory(workers) as pool:
		return pool.starmap(worker, arguments)


def apply_array(
	image: np.ndarray,
	operation: Callable[..., np.ndarray],
	operation_args: tuple = (),
) -> np.ndarray:
	"""Apply a composable pure per-file operation to an in-memory image."""
	return np.asarray(operation(image, *operation_args))


def _apply_image_item(
	item: StackMapItem,
	operation: Callable[..., np.ndarray],
	operation_args: tuple,
	compression: str | None,
	extra: str,
) -> Path:
	tifffile = require(
		"tifffile",
		extra,
		purpose="tifffile is required for TIFF stack transforms",
	)
	image = tifffile.imread(item.source)
	output = apply_array(image, operation, operation_args)
	write_tiff_stack(
		lambda _index: output,
		1,
		item.target,
		mode="image",
		compression=compression,
		extra=extra,
	)
	log.write("File Written", str(item.target), log_level=LOG.INFO)
	return item.target


def apply_image_stack(
	items: Iterable[StackMapItem],
	operation: Callable[..., np.ndarray],
	*,
	operation_args: tuple = (),
	compression: str | None = None,
	workers: int = 1,
	dry_run: bool = False,
	extra: str = "transform",
	pool_factory=Pool,
) -> tuple[Path, ...]:
	"""Execute a one-input/one-output TIFF map with shared dry-run logging."""
	items = tuple(items)
	if dry_run:
		for item in items:
			log.write(
				"Dry Run",
				f"Would write {item.target} from {item.source}",
				log_level=LOG.INFO,
			)
		return tuple(item.target for item in items)

	run_parallel(
		_apply_image_item,
		(
			(item, operation, operation_args, compression, extra)
			for item in items
		),
		workers,
		pool_factory=pool_factory,
	)
	return tuple(item.target for item in items)


def named_output_paths(
	output_dir: str | Path,
	names: Iterable[str],
) -> tuple[Path, ...]:
	"""Resolve a fixed set of named transform outputs."""
	output_dir = Path(output_dir)
	return tuple(output_dir / name for name in names)


def write_named_images(
	images: Mapping[str, np.ndarray],
	output_dir: str | Path,
	*,
	dry_run: bool = False,
) -> tuple[Path, ...]:
	"""Write or plan a small fixed mapping of names to in-memory images."""
	paths = named_output_paths(output_dir, images)
	for name, target in zip(images, paths):
		if dry_run:
			log.write("Dry Run", f"Would write {target}", log_level=LOG.INFO)
			continue
		write_tiff_stack(
			lambda _index, image=images[name]: image,
			1,
			target,
			mode="image",
		)
		log.write("File Written", str(target), log_level=LOG.INFO)
	return paths
