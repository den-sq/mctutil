"""Canonical TIFF image, stack, and per-Z write tail."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mctutil.shared.deps import require


WriteMode = Literal["image", "stack", "slices"]
FrameReader = Callable[[int], Any]
FrameCallback = Callable[[np.ndarray, int, Path], None]
ProgressCallback = Callable[[int, int, int, Path], None]
FrameValidator = Callable[[np.ndarray, int], None]


@dataclass(frozen=True)
class SliceNaming:
	"""Numbered per-Z TIFF naming policy."""

	prefix: str
	digits: int
	separator: str = "_z"
	suffix: str = ".tif"

	def filename(self, index: int) -> str:
		return (
			f"{self.prefix}{self.separator}{index:0{self.digits}d}"
			f"{self.suffix}"
		)


def compression_for(enabled: bool, codec: str = "zlib") -> str | None:
	"""Return one canonical tifffile compression value."""
	return codec if enabled else None


def numbered_tiff_path(
	directory: Path,
	naming: SliceNaming,
	index: int,
) -> Path:
	"""Build a per-Z path through the shared naming policy."""
	return Path(directory) / naming.filename(index)


def tiff_output_path(path: Path, suffix: str = ".tiff") -> Path:
	"""Normalize a decoded image destination to a TIFF suffix."""
	return Path(path).with_suffix(suffix)


def _source_indices(
	frame_count: int,
	indices: Iterable[int] | None,
) -> tuple[int, ...]:
	values = tuple(range(frame_count)) if indices is None else tuple(indices)
	if len(values) != frame_count:
		raise ValueError(
			f"frame_count={frame_count} does not match {len(values)} source indices"
		)
	return values


def _planned_paths(
	destination: Path,
	mode: WriteMode,
	indices: tuple[int, ...],
	naming: SliceNaming | None,
) -> tuple[Path, ...]:
	if mode in {"image", "stack"}:
		return (destination,)
	if mode != "slices":
		raise ValueError(f"unsupported TIFF write mode: {mode}")
	if naming is None:
		raise ValueError("slice naming is required for per-Z TIFF output")
	return tuple(
		numbered_tiff_path(destination, naming, index)
		for index in indices
	)


def _imwrite(
	tifffile,
	path: Path,
	frame: np.ndarray,
	compression: str | None,
	bigtiff: bool | None,
) -> None:
	options = {"compression": compression}
	if bigtiff is not None:
		options["bigtiff"] = bigtiff
	tifffile.imwrite(path, frame, **options)


def write_tiff_stack(  # noqa: C901
	frame_reader: FrameReader,
	frame_count: int,
	destination: Path,
	*,
	mode: WriteMode,
	indices: Iterable[int] | None = None,
	naming: SliceNaming | None = None,
	compression: str | None = None,
	bigtiff: bool | None = None,
	contiguous: bool = False,
	dry_run: bool = False,
	extra: str = "transform",
	validate_frame: FrameValidator | None = None,
	on_frame: FrameCallback | None = None,
	on_progress: ProgressCallback | None = None,
) -> tuple[Path, ...]:
	"""Write lazily supplied frames under one shared TIFF policy.

	Dry runs resolve every destination without importing tifffile, creating
	directories, or invoking ``frame_reader``.
	"""
	if frame_count < 0:
		raise ValueError("TIFF frame count cannot be negative")
	if mode == "image" and frame_count != 1:
		raise ValueError("image mode requires exactly one frame")
	source_indices = _source_indices(frame_count, indices)
	destination = Path(destination)
	paths = _planned_paths(destination, mode, source_indices, naming)
	if dry_run:
		return paths

	tifffile = require(
		"tifffile",
		extra,
		purpose="tifffile is required for TIFF output",
	)
	if mode == "slices":
		destination.mkdir(parents=True, exist_ok=True)
	else:
		destination.parent.mkdir(parents=True, exist_ok=True)

	def read_frame(position: int, source_index: int, path: Path) -> np.ndarray:
		frame = np.asarray(frame_reader(source_index))
		if validate_frame is not None:
			validate_frame(frame, source_index)
		if on_frame is not None:
			on_frame(frame, source_index, path)
		return frame

	if mode == "image":
		frame = read_frame(0, source_indices[0], destination)
		_imwrite(tifffile, destination, frame, compression, bigtiff)
		if on_progress is not None:
			on_progress(0, 1, source_indices[0], destination)
		return paths

	if mode == "stack":
		with tifffile.TiffWriter(destination, bigtiff=bool(bigtiff)) as writer:
			for position, source_index in enumerate(source_indices):
				frame = read_frame(position, source_index, destination)
				writer.write(
					frame,
					compression=compression,
					contiguous=contiguous,
				)
				if on_progress is not None:
					on_progress(
						position,
						frame_count,
						source_index,
						destination,
					)
		return paths

	for position, (source_index, path) in enumerate(zip(source_indices, paths)):
		frame = read_frame(position, source_index, path)
		_imwrite(tifffile, path, frame, compression, bigtiff)
		if on_progress is not None:
			on_progress(position, frame_count, source_index, path)
	return paths
