"""Cheap local completeness checks for full-resolution precomputed scales."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np


VARIABLE_SIZE_ENCODINGS = {"compressed_segmentation", "compresso"}


@dataclass(frozen=True)
class Mip0Completeness:
	"""Result of a single-enumeration MIP-0 completeness check."""

	complete: bool
	scale_path: Path | None
	metric: str
	expected: int | None
	actual: int | None
	detail: str

	def summary(self) -> str:
		if self.expected is None or self.actual is None:
			return self.detail
		return (
			f"{self.metric}: expected={self.expected}, actual={self.actual}; "
			f"{self.detail}"
		)


@dataclass(frozen=True)
class Mip0Spec:
	scale_path: Path
	encoding: str
	size: tuple[int, ...]
	chunk_size: tuple[int, ...]
	num_channels: int
	dtype: np.dtype


def local_layer_path(layer_path: str | Path) -> Path | None:
	"""Resolve plain and file:// precomputed paths without importing CloudVolume."""
	value = str(layer_path).removeprefix("precomputed://")
	if "://" not in value:
		return Path(value).resolve()
	parsed = urlparse(value)
	if parsed.scheme != "file":
		return None
	return Path(unquote(parsed.path)).resolve()


def _failure(
	detail: str,
	scale_path: Path | None = None,
	metric: str = "unavailable",
	expected: int | None = None,
	actual: int | None = None,
) -> Mip0Completeness:
	return Mip0Completeness(
		complete=False,
		scale_path=scale_path,
		metric=metric,
		expected=expected,
		actual=actual,
		detail=detail,
	)


def _mip0_spec(root: Path, info: dict) -> Mip0Spec:
	scale = info["scales"][0]
	return Mip0Spec(
		scale_path=root / str(scale["key"]),
		encoding=str(scale["encoding"]),
		size=tuple(int(value) for value in scale["size"]),
		chunk_size=tuple(int(value) for value in scale["chunk_sizes"][0]),
		num_channels=int(info.get("num_channels", 1)),
		dtype=np.dtype(info["data_type"]),
	)


def _load_mip0_spec(root: Path, info: dict | None) -> Mip0Spec:
	if info is None:
		info = json.loads((root / "info").read_text(encoding="utf-8"))
	return _mip0_spec(root, info)


def _scan_scale(scale_path: Path) -> tuple[int, int]:
	file_count = 0
	byte_count = 0
	with os.scandir(scale_path) as entries:
		for entry in entries:
			if not entry.is_file(follow_symlinks=False):
				continue
			file_count += 1
			byte_count += entry.stat(follow_symlinks=False).st_size
	return file_count, byte_count


def _raw_result(spec: Mip0Spec, byte_count: int) -> Mip0Completeness:
	expected = math.prod(spec.size) * spec.dtype.itemsize * spec.num_channels
	return Mip0Completeness(
		complete=byte_count == expected,
		scale_path=spec.scale_path,
		metric="bytes",
		expected=expected,
		actual=byte_count,
		detail=(
			"raw MIP 0 is complete"
			if byte_count == expected
			else "raw MIP 0 is incomplete"
		),
	)


def _segmentation_result(
	spec: Mip0Spec,
	file_count: int,
) -> Mip0Completeness:
	expected = math.prod(
		math.ceil(length / chunk)
		for length, chunk in zip(spec.size, spec.chunk_size)
	)
	return Mip0Completeness(
		complete=file_count == expected,
		scale_path=spec.scale_path,
		metric="chunks",
		expected=expected,
		actual=file_count,
		detail=(
			"segmentation MIP 0 is structurally complete"
			if file_count == expected
			else "segmentation MIP 0 is structurally incomplete"
		),
	)


def check_mip0_completeness(
	layer_path: str | Path,
	info: dict | None = None,
) -> Mip0Completeness:
	"""Check local MIP 0 with one scale-directory enumeration and no probes."""
	root = local_layer_path(layer_path)
	if root is None:
		return _failure("completeness checks require a local file:// layer")

	try:
		spec = _load_mip0_spec(root, info)
	except (
		FileNotFoundError,
		json.JSONDecodeError,
		IndexError,
		KeyError,
		TypeError,
		ValueError,
	) as exc:
		return _failure(f"invalid or missing MIP-0 metadata: {exc}")

	try:
		file_count, byte_count = _scan_scale(spec.scale_path)
	except FileNotFoundError:
		return _failure(
			f"MIP-0 scale directory is missing: {spec.scale_path}",
			scale_path=spec.scale_path,
		)

	if spec.encoding == "raw":
		return _raw_result(spec, byte_count)
	if spec.encoding in VARIABLE_SIZE_ENCODINGS:
		return _segmentation_result(spec, file_count)

	return _failure(
		f"unsupported MIP-0 encoding for completeness check: {spec.encoding}",
		scale_path=spec.scale_path,
	)
