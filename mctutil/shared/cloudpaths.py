"""Shared path and encoding helpers for Neuroglancer precomputed layers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


def normalize_layer_path(layer_path: str | Path) -> str:
	"""Return a CloudVolume-compatible layer path, preserving URL schemes."""
	value = str(layer_path)
	if "://" in value:
		return value
	return Path(value).resolve().as_uri()


def normalize_cloudpath(layer_path: str | Path) -> str:
	"""Return the underlying CloudFiles path without a precomputed wrapper."""
	value = str(layer_path).removeprefix("precomputed://")
	return normalize_layer_path(value)


def local_layer_path(layer_path: str | Path) -> Path | None:
	"""Resolve plain and file:// precomputed paths without CloudVolume."""
	value = str(layer_path).removeprefix("precomputed://")
	if "://" not in value:
		return Path(value).resolve()
	parsed = urlparse(value)
	if parsed.scheme != "file":
		return None
	return Path(unquote(parsed.path)).resolve()


def default_queue_root(layer_path: str | Path) -> Path:
	"""Return the default durable queue root for a local layer."""
	local_path = local_layer_path(layer_path)
	if local_path is None:
		raise ValueError("--queue is required for non-local layer paths")
	return local_path / ".mctutil-queues"


def select_layer_encoding(
	requested: str,
	layer_type: str,
	source_encoding: str = "raw",
) -> str:
	"""Resolve the shared ``auto`` encoding policy for NG derived layers."""
	if requested != "auto":
		return requested
	if layer_type == "image":
		return "raw"
	if layer_type == "segmentation":
		if source_encoding == "raw":
			return "compressed_segmentation"
		return source_encoding
	return source_encoding
