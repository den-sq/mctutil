"""Small direct-file and HDF5 primitives shared by canonical readers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from mctutil.parse.import_records import decode_scalar
from mctutil.shared.deps import require


H5_SUFFIXES = (".h5", ".hdf5")


def require_h5py(purpose: str):
	"""Load the existing HDF5 extra with a source-specific error purpose."""
	return require("h5py", "als832", purpose=purpose)


def discover_files(
	inputs: Iterable[Path],
	suffixes: tuple[str, ...],
	missing_message: str,
) -> tuple[Path, ...]:
	"""Return sorted explicit or direct-child files with accepted suffixes."""
	paths: set[Path] = set()
	for raw_path in inputs:
		path = Path(raw_path)
		if path.is_dir():
			paths.update(
				item
				for item in path.iterdir()
				if item.is_file() and item.suffix.casefold() in suffixes
			)
		elif path.is_file() and path.suffix.casefold() in suffixes:
			paths.add(path)
	if not paths:
		raise ValueError(missing_message)
	return tuple(sorted(paths, key=lambda item: str(item)))


def dataset(handle, path: str):
	"""Return an HDF5 dataset-like object, never a group."""
	item = handle.get(path)
	return item if item is not None and hasattr(item, "shape") else None


def representative_value(handle, path: str):
	"""Return the finite numeric median or first non-empty text scalar."""
	item = dataset(handle, path)
	if item is None:
		return None
	try:
		values = np.asarray(item[()]).reshape(-1)
	except OSError:
		return None
	if not values.size:
		return None
	if values.dtype.kind in "iufb":
		finite = values[np.isfinite(values.astype(float, copy=False))]
		return float(np.median(finite)) if finite.size else None
	for value in values:
		decoded = decode_scalar(value)
		if decoded:
			return decoded
	return None
