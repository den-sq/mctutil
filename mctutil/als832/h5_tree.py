"""Dump HDF5 structure with ALS/DataExchange-friendly dataset value expansion."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mctutil.shared.log import LOG, log


EXPAND_HINTS = ("exchange", "flat", "dark", "white", "bright", "theta")
MAX_LIST = 8
VALUE_MAX = 64


def _require_h5py():
	try:
		import h5py
	except ImportError as exc:
		raise click.ClickException(
			"h5py is required for ALS 8.3.2 HDF5 inspection; install mctutil[als832]."
		) from exc
	return h5py


def fmt_val(value):
	array = np.asarray(value)
	flat = array.ravel()
	out = []
	for item in flat.tolist():
		out.append(item.decode() if isinstance(item, (bytes, bytearray)) else item)
	return repr(out[0]) if array.size == 1 else repr(out)


def emit(line):
	log.dump(line, log_level=LOG.STATUS)


def show_attrs(obj, indent):
	for key in obj.attrs:
		value = np.asarray(obj.attrs[key])
		text = fmt_val(value) if value.size <= VALUE_MAX else f"<{value.dtype} {value.shape}>"
		emit(f"{indent}@{key} = {text}")


def has_stack(group):
	h5py = _require_h5py()
	for key in group.keys():
		obj = group[key]
		if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
			return True
	return False


def is_expand(group):
	low = group.name.lower()
	return any(hint in low for hint in EXPAND_HINTS) or has_stack(group)


def dump(group, indent=""):
	h5py = _require_h5py()
	show_attrs(group, indent + "  ")
	datasets = [key for key in group.keys() if isinstance(group[key], h5py.Dataset)]
	subgroups = [key for key in group.keys() if isinstance(group[key], h5py.Group)]
	expand = is_expand(group) or len(datasets) <= MAX_LIST
	shown = datasets if expand else datasets[:2]

	for key in shown:
		dataset = group[key]
		line = f"{indent}  - {key}   shape={dataset.shape} dtype={dataset.dtype}"
		if dataset.size <= VALUE_MAX:
			try:
				line += f"  = {fmt_val(dataset[()])}"
			except Exception:
				pass
		emit(line)
		show_attrs(dataset, indent + "        ")

	if not expand and len(datasets) > 2:
		emit(f"{indent}  - ... ({len(datasets)} datasets total; showing first 2)")

	for key in subgroups:
		emit(f"{indent}  {key}/")
		dump(group[key], indent + "  ")


@click.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
def h5_tree(inputs):
	"""Print groups, datasets, attributes, and small values from HDF5 files."""
	if not inputs:
		raise click.UsageError("At least one HDF5 file is required.")

	h5py = _require_h5py()
	for path in inputs:
		log.write("H5 File", str(path), log_level=LOG.STATUS)
		emit("ROOT:")
		with h5py.File(path, "r") as handle:
			dump(handle)


if __name__ == "__main__":
	h5_tree()
