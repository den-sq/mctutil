"""Dump HDF5 structure with ALS/DataExchange-friendly dataset value expansion."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mctutil.shared.deps import require
from mctutil.shared.log import LOG, log


EXPAND_HINTS = ("exchange", "flat", "dark", "white", "bright", "theta")
MAX_LIST = 8
VALUE_MAX = 64
SELECTED_VALUE_MAX = 10000


def _require_h5py():
	return require(
		"h5py",
		"als832",
		purpose="h5py is required for ALS 8.3.2 HDF5 inspection",
		error_type=click.ClickException,
	)


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


def normalize_h5_path(path):
	"""Return an HDF5 path relative to the root group."""
	return path.strip("/")


def selected_value(dataset, max_values):
	"""Read and format a selected dataset when it is within the safety limit."""
	if max_values and dataset.size > max_values:
		return (
			f"<{dataset.size} values; exceeds --max-values {max_values}. "
			"Use --max-values 0 to read without a limit>"
		)
	try:
		return fmt_val(dataset[()])
	except Exception as exc:
		return f"<unreadable: {exc}>"


def dump_selected_dataset(dataset, indent="", max_values=SELECTED_VALUE_MAX):
	"""Print a selected dataset, its value, and its attributes."""
	value = selected_value(dataset, max_values)
	emit(f"{indent}{dataset.name}   shape={dataset.shape} dtype={dataset.dtype}  = {value}")
	show_attrs(dataset, indent + "  ")


def dump_selected_group(group, indent="", max_values=SELECTED_VALUE_MAX):
	"""Recursively print every dataset below a selected group."""
	h5py = _require_h5py()
	emit(f"{indent}{group.name.rstrip('/') or '/'}/")
	show_attrs(group, indent + "  ")
	for key in group.keys():
		obj = group[key]
		if isinstance(obj, h5py.Dataset):
			dump_selected_dataset(obj, indent + "  ", max_values)
		elif isinstance(obj, h5py.Group):
			dump_selected_group(obj, indent + "  ", max_values)


def dump_path(handle, requested_path, max_values=SELECTED_VALUE_MAX):
	"""Print one exact dataset or all descendants of one group."""
	h5py = _require_h5py()
	path = normalize_h5_path(requested_path)
	if path not in handle:
		raise click.ClickException(f"HDF5 path not found: /{path}")
	obj = handle[path]
	if isinstance(obj, h5py.Dataset):
		dump_selected_dataset(obj, max_values=max_values)
	elif isinstance(obj, h5py.Group):
		dump_selected_group(obj, max_values=max_values)
	else:
		raise click.ClickException(f"Unsupported HDF5 object at /{path}")


@click.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
	"--path",
	"paths",
	multiple=True,
	help=(
		"HDF5 dataset or group path to read. Repeat for multiple paths. "
		"Groups include all descendants."
	),
)
@click.option(
	"--max-values",
	type=click.IntRange(min=0),
	default=SELECTED_VALUE_MAX,
	show_default=True,
	help="Maximum values read from each selected dataset; 0 disables the limit.",
)
def h5_tree(inputs, paths, max_values):
	"""Read HDF5 structure or values without modifying the source files.

	With no --path, print the complete tree and small values. A selected
	dataset prints its value; a selected group recursively prints every
	descendant dataset and its value.
	"""
	if not inputs:
		raise click.UsageError("At least one HDF5 file is required.")

	h5py = _require_h5py()
	for path in inputs:
		log.write("H5 File", str(path), log_level=LOG.STATUS)
		with h5py.File(path, "r") as handle:
			if paths:
				for requested_path in paths:
					dump_path(handle, requested_path, max_values=max_values)
			else:
				emit("ROOT:")
				dump(handle)


if __name__ == "__main__":
	h5_tree()
