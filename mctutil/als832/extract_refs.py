"""Extract ALS Beamline 8.3.2 flat and dark reference frames."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import re

import click
import numpy as np

from mctutil.shared.deps import require
from mctutil.shared.log import LOG, log
from mctutil.shared.tiff_stack_writer import SliceNaming, write_tiff_stack


H5_PATTERNS = ("*.h5", "*.hdf5", "*.he5")
REF_STACKS = {
	"data_white": ("gains", "white"),
	"data_dark": ("darks", "dark"),
}
IMAGE_KEY = {"white": 1, "dark": 2}


def _require_h5py():
	return require(
		"h5py",
		"als832",
		purpose="h5py is required for ALS 8.3.2 reference extraction",
		error_type=click.ClickException,
	)


def natural_key(value):
	return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(value))]


def find_exchange(handle):
	"""Return the group holding data_white / data_dark, or None."""
	h5py = _require_h5py()
	if "exchange" in handle and isinstance(handle["exchange"], h5py.Group):
		group = handle["exchange"]
		if any(name in group for name in REF_STACKS):
			return group

	hits = []

	def visit(_name, obj):
		if isinstance(obj, h5py.Group) and any(name in obj for name in REF_STACKS):
			hits.append(obj)

	handle.visititems(visit)
	return hits[0] if hits else None


def _read_array(handle, paths, expected_len=None):
	for path in paths:
		if path in handle:
			if expected_len is not None and len(handle[path]) != expected_len:
				continue
			return np.asarray(handle[path][()])
	return None


def _format_timestamp(timestamp, nanosecond=None):
	try:
		value = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
		if nanosecond is not None:
			value += f".{int(nanosecond):09d}"
		return value + "Z"
	except Exception:
		return str(int(timestamp))


def _format_timestamps(timestamps, nanoseconds=None):
	formatted = []
	for index, timestamp in enumerate(timestamps):
		nanosecond = None if nanoseconds is None else nanoseconds[index]
		formatted.append(_format_timestamp(timestamp, nanosecond))
	return formatted


def _label_timestamps(label, image_key, date, nanoseconds, counts):
	key = IMAGE_KEY[label]
	selected = image_key == key
	timestamps = date[selected]
	if len(timestamps) != counts.get(label, -1):
		return None
	ns_values = nanoseconds[selected] if nanoseconds is not None else None
	return _format_timestamps(timestamps, ns_values)


def load_timestamps(handle, _exchange, counts):
	"""Map per-frame timestamps to white/dark frames via image_key + image_date."""
	image_key = _read_array(handle, ("exchange/image_key", "image_key"))
	if image_key is None:
		return {}

	date = _read_array(handle, ("process/acquisition/image_date", "exchange/image_date"), len(image_key))
	if date is None:
		return {}

	nanoseconds = _read_array(handle, ("process/acquisition/image_date_ns",), len(image_key))
	out = {}
	for label in IMAGE_KEY:
		label_values = _label_timestamps(label, image_key, date, nanoseconds, counts)
		if label_values is not None:
			out[label] = label_values
	return out


def _read_scalar(handle, *paths):
	for path in paths:
		if path in handle:
			try:
				return np.asarray(handle[path][()]).ravel()[0].item()
			except Exception:
				pass
	return ""


def read_flat_meta(handle):
	"""Read flat-field acquisition params shared by every frame in the file."""
	h5py = _require_h5py()
	base = "process/acquisition/flat_fields"
	if base not in handle:
		hits = []
		handle.visititems(
			lambda name, obj: hits.append(name)
			if isinstance(obj, h5py.Group) and name.lower().endswith("flat_fields")
			else None
		)
		base = hits[0] if hits else None

	keys = ("i0_move_x", "i0_move_y", "i0cycle")
	if base is None:
		return {key: "" for key in keys}
	return {key: _read_scalar(handle, f"{base}/{key}") for key in keys}


def _read_scalar_by_leaf(handle, leaf):
	h5py = _require_h5py()
	hits = []
	handle.visititems(
		lambda name, obj: hits.append(name)
		if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == leaf
		else None
	)
	return _read_scalar(handle, hits[0]) if hits else ""


def read_camera_meta(handle):
	"""Read camera stage geometry shared by every frame in the file."""
	base = "measurement/instrument/camera_motor_stack/setup"
	keys = ("camera_distance", "camera_elevation", "tilt_motor")
	out = {}
	for key in keys:
		value = _read_scalar(handle, f"{base}/{key}")
		out[key] = value if value != "" else _read_scalar_by_leaf(handle, key)
	return out


def _reference_counts(exchange):
	return {
		label: exchange[name].shape[0] if name in exchange else 0
		for name, (_subfolder, label) in REF_STACKS.items()
	}


def _log_white_stats(exchange):
	white = exchange["data_white"]
	means = [float(white[index].mean()) for index in range(white.shape[0])]
	low, high = min(means), max(means)
	log.write("White Means", f"min={low:.1f} max={high:.1f} spread={high - low:.1f}")
	median = float(np.median(means))
	outliers = [index for index, mean in enumerate(means) if abs(mean - median) > 0.15 * median]
	if outliers:
		log.write(
			"Flat Outliers",
			f"{len(outliers)} flat(s) deviate >15% from median: {outliers}",
			log_level=LOG.WARN,
		)


def inspect_file(path, exchange, counts, flat_meta, camera_meta):
	projection_count = exchange["data"].shape[0] if "data" in exchange else "?"
	log.write(
		"Exchange",
		(
			f"{path.name}: projections={projection_count} "
			+ " ".join(f"{label}={counts[label]}" for label in ("white", "dark"))
		),
		log_level=LOG.STATUS,
	)
	if counts.get("white"):
		_log_white_stats(exchange)

	move_x = flat_meta.get("i0_move_x")
	move_y = flat_meta.get("i0_move_y")
	if move_x == 0 and move_y == 0:
		log.write(
			"Flat Metadata",
			"i0_move_x = i0_move_y = 0; sample may not have moved out of beam for flats",
			log_level=LOG.WARN,
		)
	log.write(
		"Flat Metadata",
		f"i0_move_x={move_x} i0_move_y={move_y} i0cycle={flat_meta.get('i0cycle')}",
	)
	log.write(
		"Geometry",
		(
			f"camera_distance={camera_meta.get('camera_distance')} "
			f"camera_elevation={camera_meta.get('camera_elevation')} "
			f"tilt_motor={camera_meta.get('tilt_motor')}"
		),
	)


def _frame_stats(frame):
	return round(float(frame.mean()), 2), int(frame.min()), int(frame.max())


def _manifest_row(path, label, index, output, timestamp, stats, flat_meta, camera_meta):
	mean, minimum, maximum = stats
	return {
		"source_file": path.name,
		"type": label,
		"index": index,
		"output": output,
		"timestamp": timestamp,
		"mean": mean,
		"min": minimum,
		"max": maximum,
		"i0_move_x": flat_meta.get("i0_move_x", ""),
		"i0_move_y": flat_meta.get("i0_move_y", ""),
		"i0cycle": flat_meta.get("i0cycle", ""),
		"camera_distance": camera_meta.get("camera_distance", ""),
		"camera_elevation": camera_meta.get("camera_elevation", ""),
		"tilt_motor": camera_meta.get("tilt_motor", ""),
	}


def extract_stack(path, out_root, manifest_rows, stack_info, timestamps, meta, dry_run, write_frames):
	_name, subfolder, label, dataset = stack_info
	flat_meta, camera_meta = meta
	n_frames = dataset.shape[0]
	dest = out_root / subfolder
	need_write = write_frames and not dry_run

	frame_timestamps = timestamps.get(label)
	width = max(3, len(str(n_frames - 1)))
	naming = SliceNaming(
		prefix=f"{path.stem}__{label}",
		digits=width,
		separator="_",
	)
	stats_by_index = {}
	planned_paths = ()
	if write_frames:
		planned_paths = write_tiff_stack(
			lambda index: np.asarray(dataset[index]),
			n_frames,
			dest,
			mode="slices",
			naming=naming,
			dry_run=dry_run,
			extra="als832",
			on_frame=lambda frame, index, _target: stats_by_index.__setitem__(
				index,
				_frame_stats(frame),
			),
		)
	for index in range(n_frames):
		filename = naming.filename(index)
		relative_path = f"{subfolder}/{filename}"
		if need_write:
			stats = stats_by_index[index]
		else:
			if dry_run and write_frames:
				log.write(
					"Dry Run",
					f"Would write {planned_paths[index]}",
					log_level=LOG.INFO,
				)
			stats = ("", "", "")
		timestamp = frame_timestamps[index] if frame_timestamps else ""
		output = relative_path if write_frames else ""
		manifest_rows.append(_manifest_row(path, label, index, output, timestamp, stats, flat_meta, camera_meta))

	dest_note = f"{subfolder}/" if write_frames else "manifest only; frames not written"
	log.write("Reference Frames", f"{path.name}: {label} x{n_frames} -> {dest_note}")


def extract_file(path, exchange, out_root, manifest_rows, timestamps, meta, dry_run, write_frames):
	for name, (subfolder, label) in REF_STACKS.items():
		if name in exchange:
			stack_info = (name, subfolder, label, exchange[name])
			extract_stack(path, out_root, manifest_rows, stack_info, timestamps, meta, dry_run, write_frames)


def process_open_file(path, out_root, manifest_rows, handle, mode, dry_run, write_frames):
	exchange = find_exchange(handle)
	if exchange is None:
		log.write(
			"Skip",
			f"{path.name}: no exchange group with data_white/data_dark (keys: {list(handle.keys())[:6]})",
			log_level=LOG.WARN,
		)
		return

	counts = _reference_counts(exchange)
	timestamps = load_timestamps(handle, exchange, counts)
	flat_meta = read_flat_meta(handle)
	camera_meta = read_camera_meta(handle)
	if mode == "inspect":
		inspect_file(path, exchange, counts, flat_meta, camera_meta)
	else:
		extract_file(path, exchange, out_root, manifest_rows, timestamps, (flat_meta, camera_meta), dry_run, write_frames)


def process_file(path, out_root, manifest_rows, mode="extract", dry_run=False, write_frames=True):
	h5py = _require_h5py()
	path = Path(path)

	try:
		handle = h5py.File(path, "r")
	except Exception as exc:
		log.write("Skip", f"cannot open {path.name}: {exc}", log_level=LOG.WARN)
		return

	with handle:
		process_open_file(path, out_root, manifest_rows, handle, mode, dry_run, write_frames)


def iter_h5_inputs(inputs):
	for item in inputs:
		path = Path(item)
		if path.is_dir():
			seen = set()
			for pattern in H5_PATTERNS:
				for candidate in path.rglob(pattern):
					if candidate not in seen:
						seen.add(candidate)
						yield candidate
		elif path.is_file():
			yield path
		else:
			log.write("Input Missing", str(item), log_level=LOG.WARN)


def write_manifest(path, rows):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


@click.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
	"-o",
	"--output",
	"output_dir",
	default="als832_refs",
	show_default=True,
	type=click.Path(file_okay=False, path_type=Path),
	help="Output root folder.",
)
@click.option("--inspect", is_flag=True, help="Summarize each file, including flat means, without writing.")
@click.option(
	"--manifest-only",
	is_flag=True,
	help="Write manifest.csv from metadata only; skip frame TIFFs and per-frame pixel reads.",
)
@click.option("--dry-run", is_flag=True, help="Plan extraction without writing files.")
def extract_refs(inputs, output_dir, inspect, manifest_only, dry_run):
	"""Extract ALS 8.3.2 flat/bright and dark reference frames."""
	if not inputs:
		raise click.UsageError("At least one HDF5 file or directory is required.")

	log.start()
	files = sorted(iter_h5_inputs(inputs), key=natural_key)
	if not files:
		raise click.ClickException("No .h5/.hdf5/.he5 files found.")

	mode = "inspect" if inspect else "extract"
	write_frames = not manifest_only
	if inspect:
		heading = "Inspecting"
	elif dry_run:
		heading = "Planning"
	elif manifest_only:
		heading = "Building manifest for"
	else:
		heading = "Processing"
	log.write("ALS 8.3.2", f"{heading} {len(files)} file(s)" + ("" if inspect or dry_run else f" -> {output_dir}"))

	manifest_rows = []
	for path in files:
		process_file(
			path,
			output_dir,
			manifest_rows,
			mode=mode,
			dry_run=dry_run,
			write_frames=write_frames,
		)

	if mode == "extract" and manifest_rows:
		ngain = sum(row["type"] == "white" for row in manifest_rows)
		ndark = sum(row["type"] == "dark" for row in manifest_rows)
		if dry_run:
			log.write("Dry Run", f"Would write manifest.csv with gains={ngain} darks={ndark}")
			return

		manifest_path = output_dir / "manifest.csv"
		write_manifest(manifest_path, manifest_rows)
		verb = "Catalogued" if not write_frames else "Extracted"
		log.write("References", f"{verb} gains={ngain} darks={ndark} across {len(files)} file(s)")
		log.write("Manifest", str(manifest_path))


if __name__ == "__main__":
	extract_refs()
