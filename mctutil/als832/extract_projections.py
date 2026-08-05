"""Extract ALS Beamline 8.3.2 projection stacks from Data Exchange HDF5 files."""

from __future__ import annotations

from pathlib import Path
import re

import click
import numpy as np

from mctutil.shared.deps import require
from mctutil.shared.log import LOG, log


H5_PATTERNS = ("*.h5", "*.hdf5", "*.he5")


def _require_h5py():
	return require(
		"h5py",
		"als832",
		purpose="h5py is required for ALS 8.3.2 extraction",
		error_type=click.ClickException,
	)


def _require_tifffile():
	return require(
		"tifffile",
		"als832",
		purpose="tifffile is required for ALS 8.3.2 extraction",
		error_type=click.ClickException,
	)


def natural_key(value):
	return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(value))]


def find_projection_ds(handle):
	"""Return the projection dataset, preferring the Data Exchange exchange/data path."""
	h5py = _require_h5py()
	if "exchange/data" in handle and isinstance(handle["exchange/data"], h5py.Dataset):
		return handle["exchange/data"]

	hits = []

	def visit(name, obj):
		if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == "data" and obj.ndim == 3:
			hits.append(obj)

	handle.visititems(visit)
	return hits[0] if hits else None


def human_gb(nbytes):
	return f"{nbytes / 1e9:.2f} GB"


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


def process_file(path, out_root, step=1, projection_range=None, multipage=False, dry_run=False):
	h5py = _require_h5py()
	path = Path(path)

	try:
		handle = h5py.File(path, "r")
	except Exception as exc:
		log.write("Skip", f"cannot open {path.name}: {exc}", log_level=LOG.WARN)
		return 0

	with handle:
		dataset = find_projection_ds(handle)
		if dataset is None:
			log.write("Skip", f"{path.name}: no projection dataset (exchange/data)", log_level=LOG.WARN)
			return 0

		n_frames = dataset.shape[0]
		start, stop = projection_range if projection_range else (0, n_frames)
		start = max(0, start)
		stop = min(stop, n_frames)
		indices = list(range(start, stop, step))
		frame_bytes = int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize
		estimate = len(indices) * frame_bytes
		log.write(
			"Projection Plan",
			(
				f"{path.name}: {n_frames} projections {dataset.shape[1:]} {dataset.dtype}; "
				f"{'would write' if dry_run else 'writing'} {len(indices)} frame(s) "
				f"(~{human_gb(estimate)})"
				+ (" as one multipage BigTIFF" if multipage else "")
			),
			log_level=LOG.STATUS,
		)
		if dry_run:
			return len(indices)

		tifffile = _require_tifffile()
		width = max(4, len(str(n_frames - 1)))
		tick = max(1, len(indices) // 20)

		if multipage:
			folder = out_root
			folder.mkdir(parents=True, exist_ok=True)
			target = folder / f"{path.stem}_projections.tif"
			with tifffile.TiffWriter(str(target), bigtiff=True) as writer:
				for written_index, frame_index in enumerate(indices):
					writer.write(np.asarray(dataset[frame_index]), contiguous=True)
					if written_index % tick == 0 or written_index == len(indices) - 1:
						log.write("Progress", f"{path.name}: {written_index + 1}/{len(indices)}", log_level=LOG.INFO)
			log.write("File Written", str(target))
		else:
			folder = out_root / path.stem
			folder.mkdir(parents=True, exist_ok=True)
			for written_index, frame_index in enumerate(indices):
				tifffile.imwrite(str(folder / f"{path.stem}_{frame_index:0{width}d}.tif"), np.asarray(dataset[frame_index]))
				if written_index % tick == 0 or written_index == len(indices) - 1:
					log.write("Progress", f"{path.name}: {written_index + 1}/{len(indices)}", log_level=LOG.INFO)
			log.write("Directory Written", str(folder))

		return len(indices)


@click.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
	"-o",
	"--output",
	"output_dir",
	default="als832_projections",
	show_default=True,
	type=click.Path(file_okay=False, path_type=Path),
	help="Output root folder.",
)
@click.option("--step", type=click.IntRange(1), default=1, show_default=True, help="Take every Nth projection.")
@click.option(
	"--range",
	"projection_range",
	type=int,
	nargs=2,
	metavar="START STOP",
	help="Projection index range [START, STOP).",
)
@click.option("--multipage", is_flag=True, help="Write one multipage BigTIFF per source file.")
@click.option("--dry-run", is_flag=True, help="Plan extraction without writing files.")
def extract_projections(inputs, output_dir, step, projection_range, multipage, dry_run):
	"""Extract projection frames from ALS 8.3.2 HDF5 files or directories."""
	if not inputs:
		raise click.UsageError("At least one HDF5 file or directory is required.")
	if projection_range is not None and projection_range[1] <= projection_range[0]:
		raise click.BadParameter("STOP must be greater than START.", param_hint="--range")

	log.start()
	files = sorted(iter_h5_inputs(inputs), key=natural_key)
	if not files:
		raise click.ClickException("No .h5/.hdf5/.he5 files found.")

	verb = "Planning" if dry_run else "Processing"
	log.write("ALS 8.3.2", f"{verb} {len(files)} file(s)" + ("" if dry_run else f" -> {output_dir}"))
	total = 0
	for path in files:
		total += process_file(
			path,
			output_dir,
			step=step,
			projection_range=projection_range,
			multipage=multipage,
			dry_run=dry_run,
		)
	log.write("Total", f"{'would write' if dry_run else 'wrote'} {total} projection frame(s)")


if __name__ == "__main__":
	extract_projections()
