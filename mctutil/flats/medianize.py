"""Median-combine TIFF flat frames grouped by filename stem."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mctutil.shared.log import LOG, log


def _require_tifffile():
	try:
		import tifffile
	except ImportError as exc:
		raise click.ClickException(
			"tifffile is required for flat medianization; install mctutil[flats]."
		) from exc
	return tifffile


def flat_key(path):
	stem = path.stem
	return stem.rsplit("_", 1)[0] if "_" in stem else stem


def collect_flat_sets(input_dir):
	flat_sets = {}
	for full_path in sorted(Path(input_dir).iterdir()):
		if full_path.suffix.lower() not in {".tif", ".tiff"}:
			continue
		key = flat_key(full_path)
		flat_sets.setdefault(key, []).append(full_path)
	return flat_sets


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Plan median outputs without reading or writing frames.")
def medianize(input_dir, output_dir, dry_run):
	"""Median TIFF flats by filename prefix and write one median image per group."""
	log.start()
	flat_sets = collect_flat_sets(input_dir)
	if not flat_sets:
		raise click.ClickException(f"No TIFF flats found in {input_dir}.")

	if dry_run:
		for key, paths in flat_sets.items():
			log.write("Dry Run", f"Would medianize {len(paths)} frame(s) for {key}", log_level=LOG.INFO)
		return

	tifffile = _require_tifffile()
	output_dir.mkdir(exist_ok=True, parents=True)
	for key, paths in flat_sets.items():
		flats_data = [tifffile.imread(path) for path in paths]
		median_flat = np.median(flats_data, axis=0)
		shape_dir = output_dir / str(median_flat.shape)
		shape_dir.mkdir(exist_ok=True, parents=True)
		output_path = shape_dir / f"{key}_median.tif"
		tifffile.imwrite(output_path, median_flat)
		log.write("File Written", f"{output_path} from {len(paths)} frame(s)")


if __name__ == "__main__":
	medianize()
