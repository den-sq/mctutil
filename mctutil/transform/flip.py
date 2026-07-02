"""Flip a directory-backed TIFF stack along a selected volume axis."""

from pathlib import Path

import click
import numpy as np
import tifffile as tf

from mctutil.shared.log import LOG, log


TIFF_SUFFIXES = {".tif", ".tiff"}


def tiff_paths(input_dir):
	return sorted(path for path in Path(input_dir).iterdir() if path.suffix.lower() in TIFF_SUFFIXES)


def validate_inputs(input_dir, flip_axis):
	paths = tiff_paths(input_dir)
	if not paths:
		raise click.ClickException(f"No TIFF files found in {input_dir}.")
	if flip_axis not in {0, 1, 2}:
		raise click.BadParameter("Use 0 for stack order, 1 for image rows, or 2 for image columns.", param_hint="--flip-axis")
	return paths


def flipped_image(image, flip_axis):
	if flip_axis == 0:
		return image
	return np.flip(image, axis=flip_axis - 1)


@click.command()
@click.option(
	"-f",
	"--flip-axis",
	type=click.IntRange(0, 2),
	required=True,
	help="Volume axis to flip: 0=stack order, 1=image rows, 2=image columns.",
)
@click.option("--dry-run", is_flag=True, help="Plan output files without writing them.")
@click.argument("inputdir", type=click.Path(exists=True, path_type=Path, file_okay=False), required=True)
@click.argument("outputdir", type=click.Path(path_type=Path, file_okay=False), required=True)
def flip_stack(flip_axis, dry_run, inputdir, outputdir):
	"""Flip a TIFF stack along depth, row, or column axis."""
	log.start()
	paths = validate_inputs(inputdir, flip_axis)
	output_paths = [outputdir / path.name for path in paths]
	source_paths = list(reversed(paths)) if flip_axis == 0 else paths

	log.write("Flip Setup", f"{len(paths)} frame(s); axis={flip_axis}; output={outputdir}")
	if dry_run:
		for source, target in zip(source_paths, output_paths):
			log.write("Dry Run", f"Would write {target} from {source}", log_level=LOG.INFO)
		return

	outputdir.mkdir(parents=True, exist_ok=True)
	for source, target in zip(source_paths, output_paths):
		image = tf.imread(source)
		tf.imwrite(target, flipped_image(image, flip_axis))
		log.write("File Written", str(target), log_level=LOG.INFO)
	log.write("Images Written", f"{len(paths)} frame(s)")


if __name__ == "__main__":
	flip_stack()
