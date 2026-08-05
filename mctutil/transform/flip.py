"""Flip a directory-backed TIFF stack along a selected volume axis."""

from pathlib import Path

import click
import numpy as np

from mctutil.shared.log import log
from mctutil.shared.stack_apply import (
	apply_image_stack,
	plan_stack_map,
	require_tiff_paths,
)


def validate_inputs(input_dir, flip_axis):
	try:
		paths = require_tiff_paths(input_dir)
	except ValueError as exc:
		raise click.ClickException(str(exc)) from exc
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
	source_paths = tuple(reversed(paths)) if flip_axis == 0 else paths
	items = plan_stack_map(
		source_paths,
		outputdir,
		target_names=(path.name for path in paths),
	)

	log.write("Flip Setup", f"{len(paths)} frame(s); axis={flip_axis}; output={outputdir}")
	apply_image_stack(
		items,
		flipped_image,
		operation_args=(flip_axis,),
		dry_run=dry_run,
	)
	if dry_run:
		return
	log.write("Images Written", f"{len(paths)} frame(s)")


if __name__ == "__main__":
	flip_stack()
