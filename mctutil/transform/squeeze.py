"""Remove singleton axes from TIFF files in place."""

from pathlib import Path

import click
import numpy as np

from mctutil.shared.log import log
from mctutil.shared.stack_apply import (
	apply_image_stack,
	plan_stack_map,
	require_tiff_paths,
)


@click.command()
@click.option("--dry-run", is_flag=True, help="List planned rewrites without modifying files.")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def squeeze(directory: Path, dry_run: bool):
	"""Remove every length-one axis from each TIFF in DIRECTORY, in place.

	For example, (500, 500, 1) becomes (500, 500). Pixel values and dtype
	are preserved. Files are rewritten uncompressed using the standard TIFF
	writer; original TIFF metadata is not retained. Subdirectories are ignored.
	"""
	log.start()
	try:
		paths = require_tiff_paths(directory)
	except ValueError as exc:
		raise click.ClickException(str(exc)) from exc
	apply_image_stack(plan_stack_map(paths, directory), np.squeeze, dry_run=dry_run)
	if not dry_run:
		log.write("Squeezed", f"{len(paths)} TIFF file(s)")
