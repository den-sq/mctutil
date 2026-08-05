"""Split a multi-page TIFF stack into one TIFF file per Z slice."""

from __future__ import annotations

from pathlib import Path

import click

from mctutil.shared.deps import require
from mctutil.shared.log import LOG, log
from mctutil.shared.tiff_stack_writer import SliceNaming, write_tiff_stack


def _require_tifffile():
	return require(
		"tifffile",
		"transform",
		purpose="tifffile is required for TIFF splitting",
		error_type=click.ClickException,
	)


def extract_tiff_stack(input_path: Path, output_dir: Path, prefix: str | None = None) -> None:
	"""Split a TIFF stack into per-Z TIFF files under ``output_dir``.

		:param input_path: Multi-page TIFF stack.
		:param output_dir: Directory to receive the per-Z output files.
		:param prefix: Filename stem for the output files. Defaults to the input's stem.
	"""
	tifffile = _require_tifffile()

	if prefix is None:
		prefix = input_path.stem

	with tifffile.TiffFile(input_path) as tif:
		num_pages = len(tif.pages)
		if num_pages == 0:
			raise click.ClickException("No pages found in TIFF file.")

		first_page = tif.pages[0]
		height, width = first_page.shape[:2]
		dtype = first_page.dtype

		log.write("Stack Input", str(input_path), log_level=LOG.STATUS)
		log.write(
			"Stack Shape",
			f"width={width} height={height} depth={num_pages} dtype={dtype}",
			log_level=LOG.INFO,
		)
		log.write("Stack Output", str(output_dir), log_level=LOG.STATUS)

		digits = max(4, len(str(num_pages)))

		def validate_frame(arr, z):
			# Guard against inconsistent pages, which tifffile allows in principle
			# but our per-slice-per-file split cannot represent.
			if arr.shape[:2] != (height, width):
				raise click.ClickException(
					f"Page {z} has shape {arr.shape}, expected ({height}, {width})"
				)

		write_tiff_stack(
			lambda z: tif.pages[z].asarray(),
			num_pages,
			output_dir,
			mode="slices",
			naming=SliceNaming(prefix=prefix, digits=digits),
			validate_frame=validate_frame,
			on_progress=lambda position, total, _index, path: (
				log.write(
					"Stack Progress",
					f"slice {position + 1}/{total}: {path.name}",
					log_level=LOG.INFO,
				)
				if position % 100 == 0 or position == total - 1
				else None
			),
		)

	log.write("Stack Done", str(output_dir), log_level=LOG.STATUS)


@click.command()
@click.argument("input_tiff", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
	"-o", "--output-dir",
	type=click.Path(file_okay=False, path_type=Path),
	default=None,
	help="Output directory. Defaults to <input_stem>_slices alongside the input.",
)
@click.option(
	"--prefix",
	type=str,
	default=None,
	help="Prefix for output filenames. Defaults to the input filename's stem.",
)
def stack_split(input_tiff: Path, output_dir: Path | None, prefix: str | None) -> None:
	"""Split a multi-page TIFF stack into one TIFF file per Z slice."""
	if output_dir is None:
		output_dir = input_tiff.with_suffix("").parent / f"{input_tiff.stem}_slices"

	extract_tiff_stack(input_path=input_tiff, output_dir=output_dir, prefix=prefix)


if __name__ == "__main__":
	stack_split()
