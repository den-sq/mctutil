"""Extract orthogonal slices through a directory-backed TIFF stack."""

from pathlib import Path

import click
import numpy as np
import tifffile as tf

from mctutil.shared.cli import XYZ
from mctutil.shared.log import log
from mctutil.shared.stack_apply import require_tiff_paths, write_named_images


def stack_shape(paths):
	if not paths:
		raise click.ClickException("No TIFF files found in input folder.")
	with tf.TiffFile(paths[0]) as tiff:
		y, x = tiff.pages[0].shape
	return len(paths), y, x


def validate_coordinate(coord, shape):
	x, y, z = coord
	z_size, y_size, x_size = shape
	if not (0 <= x < x_size):
		raise click.BadParameter(f"x={x} is outside [0, {x_size}).", param_hint="--reslice")
	if not (0 <= y < y_size):
		raise click.BadParameter(f"y={y} is outside [0, {y_size}).", param_hint="--reslice")
	if not (0 <= z < z_size):
		raise click.BadParameter(f"z={z} is outside [0, {z_size}).", param_hint="--reslice")


def read_reslices(paths, coord):
	x, y, z = coord
	xy = tf.imread(paths[z])
	xz = np.empty((len(paths), xy.shape[1]), dtype=xy.dtype)
	yz = np.empty((len(paths), xy.shape[0]), dtype=xy.dtype)

	for z_index, path in enumerate(paths):
		image = tf.imread(path)
		xz[z_index, :] = image[y, :]
		yz[z_index, :] = image[:, x]

	return {
		f"xy_z{z}.tif": xy,
		f"xz_y{y}.tif": xz,
		f"yz_x{x}.tif": yz,
	}


@click.command()
@click.option("--reslice", "-r", "coord", type=XYZ, required=True, help="Coordinate as x,y,z.")
@click.option("--dry-run", is_flag=True, help="Plan output slices without writing them.")
@click.argument("input_folder", type=click.Path(exists=True, path_type=Path, file_okay=False))
@click.argument("output_folder", type=click.Path(path_type=Path, file_okay=False))
def reslice(coord, dry_run, input_folder, output_folder):
	"""Write XY, XZ, and YZ TIFF slices through a stack coordinate."""
	log.start()
	try:
		paths = require_tiff_paths(
			input_folder,
			"No TIFF files found in input folder.",
		)
	except ValueError as exc:
		raise click.ClickException(str(exc)) from exc
	shape = stack_shape(paths)
	validate_coordinate(coord, shape)
	log.write("Reslice Setup", f"shape={shape}; coord={coord}; output={output_folder}")

	x, y, z = coord
	output_names = [f"xy_z{z}.tif", f"xz_y{y}.tif", f"yz_x{x}.tif"]
	if dry_run:
		write_named_images(
			dict.fromkeys(output_names),
			output_folder,
			dry_run=True,
		)
		return

	write_named_images(read_reslices(paths, coord), output_folder)


if __name__ == "__main__":
	reslice()
