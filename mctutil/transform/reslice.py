"""Extract orthogonal slices through a directory-backed TIFF stack."""

from collections import namedtuple
from pathlib import Path

import click
import numpy as np
import tifffile as tf

from mctutil.shared.log import LOG, log


Coord = namedtuple("Coord", ["x", "y", "z"])
TIFF_SUFFIXES = {".tif", ".tiff"}


class Coordinates(click.ParamType):
	name = "x,y,z"

	def convert(self, value, param, ctx):
		try:
			fields = [int(field) for field in value.split(",")]
			if len(fields) != 3:
				raise ValueError
			return Coord(*fields)
		except (ValueError, TypeError):
			self.fail(f"{value} is not a 3-value integer coordinate.", param, ctx)


COORDINATES = Coordinates()


def tiff_paths(input_folder):
	return sorted(path for path in Path(input_folder).iterdir() if path.suffix.lower() in TIFF_SUFFIXES)


def stack_shape(paths):
	if not paths:
		raise click.ClickException("No TIFF files found in input folder.")
	with tf.TiffFile(paths[0]) as tiff:
		y, x = tiff.pages[0].shape
	return len(paths), y, x


def validate_coordinate(coord, shape):
	z_size, y_size, x_size = shape
	if not (0 <= coord.x < x_size):
		raise click.BadParameter(f"x={coord.x} is outside [0, {x_size}).", param_hint="--reslice")
	if not (0 <= coord.y < y_size):
		raise click.BadParameter(f"y={coord.y} is outside [0, {y_size}).", param_hint="--reslice")
	if not (0 <= coord.z < z_size):
		raise click.BadParameter(f"z={coord.z} is outside [0, {z_size}).", param_hint="--reslice")


def read_reslices(paths, coord):
	xy = tf.imread(paths[coord.z])
	xz = np.empty((len(paths), xy.shape[1]), dtype=xy.dtype)
	yz = np.empty((len(paths), xy.shape[0]), dtype=xy.dtype)

	for z_index, path in enumerate(paths):
		image = tf.imread(path)
		xz[z_index, :] = image[coord.y, :]
		yz[z_index, :] = image[:, coord.x]

	return {
		f"xy_z{coord.z}.tif": xy,
		f"xz_y{coord.y}.tif": xz,
		f"yz_x{coord.x}.tif": yz,
	}


@click.command()
@click.option("--reslice", "-r", "coord", type=COORDINATES, required=True, help="Coordinate as x,y,z.")
@click.option("--dry-run", is_flag=True, help="Plan output slices without writing them.")
@click.argument("input_folder", type=click.Path(exists=True, path_type=Path, file_okay=False))
@click.argument("output_folder", type=click.Path(path_type=Path, file_okay=False))
def reslice(coord, dry_run, input_folder, output_folder):
	"""Write XY, XZ, and YZ TIFF slices through a stack coordinate."""
	log.start()
	paths = tiff_paths(input_folder)
	shape = stack_shape(paths)
	validate_coordinate(coord, shape)
	log.write("Reslice Setup", f"shape={shape}; coord={coord}; output={output_folder}")

	output_names = [f"xy_z{coord.z}.tif", f"xz_y{coord.y}.tif", f"yz_x{coord.x}.tif"]
	if dry_run:
		for name in output_names:
			log.write("Dry Run", f"Would write {output_folder / name}", log_level=LOG.INFO)
		return

	output_folder.mkdir(exist_ok=True, parents=True)
	for name, image in read_reslices(paths, coord).items():
		target = output_folder / name
		tf.imwrite(target, image)
		log.write("File Written", str(target), log_level=LOG.INFO)


if __name__ == "__main__":
	reslice()
