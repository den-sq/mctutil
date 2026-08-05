from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import click
from osgeo import gdal

from mctutil.shared.log import log, LOG
from mctutil.shared.tiff_stack_writer import write_tiff_stack

startTime = datetime.now()
gdal.UseExceptions() 	# Throws many warnings if we don't set whether we want exceptions.


def get_image_paths(folder: Path):
	""" Finds all HDF files in raw subdirectory of folder.

		:param folder: Base Path containing raw folder.
		:return: Sorted list of HDF files.
	"""
	return sorted(folder.glob('raw/*.hdf'))


def image_conv(image_path: Path, target_dir: Path, execute: bool = True):
	""" Converts hdf5 file to a tiff file and writes the result in a new location.

		:param image_path: Full path to an image.
		:param target_dir: Folder to write tiff to, retaining filename (except suffix).
		:param execute: If False, plan the conversion without reading or writing.
	"""
	target_path = target_dir.joinpath(image_path.with_suffix(".tiff").name)

	if not execute:
		write_tiff_stack(
			lambda _index: (_ for _ in ()).throw(
				AssertionError("dry run decoded HDF data")
			),
			1,
			target_path,
			mode="image",
			dry_run=True,
		)
		log.write("HDF Convert", f"Would convert {image_path} -> {target_path}", log_level=LOG.INFO)
		return

	with open(target_dir.parent.joinpath(f"{target_dir.name}.log"), "a") as logfile:
		src_ds = gdal.Open(str(image_path))
		log.write("HDF Convert", f"{image_path} read", log_level=LOG.STATUS)
		logfile.write(f"{image_path} Read\n")

		out_ds = gdal.Translate('/vsimem/in_memory_output.tif', src_ds, format='GTiff', bandList=[1])
		out_arr = out_ds.ReadAsArray()

		write_tiff_stack(
			lambda _index: out_arr,
			1,
			target_path,
			mode="image",
		)
		log.write("HDF Convert", f"{target_path} written", log_level=LOG.STATUS)
		logfile.write(f"{target_path} Written\n")


@click.command()
@click.option("--target-dir", "-t", default=Path("."), type=click.Path(path_type=Path),
				help="Target directory to write to, defaulting to current working directory.")
@click.option("--processes", "-p", default=60, type=click.INT,
				help="Number of simulatenous processes to use for reading, default 60.  If 1, it will not use multiprocessing.")
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually run the conversions or just plan the work.")
@click.argument("input", nargs=-1,
				type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path))
def hdf_convert(target_dir, processes, execute, input):
	""" Converts files from hdf format to tiff files, usually for hdf4.

		Files are searched for in INPUT/raw/;  any number of INPUT directories are allowed."""
	for proj_dir in input:
		file_paths = get_image_paths(proj_dir)
		total = len(file_paths)

		if total == 0:
			log.write("HDF Convert", f"{proj_dir}: No images found, skipping.", log_level=LOG.WARN)
			continue
		else:
			target_subdir = target_dir.joinpath("tiff_sets", proj_dir.parent.name, proj_dir.name)
			log.write("HDF Convert",
					f"{proj_dir}: {total} images found; writing to {target_subdir}",
					log_level=LOG.STATUS)
			if execute:
				target_subdir.mkdir(parents=True, exist_ok=True)

		if execute:
			with open(target_subdir.parent.joinpath(f"{target_subdir.name}.log"), "a") as logfile:
				logfile.write(f"{proj_dir}: {total} images found; writing to {target_subdir}\n")

		if processes == 1 or not execute:
			for image_path in file_paths:
				image_conv(image_path, target_subdir, execute=execute)
		else:
			with Pool(processes=processes) as pool:
				pool.starmap(image_conv, [(file_path, target_subdir, execute) for file_path in file_paths])

	log.write("HDF Convert",
			f"HDF Convert {'complete' if execute else 'planned'}: {datetime.now() - startTime}",
			log_level=LOG.STATUS)


if __name__ == '__main__':
	hdf_convert()
