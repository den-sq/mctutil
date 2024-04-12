from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import click
from osgeo import gdal
import tifffile

startTime = datetime.now()
gdal.UseExceptions() 	# Throws many warnings if we don't set whether we want exceptions.


def get_image_paths(folder: Path):
	""" Finds all HDF files in raw subdirectory of folder.

		:param folder: Base Path containing raw folder.
		:return: Sorted list of HDF files.
	"""
	return sorted(folder.glob('raw\\*.hdf'))


def image_conv(image_path: Path, target_dir: Path):
	""" Converts hdf5 file to a tiff file and writes the result in a new location.

		:param image_path: Full path to an image.
		:param target_dir: Folder to write tiff to, retaining filename (except suffix).
	"""
	with open(target_dir.parent.joinpath(f"{target_dir.name}.log"), "a") as log:
		target_path = target_dir.joinpath(image_path.with_suffix(".tiff").name)

		src_ds = gdal.Open(str(image_path))
		print(f"{image_path} Read")
		log.write(f"{image_path} Read\n")

		out_ds = gdal.Translate('/vsimem/in_memory_output.tif', src_ds, format='GTiff', bandList=[1])
		out_arr = out_ds.ReadAsArray()

		tifffile.imwrite(target_path, out_arr)
		print(f"{target_path} Written")
		log.write(f"{target_path} Written\n")


@click.command()
@click.option("--target-dir", "-t", default=Path("."), type=click.Path(path_type=Path),
				help="Target directory to write to, defaulting to current working directory.")
@click.option("--processes", "-p", default=60, type=click.INT,
				help="Number of simulatenous processes to use for reading, default 60.  If 1, it will not use multiprocessing.")
@click.argument("input", nargs=-1,
				type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path))
def hdf_convert(target_dir, processes, input):
	""" Converts files from hdf format to tiff files, usually for hdf4.

		Files are searched for in INPUT/raw/;  any number of INPUT directories are allowed."""
	for proj_dir in input:
		file_paths = get_image_paths(proj_dir)
		total = len(file_paths)

		if total == 0:
			print(f"{proj_dir}: No images found, skipping.")
			continue
		else:
			target_subdir = target_dir.joinpath("tiff_sets", proj_dir.parent.name, proj_dir.name)
			print(f"{proj_dir}: {total} images found; writing to {target_subdir}")
			target_subdir.mkdir(parents=True, exist_ok=True)

		with open(target_subdir.parent.joinpath(f"{target_subdir.name}.log"), "a") as log:
			log.write(f"{proj_dir}: {total} images found; writing to {target_subdir}\n")

		if processes == 1:
			for image_path in file_paths:
				image_conv(image_path, target_dir)
		else:
			with Pool(processes=processes) as pool:
				pool.starmap(image_conv, [(file_path, target_subdir) for file_path in file_paths])

	print(f"HDF Convert Complete: {datetime.now() - startTime}")


if __name__ == '__main__':
	hdf_convert()
