from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
import psutil
import tifffile as tf


from mctutil.shared.log import log
from mctutil.shared.io_helpers import byteread_helper
from mctutil.shared.mem import SharedNP, ReconOrder


MODE = click.Choice(["shared", "naive"], case_sensitive=False)


def get_details(path, stack_levels):
	flist = list(Path(path).iterdir())
	with tf.TiffFile(flist[0]) as tif:
		page = tif.pages[0]
		return page.dtype, (ReconOrder(len(flist), stack_levels, page.shape[1])), page.dataoffsets[0]


def transpose_write(recon_mem: SharedNP, path, i):
	with recon_mem as recon:
		view = np.transpose(recon, [1, 2, 0])
		tf.imwrite(path, view[i, :, :])


def transpose_write_array(view, path, i):
	tf.imwrite(path, view[i, :, :])


def transpose_naive(path: Path, output_path: Path, out_name: str):
	im_list = sorted(list(path.iterdir()))
	with tf.TiffFile(im_list[0]) as im:
		old_shape = (len(im_list), im.pages[0].shape[0], im.pages[0].shape[1])
		old_dtype = im.pages[0].dtype

	full_data = np.empty(old_shape, dtype=old_dtype)
	for i, image in enumerate(im_list):
		full_data[i, :, :] = tf.imread(image)
	transposed_data = np.transpose(full_data, [2, 1, 0])

	output_path.mkdir(parents=True, exist_ok=True)
	for i in range(old_shape[2]):
		tf.imwrite(output_path.joinpath(f"{out_name}_{str(i).zfill(4)}.tif"), transposed_data[i])


def validate_shared_mode(mode: str, stack_start, stack_levels):
	if mode == "shared":
		if stack_start is None:
			raise click.UsageError("--stack-start is required when --mode=shared.")
		if stack_levels is None:
			raise click.UsageError("--stack-levels is required when --mode=shared.")


@click.command()
@click.option("--mode", type=MODE, default="shared", show_default=True)
@click.option("-p", "--path", type=click.Path(exists=True, path_type=Path), help="Reconstruction path", required=True)
@click.option("-s", "--stack-start", type=click.INT, help="Y value to start reading from images.", required=False)
@click.option("-l", "--stack-levels", type=click.INT, help="Vertical stacks to use for transpose.", required=False)
@click.option("-x", "--pixel-shift", type=click.FLOAT, default=0.0,
				help="Vertical shift per pixel to track angular movement")
@click.option("-n", "--out-name", type=click.STRING, help="Name prefix for files", default="tp", show_default=True)
@click.argument("out-path", required=True, type=click.Path(path_type=Path))
def transpose_stack(mode, path, stack_start, stack_levels, pixel_shift, out_name, out_path):
	mode = mode.lower()
	validate_shared_mode(mode, stack_start, stack_levels)
	if mode == "naive":
		transpose_naive(path, out_path, out_name)
		return

	recon_dtype, recon_shape, base_offset = get_details(path, stack_levels)
	im_list = sorted(list(Path(path).iterdir()))
	log.write("Setup", f"Shape {recon_shape}; Type {recon_dtype}; offset {base_offset}")
	with SharedNP("Tranpose_Source", recon_dtype, recon_shape, create=True) as tp_mem:
		itemsize = np.dtype(recon_dtype).itemsize
		source_offset = base_offset + recon_shape.X * itemsize * stack_start
		target_offset = tp_mem[0].buffer_address.start
		line_size = recon_shape.X * itemsize
		chunk_size = line_size * recon_shape.Z

		log.write("Setup", f"Itemsize {itemsize}; Offset {source_offset}; Line Size {line_size}")

		with Pool(psutil.cpu_count()) as pool:
			def get_offset(i):
				return [{"source": source_offset + int(i * pixel_shift) * line_size,
							"target": target_offset + chunk_size * i}]

			pool.starmap(byteread_helper, [(tp_mem.name, im_list[i], recon_dtype, get_offset(i), chunk_size)
										for i in range(len(im_list))])

		log.write("Images Loaded")
		Path(out_path).mkdir(parents=True, exist_ok=True)

		with Pool(psutil.cpu_count()) as pool:
			pool.starmap(transpose_write, [(tp_mem, Path(out_path, f"{out_name}_{i}.tif"), i)
										for i in range(recon_shape.Z)])

		log.write("Images Written")


if __name__ == "__main__":
	transpose_stack()
