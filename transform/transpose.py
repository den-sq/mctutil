from multiprocessing import Pool, shared_memory
from pathlib import Path
import sys

import click
import numpy as np
import psutil
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402
from shared.mem import SharedNP, ReconOrder 	# noqa::E402


def get_details(path, stack_levels):
	flist = list(Path(path).iterdir())
	with tf.TiffFile(flist[0]) as tif:
		page = tif.pages[0]
		return page.dtype, (ReconOrder(len(flist), stack_levels, page.shape[1])), page.dataoffsets[0]


def byteread_helper(target, image, i_dtype, offsets, size):
	""" Sinogram order-capable reader using direct buffer reading.

		TODO: Figure out if you can get rid of for loop - order matters.

		:param target: Shared memory to read into.
		:param image: Path to image to read from.
		:param i_dtype: Data type of image.  Unused here, since reading is direct buffer.
		:param offsets: Array of memory offsets to read into.  Should be sequential in file.
		:param size: Size of chunk of memory to read.
	"""
	sm = shared_memory.SharedMemory(name=target)
	with open(image, "rb") as handle:
		handle.seek(offsets[0]["source"])
		for offset in offsets:
			handle.readinto(sm.buf[offset["target"]:offset["target"] + size])
	sm.close()


def transpose_write(recon_mem: SharedNP, path, i):
	""" Writes a portion of reconstruction to disk after transposing it.

		:param recon_mem: Metadata for reconstruction shared memory.
		:param path: Path on disk to write to.
		:param i: Vertical slice(s) (y) of reconstruction to write.
	"""
	with recon_mem as recon:
		view = np.transpose(recon, [1, 2, 0])
		tf.imwrite(path, view[i, :, :])


@click.command()
@click.option("-p", "--path", type=click.Path(), help="Reconstruction Path", required=True)
@click.option("-s", "--stack-start", type=click.INT, help="Y value to start reading from images.", required=True)
@click.option("-l", "--stack-levels", type=click.INT, help="Vertical Stacks to use for transpose.", required=True)
@click.option("-x", "--pixel-shift", type=click.FLOAT, default=0.0,
				help="Vertical shift per pixel to track angular movement")
@click.option("-n", "--out-name", type=click.STRING, help="Name Prefix for Files", required=True)
@click.argument("out-path", required=True)
def transpose_stack(path, stack_start, stack_levels, pixel_shift, out_name, out_path):
	recon_dtype, recon_shape, base_offset = get_details(path, stack_levels)
	im_list = sorted(list(Path(path).iterdir()))
	log.log("Setup", f"Shape {recon_shape}; Type {recon_dtype}; offset {base_offset}")
	with SharedNP("Tranpose_Source", recon_dtype, recon_shape, create=True) as tp_mem:
		itemsize = np.dtype(recon_dtype).itemsize
		source_offset = base_offset + recon_shape.X * itemsize * stack_start

		# Find the offset values for start of blocks.
		# This is hilariouslyy stupid and needs a rewrite
		target_offset = tp_mem[0].buffer_address.start
		line_size = recon_shape.X * itemsize
		chunk_size = line_size * recon_shape.Z

		log.log("Setup", f"Itemsize {itemsize}; Offset {source_offset}; Line Size {line_size}")

		with Pool(psutil.cpu_count()) as pool:
			def get_offset(i):
				return [{"source": source_offset + int(i * pixel_shift) * line_size,
							"target": target_offset + chunk_size * i}]

			pool.starmap(byteread_helper, [(tp_mem.name, im_list[i], recon_dtype, get_offset(i), chunk_size)
											for i in range(len(im_list))])

		log.log("Images Loaded")

		Path(out_path).mkdir(parents=True, exist_ok=True)

		with Pool(psutil.cpu_count()) as pool:
			pool.starmap(transpose_write, [(tp_mem, Path(out_path, f"{out_name}_{i}.tif"), i)
											for i in range(recon_shape.Z)])

		log.log("Images Written")


if __name__ == "__main__":
	transpose_stack()
