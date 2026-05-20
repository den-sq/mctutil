from multiprocessing import Pool
from pathlib import Path
import sys

import click
import numpy as np
import psutil
import tifffile as tf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402
from shared.io_helpers import byteread_helper 	# noqa::E402
from shared.mem import SharedNP, ReconOrder 	# noqa::E402


def get_details(path, stack_levels):
	flist = list(Path(path).iterdir())
	with tf.TiffFile(flist[0]) as tif:
		page = tif.pages[0]
		return page.dtype, (ReconOrder(len(flist), stack_levels, page.shape[1])), page.dataoffsets[0]


def transpose_write(recon_mem: SharedNP, path, i):
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
