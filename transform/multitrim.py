from multiprocessing import Pool
from os import PathLike
import sys
from pathlib import Path  # if you haven't already done so

import click
import numpy as np
import tifffile as tf
import psutil

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import cli 	# noqa::E402
from shared import log 	# noqa::E402
from shared.mem import SharedNP, ProjOrder 	# noqa::E402


def batch(iterable, n=1):
	length = len(iterable)
	for ndx in range(0, length, n):
		yield iterable[ndx:min(ndx + n, length)]


def memreader(mem, i, path):
	with mem as mem_array:
		mem_array[i] = tf.imread(path)


def mem_write(mem: SharedNP, path: PathLike, i, slice, dtype):
	""" Writes to disk in distributed fashion

	:param mem: Metadata for reconstruction shared memory.
	:param path: Path on disk to write to.
	:param i: Vertical slice(s) (y) of reconstruction to write.
	"""
	with mem[i] as out_data:
		tf.imwrite(path, out_data.astype(dtype), dtype=dtype)


@click.command
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for original dataset.', required=True)
@click.option('-o', '--output-dir', type=click.Path(), help='Output path for transformed dataset.', required=True)
@click.option('-v', '--vertical-trim', type=click.FLOAT, default=0.0,
				help='Vertical trim (top and bottom) as a percent')
@click.option('-h', '--horizontal-trim', type=click.FLOAT, default=0.0,
				help='Horizontal trim (top and bottom) as a percent')
@click.option('-t', "--out-dtype", type=cli.NUMPYTYPE, default=np.uint8, help="Datatype of Output.")
def trim(data_dir, output_dir, vertical_trim, horizontal_trim, out_dtype):
	log.start()
	processes = psutil.cpu_count()
	indices = list(range(processes))
	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	inputs = [x for x in Path(data_dir).iterdir() if ".tif" in x.name]
	batched_input = list(batch(inputs, processes))

	with tf.TiffFile(inputs[0]) as tif:
		mem_shape = ProjOrder(processes, tif.pages[0].shape[0], tif.pages[0].shape[1])
		dim = tif.pages[0].shape
		out_dim = np.s_[int(dim[0] * horizontal_trim):int(dim[0] * (1 - horizontal_trim)),
						int(dim[1] * vertical_trim):int(dim[1] * (1 - vertical_trim))]
		log.log("Dimensions", f"{dim}-{out_dim}")

	with SharedNP('Normalize_Mem', np.float32, mem_shape, create=True) as norm_mem:
		for input_set in batched_input:
			with Pool(processes) as pool:
				pool.starmap(memreader, [(norm_mem, i, input_set[i]) for i in indices])
			log.log("Image Load", f"{len(indices)} Images Loaded")
			with Pool(processes) as pool:
				pool.starmap(mem_write, [(norm_mem, Path(output_dir, input_set[i].name), i, out_dim, out_dtype.nptype)
											for i in indices])
			log.log("Image Writing", f"{len(indices)} Images Written")


if __name__ == "__main__":
	trim()
