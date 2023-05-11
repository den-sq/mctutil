from multiprocessing import Pool
from os import PathLike
from pathlib import Path
import sys

import click
import numpy as np
import psutil
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log
from shared.cli import FRANGE
from shared.mem import SharedNP, ProjOrder

def norm_helper(image_mem, i, floor, ceiling):
	with image_mem[i] as image:
		image[image > ceiling] = ceiling
		image[image < floor] = floor

		image[:, :] -= floor
		image[:, :] /= (ceiling - floor)


def normalize(image_mem, index, bottom_threshold, top_threshold, thread_max):
	""" Straightforward image normalization, disposing of values at edges """
	with image_mem[index] as image:
		floor = np.percentile(image, bottom_threshold)
		ceiling = np.percentile(image, top_threshold)
		
		log.log('Normalization',
			f"{np.min(image)}-{np.max(image)}: {bottom_threshold}-{top_threshold} is {floor:.4g}-{ceiling:.4g}",
			log_level=log.DEBUG.INFO)

		with Pool(thread_max) as pool:
			pool.starmap(norm_helper, [(image_mem, i, floor, ceiling) for i in index])
		log.log('Normalization', f"{bottom_threshold} to {top_threshold}: {floor:.4g} to {ceiling:.4g} {(ceiling - floor):.4g}",
				log_level=log.DEBUG.INFO)


def convert(source_mem, target_mem, i, j):
	with source_mem[i] as source, target_mem[j] as target:
		target[:] = source.astype(source_mem.dtype)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def memreader(mem, i, path):
	with mem as mem_array:
		mem_array[i] = tf.imread(path)


def mem_write(mem: SharedNP, path: PathLike, i):
	""" Writes to disk in distributed fashion

		:param mem: Metadata for reconstruction shared memory.
		:param path: Path on disk to write to.
		:param i: Vertical slice(s) (y) of reconstruction to write.
	"""
	with mem[i] as out_data:
		tf.imwrite(path, out_data)


@click.command()
@click.option('-n', '--normalize-over', type=FRANGE, help="Range of retained values to normalize over, by percentiles.")
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for noisy dataset')
@click.option('-o', '--output-dir', type=click.Path(), help='Output path for cleaned images', default='data/clean/')
@click.option('-p', '--processes', type=click.Path(), help='Process Count (for simulatenous images)', default=psutil.cpu_count())
def norm(normalize_over, data_dir, output_dir, processes):
	log.start()

	indices = list(range(processes))

	Path(output_dir).mkdir(parents=True, exist_ok=True)
	inputs = [x for x in Path(data_dir).iterdir() if ".tif" in x.name]
	batched_input = list(batch(inputs, processes))

	log.log("Initialize", "Inputs Batched")

	with tf.TiffFile(inputs[0]) as tif:
		mem_shape = ProjOrder(processes, tif.pages[0].shape[0], tif.pages[0].shape[1])
	 
	log.log("Initialize", "Tiff Dimensions Fetched")

	with SharedNP('Normalize_Mem', np.float32, mem_shape, create=True) as norm_mem:
		for input_set in batched_input:
			with Pool(processes) as pool:
				pool.starmap(memreader, [(norm_mem, i, input_set[i]) for i in indices])
			log.log("Image Load", f"{len(indices)} Images Loaded")
			normalize(norm_mem, indices, normalize_over.start, normalize_over.stop, processes)
			# log.log("Image Normalization", f"{len(indices)} Images Normalized")
			with Pool(processes) as pool:
				pool.starmap(mem_write, [(norm_mem, Path(output_dir, input_set[i].name), i) for i in indices])
			log.log("Image Writing", f"{len(indices)} Images Written")
		

if __name__ == '__main__':
	norm()