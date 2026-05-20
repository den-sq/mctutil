from multiprocessing import Pool
from os import PathLike
from pathlib import Path
import sys

import click
import numpy as np
import psutil
import tifffile as tf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402
from shared.cli import FRANGE 	# noqa::E402
from shared.mem import SharedNP, ProjOrder 	# noqa::E402
from shared.np_convert import np_convert 	# noqa::E402


def norm_helper(image_mem, i, floor, ceiling):
	with image_mem[i] as image:
		image[image > ceiling] = ceiling
		image[image < floor] = floor

		image[:, :] -= floor
		image[:, :] /= (ceiling - floor)


def normalize(image_mem, index, bottom_threshold, top_threshold, thread_max):
	"""Straightforward image normalization, disposing of values at edges."""
	with image_mem[index] as image:
		floor = np.percentile(image, bottom_threshold)
		ceiling = np.percentile(image, top_threshold)

		log.log('Normalization',
			f"{np.min(image)}-{np.max(image)}: {bottom_threshold}-{top_threshold} is {floor:.4g}-{ceiling:.4g}",
			log_level=log.DEBUG.INFO)

		with Pool(thread_max) as pool:
			pool.starmap(norm_helper, [(image_mem, i, floor, ceiling) for i in index])
		log.log('Normalization',
				f"{bottom_threshold} to {top_threshold}: {floor:.4g} to {ceiling:.4g} {(ceiling - floor):.4g}",
				log_level=log.DEBUG.INFO)


def convert(source_mem, target_mem, i, j):
	with source_mem[i] as source, target_mem[j] as target:
		target[:] = source.astype(source_mem.dtype)


def batch(iterable, n=1):
	length = len(iterable)
	for ndx in range(0, length, n):
		yield iterable[ndx:min(ndx + n, length)]


def memreader(mem, i, path):
	with mem as mem_array:
		mem_array[i] = tf.imread(path)


def mem_write(mem: SharedNP, path: PathLike, i, dtype, execute=True):
	"""Writes to disk in distributed fashion."""
	with mem[i] as out_data:
		if execute:
			tf.imwrite(path, np_convert(dtype, out_data), dtype=dtype)
			log.log("File Written", path.name)
		else:
			log.log("Dry Run", f"Would write {path.name}")


@click.command()
@click.option('-n', '--normalize-over', type=FRANGE, help="Range of retained values to normalize over, by percentiles.")
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for noisy dataset')
@click.option('-o', '--output-dir', type=click.Path(),
				help='Output path for cleaned images', default='data/clean/')
@click.option('-p', '--processes', type=click.INT, default=psutil.cpu_count(),
				help='Process Count (for simulatenous images)')
@click.option('--execute/--dry-run', default=True,
				help='Whether to write normalized files or only log the planned outputs.')
def norm(normalize_over, data_dir, output_dir, processes, execute):
	log.start()

	if execute:
		Path(output_dir).mkdir(parents=True, exist_ok=True)
	inputs = sorted([x for x in Path(data_dir).iterdir() if ".tif" in x.name])
	batched_input = list(batch(inputs, processes))

	log.log("Initialize", "Inputs Batched")

	with tf.TiffFile(inputs[0]) as tif:
		mem_shape = ProjOrder(processes, tif.pages[0].shape[0], tif.pages[0].shape[1])
		dtype = tif.pages[0].dtype

	log.log("Initialize", "Tiff Dimensions Fetched")

	with SharedNP('Normalize_Mem', np.float32, mem_shape, create=True) as norm_mem:
		for input_set in batched_input:
			active_indices = list(range(len(input_set)))
			with Pool(processes) as pool:
				pool.starmap(memreader, [(norm_mem, i, input_set[i]) for i in active_indices])
			log.log("Image Load", f"{len(active_indices)} Images Loaded")
			normalize(norm_mem, active_indices, normalize_over.start, normalize_over.stop, processes)
			with Pool(processes) as pool:
				pool.starmap(
					mem_write,
					[(norm_mem, Path(output_dir, input_set[i].name), i, dtype, execute) for i in active_indices],
				)
			log.log("Image Writing", f"{len(active_indices)} Images {'Written' if execute else 'Planned'}")


if __name__ == '__main__':
	norm()
