from multiprocessing import Pool
from os import PathLike
from pathlib import Path

import click
import numpy as np
import psutil
import tifffile as tf


from mctutil.shared.log import log, LOG
from mctutil.shared.cli import FRANGE
from mctutil.shared.mem import SharedNP, ProjOrder
from mctutil.shared.np_convert import np_convert
from mctutil.shared.stack_apply import apply_array, batched, run_parallel, tiff_paths
from mctutil.shared.tiff_stack_writer import write_tiff_stack


def normalized_image(image, floor, ceiling):
	"""Return one normalized image without mutating its input view."""
	result = np.array(image, dtype=np.float32, copy=True)
	result[result > ceiling] = ceiling
	result[result < floor] = floor
	result -= floor
	result /= (ceiling - floor)
	return result


def normalization_bounds(image, bottom_threshold, top_threshold):
	"""Return the percentile bounds used by normalization commands."""
	floor, ceiling = np.percentile(
		np.asarray(image),
		(bottom_threshold, top_threshold),
	)
	return float(floor), float(ceiling)


def norm_helper(image_mem, i, floor, ceiling):
	with image_mem[i] as image:
		image[:] = apply_array(
			image,
			normalized_image,
			(floor, ceiling),
		)


def normalize(image_mem, index, bottom_threshold, top_threshold, thread_max):
	"""Straightforward image normalization, disposing of values at edges."""
	with image_mem[index] as image:
		floor, ceiling = normalization_bounds(
			image,
			bottom_threshold,
			top_threshold,
		)

		log.write('Normalization',
			f"{np.min(image)}-{np.max(image)}: {bottom_threshold}-{top_threshold} is {floor:.4g}-{ceiling:.4g}",
			log_level=LOG.INFO)

		run_parallel(
			norm_helper,
			((image_mem, i, floor, ceiling) for i in index),
			thread_max,
			pool_factory=Pool,
		)
		log.write('Normalization',
				f"{bottom_threshold} to {top_threshold}: {floor:.4g} to {ceiling:.4g} {(ceiling - floor):.4g}",
				log_level=LOG.INFO)


def convert(source_mem, target_mem, i, j):
	with source_mem[i] as source, target_mem[j] as target:
		target[:] = source.astype(source_mem.dtype)


def memreader(mem, i, path):
	with mem as mem_array:
		mem_array[i] = tf.imread(path)


def mem_write(mem: SharedNP, path: PathLike, i, dtype, execute=True):
	"""Writes to disk in distributed fashion."""
	with mem[i] as out_data:
		write_tiff_stack(
			lambda _index: np_convert(dtype, out_data),
			1,
			path,
			mode="image",
			dry_run=not execute,
		)
		if execute:
			log.write("File Written", path.name)
		else:
			log.write("Dry Run", f"Would write {path.name}")


@click.command()
@click.option('-n', '--normalize-over', type=FRANGE, help="Range of retained values to normalize over, by percentiles.")
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for noisy dataset')
@click.option('-o', '--output-dir', type=click.Path(),
				help='Output path for cleaned images', default='data/clean/')
@click.option('-p', '--processes', type=click.INT, default=psutil.cpu_count(),
				help='Process Count (for simulatenous images)')
@click.option("--hard-cut/--relative-cut", type=bool, default=False,
				help="Whether to use hard or relative values for normalizing.")
@click.option('--execute/--dry-run', default=True,
				help='Whether to write normalized files or only log the planned outputs.')
def norm(normalize_over, data_dir, output_dir, processes, hard_cut, execute):
	log.start()

	if execute:
		Path(output_dir).mkdir(parents=True, exist_ok=True)
	inputs = tiff_paths(data_dir)
	batched_input = batched(inputs, processes)

	log.write("Initialize", "Inputs Batched")

	with tf.TiffFile(inputs[0]) as tif:
		mem_shape = ProjOrder(processes, tif.pages[0].shape[0], tif.pages[0].shape[1])
		dtype = tif.pages[0].dtype

	log.write("Initialize", "Tiff Dimensions Fetched")

	with SharedNP('Normalize_Mem', np.float32, mem_shape, create=True) as norm_mem:
		for input_set in batched_input:
			active_indices = list(range(len(input_set)))
			run_parallel(
				memreader,
				((norm_mem, i, input_set[i]) for i in active_indices),
				processes,
				pool_factory=Pool,
			)
			log.write("Image Load", f"{len(active_indices)} Images Loaded")
			normalize(norm_mem, active_indices, normalize_over.start, normalize_over.stop, processes)
			run_parallel(
				mem_write,
				(
					(
						norm_mem,
						Path(output_dir, input_set[i].name),
						i,
						dtype,
						execute,
					)
					for i in active_indices
				),
				processes,
				pool_factory=Pool,
			)
			log.write("Image Writing", f"{len(active_indices)} Images {'Written' if execute else 'Planned'}")


if __name__ == '__main__':
	norm()
