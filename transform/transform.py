from multiprocessing import Pool
from os import PathLike
from pathlib import Path
import sys

import click
import numpy as np
import psutil
import tifffile as tf
from tomopy import circ_mask

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402
from shared import cli 	# noqa::E402
from shared.cli import FRANGE 	# noqa::E402
from shared.mem import SharedNP, ProjOrder 	# noqa::E402


def normalize(image_mem, part, floor, ceiling):
	with image_mem[part] as image:
		image[image > ceiling] = ceiling
		image[image < floor] = floor

		image[:, :] -= floor
		image[:, :] /= (ceiling - floor)


def normalize_calc(image_mem, index, bottom_threshold, top_threshold, slice):
	with image_mem[index] as image:
		im_part = (np.s_[:],) + slice
		log.log("Normalize Setup", f"{image.shape}:{im_part}")
		floor = np.percentile(image[im_part], bottom_threshold)
		ceiling = np.percentile(image[im_part], top_threshold)
		log.log('Normalization Setup',
			f"{np.min(image)}-{np.max(image)}: {bottom_threshold}-{top_threshold} is {floor:.4g}-{ceiling:.4g}",
			log_level=log.DEBUG.INFO)

	return floor, ceiling


def convert(source_mem, target_mem, i, j):
	with source_mem[i] as source, target_mem[j] as target:
		target[:] = source.astype(source_mem.dtype)


def batch(iterable, n=1, extra=0):
	length = len(iterable)
	for ndx in range(extra, length, n):
		yield iterable[ndx - extra:min(ndx + n, length)]


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
		if np.issubdtype(dtype, np.integer):
			tf.imwrite(path, np.multiply(out_data[slice], np.iinfo(dtype).max).astype(dtype), dtype=dtype)
		else:
			tf.imwrite(path, out_data[slice].astype(dtype), dtype=dtype)


@click.command()
@click.option('-n', '--normalize-over', type=FRANGE, help="Range of retained values to normalize over, by percentiles.")
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for noisy dataset')
@click.option('-o', '--output-dir', type=click.Path(), help='Output path for cleaned images', default='data/clean/')
@click.option('-v', '--vertical-trim', type=click.STRING, default=0.0,
				help='Vertical trim as a percent (0.0-1.0).  One # defines both, two is top/bottom.')
@click.option('-h', '--horizontal-trim', type=click.STRING, default=0.0,
				help='Horizontal trim as a percent (0.0-1.0).  One # defines both, two is left/right.')
@click.option('-t', "--out-dtype", type=cli.NUMPYTYPE, default=np.uint8, help="Datatype of Output.")
@click.option('-p', '--processes', type=click.INT, default=psutil.cpu_count(),
				help='Process Count (for simulatenous images)')
@click.option('-m', '--mips', type=click.INT, help='# of projections to take maximum intensity from.', default=0)
@click.option("-c", "--circ-mask-ratio", type=click.FLOAT, default=None,
				help="Circular Mask Ratio, if wanted.")
def norm(normalize_over, data_dir, output_dir, vertical_trim, horizontal_trim, out_dtype, processes, mips,
			circ_mask_ratio):
	log.start()

	try:
		top_trim = bottom_trim = float(vertical_trim)
	except ValueError:
		top_trim, bottom_trim = [float(x) for x in vertical_trim.split(",")]

	try:
		left_trim = right_trim = float(horizontal_trim)
	except ValueError:
		left_trim, right_trim = [float(x) for x in horizontal_trim.split(",")]

	mips_offset = max(mips, 1) - 1
	indices = list(range(0, mips_offset + processes))

	Path(output_dir).mkdir(parents=True, exist_ok=True)
	inputs = sorted([x for x in Path(data_dir).iterdir() if ".tif" in x.name])

	# Because samples vary over rage, we want to normalize across the range.
	# Interval is calculated so size of normalize dataset is one batch.
	normalize_interval = int(np.ceil(len(inputs) / processes))

	batched_input = list(batch(inputs, processes, mips_offset))

	log.log("Initialize", "Inputs Batched")

	with tf.TiffFile(inputs[0]) as tif:
		mem_shape = ProjOrder(processes + mips_offset, tif.pages[0].shape[0], tif.pages[0].shape[1])
		dim = tif.pages[0].shape
		out_dim = np.s_[int(dim[0] * top_trim):int(dim[0] * (1 - bottom_trim)),
						int(dim[1] * left_trim):int(dim[1] * (1 - right_trim))]
		log.log("Dimensions", f"{dim}-{out_dim}")

	log.log("Initialize", "Tiff Dimensions Fetched")

	if circ_mask_ratio:
		actual_start = normalize_over.start + 100 * (1.0 - np.pi * ((circ_mask_ratio / 2) ** 2))

	with SharedNP('Normalize_Mem', np.float32, mem_shape, create=True) as norm_mem:
		# Normalization calculation
		with Pool(processes) as pool:
			image_count = len(inputs) // normalize_interval
			log.log("Image Load", f"{len(inputs)}:{normalize_interval}:{processes}:{len(inputs) // normalize_interval}")
			pool.starmap(memreader, [(norm_mem, i, inputs[j]) for i, j in enumerate(range(0, len(inputs), image_count))])
			log.log("Image Load",
					f"{len(range(0, len(inputs), len(inputs) % normalize_interval))}|{normalize_interval}"
					" Images Loaded For Normalization")

			if circ_mask_ratio:
				with norm_mem as mips_mem:
					circ_mask(mips_mem, axis=0, ratio=circ_mask_ratio, val=np.min(mips_mem[0:image_count]))
				log.log("Image Masking", f"Images Masked at {circ_mask_ratio}")

			floor, ceiling = normalize_calc(norm_mem, np.s_[0:image_count], actual_start, normalize_over.stop, np.s_[:, :])

		for input_set in batched_input:
			indices = [i for i in indices if i < len(input_set)]

			with Pool(processes) as pool:
				pool.starmap(memreader, [(norm_mem, i, input_set[i]) for i in indices])
			log.log("Image Load", f"{len(indices)}|{len(input_set)} Images Loaded")

			if circ_mask_ratio:
				with norm_mem as mips_mem:
					circ_mask(mips_mem, axis=0, ratio=circ_mask_ratio, val=floor)
				log.log("Image Masking", f"Images Masked at {circ_mask_ratio}")

			if mips > 1:
				with norm_mem as mips_mem:
					for i in reversed(indices[mips_offset:]):
						# log.log("Making MIP", f"{i}:{i - mips_offset}:{mips_mem.shape}:{np.min(mips_mem[i])}:{np.max(mips_mem[i])}",
						# 			log_level=log.DEBUG.INFO)
						mips_mem[i] = np.max(mips_mem[i - mips_offset: i + 1], axis=0)
						# log.log("Making MIP", f"{i}:{i - mips_offset}:{mips_mem.shape}:{np.min(mips_mem[i])}:{np.max(mips_mem[i])}",
						# 			log_level=log.DEBUG.INFO)

			with Pool(processes) as pool:
				pool.starmap(normalize, [(norm_mem, i, floor, ceiling) for i in indices[mips_offset:]])

			log.log("Image Normalization",
					f"{len(indices[mips_offset:])} Images Normalized at {floor:.4g} to {ceiling:.4g} over {(ceiling - floor):.4g}")

			with Pool(processes) as pool:
				pool.starmap(mem_write, [(norm_mem, Path(output_dir, input_set[i].name), i, out_dim, out_dtype.nptype)
								for i in indices[mips_offset:]])

			log.log("Image Writing", f"{len([i for i in indices[mips_offset:]])} Images Written")
	log.log("Complete")


if __name__ == '__main__':
	norm()
