from collections import namedtuple
from enum import Enum
from os import PathLike
from pathlib import Path
from multiprocessing import Pool, shared_memory
import sys
from typing import Mapping, Callable
import uuid

import click
import natsort
import numpy as np
from numpy.typing import ArrayLike
from psutil import cpu_count
from skimage.restoration import denoise_nl_means, estimate_sigma
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402
from shared import cli 	# noqa::E402
from shared.cli import FRANGE 	# noqa::E402
from shared.mem import SharedNP, ProjOrder, SinoOrder 	# noqa::E402

FlatPair = namedtuple("FlatPair", ['Index', 'Offset'])


class FLAT(Enum):
	PREGAIN = FlatPair(0, -1)
	POSTGAIN = FlatPair(1, 1)
	PREDARK = FlatPair(2, -2)
	POSTDARK = FlatPair(3, 2)

	def __str__(self):
		return str(self.name.lower())

	def opp(self):
		tens = self._value_.Index // 2
		ones = self._value_.Index % 2
		return FLAT(2 * tens + 1 - ones)

	def __getitem__(self):
		return self._value_.Index

	@property
	def index(self):
		return self._value_.Index

	@property
	def offset(self):
		return self._value_.Offset


# Normalziation cacluation using weighted pre/post.
def weighted_normalize(sino_mem: SharedNP, input_mem: SharedNP, flats_mem: SharedNP, window, int_window,
						projection: int, projection_count: int, debug_folder: Path = None):
	"""Normalizes single projection using gain and dark flats, weighing pre and post darks based on projection #.

		:param sino_mem: Shared memory metadata for sinogram.
		:param input_mem: Shared memory holding read input.
		:param flats_mem: Shared memory metadata for flats (gains, darks).
		:param work_mem: Shared memory metadata for working memory.
		:param window: Working window of full sinogram.
		:param int_window: Matching window of shared memory sinogram.
		:param projection: Projection # to normalize.
		:param projection_count: Total # of projections.
	"""

	with sino_mem[int_window] as sino, flats_mem as flats, input_mem as source:
		dark_map = np.average(flats[FLAT.PREDARK.index:FLAT.POSTDARK.index + 1, window, :], axis=0,
								weights=[projection_count - projection, projection])
		gain_map = np.subtract(np.average(flats[FLAT.PREGAIN.index:FLAT.POSTGAIN.index + 1, window, :], axis=0,
								weights=[projection_count - projection, projection]), dark_map)
		temp = np.subtract(source[projection, int_window, :].astype(sino_mem.dtype), dark_map)

		sino[:, projection, :] = np.divide(temp, gain_map)


def memmap_helper(target, image, i_dtype, offsets, size):
	""" Sinogram order-capable reader using direct buffer reading.

		:param target: Shared memory to read into.
		:param image: Path to image to read from.
		:param i_dtype: Data type of image.
		:param offsets: Array of memory offsets to read into.  Should be sequential in file.
		:param size: Size of chunk of memory to read.
	"""
	sm = shared_memory.SharedMemory(name=target)
	shape = size // np.dtype(i_dtype).itemsize

	def set_array(offset):
		target_array = np.ndarray(shape, dtype=i_dtype, buffer=sm.buf[offset["target"]:offset["target"] + size])
		target_array[:] = np.memmap(image, dtype=i_dtype, mode="r+", offset=offset["source"], shape=shape, order='C')

	map(set_array, offsets)
	sm.close()


def byteread_helper(target: SharedNP, image: PathLike, i_dtype: np.dtype, offsets: ArrayLike, size: int):
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


def distribute_read(target_mem: SharedNP, pj: Mapping, window, int_window,
					image_order: ArrayLike, thread_max: int = cpu_count(),
					read_func: Callable = byteread_helper, sino_order: bool = True):
	""" Distributes file reading across multiple threads.

		Currently can work in projection or sinogram order, maybe?

		:param target_mem: Memory to read files into.
		:param mem_shape: Shape of memory we are reading into.
		:param pj: Information about tiff file structure grabbed from first.
		:param window: Vertical portion of images to fetch.
		:param int_window: Internal memory range matching window.
		:param int_offset: Internal memory offset before start of internal window.
		:param image_order: Order of images to read; may not be directly followed due to starmapping.
		:param thread_max: Maximum # of threads to use.
	"""
	# Steps for memory space jumps.
	h_step = pj["x"] * pj["bytesize"]

	# Size of Sinogram Block
	sino_block_size = target_mem.shape.Theta * h_step

	# Size of Proj block
	proj_block_size = len(int_window) * h_step

	# Find the offset values for start of blocks.
	# This is hilariouslyy stupid and needs a rewrite
	base_offset = target_mem[int_window].buffer_address.start

	def generate_offset_pairs_sino(i):
		return [{"source": pj["offset"] + (window.start + j) * h_step,
				"target": int(base_offset + j * sino_block_size + i * h_step)}
					for j in range(len(int_window))]

	def generate_offset_pairs_proj(i):
		return [{"source": pj["offset"] + (window.start) * h_step, "target": int(base_offset + i * proj_block_size)}]

	if sino_order:
		log.log("Files Into Memory", f"Writing (in {target_mem.name} | {target_mem.shape}) {base_offset}"
			+ f" to {base_offset + len(int_window) * sino_block_size}", log_level=log.DEBUG.INFO)
		pairs_func = generate_offset_pairs_sino
		size = h_step
	else:
		log.log("Files Into Memory", f"Writing (in {target_mem.name} | {target_mem.shape}) {base_offset}"
			+ f" to {base_offset + len(int_window) * proj_block_size} out of {target_mem[int_window].buffer_address}",
			log_level=log.DEBUG.INFO)
		pairs_func = generate_offset_pairs_proj
		size = proj_block_size

	# Load initial data.
	with Pool(thread_max) as pool:
		pool.starmap(read_func,
			[(target_mem.name, image, pj["dtype"], pairs_func(i), size) for i, image in image_order])


def sino_write(sino_mem: SharedNP, path: PathLike, i, out_type: cli.NumpyCLI = None):
	""" Writes a portion of sinogram to disk.

		:param sino_mem: Metadata for reconstruction shared memory.
		:param path: Path on disk to write to.
		:param i: Vertical slice(s) (y) of reconstruction to write.
	"""
	with sino_mem as sino:
		if out_type is None:
			tf.imwrite(path, sino[i, :, :])
		else:
			tf.imwrite(path, out_type.convert_ar(sino[i, :, :]))


def image_bounds(sino_mem: SharedNP):
	with sino_mem as sino:
		np.array([np.min(sino), np.max(sino)])


def minmaxscale(sino_mem, i, minval=None, maxval=None):
	with sino_mem[i] as sino:
		if minval is None:
			minval = np.min(sino)
		if maxval is None:
			maxval = np.max(sino)
		sino[:, :] = (sino - minval) / (maxval - minval)


def remove_outlier(sino_mem, i):
	with sino_mem[i] as sino:
		a_sigma_est = estimate_sigma(sino, channel_axis=None, average_sigmas=True)
		sino[:, :] = denoise_nl_means(sino, patch_size=9, patch_distance=5,
								fast_mode=True, sigma=0.001 * a_sigma_est,
								preserve_range=False, channel_axis=None)


def preprocess(sino_mem, i, minval=None, maxval=None):
	minmaxscale(sino_mem, i, minval, maxval)
	remove_outlier(sino_mem, i)


def sh_imread(sino_mem, i, path):
	with sino_mem[i] as sino:
		sino[:, :] = tf.imread(path)


@click.command()
@click.option("-i", "--input-dir", type=click.Path(path_type=Path, file_okay=False), required=True,
				help="Directory of Input Projections.")
@click.option("-o", "--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True,
				help="Directory of Output Sinograms.")
@click.option("-p", "--process-count", type=click.INT, default=cpu_count(),
				help="# of simulatenous processes during conversion.  Also used as window size.")
@click.option("--min-val", type=click.FLOAT, default=None, help="Minimum Value of Sinogram Set")
@click.option("--max-val", type=click.FLOAT, default=None, help="Maximum Value of Sinogram Set")
def sino_convert(input_dir: Path, output_dir: Path, process_count: int, min_val: int, max_val: int):
	image_paths = natsort.natsorted(list(input_dir.glob("**/*.tif*")))
	output_dir.mkdir(parents=True, exist_ok=True)
	output_paths = [output_dir.joinpath(x.name) for x in image_paths]

	segment_id = str(uuid.uuid4())
	internal_dtype = np.float32

	with tf.TiffFile(image_paths[0]) as tif:
		page = tif.pages[0]
		pj = {"dtype": page.dtype, "bytesize": page.dtype.itemsize, "offset": page.dataoffsets[0],
				"x": page.shape[1], "y": page.shape[0]}

	sino_shape = SinoOrder(process_count, pj["y"], pj["x"])
	bounds = []

	log.log("Setup", f"{pj}")

	if min_val is None or max_val is None:
		with (SharedNP(f"sino_{segment_id}", internal_dtype, sino_shape, create=True) as sino_mem):
			for x in range(0, len(image_paths), process_count):
				window = range(x, min(x + process_count, len(image_paths)))
				internal_window = range(0, len(window))

				log.log("Preprocess Cycle Start", f"Window {window}; Shape {sino_shape}")

				with Pool(process_count) as pool:
					pool.starmap(sh_imread, [(sino_mem, i, image_paths[window[i]]) for i in internal_window])

				log.log("Files Read", f"Window {window}; Shape {sino_shape}")

				with Pool(process_count) as pool:
					bounds += pool.map(image_bounds, sino_mem)

				log.log("Bounds Calculated", f"{window}", log.DEBUG.TIME)

		min_val = np.min(bounds[:, 0])
		max_val = np.max(bounds[:, 1])
		log.log("Final Bounds Calculated", f"{min_val} : {max_val}", log.DEBUG.TIME)

	with (SharedNP(f"sino_{segment_id}", internal_dtype, sino_shape, create=True) as sino_mem):
		for x in range(0, len(image_paths), process_count):
			window = range(x, min(x + process_count, len(image_paths)))
			internal_window = range(0, len(window))

			log.log("Preprocess Cycle Start", f"Window {window}; Shape {sino_shape}")

			with Pool(process_count) as pool:
				pool.starmap(sh_imread, [(sino_mem, i, image_paths[window[i]]) for i in internal_window])

			log.log("Files Read", f"Window {window}; Shape {sino_shape}")

			with Pool(process_count) as pool:
				pool.starmap(preprocess, [(sino_mem, i, min_val, max_val) for i in internal_window])

			log.log("Sinogram Preprocessing", f"{min_val} : {max_val}", log.DEBUG.TIME)

			with Pool(process_count) as pool:
				pool.starmap(sino_write, [(sino_mem, output_paths[i + window.start], i) for i in internal_window])

			log.log("Files Written", f"{output_dir} : {window}", log.DEBUG.TIME)


if __name__ == "__main__":
	sino_convert()
