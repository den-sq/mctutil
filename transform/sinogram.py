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
def weighted_normalize(sino_mem: SharedNP, flats_mem: SharedNP, window, int_window,
						projection: int, projection_count: int):
	"""Normalizes single projection using gain and dark flats, weighing pre and post darks based on projection #.

		:param sino_mem: Shared memory metadata for sinogram.
		:param flats_mem: Shared memory metadata for flats (gains, darks).
		:param work_mem: Shared memory metadata for working memory.
		:param window: Working window of full sinogram.
		:param int_window: Matching window of shared memory sinogram.
		:param projection: Projection # to normalize.
		:param projection_count: Total # of projections.
	"""

	with sino_mem[int_window] as sino, flats_mem as flats:
		dark_map = np.average(flats[FLAT.PREDARK.index:FLAT.POSTDARK.index + 1, window, :], axis=0,
								weights=[projection_count - projection, projection])
		gain_map = np.subtract(np.average(flats[FLAT.PREGAIN.index:FLAT.POSTGAIN.index + 1, window, :], axis=0,
								weights=[projection_count - projection, projection]), dark_map)
		temp = np.subtract(sino[:, projection, :].astype(sino_mem.dtype), dark_map)

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


def distribute_read(target_mem: SharedNP, mem_shape, pj: Mapping, window, int_window,
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
	block_size = mem_shape.Theta * h_step

	# Find the offset values for start of blocks.
	# This is hilariouslyy stupid and needs a rewrite
	base_offset = target_mem[int_window].buffer_address.start

	def generate_offset_pairs_sino(i):
		return [{"source": pj["offset"] + (window.start + j) * h_step,
				"target": int(base_offset + j * block_size + i * h_step)}
					for j in range(len(int_window))]

	def generate_offset_pairs_proj(i):
		return [{"source": pj["offset"] + (window.start) * h_step, "target": int(base_offset + i * h_step)}]

	log("Files Into Memory", f"Writing (in {target_mem.name}) {base_offset}"
		+ f" to {base_offset + len(int_window) * block_size}", log_level=log.DEBUG.INFO)

	if sino_order:
		pairs_func = generate_offset_pairs_sino
		size = h_step
	else:
		pairs_func = generate_offset_pairs_proj
		size = h_step * len(int_window)

	# Load initial data.
	with Pool(thread_max) as pool:
		pool.starmap(read_func,
			[(target_mem.name, image, pj["dtype"], pairs_func(i), size) for i, image in image_order])


def sino_write(sino_mem: SharedNP, path: PathLike, i, out_type: cli.NumpyCLI):
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


@click.command()
@click.option("-i", "--input-dir", type=click.Path(path_type=Path, file_okay=False),
				help="Directory of Input Projections.")
@click.option("-o", "--output-dir", type=click.Path(path_type=Path, file_okay=False),
				help="Directory of Output Sinograms.")
@click.option("-f", "--flat-dir", type=click.Path(path_type=Path, file_okay=False),
				help="Directory of Flats.")
@click.option("-p", "--process-count", type=click.INT,
				help="# of simulatenous processes during conversion.  Also used as window size.")
def sino_convert(input_dir: Path, output_dir: Path, flat_dir: Path, process_count: int):
	image_paths = natsort.natsorted(list(input_dir.glob("**/*.tif*")))
	output_paths = [output_dir.joinpath(x.name) for x in image_paths]

	segment_id = str(uuid.uuid4())
	int_dtype = np.uint16

	with tf.TiffFile(image_paths[0]) as tif:
		page = tif.pages[0]
		pj = {"dtype": page.dtype, "bytesize": page.dtype.itemsize, "offset": page.dataoffsets[0],
				"x": page.shape[1], "y": page.shape[0]}

	sino_shape = SinoOrder(pj["y"], len(image_paths), pj["x"])

	for x in range(0, len(image_paths), process_count):
		window = range(x, min(x + process_count, len(image_paths)))
		int_window = range(0, process_count)

		with (SharedNP(f'flats_{segment_id}', int_dtype, ProjOrder(len(FLAT), pj["y"], pj["x"]), create=True) as flats_mem,
				SharedNP(f"sino_{segment_id}", int_dtype, sino_shape) as sino_mem):

			distribute_read(sino_mem, sino_mem.shape, pj, window, int_window, enumerate(image_paths), thread_max=process_count)

			with flats_mem as flat_set:
				for flat in list(FLAT):
					flat_set[flat.index, :, :] = tf.imread(flat_dir.joinpath(f"{flat}_median.tif")).astype(int_dtype)

			with Pool(process_count) as pool:
				pool.starmap(weighted_normalize, [(sino_mem, flats_mem, window, int_window, i, sino_mem.shape.Theta)
													for i in range(sino_mem.shape.Theta)])

			log("Gain Correction", f"{int_window} | {process_count}", log.DEBUG.TIME)

			with Pool(process_count) as pool:
				pool.starmap(sino_write, [(sino_mem, output_paths[i + window.start], i) for i in int_window])


if __name__ == "__main__":
	sino_convert()