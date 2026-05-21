from collections import namedtuple
from enum import Enum
from multiprocessing import Pool, shared_memory
from os import PathLike
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike
from psutil import cpu_count

from mctutil.shared.log import log, LOG
from mctutil.shared.mem import SharedNP

FlatPair = namedtuple("FlatPair", ["Index", "Offset"])


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


def memmap_helper(target, image, i_dtype, offsets, size):
	"""Sinogram order-capable reader using direct buffer reading."""
	sm = shared_memory.SharedMemory(name=target)
	shape = size // np.dtype(i_dtype).itemsize

	for offset in offsets:
		target_array = np.ndarray(shape, dtype=i_dtype, buffer=sm.buf[offset["target"]:offset["target"] + size])
		target_array[:] = np.memmap(image, dtype=i_dtype, mode="r+", offset=offset["source"], shape=shape, order='C')
	sm.close()


def byteread_helper(target: SharedNP, image: PathLike, _i_dtype: np.dtype, offsets: ArrayLike, size: int):
	"""Sinogram order-capable reader using direct buffer reads."""
	sm = shared_memory.SharedMemory(name=target)
	with open(image, "rb") as handle:
		handle.seek(offsets[0]["source"])
		for offset in offsets:
			handle.readinto(sm.buf[offset["target"]:offset["target"] + size])
	sm.close()


def distribute_read(target_mem: SharedNP, pj: Mapping, window, int_window,
					image_order: ArrayLike, thread_max: int = cpu_count(),
					read_func: Callable = byteread_helper, sino_order: bool = True):
	"""Distribute direct reads across workers."""
	h_step = pj["x"] * pj["bytesize"]
	sino_block_size = target_mem.shape.Theta * h_step
	proj_block_size = len(int_window) * h_step
	base_offset = target_mem[int_window].buffer_address.start

	def generate_offset_pairs_sino(i):
		return [{"source": pj["offset"] + (window.start + j) * h_step,
				"target": int(base_offset + j * sino_block_size + i * h_step)}
					for j in range(len(int_window))]

	def generate_offset_pairs_proj(i):
		return [{"source": pj["offset"] + window.start * h_step, "target": int(base_offset + i * proj_block_size)}]

	if sino_order:
		log.write("Files Into Memory", f"Writing (in {target_mem.name} | {target_mem.shape}) {base_offset}"
			+ f" to {base_offset + len(int_window) * sino_block_size}", log_level=LOG.INFO)
		pairs_func = generate_offset_pairs_sino
		size = h_step
	else:
		log.write("Files Into Memory", f"Writing (in {target_mem.name} | {target_mem.shape}) {base_offset}"
			+ f" to {base_offset + len(int_window) * proj_block_size} out of {target_mem[int_window].buffer_address}",
			log_level=LOG.INFO)
		pairs_func = generate_offset_pairs_proj
		size = proj_block_size

	with Pool(thread_max) as pool:
		pool.starmap(read_func,
			[(target_mem.name, image, pj["dtype"], pairs_func(i), size) for i, image in image_order])
