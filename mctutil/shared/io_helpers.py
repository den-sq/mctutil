from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool, shared_memory
from os import PathLike
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from psutil import cpu_count

from mctutil.shared.mem import SharedNP

FlatPair = namedtuple("FlatPair", ["Index", "Offset"])


@dataclass(frozen=True)
class RawOffsetRead:
	"""One raw byte span copied from a file into shared memory."""

	source: PathLike
	source_offset: int
	target_offset: int
	size: int

	def __post_init__(self):
		for name in ("source_offset", "target_offset", "size"):
			value = getattr(self, name)
			if value < 0:
				raise ValueError(f"{name} must be non-negative, got {value}")


def offset_reads(
	source: PathLike,
	*,
	source_offset: int,
	target_offset: int,
	size: int,
	count: int = 1,
	source_stride: int | None = None,
	target_stride: int | None = None,
) -> tuple[RawOffsetRead, ...]:
	"""Build layout-agnostic strided raw-byte reads."""
	if count < 0:
		raise ValueError(f"count must be non-negative, got {count}")
	source_stride = size if source_stride is None else source_stride
	target_stride = size if target_stride is None else target_stride
	return tuple(
		RawOffsetRead(
			source=source,
			source_offset=source_offset + index * source_stride,
			target_offset=target_offset + index * target_stride,
			size=size,
		)
		for index in range(count)
	)


def readinto_offset(source_handle, target_buffer, read: RawOffsetRead) -> None:
	"""Copy one exact raw file span into an arbitrary shared-memory span."""
	target_stop = read.target_offset + read.size
	if target_stop > len(target_buffer):
		raise ValueError(
			f"raw read target exceeds shared memory: stop={target_stop}, "
			f"available={len(target_buffer)}"
		)
	source_handle.seek(read.source_offset)
	target = target_buffer[read.target_offset:target_stop]
	try:
		read_count = source_handle.readinto(target)
	finally:
		target.release()
	if read_count != read.size:
		raise EOFError(
			f"short raw read from {read.source}: expected {read.size} bytes at "
			f"offset {read.source_offset}, got {read_count}"
		)


def _readinto_shared(target: str, source: PathLike, reads: tuple[RawOffsetRead, ...]) -> None:
	"""Attach to shared memory and perform one source file's raw reads."""
	memory = shared_memory.SharedMemory(name=target)
	try:
		with open(source, "rb", buffering=0) as source_handle:
			for read in reads:
				readinto_offset(source_handle, memory.buf, read)
	finally:
		memory.close()


def distribute_read(
	target_mem: SharedNP | shared_memory.SharedMemory | str,
	reads: Iterable[RawOffsetRead],
	thread_max: int | None = None,
) -> None:
	"""Distribute layout-agnostic raw-offset reads into shared memory."""
	grouped: dict[PathLike, list[RawOffsetRead]] = {}
	for read in reads:
		grouped.setdefault(read.source, []).append(read)
	if not grouped:
		return

	target_name = target_mem if isinstance(target_mem, str) else target_mem.name
	if thread_max is not None and thread_max < 1:
		raise ValueError(f"thread_max must be positive, got {thread_max}")
	requested_workers = thread_max or cpu_count() or 1
	worker_count = min(requested_workers, len(grouped))
	jobs = [
		(target_name, source, tuple(source_reads))
		for source, source_reads in grouped.items()
	]
	if worker_count == 1:
		for job in jobs:
			_readinto_shared(*job)
		return
	with Pool(worker_count) as pool:
		pool.starmap(_readinto_shared, jobs)


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
	"""Compatibility wrapper for callers using the legacy offset dictionaries."""
	reads = tuple(
		RawOffsetRead(
			source=image,
			source_offset=int(offset["source"]),
			target_offset=int(offset["target"]),
			size=size,
		)
		for offset in offsets
	)
	_readinto_shared(target, image, reads)
