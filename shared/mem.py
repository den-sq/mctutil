from collections import namedtuple
from multiprocessing import shared_memory
from sys import stdout
from time import sleep
from typing import TextIO

import numpy as np
import psutil

from shared.log import log, DEBUG, attach_func

NPString = namedtuple("NPString", 'T')
SinoOrder = namedtuple("SinoOrder", ['Y', 'Theta', 'X'])
ProjOrder = namedtuple("ProjOrder", ['Theta', 'Y', 'X'])
ReconOrder = namedtuple("ReconOrder", ['Y', 'Z', 'X'])
ThetaSlice = namedtuple("ThetaSlice", ['Y', 'X'])
Proj = namedtuple("Proj", ['Y', 'X'])

MEM_INTERVAL_TIMER = 1.0


def shrink_nt(source: namedtuple) -> namedtuple:
	return namedtuple(f"{type(source).__name__}_Slice", source._fields[1:])(*source[1:])


class SharedNP:
	"""Manages a shared memory numpy array.

		:param name: Name of shared memory to use.
		:param dtype: Datatype of numpy array.
		:param shape: Shape of numpy array.
		:param size: Size of shared memory to create, if different from array size.

		:param ar_size: Size of array in memory.
	"""

	def __init__(self, name: str, dtype: np.dtype, shape: namedtuple,
					buffer_index=None, create=True, create_for=[], recycle=True, size=None):
		"""Constructor"""
		self.__name = name
		self.__dtype = np.dtype(dtype)
		self.__shape = shape
		self.__ar_size = int(np.prod(shape, dtype=np.float64) * self.__dtype.itemsize)
		if size is None:
			self.__full_size = self.__ar_size
		else:
			self.__full_size = max(size, self.__ar_size)
		self.__sm = None
		self.__unlink = False
		self.__mem_list = []
		self.__slice = self.__slice_to_byterange(buffer_index)
		if create:
			if len(create_for) > 0:
				self.create_for(create_for, recycle)
			else:
				self.create(recycle)

	def __slice_to_byterange(self, buffer_index):
		if buffer_index is None:
			return slice(None)
		else:
			if isinstance(buffer_index, int):
				start = self.size([buffer_index])
				stop = self.size([buffer_index + 1])
				self.__shape = shrink_nt(self.__shape)
			elif (isinstance(buffer_index, slice) or isinstance(buffer_index, range)) and buffer_index.step in [None, 1]:
				start = self.size([buffer_index.start])
				stop = self.size([buffer_index.stop])
				self.__shape = self.__shape._make([buffer_index.stop - buffer_index.start] + list(self.__shape[1:]))
			elif isinstance(buffer_index, list) and (buffer_index[-1] == (buffer_index[0] + len(buffer_index) - 1)):
				start = self.size([buffer_index[0]])
				stop = self.size([buffer_index[-1] + 1])
				self.__shape = self.__shape._make([len(buffer_index)] + list(self.__shape[1:]))
			else:
				log("Shared Memory", "Please use only single contiguous ranges for memory allocation;"
										+ f" {buffer_index} for {self.__name}, shape {self.__shape} failed.",
										log_level=DEBUG.ERROR)
				raise MemoryError
			return slice(start, stop)

	def __create(self, recycle=True):
		"""Creates shared memory space, assigns it to global.

			:param recycle: Whether to delete & recreate memory if it already exists.  Just reused if false.
			"""
		try:
			shared_memory.SharedMemory(create=True, size=self.__full_size, name=self.__name)
		except FileExistsError:
			log("Shared Memory", f"Shared Memory Name {self.__name} Pre-Existed", log_level=DEBUG.INFO)
			mem = shared_memory.SharedMemory(name=self.__name)
			if recycle:
				mem.close()
				mem.unlink()
				mem = shared_memory.SharedMemory(create=True, size=self.__full_size, name=self.__name)
				log("Shared Memory", f"Shared Memory Name {self.__name} Recreated", log_level=DEBUG.INFO)
			elif len(mem.buf) < self.__ar_size:
				log("Shared Memory", f"Existing {self.__name} Memory Too Small for Array ({len(mem.buf)} vs {self.__ar_size})",
					log_level=DEBUG.ERROR)
				raise MemoryError
			else:
				if len(mem.buf) < self.__full_size:
					log("Shared Memory", f"Existing {self.__name} Size {len(mem.buf)} Less Than Full Size {self.__full_size}.",
							log_level=DEBUG.WARN)
				log("Shared Memory", f"Using Existing {self.__name} Size {len(mem.buf)}", log_level=DEBUG.INFO)
		self.__unlink = True

	def __load(self):
		"""Allocates a numpy array in the shared memory."""
		try:
			self.__sm = shared_memory.SharedMemory(name=self.__name)
			return np.ndarray(shape=self.__shape, dtype=self.__dtype, buffer=self.__sm.buf[self.__slice])
		except FileNotFoundError:
			log("Shared Memory", f"Shared Memory {self.__name} Has Not Been Created", log_level=DEBUG.ERROR)
			return None

	def __branch(self, buffer_index=None, dtype=None, transpose=False, shape=None):
		# Carry over __slice somehow
		return SharedNP(self.__name, dtype if dtype is not None else self.__dtype,
						shape if shape is not None else self.__shape,
						create=False, buffer_index=buffer_index, size=self.__full_size)

	def create(self, recycle=True):
		self.__create(recycle)
		return self.__branch()

	def create_for(self, mem_list, recycle=True):
		self.__full_size = max([self.__full_size] + [mem.size() for mem in mem_list])
		self.__create(recycle)
		self.__mem_list = [self.__branch()] + mem_list
		return self

	def load(self):
		return self.__load()

	@property
	def buffer_address(self):
		if self.__slice is not None:
			return self.__slice
		else:
			return slice(0, self.size)

	def close(self):
		"""Closes this shared memory space."""
		if self.__sm is not None:
			self.__sm.close()

	def unlink(self):
		"""Closes and unlinks shared memory."""
		self.close()
		if self.__sm is not None:
			self.__sm.unlink()
			log("Shared Memory", f"Shared Memory {self.__name} Unlinked", log_level=DEBUG.STATUS)
		else:
			log("Shared Memory", f"Shared Memory {self.__name} Is None", log_level=DEBUG.STATUS)

	def transpose(self, order=[1, 0], change_data=False):
		if change_data:
			return self.__branch() 	# TODO: Actually Implement
		elif len(self.__shape) < 2:
			return self.__branch()
		elif len(self.__shape) != len(order):
			return self.__branch() 	# TODO: Error Messaage
		else:
			new_shape = namedtuple(f"{type(self.__shape).__name__}_Transpose",
									[self.__shape._fields[x] for x in order]
									)(*[self.__shape[x] for x in order])
			return self.__branch(shape=new_shape)

	@property
	def name(self):
		return self.__name

	@property
	def dtype(self):
		return self.__dtype

	@property
	def shape(self):
		return self.__shape

	def size(self, swap=None):
		""" Size of array in memory.

			:param swap: Tuple overriding shape parameters, for a custom size.
			:retval: Size in bytes.
		"""
		if swap is None:
			return self.__ar_size
		else:
			shape = [swap[x] if swap[x] is not None else self.__shape[x] for x in range(len(swap))]
			shape += [self.__shape[x] for x in range(len(shape), len(self.__shape))]
			return int(np.prod(shape, dtype=np.float64) * self.__dtype.itemsize)

	@property
	def full_size(self):
		""" Full size of memory block created for array usage.

			:retval: Size in bytes.
		"""
		return self.__full_size

	def __enter__(self):
		"""Context handler to load shared memory."""
		if self.__unlink:
			if len(self.__mem_list) > 0:
				return [mem for mem in self.__mem_list]
			else:
				return self.__branch()
		else:
			return self.__load()

	def __exit__(self, exc_type, exc_value, traceback):
		"""Context handler to close shared memory.  Unlinks if this created the memory."""
		self.close()
		if self.__unlink:
			log("Shared Memory", f"Unlinking {self.name}", log_level=DEBUG.INFO)
			self.unlink()
		if exc_type is not None:
			log("Shared Memory", f"Exception ({exc_type}: {exc_value})", log_level=DEBUG.ERROR)
			log("Shared Memory", traceback, log_level=DEBUG.ERROR)

	def __del__(self):
		"""Delete closes; unlinks only if this created the memory.."""
		self.close()

	def __getitem__(self, key):
		return self.__branch(key)

	# TODO: Make Work After Slicing
	def as_dtype(self, dtype):
		return self.__branch(dtype=dtype)


def cleanup_mem(*shm_objects):
	""" Close and unlink shared memory objects.

		:param shm_objects: Shared memory objects to shut down.
	"""
	for shm in shm_objects:
		if shm is not None:
			shm.close()
			shm.unlink()


def exit_cleanly(step: str, *shm_objects, return_code: int = 0, statement: str = '', log_level: DEBUG = DEBUG.TIME,
					out: TextIO = stdout, throw: Exception = None):
	""" Exit while cleaning up shared memory.

		:param step: Step of reconstruction process we are exiting during.
		:param shm_objects: Shared memory objects to shut down.
		:param return_code: Process return code to send.
	"""
	log(step, statement, log_level, out)
	cleanup_mem(*shm_objects)

	sleep(MEM_INTERVAL_TIMER)
	if throw is not None:
		raise throw

	exit(return_code)


def mem_monitor(mem_file, mem_store, pid):
	with open(mem_file, "w") as out, mem_store as mem_branch:
		with mem_branch as last_step_arr:
			last_step = last_step_arr.tobytes().decode()

			while last_step.strip() not in ["Error Exit", "Script Completion", "Keyboard Exit", "Exit",
											"Center Find Done", "Scan Path Failure"]:
				last_step = last_step_arr.tobytes().decode()
				log(last_step, out=out, pid=pid)
				sleep(MEM_INTERVAL_TIMER)


def __update_last_step(step, pid):
	step_buf = shared_memory.SharedMemory(name=mem_tracker.name)
	last_step = np.ndarray(mem_tracker.shape, mem_tracker.dtype, step_buf.buf)
	last_step[:] = np.frombuffer(step.ljust(20)[:20].encode(), dtype=np.uint8)


def init_mem_tracker():
	last_step = SharedNP(dtype=np.uint8, name=f"last_step_{psutil.Process().pid}", shape=NPString(T=20), create=False)
	with last_step.create() as last_step_arr:
		last_step_arr[:] = np.frombuffer("Script Inactive     ".encode(), dtype=last_step.dtype)
	attach_func(__update_last_step)
	return last_step


mem_tracker = init_mem_tracker()
