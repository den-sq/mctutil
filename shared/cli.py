from enum import Enum, Flag, auto

import click
import numpy as np
from ruamel.yaml import YAML, yaml_object

yaml = YAML()


@yaml_object(yaml)
class FloatRange:
	start: float
	stop: float
	step: float
	yaml_tag = '!FloatRange'

	def __init__(self, start, stop, step):
		self.start = start
		self.stop = stop
		self.step = step

	@classmethod
	def to_yaml(cls, representer, node):
		return representer.represent_scalar(cls.yaml_tag, str(node))

	@classmethod
	def from_yaml(cls, constructor, node):
		return cls(*node.value.split(","))

	def __str__(self):
		return f"{self.start},{self.stop},{self.step}"

	def as_array(self):
		steps = int((self.start - self.stop) // self.step) + 1
		return np.linspace(self.start, self.stop, steps)


@yaml_object(yaml)
class NumpyCLI:
	_nptype: np.dtype
	yaml_tag = '!NumpyCLI'

	def __init__(self, type_value):
		self._nptype = np.dtype(type_value)

	@classmethod
	def to_yaml(cls, representer, node):
		return representer.represent_scalar(cls.yaml_tag, str(node))

	@classmethod
	def from_yaml(cls, constructor, node):
		return cls(str(node))

	@property
	def nptype(self):
		return self._nptype

	def convert_ar(self, alt_ar):
		if np.issubdtype(self._nptype, np.integer):
			dtype_range = np.iinfo(self._nptype).max - np.iinfo(self._nptype).min
			source_range = int(np.max(alt_ar) - np.min(alt_ar))
			if source_range == 0:
				source_range = dtype_range
			return (alt_ar * (dtype_range // source_range))
		elif np.issubdtype(self._nptype, np.floating):
			return alt_ar.astype(self._nptype)
		else:
			raise TypeError(f"Cannot convert to non-numeric datatype {self._nptype}.")

	def __str__(self):
		return f"{self._nptype}"


# Click Parameter: Comma-separated key=value list of options.
class OptionList(click.ParamType):
	name = "Option List"

	def convert(self, value, param, ctx):
		try:
			return {param.split('=')[0]: param.split("=")[1] for param in value.split(",")}
		except ValueError:
			self.fail(f'{value} is not a list of comma-separated options.')


# Click Parameter:
class Range(click.ParamType):
	name = "Integer Range"

	def convert(self, value, param, ctx):
		try:
			params = [int(x) for x in value.split(",")]
			start, stop, step = ([0] if len(params) == 1 else []) + params + ([1] if len(params) in [1, 2] else [])
			return range(start, stop, step)
		except ValueError:
			self.fail(f'{value} is not a python range.')


# Click Parameter: Float Range (Imitated by linspace).
class Frange(click.ParamType):
	name = "Float Range"

	def convert(self, value, param, ctx):
		try:
			params = [float(x) for x in str(value).split(",")]
			start, stop, step = ([0.] if len(params) == 1 else []) + params + ([1.] if len(params) in [1, 2] else [])
			return FloatRange(start, stop, step)
		except ValueError:
			self.fail(f'{value} cannot be evaluated as a float range.')


# Click Parameter: Choice of an Enum from a list.
class EnumParameter(click.Choice):
	name = "Enumerated Value"

	def __init__(self, enum):
		self.__enum = enum
		super().__init__(enum.__members__)

	def convert(self, value, param, ctx):
		return self.__enum[super().convert(value, param, ctx)]


# Click Parameter: Choice of an Enum from a list.
class NumPyType(click.ParamType):
	name = "Numpy Datatype"

	def convert(self, value, param, ctx):
		try:
			val = NumpyCLI(value)
			return val
		except TypeError:
			self.fail(f'{value} is not a valid numpy datatype.')


# Click Parameter: Indexing Slice
class SLICE(click.ParamType):
	name = "Index Slice"

	def convert(self, value, _param, _ctx):
		try:
			if value[0] != '[' or value[-1] != ']':
				self.fail(f"{value} should be enclosed in brackets like a slice; e.g. [1:5, 2:3].")

			built_slice = ()

			for dim in value[1:-1].split(","):
				entries = [int(x) if x != '' else None for x in dim.split(':')]
				if len(entries) == 1:
					built_slice += (np.s_[entries[0]],)
				elif len(entries) == 2:
					built_slice += (np.s_[entries[0]: entries[1]],)
				elif len(entries) == 3:
					built_slice += (np.s_[entries[0]: entries[1]: entries[2]],)

			if built_slice != ():
				return built_slice
			else:
				self.fail(f'{value} must have at least one entry to be a slice.')
		except ValueError as ex:
			self.fail(f'{value} is not formatted as a valid slice; e.g. [1,2:7,4:] - {ex}')


# Different projection types for CUDA reconstruction algorithms.
class PROJ(Enum):
	PB_LINE = "line"
	PB_STRIP = "strip"
	PB_LINEAR = "linear"
	FB_LINE = "line_fanflat"
	FB_STRIP = "strip_fanflat"
	SPARSE = "sparse_matrix"
	CUDA = "cuda"

	def __str__(self):
		return str(self.value)


# Different reconstruction algorithm choices.
class RA(Enum):
	GRIDREC = "GRIDREC"
	FP_CUDA = "FP_CUDA"
	BP_CUDA = "BP_CUDA"
	FBP_CUDA = "FBP_CUDA"
	SIRT_CUDA = "SIRT_CUDA"
	SART_CUDA = "SART_CUDA"
	CGLS_CUDA = "CGLS_CUDA"
	EM_CUDA = "EM_CUDA"

	def __str__(self):
		return str(self.value)


class CF(Enum):
	NONE = 0
	VO = 1
	ENTROPY_TP = 2
	ENTROPY_LOCAL = 3

	def __str__(self):
		return str(self.name.lower())

	def __getitem__(self):
		return self._value_


class RFLAG(Flag):
	LOCAL_GAINS = auto()					# Use local gains normalization algorithm.
	LOCAL_THETA = auto()					# Use local theta calclation based on shutter timings.
	SKIP_CENTER_NORMALIZATION = auto()		# Whether to normalize before center finding.
	THREAD_READ = auto()					# Whether to use a thread pool instead of a process pool for reading.


OPTION_LIST = OptionList()
RANGE = Range()
FRANGE = Frange()
NUMPYTYPE = NumPyType()
FLAGS = []							# List of toggle flags set on command line.
