from enum import Enum, Flag, auto
import re

import click
import numpy as np
from ruamel.yaml import YAML, yaml_object

from mctutil.shared.np_convert import np_convert

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
		if np.issubdtype(self._nptype, np.number) or self._nptype == np.dtype(bool):
			return np_convert(self._nptype, alt_ar)
		raise TypeError(f"Cannot convert to non-numeric datatype {self._nptype}.")

	def __str__(self):
		return f"{self._nptype}"


class OptionList(click.ParamType):
	name = "Option List"

	def convert(self, value, param, ctx):
		try:
			return {param.split('=')[0]: param.split("=")[1] for param in value.split(",")}
		except ValueError:
			self.fail(f'{value} is not a list of comma-separated options.')


class Range(click.ParamType):
	name = "Integer Range"

	def convert(self, value, param, ctx):
		try:
			params = [int(x) for x in value.split(",")]
			start, stop, step = ([0] if len(params) == 1 else []) + params + ([1] if len(params) in [1, 2] else [])
			return range(start, stop, step)
		except ValueError:
			self.fail(f'{value} is not a python range.')


class Frange(click.ParamType):
	name = "Float Range"

	def convert(self, value, param, ctx):
		try:
			params = [float(x) for x in str(value).split(",")]
			start, stop, step = ([0.] if len(params) == 1 else []) + params + ([1.] if len(params) in [1, 2] else [])
			return FloatRange(start, stop, step)
		except ValueError:
			self.fail(f'{value} cannot be evaluated as a float range.')


class EnumParameter(click.Choice):
	name = "Enumerated Value"

	def __init__(self, enum):
		self.__enum = enum
		super().__init__(enum.__members__)

	def convert(self, value, param, ctx):
		return self.__enum[super().convert(value, param, ctx)]


class NumPyType(click.ParamType):
	name = "Numpy Datatype"

	def convert(self, value, param, ctx):
		try:
			val = NumpyCLI(value)
			return val
		except TypeError:
			self.fail(f'{value} is not a valid numpy datatype.')


class DelimitedRecord(click.ParamType):
	def __init__(self, record_type, field_parsers, delimiter=":", defaults=None, min_fields=None, name=None):
		self.record_type = record_type
		self.field_parsers = tuple(field_parsers)
		self.delimiter = delimiter
		self.defaults = tuple(defaults) if defaults is not None else (None,) * len(self.field_parsers)
		self.min_fields = len(self.field_parsers) if min_fields is None else min_fields
		self.name = name if name is not None else getattr(record_type, "__name__", "Delimited Record")

		if len(self.defaults) != len(self.field_parsers):
			raise ValueError("defaults must match the parser count.")

	def convert(self, value, param, ctx):
		fields = str(value).split(self.delimiter)

		if len(fields) < self.min_fields or len(fields) > len(self.field_parsers):
			self.fail(f"{value} must have between {self.min_fields} and {len(self.field_parsers)} fields.")

		fields += [self.defaults[i] for i in range(len(fields), len(self.field_parsers))]

		try:
			return self.record_type(*[
				parser(field)
				for parser, field in zip(self.field_parsers, fields)
			])
		except (AttributeError, KeyError, TypeError, ValueError) as ex:
			self.fail(str(ex))


class CropNumberType(click.ParamType):
	name = "CropNumber"

	def convert(self, value, param, ctx):
		pair = str(value).split(",")
		if len(pair) == 1:
			pair = [pair[0], pair[0]]
		elif len(pair) != 2:
			self.fail(f"{value} must be a single or pair of values.")

		converted = []
		for item in pair:
			if str(item).isnumeric():
				converted.append(int(item))
			else:
				try:
					converted.append(float(item))
				except ValueError:
					self.fail(f"{value} must contain ints or floats: {item} is not.")
		return converted


class IntegerTriple(click.ParamType):
	name = "X,Y,Z"

	def convert(self, value, param, ctx):
		if isinstance(value, tuple) and len(value) == 3:
			return value
		try:
			parts = tuple(int(item) for item in re.split(r"[x,]", str(value)))
		except ValueError:
			self.fail(f"{value} must contain three integers.", param, ctx)
		if len(parts) != 3:
			self.fail(f"{value} must contain exactly three integers.", param, ctx)
		return parts


class SLICE(click.ParamType):
	name = "Index Slice"

	def convert(self, value, _param, _ctx):
		try:
			if value[0] != '[' or value[-1] != ']':
				self.fail(f"{value} should be enclosed in brackets like a slice; e.g. [1:5,2:3].")

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
	LOCAL_GAINS = auto()
	LOCAL_THETA = auto()
	SKIP_CENTER_NORMALIZATION = auto()
	THREAD_READ = auto()


def crop_val(crop, dim):
	return np.s_[int(crop[0] * dim) if isinstance(crop[0], float) else crop[0]:
			int((1.0 - crop[1]) * dim) if isinstance(crop[1], float) else dim - crop[1]]


OPTION_LIST = OptionList()
RANGE = Range()
FRANGE = Frange()
NUMPYTYPE = NumPyType()
CROP_NUMBER = CropNumberType()
XYZ = IntegerTriple()
FLAGS = []
