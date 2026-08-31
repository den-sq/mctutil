"""Typed readers over artifacts that adapters have already opened.

Readers deliberately have no filesystem-opening API.  A mapping can redirect a
locator inside an :class:`OpenedArtifact`, but cannot name or open another
source file.
"""

from __future__ import annotations

import configparser
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence


class ReaderError(ValueError):
	"""A declared read cannot be satisfied by its opened artifact."""


@dataclass(frozen=True)
class OpenedArtifact:
	"""An adapter-owned artifact handle exposed to declarative readers."""

	role: str
	path: Path
	media_type: str
	content: object
	metadata: Mapping[str, object] = field(default_factory=dict)

	def __post_init__(self) -> None:
		if not self.role.strip() or not self.media_type.strip():
			raise ValueError("artifact role and media_type must be non-empty")
		if not isinstance(self.path, Path):
			raise TypeError("artifact path must be a pathlib.Path")
		object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ReadResult:
	value: object
	artifact: OpenedArtifact
	locator: str


Reader = Callable[[OpenedArtifact, Mapping[str, object]], ReadResult]


def _require(locator: Mapping[str, object], key: str) -> object:
	try:
		return locator[key]
	except KeyError as exc:
		raise ReaderError(f"reader locator requires {key!r}") from exc


def _artifact_reader(
	artifact: OpenedArtifact, locator: Mapping[str, object]
) -> ReadResult:
	property_name = str(_require(locator, "property"))
	properties: dict[str, object] = {
		"filename": artifact.path.name,
		"stem": artifact.path.stem,
		"parent": str(artifact.path.parent),
		"role": artifact.role,
	}
	if "size" in artifact.metadata:
		properties["size"] = artifact.metadata["size"]
	try:
		value = properties[property_name]
	except KeyError as exc:
		raise ReaderError(
			f"artifact property {property_name!r} is unavailable without filesystem access"
		) from exc
	return ReadResult(value, artifact, f"artifact:{property_name}")


def _lookup_hdf5(artifact: OpenedArtifact, path: str) -> object:
	try:
		return artifact.content[path]  # type: ignore[index]
	except (KeyError, TypeError, IndexError) as exc:
		raise ReaderError(f"HDF5 dataset is unavailable: {path}") from exc


def _hdf5_reader(
	artifact: OpenedArtifact, locator: Mapping[str, object]
) -> ReadResult:
	path = str(_require(locator, "path"))
	dataset = _lookup_hdf5(artifact, path)
	if hasattr(dataset, "shape") and hasattr(dataset, "__getitem__"):
		try:
			value = dataset[()]
		except (KeyError, TypeError, ValueError) as exc:
			raise ReaderError(f"cannot read HDF5 dataset: {path}") from exc
	else:
		value = dataset
	return ReadResult(value, artifact, path)


def _hdf5_shape_reader(
	artifact: OpenedArtifact, locator: Mapping[str, object]
) -> ReadResult:
	path = str(_require(locator, "path"))
	property_name = str(_require(locator, "property"))
	dataset = _lookup_hdf5(artifact, path)
	if property_name == "shape":
		value = getattr(dataset, "shape", None)
	elif property_name == "dtype":
		value = getattr(dataset, "dtype", None)
	else:
		raise ReaderError(f"unsupported HDF5 metadata property: {property_name}")
	if value is None:
		raise ReaderError(f"HDF5 {property_name} is unavailable: {path}")
	return ReadResult(value, artifact, f"{path}@{property_name}")


def _ini_reader(artifact: OpenedArtifact, locator: Mapping[str, object]) -> ReadResult:
	section = str(_require(locator, "section"))
	key = str(_require(locator, "key"))
	if not isinstance(artifact.content, configparser.ConfigParser):
		raise ReaderError("INI reader requires an opened ConfigParser")
	try:
		value = artifact.content[section][key]
	except KeyError as exc:
		raise ReaderError(f"INI value is unavailable: [{section}] {key}") from exc
	return ReadResult(value, artifact, f"[{section}] {key}")


def _decode_pointer_part(part: str) -> str:
	return part.replace("~1", "/").replace("~0", "~")


def _json_pointer_reader(
	artifact: OpenedArtifact, locator: Mapping[str, object]
) -> ReadResult:
	pointer = str(_require(locator, "pointer"))
	if pointer == "":
		return ReadResult(artifact.content, artifact, pointer)
	if not pointer.startswith("/"):
		raise ReaderError("JSON pointer must be empty or start with '/'")
	value = artifact.content
	for raw_part in pointer[1:].split("/"):
		part = _decode_pointer_part(raw_part)
		try:
			if isinstance(value, Mapping):
				value = value[part]
			elif isinstance(value, Sequence) and not isinstance(
				value, (str, bytes, bytearray)
			):
				value = value[int(part)]
			else:
				raise TypeError
		except (KeyError, IndexError, TypeError, ValueError) as exc:
			raise ReaderError(f"JSON pointer is unavailable: {pointer}") from exc
	return ReadResult(value, artifact, pointer)


def cli_option_raw(tokens: str | Sequence[str], option: str) -> str | None:
	"""Return only the exact option's raw token value.

	No aliases are recognized, no defaults are filled, and no configured/effective
	precedence is applied here.
	"""

	parts = shlex.split(tokens) if isinstance(tokens, str) else list(tokens)
	for index, token in enumerate(parts):
		if token == option:
			if index + 1 >= len(parts) or parts[index + 1].startswith("--"):
				return None
			return parts[index + 1]
		prefix = f"{option}="
		if token.startswith(prefix):
			return token[len(prefix):]
	return None


def _cli_option_reader(
	artifact: OpenedArtifact, locator: Mapping[str, object]
) -> ReadResult:
	option = str(_require(locator, "option"))
	if not isinstance(artifact.content, (str, list, tuple)):
		raise ReaderError("CLI reader requires a command string or token sequence")
	value = cli_option_raw(artifact.content, option)
	return ReadResult(value, artifact, option)


def _xlsx_column_reader(
	artifact: OpenedArtifact, locator: Mapping[str, object]
) -> ReadResult:
	sheet = str(_require(locator, "sheet"))
	header = str(_require(locator, "header"))
	row_index_value = locator.get("row", artifact.metadata.get("row_index"))
	if row_index_value is None:
		raise ReaderError("XLSX reader requires a selected row")
	row_index = int(row_index_value)
	try:
		rows = artifact.content[sheet]  # type: ignore[index]
		row = rows[row_index]
		value = row[header]
	except (KeyError, IndexError, TypeError) as exc:
		raise ReaderError(
			f"XLSX value is unavailable: {sheet!r} row {row_index} column {header!r}"
		) from exc
	return ReadResult(value, artifact, f"{sheet}!{header}[{row_index}]")


READERS: Mapping[str, Reader] = MappingProxyType(
	{
		"artifact": _artifact_reader,
		"hdf5": _hdf5_reader,
		"hdf5_shape": _hdf5_shape_reader,
		"ini": _ini_reader,
		"json_pointer": _json_pointer_reader,
		"cli_option": _cli_option_reader,
		"xlsx_column": _xlsx_column_reader,
	}
)


@dataclass(frozen=True)
class ReaderContext:
	"""Role-indexed handles that bound everything a mapping can read."""

	artifacts: tuple[OpenedArtifact, ...]

	def __post_init__(self) -> None:
		object.__setattr__(self, "artifacts", tuple(self.artifacts))
		roles = [artifact.role for artifact in self.artifacts]
		if len(roles) != len(set(roles)):
			raise ValueError("opened artifact roles must be unique")

	def read(
		self,
		kind: str,
		role: str,
		locator: Mapping[str, object],
	) -> ReadResult:
		try:
			reader = READERS[kind]
		except KeyError as exc:
			raise ReaderError(f"unknown reader: {kind}") from exc
		try:
			artifact = next(item for item in self.artifacts if item.role == role)
		except StopIteration as exc:
			raise ReaderError(f"opened artifact role is unavailable: {role}") from exc
		return reader(artifact, locator)
