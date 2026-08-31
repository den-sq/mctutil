"""Canonical, source-independent importer value objects."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping


CANONICAL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
EXTENSION_KEY_PATTERN = re.compile(
	r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*\.[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"
)

RecordKind = Literal["scan", "reconstruction"]
EvidenceState = Literal["configured", "effective", "observed", "derived"]
ValueState = Literal[
	"configured", "effective", "observed", "derived", "missing"
]


class FieldType(str, Enum):
	"""Closed canonical value types understood by schema validation."""

	STRING = "str"
	INTEGER = "int"
	FLOAT = "float"
	BOOLEAN = "bool"
	DATETIME = "datetime"
	STRING_LIST = "str_list"
	INTEGER_TUPLE = "int_tuple"
	FLOAT_TUPLE = "float_tuple"


def _validate_string(key: str, value: object) -> None:
	if not isinstance(value, str) or not value.strip():
		raise TypeError(f"{key} must be a non-empty string")


def _validate_integer(key: str, value: object) -> None:
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError(f"{key} must be an int")


def _validate_float(key: str, value: object) -> None:
	if isinstance(value, bool) or not isinstance(value, float):
		raise TypeError(f"{key} must be a float")
	if not math.isfinite(value):
		raise ValueError(f"{key} must be finite")


def _validate_boolean(key: str, value: object) -> None:
	if not isinstance(value, bool):
		raise TypeError(f"{key} must be a bool")


def _validate_datetime(key: str, value: object) -> None:
	if not isinstance(value, datetime):
		raise TypeError(f"{key} must be a datetime")
	if value.tzinfo is None or value.utcoffset() is None:
		raise ValueError(f"{key} must be timezone-aware")


def _validate_string_list(key: str, value: object) -> None:
	if not isinstance(value, list) or any(
		not isinstance(item, str) or not item.strip() for item in value
	):
		raise TypeError(f"{key} must be a list of non-empty strings")


def _validate_integer_tuple(key: str, value: object) -> None:
	if not isinstance(value, tuple) or any(
		isinstance(item, bool) or not isinstance(item, int) for item in value
	):
		raise TypeError(f"{key} must be a tuple of ints")


def _validate_float_tuple(key: str, value: object) -> None:
	if not isinstance(value, tuple) or any(
		isinstance(item, bool)
		or not isinstance(item, float)
		or not math.isfinite(item)
		for item in value
	):
		raise TypeError(f"{key} must be a finite tuple of floats")


_FIELD_VALIDATORS: Mapping[FieldType, Callable[[str, object], None]] = {
	FieldType.STRING: _validate_string,
	FieldType.INTEGER: _validate_integer,
	FieldType.FLOAT: _validate_float,
	FieldType.BOOLEAN: _validate_boolean,
	FieldType.DATETIME: _validate_datetime,
	FieldType.STRING_LIST: _validate_string_list,
	FieldType.INTEGER_TUPLE: _validate_integer_tuple,
	FieldType.FLOAT_TUPLE: _validate_float_tuple,
}


@dataclass(frozen=True)
class FieldSpec:
	"""The single code-owned definition of one canonical field."""

	key: str
	value_type: FieldType
	unit: str | None
	nullable: bool
	label: str
	description: str
	min_items: int | None = None
	max_items: int | None = None

	def __post_init__(self) -> None:
		if not CANONICAL_KEY_PATTERN.fullmatch(self.key):
			raise ValueError(f"invalid canonical field key: {self.key!r}")
		if not isinstance(self.value_type, FieldType):
			raise TypeError("value_type must be a FieldType")
		if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
			raise ValueError("unit must be a non-empty string or None")
		if not isinstance(self.nullable, bool):
			raise TypeError("nullable must be bool")
		if not self.label.strip() or not self.description.strip():
			raise ValueError("label and description must be non-empty")

		is_sequence = self.value_type in {
			FieldType.STRING_LIST,
			FieldType.INTEGER_TUPLE,
			FieldType.FLOAT_TUPLE,
		}
		if not is_sequence and (self.min_items is not None or self.max_items is not None):
			raise ValueError("item bounds are valid only for sequence fields")
		if self.min_items is not None and self.min_items < 0:
			raise ValueError("min_items must be non-negative")
		if self.max_items is not None and self.max_items < 1:
			raise ValueError("max_items must be positive")
		if (
			self.min_items is not None
			and self.max_items is not None
			and self.min_items > self.max_items
		):
			raise ValueError("min_items cannot exceed max_items")

	def validate(self, value: object | None) -> None:
		"""Raise at the canonical boundary when *value* violates this spec."""

		if value is None:
			if not self.nullable:
				raise ValueError(f"{self.key} is not nullable")
			return

		_FIELD_VALIDATORS[self.value_type](self.key, value)

		if not isinstance(value, (list, tuple)):
			return
		if self.min_items is not None and len(value) < self.min_items:
			raise ValueError(f"{self.key} requires at least {self.min_items} items")
		if self.max_items is not None and len(value) > self.max_items:
			raise ValueError(f"{self.key} allows at most {self.max_items} items")


@dataclass(frozen=True)
class SourceEvidence:
	"""Immutable provenance for a canonical-value candidate."""

	artifact: Path
	artifact_role: str
	locator: str
	raw_value: object
	source_unit: str | None
	state: EvidenceState

	def __post_init__(self) -> None:
		if not isinstance(self.artifact, Path):
			raise TypeError("artifact must be a pathlib.Path")
		if not self.artifact_role.strip() or not self.locator.strip():
			raise ValueError("artifact_role and locator must be non-empty")
		if self.source_unit is not None and not self.source_unit:
			raise ValueError("source_unit must be a non-empty string or None")
		if self.state not in {"configured", "effective", "observed", "derived"}:
			raise ValueError(f"unknown evidence state: {self.state!r}")


@dataclass(frozen=True)
class CanonicalValue:
	"""The sole authoritative value for one canonical field."""

	value: object | None
	state: ValueState
	evidence: tuple[SourceEvidence, ...] = ()

	def __post_init__(self) -> None:
		object.__setattr__(self, "evidence", tuple(self.evidence))
		if self.state not in {
			"configured",
			"effective",
			"observed",
			"derived",
			"missing",
		}:
			raise ValueError(f"unknown canonical value state: {self.state!r}")
		if self.value is None and self.state != "missing":
			raise ValueError("a null canonical value must have state 'missing'")
		if self.value is not None and self.state == "missing":
			raise ValueError("a non-null canonical value cannot have state 'missing'")
		if self.value is not None and not self.evidence:
			raise ValueError("a non-null canonical value requires source evidence")


@dataclass(frozen=True)
class PrecedencePolicy:
	"""Adapter-supplied ordering data consumed by the engine resolve loop."""

	field_key: str
	order: tuple[EvidenceState, ...]

	def __post_init__(self) -> None:
		if not CANONICAL_KEY_PATTERN.fullmatch(self.field_key):
			raise ValueError(f"invalid policy field key: {self.field_key!r}")
		object.__setattr__(self, "order", tuple(self.order))
		if not self.order or len(set(self.order)) != len(self.order):
			raise ValueError("precedence order must be non-empty and unique")
		unknown = set(self.order) - {
			"configured",
			"effective",
			"observed",
			"derived",
		}
		if unknown:
			raise ValueError(f"unknown precedence states: {sorted(unknown)!r}")


def _immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
	return MappingProxyType(dict(values))


@dataclass(frozen=True)
class ImportRecord:
	"""A source-independent record consumed by validation and exporters."""

	kind: RecordKind
	schema_version: str
	source_id: str
	values: Mapping[str, CanonicalValue]
	extensions: Mapping[str, CanonicalValue] = field(default_factory=dict)
	diagnostics: tuple[Any, ...] = ()

	def __post_init__(self) -> None:
		if self.kind not in {"scan", "reconstruction"}:
			raise ValueError(f"unknown record kind: {self.kind!r}")
		if not self.schema_version.strip() or not self.source_id.strip():
			raise ValueError("schema_version and source_id must be non-empty")
		if any(not isinstance(value, CanonicalValue) for value in self.values.values()):
			raise TypeError("record values must be CanonicalValue instances")
		if any(
			not EXTENSION_KEY_PATTERN.fullmatch(key)
			for key in self.extensions
		):
			raise ValueError("extension keys must be source-namespaced")
		if any(
			not isinstance(value, CanonicalValue) for value in self.extensions.values()
		):
			raise TypeError("record extensions must be CanonicalValue instances")
		if set(self.values) & set(self.extensions):
			raise ValueError("canonical and extension keys must be disjoint")

		object.__setattr__(self, "values", _immutable_mapping(self.values))
		object.__setattr__(self, "extensions", _immutable_mapping(self.extensions))
		object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
