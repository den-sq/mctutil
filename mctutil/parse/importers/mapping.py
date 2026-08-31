"""Declarative mapping loader, validator, selectors, and transforms.

User mappings are reviewable data, not executable plugins.  They can select a
different locator inside an adapter-opened artifact, so an incorrect mapping
can still mislabel data.  They cannot execute code or reach the filesystem
beyond those opened artifacts.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from ruamel.yaml import YAML

from .model import EvidenceState, FieldSpec, FieldType
from .readers import READERS, ReadResult, ReaderContext
from .schema import CanonicalSchema


MAPPING_FORMAT_VERSION = 1
SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

SELECTORS = frozenset(
	{
		"scalar",
		"first",
		"last",
		"first_nonempty",
		"median",
		"minimum",
		"maximum",
		"count",
		"shape_axis",
		"dtype_bits",
		"ordered_coalesce",
	}
)
TRANSFORMS = frozenset(
	{"scale", "offset", "bool_parse", "unit_convert", "whitespace", "timestamp"}
)
UNIT_CONVERSIONS: Mapping[tuple[str, str], float] = MappingProxyType(
	{
		("s", "ms"): 1_000.0,
		("s", "us"): 1_000_000.0,
		("ms", "s"): 0.001,
		("ms", "us"): 1_000.0,
		("us", "s"): 0.000_001,
		("us", "ms"): 0.001,
		("mm", "um"): 1_000.0,
		("um", "mm"): 0.001,
		("rad", "deg"): 180.0 / math.pi,
		("deg", "rad"): math.pi / 180.0,
	}
)

READER_LOCATORS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = (
	MappingProxyType(
		{
			"artifact": (frozenset({"property"}), frozenset({"property"})),
			"hdf5": (frozenset({"path"}), frozenset({"path"})),
			"hdf5_shape": (
				frozenset({"path", "property"}),
				frozenset({"path", "property"}),
			),
			"ini": (
				frozenset({"section", "key"}),
				frozenset({"section", "key"}),
			),
			"json_pointer": (frozenset({"pointer"}), frozenset({"pointer"})),
			"cli_option": (frozenset({"option"}), frozenset({"option"})),
			"xlsx_column": (
				frozenset({"sheet", "header"}),
				frozenset({"sheet", "header", "row"}),
			),
		}
	)
)


class MappingValidationError(ValueError):
	"""A mapping is outside the supported declarative vocabulary."""


@dataclass(frozen=True)
class SelectorRule:
	name: str = "scalar"
	parameters: Mapping[str, object] = field(default_factory=dict)

	def __post_init__(self) -> None:
		object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True)
class TransformRule:
	name: str
	parameters: Mapping[str, object] = field(default_factory=dict)

	def __post_init__(self) -> None:
		object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True)
class ReadRule:
	kind: str
	artifact_role: str
	locator: Mapping[str, object]
	source_unit: str | None
	state: EvidenceState

	def __post_init__(self) -> None:
		object.__setattr__(self, "locator", MappingProxyType(dict(self.locator)))


@dataclass(frozen=True)
class FieldRule:
	field_key: str
	reads: tuple[ReadRule, ...]
	selector: SelectorRule
	transforms: tuple[TransformRule, ...]

	def __post_init__(self) -> None:
		object.__setattr__(self, "reads", tuple(self.reads))
		object.__setattr__(self, "transforms", tuple(self.transforms))


@dataclass(frozen=True)
class MappingDefinition:
	mapping_format: int
	schema: str
	source: str
	fields: tuple[FieldRule, ...]

	def __post_init__(self) -> None:
		object.__setattr__(
			self, "fields", tuple(sorted(self.fields, key=lambda item: item.field_key))
		)

	@property
	def by_field(self) -> Mapping[str, FieldRule]:
		return MappingProxyType({rule.field_key: rule for rule in self.fields})


@dataclass(frozen=True)
class MappedValue:
	value: object
	source_unit: str | None
	state: EvidenceState
	reads: tuple[ReadResult, ...]


def _normalize_name(value: object) -> str:
	return str(value).strip().lower().replace("-", "_")


def _parse_named_rule(value: object, *, default: str | None = None) -> tuple[str, dict]:
	if value is None:
		if default is None:
			raise MappingValidationError("a rule name is required")
		return default, {}
	if isinstance(value, str):
		return _normalize_name(value), {}
	if not isinstance(value, Mapping) or len(value) != 1:
		raise MappingValidationError("a selector/transform must be a name or one-key map")
	name, raw_parameters = next(iter(value.items()))
	if raw_parameters is None:
		parameters = {}
	elif isinstance(raw_parameters, Mapping):
		parameters = dict(raw_parameters)
	else:
		parameters = {"value": raw_parameters}
	return _normalize_name(name), parameters


def _parse_read(value: object) -> ReadRule:
	if not isinstance(value, Mapping):
		raise MappingValidationError("read must be a mapping")
	raw = dict(value)
	kind = _normalize_name(raw.pop("kind", ""))
	role = str(raw.pop("role", "primary"))
	source_unit_value = raw.pop("source_unit", None)
	source_unit = None if source_unit_value is None else str(source_unit_value)
	state = str(raw.pop("state", "configured"))
	return ReadRule(kind, role, raw, source_unit, state)  # type: ignore[arg-type]


def _parse_field(field_key: str, value: object) -> FieldRule:
	if not isinstance(value, Mapping):
		raise MappingValidationError(f"field {field_key!r} must be a mapping")
	raw = dict(value)
	unknown = set(raw) - {"read", "select", "transforms"}
	if unknown:
		raise MappingValidationError(
			f"field {field_key!r} has unknown properties: {sorted(unknown)!r}"
		)
	read_value = raw.get("read")
	if isinstance(read_value, list):
		reads = tuple(_parse_read(item) for item in read_value)
	else:
		reads = (_parse_read(read_value),)
	selector_name, selector_parameters = _parse_named_rule(
		raw.get("select"), default="scalar"
	)
	transforms_value = raw.get("transforms", [])
	if not isinstance(transforms_value, list):
		raise MappingValidationError("transforms must be a list")
	transforms = tuple(
		TransformRule(*_parse_named_rule(item)) for item in transforms_value
	)
	return FieldRule(
		field_key,
		reads,
		SelectorRule(selector_name, selector_parameters),
		transforms,
	)


def _parse_mapping(data: object) -> MappingDefinition:
	if not isinstance(data, Mapping):
		raise MappingValidationError("mapping document must be a mapping")
	raw = dict(data)
	unknown = set(raw) - {"mapping_format", "schema", "source", "fields"}
	if unknown:
		raise MappingValidationError(f"unknown top-level properties: {sorted(unknown)!r}")
	try:
		mapping_format = int(raw["mapping_format"])
		schema = str(raw["schema"])
		source = str(raw["source"])
		fields_value = raw["fields"]
	except KeyError as exc:
		raise MappingValidationError(f"missing top-level property: {exc.args[0]}") from exc
	if not isinstance(fields_value, Mapping):
		raise MappingValidationError("fields must be a mapping")
	fields = tuple(
		_parse_field(str(field_key), value)
		for field_key, value in fields_value.items()
	)
	return MappingDefinition(mapping_format, schema, source, fields)


def _validate_read(read: ReadRule, field_spec: FieldSpec) -> None:
	if read.kind not in READERS:
		raise MappingValidationError(f"unknown reader: {read.kind}")
	if not read.artifact_role.strip():
		raise MappingValidationError("artifact role must be non-empty")
	if read.state not in {"configured", "effective", "observed", "derived"}:
		raise MappingValidationError(f"unknown evidence state: {read.state}")
	required, allowed = READER_LOCATORS[read.kind]
	missing = required - set(read.locator)
	unknown = set(read.locator) - allowed
	if missing:
		raise MappingValidationError(
			f"{read.kind} reader is missing locator properties: {sorted(missing)!r}"
		)
	if unknown:
		raise MappingValidationError(
			f"{read.kind} reader has unknown locator properties: {sorted(unknown)!r}"
		)
	if read.source_unit and field_spec.unit and read.source_unit != field_spec.unit:
		if (read.source_unit, field_spec.unit) not in UNIT_CONVERSIONS:
			raise MappingValidationError(
				f"unknown unit conversion: {read.source_unit} -> {field_spec.unit}"
			)
	if read.source_unit and field_spec.unit is None:
		raise MappingValidationError(
			f"field {field_spec.key!r} is unitless but the read declares {read.source_unit!r}"
		)


def _validate_transform(transform: TransformRule, field_spec: FieldSpec) -> None:
	if transform.name not in TRANSFORMS:
		raise MappingValidationError(f"unknown transform: {transform.name}")
	required_type: FieldType | None = None
	if transform.name in {"scale", "offset", "unit_convert"}:
		required_type = FieldType.FLOAT
	elif transform.name == "bool_parse":
		required_type = FieldType.BOOLEAN
	elif transform.name == "whitespace":
		required_type = FieldType.STRING
	elif transform.name == "timestamp":
		required_type = FieldType.DATETIME
	if required_type is not None and field_spec.value_type is not required_type:
		raise MappingValidationError(
			f"transform {transform.name!r} produces {required_type.value}, "
			f"not {field_spec.value_type.value} for {field_spec.key!r}"
		)
	if transform.name in {"scale", "offset"} and "value" not in transform.parameters:
		raise MappingValidationError(
			f"transform {transform.name!r} requires a numeric value"
		)


def _validate_selector(selector: SelectorRule) -> None:
	if selector.name not in SELECTORS:
		raise MappingValidationError(f"unknown selector: {selector.name}")
	if selector.name == "shape_axis" and "axis" not in selector.parameters:
		raise MappingValidationError("shape_axis selector requires an axis")


def _validate_mapping_identity(
	mapping: MappingDefinition, schema: CanonicalSchema, expected_source: str
) -> None:
	if mapping.mapping_format != MAPPING_FORMAT_VERSION:
		raise MappingValidationError(
			f"unsupported mapping format: {mapping.mapping_format}"
		)
	if mapping.schema != schema.identifier:
		raise MappingValidationError(
			f"mapping schema {mapping.schema!r} does not match {schema.identifier!r}"
		)
	if mapping.source != expected_source:
		raise MappingValidationError(
			f"mapping source {mapping.source!r} does not match {expected_source!r}"
		)
	if not SOURCE_ID_PATTERN.fullmatch(mapping.source):
		raise MappingValidationError(f"invalid mapping source ID: {mapping.source!r}")
	keys = [rule.field_key for rule in mapping.fields]
	if len(keys) != len(set(keys)):
		raise MappingValidationError("a canonical field may be mapped only once")


def _requires_unit_conversion(rule: FieldRule, field_spec: FieldSpec) -> bool:
	return any(
		read.source_unit
		and field_spec.unit
		and read.source_unit != field_spec.unit
		for read in rule.reads
	)


def _validate_field_rule(rule: FieldRule, schema: CanonicalSchema) -> None:
	try:
		field_spec = schema.field(rule.field_key)
	except KeyError as exc:
		raise MappingValidationError(str(exc)) from exc
	if not rule.reads:
		raise MappingValidationError(f"field {rule.field_key!r} has no reads")
	_validate_selector(rule.selector)
	if len(rule.reads) > 1 and rule.selector.name != "ordered_coalesce":
		raise MappingValidationError(
			f"field {rule.field_key!r} has multiple reads without ordered_coalesce"
		)
	for read in rule.reads:
		_validate_read(read, field_spec)
	for transform in rule.transforms:
		_validate_transform(transform, field_spec)
	if _requires_unit_conversion(rule, field_spec):
		if "unit_convert" not in {item.name for item in rule.transforms}:
			raise MappingValidationError(
				f"field {rule.field_key!r} requires an explicit unit_convert transform"
			)


def validate_mapping(
	mapping: MappingDefinition,
	*,
	schema: CanonicalSchema,
	expected_source: str,
) -> MappingDefinition:
	_validate_mapping_identity(mapping, schema, expected_source)
	for rule in mapping.fields:
		_validate_field_rule(rule, schema)
	return mapping


def _yaml() -> YAML:
	loader = YAML(typ="safe")
	loader.allow_duplicate_keys = False
	return loader


def load_mapping_text(
	text: str,
	*,
	schema: CanonicalSchema,
	expected_source: str,
) -> MappingDefinition:
	try:
		data = _yaml().load(text)
	except Exception as exc:
		raise MappingValidationError(f"invalid mapping YAML: {exc}") from exc
	return validate_mapping(
		_parse_mapping(data), schema=schema, expected_source=expected_source
	)


def load_mapping(
	path: Path,
	*,
	schema: CanonicalSchema,
	expected_source: str,
) -> MappingDefinition:
	"""Load a complete replacement mapping from the explicitly supplied path."""

	return load_mapping_text(
		path.read_text(encoding="utf-8"),
		schema=schema,
		expected_source=expected_source,
	)


def load_builtin_mapping(
	kind: str,
	source: str,
	*,
	schema: CanonicalSchema,
) -> MappingDefinition:
	if kind not in {"scan", "reconstruction"} or not SOURCE_ID_PATTERN.fullmatch(source):
		raise MappingValidationError("invalid built-in mapping identity")
	resource = resources.files("mctutil.parse.importers").joinpath(
		"mappings", kind, f"{source}.yaml"
	)
	try:
		text = resource.read_text(encoding="utf-8")
	except FileNotFoundError as exc:
		raise MappingValidationError(
			f"no built-in mapping for {kind}/{source}"
		) from exc
	return load_mapping_text(text, schema=schema, expected_source=source)


def _is_empty(value: object) -> bool:
	return value is None or (isinstance(value, str) and not value.strip())


Selector = Callable[[object, Mapping[str, object]], object]


def _select_scalar(value: object, parameters: Mapping[str, object]) -> object:
	if hasattr(value, "item"):
		try:
			return value.item()  # type: ignore[union-attr]
		except ValueError:
			pass
	if isinstance(value, (list, tuple)):
		if len(value) != 1:
			raise MappingValidationError("scalar selector requires exactly one value")
		return value[0]
	return value


def _select_dtype_bits(value: object, parameters: Mapping[str, object]) -> object:
	dtype = value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value
	itemsize = getattr(dtype, "itemsize", None)
	if itemsize is not None:
		return int(itemsize) * 8
	match = re.search(r"(\d+)$", str(dtype))
	if not match:
		raise MappingValidationError(f"cannot derive dtype bits from {dtype!r}")
	return int(match.group(1))


def _select_first(value: object, parameters: Mapping[str, object]) -> object:
	return value[0]  # type: ignore[index]


def _select_last(value: object, parameters: Mapping[str, object]) -> object:
	return value[-1]  # type: ignore[index]


def _select_first_nonempty(
	value: object, parameters: Mapping[str, object]
) -> object:
	return next((item for item in value if not _is_empty(item)), None)  # type: ignore[union-attr]


def _select_median(value: object, parameters: Mapping[str, object]) -> object:
	return statistics.median(value)  # type: ignore[arg-type]


def _select_minimum(value: object, parameters: Mapping[str, object]) -> object:
	return min(value)  # type: ignore[arg-type]


def _select_maximum(value: object, parameters: Mapping[str, object]) -> object:
	return max(value)  # type: ignore[arg-type]


def _select_count(value: object, parameters: Mapping[str, object]) -> object:
	return len(value)  # type: ignore[arg-type]


def _select_shape_axis(value: object, parameters: Mapping[str, object]) -> object:
	return value[int(parameters["axis"])]  # type: ignore[index]


_SCALAR_SELECTORS: Mapping[str, Selector] = MappingProxyType(
	{"scalar": _select_scalar, "dtype_bits": _select_dtype_bits}
)
_SEQUENCE_SELECTORS: Mapping[str, Selector] = MappingProxyType(
	{
		"first": _select_first,
		"last": _select_last,
		"first_nonempty": _select_first_nonempty,
		"ordered_coalesce": _select_first_nonempty,
		"median": _select_median,
		"minimum": _select_minimum,
		"maximum": _select_maximum,
		"count": _select_count,
		"shape_axis": _select_shape_axis,
	}
)


def apply_selector(rule: SelectorRule, value: object) -> object:
	if rule.name in _SCALAR_SELECTORS:
		return _SCALAR_SELECTORS[rule.name](value, rule.parameters)
	if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
		raise MappingValidationError(f"selector {rule.name!r} requires a sequence")
	try:
		selector = _SEQUENCE_SELECTORS[rule.name]
	except KeyError as exc:
		raise MappingValidationError(f"unknown selector: {rule.name}") from exc
	return selector(list(value), rule.parameters)


def _transform_parameter(rule: TransformRule, default: object | None = None) -> object:
	return rule.parameters.get("value", default)


Transform = Callable[[TransformRule, object, str | None, str | None], object]


def _transform_scale(
	rule: TransformRule, value: object, source_unit: str | None, target_unit: str | None
) -> object:
	return float(value) * float(_transform_parameter(rule))


def _transform_offset(
	rule: TransformRule, value: object, source_unit: str | None, target_unit: str | None
) -> object:
	return float(value) + float(_transform_parameter(rule))


def _transform_unit_convert(
	rule: TransformRule, value: object, source_unit: str | None, target_unit: str | None
) -> object:
	from_unit = str(rule.parameters.get("from", source_unit))
	to_unit = str(rule.parameters.get("to", target_unit))
	try:
		factor = UNIT_CONVERSIONS[(from_unit, to_unit)]
	except KeyError as exc:
		raise MappingValidationError(
			f"unknown unit conversion: {from_unit} -> {to_unit}"
		) from exc
	return float(value) * factor


def _transform_bool_parse(
	rule: TransformRule, value: object, source_unit: str | None, target_unit: str | None
) -> object:
	if isinstance(value, bool):
		return value
	text = str(value).strip().lower()
	if text in {"1", "true", "yes", "on"}:
		return True
	if text in {"0", "false", "no", "off"}:
		return False
	raise MappingValidationError(f"cannot parse boolean: {value!r}")


def _transform_whitespace(
	rule: TransformRule, value: object, source_unit: str | None, target_unit: str | None
) -> object:
	mode = str(_transform_parameter(rule, "strip"))
	if mode == "strip":
		return str(value).strip()
	if mode == "collapse":
		return " ".join(str(value).split())
	raise MappingValidationError(f"unknown whitespace mode: {mode}")


def _transform_timestamp(
	rule: TransformRule, value: object, source_unit: str | None, target_unit: str | None
) -> object:
	mode = str(_transform_parameter(rule, "iso8601"))
	if mode == "iso8601":
		parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
		if parsed.tzinfo is None or parsed.utcoffset() is None:
			raise MappingValidationError("timestamp has no declared timezone")
		return parsed
	divisors = {"unix_s": 1.0, "unix_ms": 1_000.0, "unix_us": 1_000_000.0}
	try:
		divisor = divisors[mode]
	except KeyError as exc:
		raise MappingValidationError(f"unknown timestamp epoch: {mode}") from exc
	return datetime.fromtimestamp(float(value) / divisor, tz=timezone.utc)


_TRANSFORMS: Mapping[str, Transform] = MappingProxyType(
	{
		"scale": _transform_scale,
		"offset": _transform_offset,
		"unit_convert": _transform_unit_convert,
		"bool_parse": _transform_bool_parse,
		"whitespace": _transform_whitespace,
		"timestamp": _transform_timestamp,
	}
)


def apply_transform(
	rule: TransformRule,
	value: object,
	*,
	source_unit: str | None,
	target_unit: str | None,
) -> object:
	if value is None:
		return None
	try:
		transform = _TRANSFORMS[rule.name]
	except KeyError as exc:
		raise MappingValidationError(f"unknown transform: {rule.name}") from exc
	return transform(rule, value, source_unit, target_unit)


def evaluate_field_rule(
	rule: FieldRule,
	*,
	context: ReaderContext,
	field_spec: FieldSpec,
) -> MappedValue:
	read_results = tuple(
		context.read(read.kind, read.artifact_role, read.locator) for read in rule.reads
	)
	raw_values = [result.value for result in read_results]
	selector_input: object = raw_values if len(raw_values) > 1 else raw_values[0]
	value = apply_selector(rule.selector, selector_input)
	selected_read = next(
		(
			read
			for read, result in zip(rule.reads, read_results)
			if not _is_empty(result.value)
		),
		rule.reads[0],
	)
	for transform in rule.transforms:
		value = apply_transform(
			transform,
			value,
			source_unit=selected_read.source_unit,
			target_unit=field_spec.unit,
		)
	return MappedValue(
		value,
		field_spec.unit if any(item.name == "unit_convert" for item in rule.transforms)
		else selected_read.source_unit,
		selected_read.state,
		read_results,
	)
