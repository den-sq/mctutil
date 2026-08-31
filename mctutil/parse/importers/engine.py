"""Canonical candidate normalization and single-site precedence resolution."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Mapping

from .diagnostics import Diagnostic, Severity
from .mapping import (
	MappingDefinition,
	MappingValidationError,
	UNIT_CONVERSIONS,
	evaluate_field_rule,
)
from .model import (
	CanonicalValue,
	EvidenceState,
	FieldSpec,
	FieldType,
	ImportRecord,
	PrecedencePolicy,
	SourceEvidence,
)
from .readers import ReaderContext
from .registry import SourceAdapter, SourceBundle
from .schema import CanonicalSchema


DEFAULT_PRECEDENCE_ORDER: tuple[EvidenceState, ...] = (
	"observed",
	"effective",
	"configured",
	"derived",
)


class UnitNormalizationError(ValueError):
	pass


@dataclass(frozen=True)
class ValueCandidate:
	"""Ephemeral normalized input to the engine-owned resolve loop."""

	field_key: str
	value: object
	state: EvidenceState
	evidence: tuple[SourceEvidence, ...]

	def __post_init__(self) -> None:
		object.__setattr__(self, "evidence", tuple(self.evidence))
		if not self.evidence:
			raise ValueError("a value candidate requires evidence")


def _convert_numeric_value(value: object, factor: float, field_key: str) -> object:
	if isinstance(value, (list, tuple)):
		converted = tuple(float(item) * factor for item in value)
		return type(value)(converted)
	if isinstance(value, bool) or not isinstance(value, numbers.Real):
		try:
			value = float(str(value).strip())
		except (TypeError, ValueError) as exc:
			raise UnitNormalizationError(
				f"cannot unit-convert non-numeric value for {field_key!r}: {value!r}"
			) from exc
	return float(value) * factor


def _normalize_unit(value: object, source_unit: str | None, spec: FieldSpec) -> object:
	if spec.unit is None:
		if source_unit is not None:
			raise UnitNormalizationError(
				f"unitless field {spec.key!r} received source unit {source_unit!r}"
			)
		return value
	if source_unit is None:
		raise UnitNormalizationError(
			f"field {spec.key!r} requires source unit {spec.unit!r}"
		)
	if source_unit == spec.unit:
		return value
	try:
		factor = UNIT_CONVERSIONS[(source_unit, spec.unit)]
	except KeyError as exc:
		raise UnitNormalizationError(
			f"unknown unit conversion: {source_unit} -> {spec.unit}"
		) from exc
	return _convert_numeric_value(value, factor, spec.key)


def _normalize_integer_item(value: object) -> int:
	if isinstance(value, bool):
		raise TypeError("bool is not a canonical integer")
	if isinstance(value, numbers.Integral):
		return int(value)
	if isinstance(value, numbers.Real) and float(value).is_integer():
		return int(value)
	return int(str(value).strip())


def _normalize_string(value: object) -> object | None:
	if isinstance(value, bytes):
		value = value.decode("utf-8")
	text = str(value).strip()
	return text or None


def _normalize_integer(value: object) -> object:
	return _normalize_integer_item(value)


def _normalize_float(value: object) -> object:
	if isinstance(value, bool):
		raise TypeError("bool is not a canonical float")
	normalized = float(value)
	if not math.isfinite(normalized):
		raise ValueError("canonical floats must be finite")
	return normalized


def _normalize_boolean(value: object) -> object:
	if isinstance(value, bool):
		return value
	text = str(value).strip().lower()
	if text in {"1", "true", "yes", "on"}:
		return True
	if text in {"0", "false", "no", "off"}:
		return False
	raise ValueError(f"cannot normalize boolean: {value!r}")


def _normalize_datetime(value: object) -> object:
	if isinstance(value, datetime):
		normalized = value
	else:
		normalized = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
	if normalized.tzinfo is None or normalized.utcoffset() is None:
		raise ValueError("canonical datetimes must be timezone-aware")
	return normalized


def _normalize_string_list(value: object) -> object:
	if not isinstance(value, (list, tuple)):
		raise TypeError("string-list value must be a list or tuple")
	return [str(item).strip() for item in value if str(item).strip()]


def _normalize_integer_tuple(value: object) -> object:
	if not isinstance(value, (list, tuple)):
		raise TypeError("integer-tuple value must be a list or tuple")
	return tuple(_normalize_integer_item(item) for item in value)


def _normalize_float_tuple(value: object) -> object:
	if not isinstance(value, (list, tuple)):
		raise TypeError("float-tuple value must be a list or tuple")
	items = tuple(float(item) for item in value)
	if any(not math.isfinite(item) for item in items):
		raise ValueError("canonical float tuples must be finite")
	return items


TypeNormalizer = Callable[[object], object | None]
_TYPE_NORMALIZERS: Mapping[FieldType, TypeNormalizer] = {
	FieldType.STRING: _normalize_string,
	FieldType.INTEGER: _normalize_integer,
	FieldType.FLOAT: _normalize_float,
	FieldType.BOOLEAN: _normalize_boolean,
	FieldType.DATETIME: _normalize_datetime,
	FieldType.STRING_LIST: _normalize_string_list,
	FieldType.INTEGER_TUPLE: _normalize_integer_tuple,
	FieldType.FLOAT_TUPLE: _normalize_float_tuple,
}


def _normalize_type(value: object, spec: FieldSpec) -> object | None:
	if value is None:
		return None
	return _TYPE_NORMALIZERS[spec.value_type](value)


def normalize_candidate_value(
	raw_value: object, *, source_unit: str | None, spec: FieldSpec
) -> object | None:
	unit_normalized = _normalize_unit(raw_value, source_unit, spec)
	type_normalized = _normalize_type(unit_normalized, spec)
	if type_normalized is not None:
		spec.validate(type_normalized)
	return type_normalized


class CandidateCollector:
	"""Schema-aware sink shared by mapping evaluation and adapter derivation."""

	def __init__(self, schema: CanonicalSchema) -> None:
		self.schema = schema
		self._candidates: dict[str, list[ValueCandidate]] = {}
		self._diagnostics: list[Diagnostic] = []

	@property
	def candidates(self) -> Mapping[str, tuple[ValueCandidate, ...]]:
		return {
			key: tuple(values)
			for key, values in sorted(self._candidates.items())
		}

	@property
	def diagnostics(self) -> tuple[Diagnostic, ...]:
		return tuple(self._diagnostics)

	def add_candidate(
		self,
		field_key: str,
		raw_value: object,
		evidence: SourceEvidence,
		*,
		source_unit: str | None = None,
	) -> None:
		self.add_with_evidence(
			field_key,
			raw_value,
			(evidence,),
			state=evidence.state,
			source_unit=evidence.source_unit if source_unit is None else source_unit,
		)

	def add_with_evidence(
		self,
		field_key: str,
		raw_value: object,
		evidence: tuple[SourceEvidence, ...],
		*,
		state: EvidenceState,
		source_unit: str | None,
	) -> None:
		try:
			spec = self.schema.field(field_key)
		except KeyError:
			self._diagnostics.append(
				Diagnostic(
					Severity.ERROR,
					"mapping.unknown_field",
					f"unknown canonical field: {field_key}",
					field_key,
					evidence,
				)
			)
			return
		try:
			value = normalize_candidate_value(
				raw_value, source_unit=source_unit, spec=spec
			)
		except UnitNormalizationError as exc:
			self._diagnostics.append(
				Diagnostic(
					Severity.ERROR,
					"value.unit_unknown",
					str(exc),
					field_key,
					evidence,
				)
			)
			return
		except (TypeError, ValueError, OverflowError) as exc:
			self._diagnostics.append(
				Diagnostic(
					Severity.ERROR,
					"value.normalization_failed",
					f"cannot normalize {field_key!r}: {exc}",
					field_key,
					evidence,
				)
			)
			return
		if value is None:
			return
		self._candidates.setdefault(spec.key, []).append(
			ValueCandidate(spec.key, value, state, evidence)
		)


def _candidate_location(candidate: ValueCandidate) -> tuple[str, ...]:
	return tuple(
		f"{item.artifact}:{item.locator}:{item.state}" for item in candidate.evidence
	)


def _policy_order(
	field_key: str, policies: Mapping[str, PrecedencePolicy]
) -> tuple[EvidenceState, ...]:
	policy = policies.get(field_key)
	if policy is None:
		return DEFAULT_PRECEDENCE_ORDER
	return policy.order + tuple(
		state for state in DEFAULT_PRECEDENCE_ORDER if state not in policy.order
	)


def _resolve_candidates(
	candidates: Mapping[str, tuple[ValueCandidate, ...]],
	policies: Iterable[PrecedencePolicy],
) -> tuple[dict[str, CanonicalValue], tuple[Diagnostic, ...]]:
	"""The one loop and call site that turns candidates into canonical values."""

	policy_items = tuple(policies)
	policy_keys = [policy.field_key for policy in policy_items]
	if len(policy_keys) != len(set(policy_keys)):
		raise ValueError("an adapter declared duplicate precedence policies")
	policy_by_field = {policy.field_key: policy for policy in policy_items}
	values: dict[str, CanonicalValue] = {}
	diagnostics: list[Diagnostic] = []
	for field_key, field_candidates in sorted(candidates.items()):
		order = _policy_order(field_key, policy_by_field)
		ranks = {state: index for index, state in enumerate(order)}
		sorted_candidates = sorted(
			field_candidates,
			key=lambda item: (ranks[item.state], _candidate_location(item)),
		)
		winning_rank = ranks[sorted_candidates[0].state]
		top = [item for item in sorted_candidates if ranks[item.state] == winning_rank]
		if any(item.value != top[0].value for item in top[1:]):
			evidence = tuple(item for candidate in sorted_candidates for item in candidate.evidence)
			diagnostics.append(
				Diagnostic(
					Severity.ERROR,
					"value.conflict",
					f"equal-precedence candidates disagree for {field_key!r}",
					field_key,
					evidence,
				)
			)
			continue
		winner = top[0]
		all_evidence = tuple(
			item for candidate in sorted_candidates for item in candidate.evidence
		)
		if any(item.value != winner.value for item in sorted_candidates[1:]):
			diagnostics.append(
				Diagnostic(
					Severity.WARNING,
					"value.conflict",
					f"precedence selected {winner.state} value for {field_key!r}",
					field_key,
					all_evidence,
				)
			)
		values[field_key] = CanonicalValue(winner.value, winner.state, all_evidence)
	return values, tuple(diagnostics)


def _schema_diagnostics(
	schema: CanonicalSchema, values: Mapping[str, CanonicalValue]
) -> tuple[Diagnostic, ...]:
	diagnostics = []
	for spec in schema.fields:
		canonical_value = values.get(spec.key)
		if canonical_value is None:
			if not spec.nullable:
				diagnostics.append(
					Diagnostic(
						Severity.ERROR,
						"value.missing_required",
						f"missing required canonical field: {spec.key}",
						spec.key,
					)
				)
			continue
		try:
			spec.validate(canonical_value.value)
		except (TypeError, ValueError) as exc:
			diagnostics.append(
				Diagnostic(
					Severity.ERROR,
					"value.type_invalid",
					str(exc),
					spec.key,
					canonical_value.evidence,
				)
			)
	return tuple(diagnostics)


@dataclass(frozen=True)
class ImportEngine:
	schema: CanonicalSchema

	def process_bundle(
		self,
		*,
		bundle: SourceBundle,
		adapter: SourceAdapter,
		mapping: MappingDefinition,
		context: ReaderContext,
	) -> ImportRecord:
		if bundle.kind != self.schema.kind or adapter.kind != self.schema.kind:
			raise ValueError("bundle/adapter kind does not match engine schema")
		if bundle.source_id != adapter.source_id or mapping.source != adapter.source_id:
			raise ValueError("bundle, adapter, and mapping source IDs must match")
		if mapping.schema != self.schema.identifier:
			raise ValueError("mapping schema does not match engine schema")

		collector = CandidateCollector(self.schema)
		mapping_diagnostics: list[Diagnostic] = []
		for rule in mapping.fields:
			try:
				mapped = evaluate_field_rule(
					rule,
					context=context,
					field_spec=self.schema.field(rule.field_key),
				)
			except (MappingValidationError, KeyError, TypeError, ValueError) as exc:
				mapping_diagnostics.append(
					Diagnostic(
						Severity.ERROR,
						"mapping.reader_failed",
						f"cannot map {rule.field_key!r}: {exc}",
						rule.field_key,
					)
				)
				continue
			evidence = tuple(
				SourceEvidence(
					result.artifact.path,
					result.artifact.role,
					result.locator,
					result.value,
					read.source_unit,
					read.state,
				)
				for read, result in zip(rule.reads, mapped.reads)
			)
			collector.add_with_evidence(
				rule.field_key,
				mapped.value,
				evidence,
				state=mapped.state,
				source_unit=mapped.source_unit,
			)

		adapter.derive(bundle, collector)
		policies = adapter.precedence_policies()
		for policy in policies:
			try:
				self.schema.field(policy.field_key)
			except KeyError as exc:
				raise ValueError(
					f"precedence policy names an unknown field: {policy.field_key}"
				) from exc
		values, resolution_diagnostics = _resolve_candidates(
			collector.candidates, policies
		)
		diagnostics = (
			tuple(mapping_diagnostics)
			+ collector.diagnostics
			+ resolution_diagnostics
			+ _schema_diagnostics(self.schema, values)
		)
		preliminary = ImportRecord(
			bundle.kind,
			self.schema.identifier,
			adapter.source_id,
			values,
			diagnostics=diagnostics,
		)
		adapter_diagnostics = tuple(adapter.validate(preliminary, bundle))
		return ImportRecord(
			preliminary.kind,
			preliminary.schema_version,
			preliminary.source_id,
			preliminary.values,
			preliminary.extensions,
			preliminary.diagnostics + adapter_diagnostics,
		)
