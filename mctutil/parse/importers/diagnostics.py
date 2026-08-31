"""Stable, namespaced diagnostics and destination write gating."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol

from .model import SourceEvidence


CODE_PATTERN = re.compile(
	r"^(?P<namespace>[a-z][a-z0-9_]*)(?:\.(?P<name>[a-z][a-z0-9_]*))$"
)
SOURCE_NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SHARED_NAMESPACES = frozenset(
	{"source", "mapping", "value", "scan", "reconstruction"}
)


class Severity(str, Enum):
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"


class StrictnessMode(str, Enum):
	DEFAULT = "default"
	STRICT = "strict"
	VALIDATE_ONLY = "validate_only"


@dataclass(frozen=True)
class Diagnostic:
	severity: Severity
	code: str
	message: str
	field: str | None = None
	evidence: tuple[SourceEvidence, ...] = ()

	def __post_init__(self) -> None:
		if not isinstance(self.severity, Severity):
			raise TypeError("diagnostic severity must be a Severity")
		if not CODE_PATTERN.fullmatch(self.code):
			raise ValueError(f"invalid diagnostic code: {self.code!r}")
		if not self.message.strip():
			raise ValueError("diagnostic message must be non-empty")
		if self.field is not None and not self.field.strip():
			raise ValueError("diagnostic field must be non-empty or None")
		object.__setattr__(self, "evidence", tuple(self.evidence))
		if any(not isinstance(item, SourceEvidence) for item in self.evidence):
			raise TypeError("diagnostic evidence must be SourceEvidence instances")


BASE_DIAGNOSTIC_CODES: Mapping[str, frozenset[str]] = MappingProxyType(
	{
		"source": frozenset(
			{
				"ambiguous",
				"missing_artifact",
				"mixed_invocation",
				"probe_failed",
				"weak_match",
			}
		),
		"mapping": frozenset(
			{
				"invalid",
				"reader_failed",
				"unknown_field",
				"unsupported_version",
			}
		),
		"value": frozenset(
			{
				"conflict",
				"missing_required",
				"normalization_failed",
				"type_invalid",
				"unit_unknown",
			}
		),
		"scan": frozenset(
			{
				"frame_gap",
				"theta_length_mismatch",
			}
		),
		"reconstruction": frozenset(
			{
				"center_convention_unknown",
				"output_incomplete",
			}
		),
	}
)


@dataclass(frozen=True)
class DiagnosticCodeRegistry:
	"""Immutable open registry; sources extend it under their own namespace."""

	codes: Mapping[str, frozenset[str]]

	def __post_init__(self) -> None:
		normalized: dict[str, frozenset[str]] = {}
		for namespace, names in self.codes.items():
			if not SOURCE_NAMESPACE_PATTERN.fullmatch(namespace):
				raise ValueError(f"invalid diagnostic namespace: {namespace!r}")
			normalized_names = frozenset(names)
			if any(
				not SOURCE_NAMESPACE_PATTERN.fullmatch(name)
				for name in normalized_names
			):
				raise ValueError(f"invalid diagnostic name in {namespace!r}")
			normalized[namespace] = normalized_names
		object.__setattr__(self, "codes", MappingProxyType(normalized))

	@classmethod
	def shared(cls) -> "DiagnosticCodeRegistry":
		return cls(BASE_DIAGNOSTIC_CODES)

	def with_source(
		self, namespace: str, names: Iterable[str]
	) -> "DiagnosticCodeRegistry":
		if namespace in SHARED_NAMESPACES:
			raise ValueError(f"source namespace is reserved: {namespace}")
		if namespace in self.codes:
			raise ValueError(f"diagnostic namespace already registered: {namespace}")
		updated = dict(self.codes)
		updated[namespace] = frozenset(names)
		return DiagnosticCodeRegistry(updated)

	def contains(self, code: str) -> bool:
		match = CODE_PATTERN.fullmatch(code)
		if not match:
			return False
		namespace, name = match.group("namespace", "name")
		return name in self.codes.get(namespace, ())

	def validate(self, diagnostic: Diagnostic) -> None:
		if not self.contains(diagnostic.code):
			raise ValueError(f"unregistered diagnostic code: {diagnostic.code}")


class DiagnosticRecord(Protocol):
	diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class WriteDecision:
	should_write: bool
	mode: StrictnessMode
	reason: str
	record_statuses: tuple[str, ...]


def diagnostic_status(diagnostics: Iterable[Diagnostic]) -> str:
	severities = {item.severity for item in diagnostics}
	if Severity.ERROR in severities:
		return "error"
	if Severity.WARNING in severities:
		return "warning"
	return "valid"


def diagnostic_summary(diagnostics: Iterable[Diagnostic]) -> str:
	return "; ".join(
		f"{item.code}: {item.message}"
		for item in diagnostics
		if item.severity in {Severity.WARNING, Severity.ERROR}
	)


def evaluate_write(
	records: Iterable[DiagnosticRecord],
	*,
	mode: StrictnessMode = StrictnessMode.DEFAULT,
	invocation_diagnostics: Iterable[Diagnostic] = (),
) -> WriteDecision:
	"""Make the only strictness decision immediately before destination output."""

	if not isinstance(mode, StrictnessMode):
		raise TypeError("mode must be a StrictnessMode")
	records_tuple = tuple(records)
	invocation_tuple = tuple(invocation_diagnostics)
	statuses = tuple(diagnostic_status(record.diagnostics) for record in records_tuple)
	if any(item.severity is Severity.ERROR for item in invocation_tuple):
		return WriteDecision(False, mode, "invocation_error", statuses)
	if mode is StrictnessMode.VALIDATE_ONLY:
		return WriteDecision(False, mode, "validate_only", statuses)
	if mode is StrictnessMode.STRICT and "error" in statuses:
		return WriteDecision(False, mode, "strict_record_error", statuses)
	return WriteDecision(True, mode, "write", statuses)
