"""One-way JSON Lines audit serialization for canonical importer records."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import singledispatch
from pathlib import Path

from mctutil.shared.atomic_output import atomic_text_output

from .diagnostics import Diagnostic
from .export import ExportProfile
from .model import (
	EXTENSION_KEY_PATTERN,
	CanonicalValue,
	ImportRecord,
	SourceEvidence,
)
from .registry import SourceBundle
from .schema import SchemaRegistry


AUDIT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class AuditItem:
	record: ImportRecord
	bundle: SourceBundle

	def __post_init__(self) -> None:
		if self.record.kind != self.bundle.kind:
			raise ValueError("audit record and bundle kinds must match")
		if self.record.source_id != self.bundle.source_id:
			raise ValueError("audit record and bundle source IDs must match")


@singledispatch
def _json_value(value: object) -> object:
	for method_name in ("item", "tolist"):
		method = getattr(value, method_name, None)
		if callable(method):
			try:
				return _json_value(method())
			except (TypeError, ValueError):
				continue
	return {
		"type": f"{type(value).__module__}.{type(value).__qualname__}",
		"repr": repr(value),
	}


@_json_value.register(type(None))
def _json_none(value: None) -> object:
	return value


@_json_value.register(str)
@_json_value.register(int)
@_json_value.register(bool)
def _json_scalar(value: object) -> object:
	return value


@_json_value.register(float)
def _json_float(value: float) -> object:
	if math.isfinite(value):
		return value
	return {"type": "nonfinite_float", "value": repr(value)}


@_json_value.register(datetime)
def _json_datetime(value: datetime) -> object:
	return {"type": "datetime", "value": value.isoformat()}


@_json_value.register(Path)
def _json_path(value: Path) -> object:
	return {"type": "path", "value": str(value)}


@_json_value.register(bytes)
def _json_bytes(value: bytes) -> object:
	return {
		"type": "bytes",
		"encoding": "base64",
		"value": base64.b64encode(value).decode("ascii"),
	}


@_json_value.register(Enum)
def _json_enum(value: Enum) -> object:
	return value.value


@_json_value.register(Mapping)
def _json_mapping(value: Mapping) -> object:
	return {
		str(key): _json_value(item)
		for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
	}


@_json_value.register(list)
@_json_value.register(tuple)
def _json_sequence(value: list | tuple) -> object:
	return [_json_value(item) for item in value]


def _evidence_document(evidence: SourceEvidence) -> dict[str, object]:
	return {
		"artifact": str(evidence.artifact),
		"artifact_role": evidence.artifact_role,
		"locator": evidence.locator,
		"raw_value": _json_value(evidence.raw_value),
		"source_unit": evidence.source_unit,
		"state": evidence.state,
	}


def _canonical_document(value: CanonicalValue) -> dict[str, object]:
	return {
		"value": _json_value(value.value),
		"state": value.state,
		"evidence": [_evidence_document(item) for item in value.evidence],
	}


def _diagnostic_document(diagnostic: Diagnostic) -> dict[str, object]:
	return {
		"severity": diagnostic.severity.value,
		"code": diagnostic.code,
		"message": diagnostic.message,
		"field": diagnostic.field,
		"evidence": [_evidence_document(item) for item in diagnostic.evidence],
	}


def audit_document(
	item: AuditItem,
	*,
	profile: ExportProfile,
	source_version: str,
) -> dict[str, object]:
	"""Derive an audit document; no source/mapping representation is accepted."""

	record = item.record
	bundle = item.bundle
	if not source_version.strip():
		raise ValueError("source_version must be non-empty")
	if record.kind != profile.kind or record.schema_version != profile.schema_version:
		raise ValueError("audit record does not match export profile")
	document = {
		"audit_format": AUDIT_FORMAT_VERSION,
		"schema": {"id": record.schema_version},
		"profile": {
			"id": profile.profile_id,
			"header_version": profile.header_version,
		},
		"source": {"id": record.source_id, "version": source_version},
		"bundle": {
			"identity_hint": bundle.identity_hint,
			"artifacts": [
				{
					"path": str(artifact.path),
					"role": artifact.role,
					"media_type": artifact.media_type,
				}
				for artifact in bundle.all_artifacts
			],
		},
		"values": {
			key: _canonical_document(value)
			for key, value in sorted(record.values.items())
		},
		"extensions": {
			key: _canonical_document(value)
			for key, value in sorted(record.extensions.items())
		},
		"diagnostics": [
			_diagnostic_document(diagnostic) for diagnostic in record.diagnostics
		],
	}
	return document


def validate_audit_document(
	document: Mapping[str, object], *, schema_registry: SchemaRegistry
) -> None:
	"""Validate in-memory audit structure against its recorded schema version."""

	if document.get("audit_format") != AUDIT_FORMAT_VERSION:
		raise ValueError("unsupported audit format")
	schema_section = document.get("schema")
	if not isinstance(schema_section, Mapping):
		raise ValueError("audit document has no schema section")
	schema_id = schema_section.get("id")
	if not isinstance(schema_id, str):
		raise ValueError("audit document has no schema ID")
	schema = schema_registry.get(schema_id)
	values = document.get("values")
	if not isinstance(values, Mapping):
		raise ValueError("audit document values must be a mapping")
	resolved = set()
	for key in values:
		resolved_key = schema.resolve_key(str(key))
		if resolved_key in resolved:
			raise ValueError(f"duplicate resolved audit field: {resolved_key}")
		resolved.add(resolved_key)
	extensions = document.get("extensions", {})
	if not isinstance(extensions, Mapping) or any(
		not EXTENSION_KEY_PATTERN.fullmatch(str(key)) for key in extensions
	):
		raise ValueError("audit extensions must be source-namespaced")


def write_audit_jsonl(
	output: Path,
	items: Iterable[AuditItem],
	*,
	profile: ExportProfile,
	source_version: str,
	schema_registry: SchemaRegistry,
	force: bool = False,
) -> None:
	"""Write audit JSONL once; there is intentionally no sidecar read API."""

	documents = tuple(
		audit_document(item, profile=profile, source_version=source_version)
		for item in items
	)
	for document in documents:
		validate_audit_document(document, schema_registry=schema_registry)
	with atomic_text_output(output, force=force) as handle:
		for document in documents:
			handle.write(
				json.dumps(
					document,
					ensure_ascii=False,
					sort_keys=True,
					separators=(",", ":"),
					allow_nan=False,
				)
			)
			handle.write("\n")
