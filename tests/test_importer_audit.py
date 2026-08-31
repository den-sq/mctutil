from __future__ import annotations

import json
from pathlib import Path

import pytest

from mctutil.parse.importers.audit import (
	AuditItem,
	audit_document,
	validate_audit_document,
	write_audit_jsonl,
)
from mctutil.parse.importers.diagnostics import Diagnostic, Severity
from mctutil.parse.importers.export import EXPORT_PROFILE_REGISTRY
from mctutil.parse.importers.model import (
	CanonicalValue,
	FieldSpec,
	FieldType,
	ImportRecord,
	SourceEvidence,
)
from mctutil.parse.importers.registry import SourceArtifact, SourceBundle
from mctutil.parse.importers.schema import (
	SCAN_SCHEMA_REGISTRY,
	ScanSchema,
	SchemaRegistry,
	SchemaVersion,
	promote_extension,
)


def _evidence(
	raw_value: object = "raw-id", *, locator: str = "/scan_id"
) -> SourceEvidence:
	return SourceEvidence(
		Path("scan.json"), "primary", locator, raw_value, None, "observed"
	)


def _value(value: object, *, raw_value: object = "raw-id") -> CanonicalValue:
	return CanonicalValue(value, "observed", (_evidence(raw_value),))


def _item(identity: str = "scan-1") -> AuditItem:
	record = ImportRecord(
		"scan",
		"scan/1.0",
		"fixture-scan",
		{"scan_id": _value(identity)},
		{"fixture_scan.raw_note": _value("extension", raw_value=b"raw")},
		(
			Diagnostic(
				Severity.WARNING,
				"scan.frame_gap",
				"a frame was skipped",
				"scan_id",
				(_evidence(7, locator="/uid"),),
			),
		),
	)
	bundle = SourceBundle(
		"scan",
		"fixture-scan",
		SourceArtifact(Path("scan.json"), "primary", "application/json"),
		(
			SourceArtifact(Path("flat.json"), "flat", "application/json"),
		),
		identity,
	)
	return AuditItem(record, bundle)


def test_document_contains_versions_bundle_values_evidence_extensions_and_diagnostics() -> None:
	profile = EXPORT_PROFILE_REGISTRY.get("scan", "scan-v1")
	document = audit_document(
		_item(), profile=profile, source_version="fixture/1"
	)

	assert document["schema"] == {"id": "scan/1.0"}
	assert document["profile"] == {
		"id": "scan-v1",
		"header_version": "scan-header/1",
	}
	assert document["source"] == {"id": "fixture-scan", "version": "fixture/1"}
	assert [artifact["role"] for artifact in document["bundle"]["artifacts"]] == [
		"primary",
		"flat",
	]
	assert document["values"]["scan_id"]["value"] == "scan-1"
	assert document["values"]["scan_id"]["evidence"][0]["raw_value"] == "raw-id"
	assert "fixture_scan.raw_note" in document["extensions"]
	assert document["diagnostics"][0]["code"] == "scan.frame_gap"
	assert document["diagnostics"][0]["evidence"][0]["locator"] == "/uid"


def test_jsonl_writer_is_one_way_and_atomic(tmp_path: Path) -> None:
	profile = EXPORT_PROFILE_REGISTRY.get("scan", "scan-v1")
	output = tmp_path / "records.audit.jsonl"
	write_audit_jsonl(
		output,
		(_item("scan-1"), _item("scan-2")),
		profile=profile,
		source_version="fixture/1",
		schema_registry=SCAN_SCHEMA_REGISTRY,
	)

	lines = output.read_text(encoding="utf-8").splitlines()
	assert [json.loads(line)["values"]["scan_id"]["value"] for line in lines] == [
		"scan-1",
		"scan-2",
	]
	assert not list(tmp_path.glob(".records.audit.jsonl.*"))
	with pytest.raises(FileExistsError):
		write_audit_jsonl(
			output,
			(_item(),),
			profile=profile,
			source_version="fixture/1",
			schema_registry=SCAN_SCHEMA_REGISTRY,
		)


def test_raw_nonfinite_values_are_json_safe_without_changing_canonical_value() -> None:
	profile = EXPORT_PROFILE_REGISTRY.get("scan", "scan-v1")
	item = _item()
	evidence = _evidence(float("nan"))
	record = ImportRecord(
		"scan",
		"scan/1.0",
		"fixture-scan",
		{"scan_id": CanonicalValue("scan-1", "observed", (evidence,))},
	)
	document = audit_document(
		AuditItem(record, item.bundle), profile=profile, source_version="fixture/1"
	)

	assert document["values"]["scan_id"]["value"] == "scan-1"
	assert document["values"]["scan_id"]["evidence"][0]["raw_value"] == {
		"type": "nonfinite_float",
		"value": "nan",
	}
	json.dumps(document, allow_nan=False)


def test_historical_document_validates_against_its_own_schema_version() -> None:
	old = ScanSchema(
		SchemaVersion.parse("scan/1.0"),
		(
			FieldSpec(
				"scan_id", FieldType.STRING, None, False, "Scan ID", "Identifier."
			),
		),
	)
	promoted = promote_extension(
		old,
		extension_key="chenglab_camera.capture_delay_raw",
		field_spec=FieldSpec(
			"capture_delay_us",
			FieldType.FLOAT,
			"us",
			True,
			"Capture Delay",
			"Normalized capture delay.",
		),
	)
	registry = SchemaRegistry("scan", (old, promoted))
	historical = {
		"audit_format": 1,
		"schema": {"id": "scan/1.0"},
		"values": {"scan_id": {"value": "old"}},
		"extensions": {"chenglab_camera.capture_delay_raw": {"value": "4"}},
	}

	validate_audit_document(historical, schema_registry=registry)
	assert historical["schema"]["id"] == "scan/1.0"
	with pytest.raises(KeyError, match="unknown canonical field"):
		validate_audit_document(
			{**historical, "values": {"future_field": {"value": 1}}},
			schema_registry=registry,
		)
