from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mctutil.parse import tabular_log
from mctutil.parse.importers.export import (
	EXPORT_PROFILE_REGISTRY,
	ExportColumn,
	ExportProfile,
	ExportProfileRegistry,
	append_records,
	materialize_row,
	write_records_csv,
)
from mctutil.parse.importers.model import (
	CanonicalValue,
	ImportRecord,
	SourceEvidence,
)
from mctutil.parse.importers.schema import SCAN_SCHEMA_V1


def _value(value: object, *, locator: str = "/raw") -> CanonicalValue:
	evidence = SourceEvidence(
		Path("fixture.json"), "primary", locator, "different raw value", None, "observed"
	)
	return CanonicalValue(value, "observed", (evidence,))


def _record(values: dict[str, CanonicalValue]) -> ImportRecord:
	return ImportRecord("scan", "scan/1.0", "fixture-scan", values)


def test_profile_validates_canonical_keys_headers_and_registration() -> None:
	with pytest.raises(ValueError, match="unknown canonical field"):
		ExportProfile(
			"bad-v1",
			"bad-header/1",
			SCAN_SCHEMA_V1,
			(ExportColumn("not_a_field", "Bad"),),
		)
	with pytest.raises(ValueError, match="duplicate canonical"):
		ExportProfile(
			"bad-v1",
			"bad-header/1",
			SCAN_SCHEMA_V1,
			(ExportColumn("scan_id", "One"), ExportColumn("scan_id", "Two")),
		)
	profile = EXPORT_PROFILE_REGISTRY.get("scan", "scan-v1")
	with pytest.raises(ValueError, match="duplicate export profile"):
		ExportProfileRegistry((profile, profile))


def test_header_version_is_fixed_and_independent_from_schema_version() -> None:
	profile = EXPORT_PROFILE_REGISTRY.get("scan", "scan-v1")

	assert profile.header_version == "scan-header/1"
	assert profile.schema_version == "scan/1.0"
	assert profile.header_version != profile.schema_version


def test_row_uses_only_canonical_values_and_stringifies_at_boundary() -> None:
	profile = ExportProfile(
		"test-v1",
		"test-header/7",
		SCAN_SCHEMA_V1,
		(
			ExportColumn("scan_id", "ID"),
			ExportColumn("projection_count_acquired", "Count"),
			ExportColumn("rotation_start_deg", "Angle"),
			ExportColumn("detector_shape_px", "Shape"),
			ExportColumn("scan_start", "Started"),
			ExportColumn("source_files", "Files"),
			ExportColumn("sample_id", "Missing"),
		),
	)
	record = _record(
		{
			"scan_id": _value("canonical-id"),
			"projection_count_acquired": _value(1200),
			"rotation_start_deg": _value(0.5),
			"detector_shape_px": _value((2048, 2048)),
			"scan_start": _value(datetime(2026, 8, 31, 12, tzinfo=timezone.utc)),
			"source_files": _value(["scan.h5", "flat.h5"]),
		},
	)
	row = materialize_row(record, profile)

	assert row == {
		"ID": "canonical-id",
		"Count": "1200",
		"Angle": "0.5",
		"Shape": "[2048,2048]",
		"Started": "2026-08-31T12:00:00+00:00",
		"Files": '["scan.h5","flat.h5"]',
		"Missing": "",
	}
	assert "different raw value" not in row.values()


def test_csv_and_sheets_delegate_same_rows_to_tabular_log(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	profile = ExportProfile(
		"test-v1",
		"test-header/1",
		SCAN_SCHEMA_V1,
		(ExportColumn("scan_id", "Scan ID"), ExportColumn("sample_id", "Sample ID")),
	)
	record = _record({"scan_id": _value("scan-1")})
	calls = {}

	def fake_write_csv(output, fields, rows, *, force=False):
		calls["csv"] = (output, fields, tuple(dict(row) for row in rows), force)

	def fake_append_rows(
		service,
		spreadsheet,
		sheet,
		fields,
		rows,
		*,
		verify=True,
		strict_header_order=False,
	):
		calls["sheet"] = (
			service,
			spreadsheet,
			sheet,
			fields,
			tuple(dict(row) for row in rows),
			verify,
			strict_header_order,
		)
		return {"updates": 1}

	monkeypatch.setattr(tabular_log, "write_csv", fake_write_csv)
	monkeypatch.setattr(tabular_log, "append_rows", fake_append_rows)
	write_records_csv(tmp_path / "out.csv", (record,), profile, force=True)
	service = object()
	result = append_records(
		service,
		"spreadsheet",
		"sheet",
		(record,),
		profile,
		strict_header_order=True,
	)

	assert result == {"updates": 1}
	assert calls["csv"][1] == calls["sheet"][3] == ("Scan ID", "Sample ID")
	assert calls["csv"][2] == calls["sheet"][4] == (
		{"Scan ID": "scan-1", "Sample ID": ""},
	)


def test_real_csv_output_uses_profile_headers(tmp_path: Path) -> None:
	profile = ExportProfile(
		"test-v1",
		"test-header/1",
		SCAN_SCHEMA_V1,
		(ExportColumn("scan_id", "Scan ID"), ExportColumn("sample_id", "Sample ID")),
	)
	output = tmp_path / "records.csv"
	write_records_csv(output, (_record({"scan_id": _value("scan-1")}),), profile)

	with output.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle))
	assert rows == [{"Scan ID": "scan-1", "Sample ID": ""}]
