from __future__ import annotations

import pytest

from mctutil.parse import import_pipeline
from mctutil.parse.import_records import ScanRecord, empty_scan_values


def _scan(scan_id: str, location: str) -> ScanRecord:
	values = empty_scan_values()
	values["scan_id"] = scan_id
	return ScanRecord("sigray", (location,), values)


def test_pipeline_dispatches_validates_and_sorts(monkeypatch):
	monkeypatch.setattr(
		import_pipeline,
		"reader_for",
		lambda kind, source: lambda: (_scan("second", "z"), _scan("first", "a")),
	)

	records = import_pipeline.read_records("scan", "sigray")

	assert [record.values["scan_id"] for record in records] == ["first", "second"]


def test_pipeline_rejects_missing_and_wrong_typed_canonical_values(monkeypatch):
	missing = _scan("scan", "scan.h5")
	del missing.values["project_id"]
	monkeypatch.setattr(import_pipeline, "reader_for", lambda *_args: lambda: (missing,))
	with pytest.raises(ValueError, match="field mismatch"):
		import_pipeline.read_records("scan", "sigray")

	wrong_type = _scan("scan", "scan.h5")
	wrong_type.values["projection_count_acquired"] = 2.0
	monkeypatch.setattr(import_pipeline, "reader_for", lambda *_args: lambda: (wrong_type,))
	with pytest.raises(TypeError, match="projection_count_acquired"):
		import_pipeline.read_records("scan", "sigray")


def test_dispatcher_has_only_the_current_explicit_sources():
	assert import_pipeline.SCAN_SOURCES == (
		"als832",
		"sigray",
		"aps-7bm",
		"chenglab-camera",
	)
	assert import_pipeline.RECONSTRUCTION_SOURCES == (
		"als832",
		"xaid",
		"tomocupy-7bm",
	)
	with pytest.raises(ValueError, match="unknown scan source"):
		import_pipeline.reader_for("scan", "auto")
