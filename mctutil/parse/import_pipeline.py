"""Minimal dispatch and validation for canonical scan/reconstruction readers."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from mctutil.parse.import_records import (
	RECONSTRUCTION_FIELDS,
	SCAN_FIELDS,
	ReconstructionRecord,
	ScanRecord,
)


SCAN_SOURCES = ("als832", "sigray", "aps-7bm", "chenglab-camera")
RECONSTRUCTION_SOURCES = ("als832", "xaid", "tomocupy-7bm")

Record = ScanRecord | ReconstructionRecord
Reader = Callable[..., Iterable[Record]]


def reader_for(kind: str, source: str) -> Reader:
	"""Return the ordinary reader function for one explicit kind/source pair."""
	if kind == "scan":
		from mctutil.parse import scan_importers

		if source == "als832":
			return scan_importers.read_als832_scans
		if source == "sigray":
			return scan_importers.read_sigray_scans
		if source == "aps-7bm":
			return scan_importers.read_aps_7bm_scans
		if source == "chenglab-camera":
			return scan_importers.read_chenglab_camera_scans
	elif kind == "reconstruction":
		from mctutil.parse import reconstruction_importers

		if source == "als832":
			return reconstruction_importers.read_als832_reconstructions
		if source == "xaid":
			return reconstruction_importers.read_xaid_reconstructions
		if source == "tomocupy-7bm":
			return reconstruction_importers.read_tomocupy_7bm_reconstructions
	else:
		raise ValueError(f"unknown record kind: {kind!r}")
	raise ValueError(f"unknown {kind} source: {source!r}")


def _validate_record(kind: str, source: str, record: Record) -> None:
	fields = SCAN_FIELDS if kind == "scan" else RECONSTRUCTION_FIELDS
	record_type = ScanRecord if kind == "scan" else ReconstructionRecord
	if type(record) is not record_type:
		raise TypeError(f"{source} returned {type(record).__name__}, expected {record_type.__name__}")
	if record.source_id != source:
		raise ValueError(
			f"reader {source!r} returned source_id {record.source_id!r}"
		)
	if not record.source_locations or not all(record.source_locations):
		raise ValueError(f"{source} record has no source location")

	expected = {item.name for item in fields}
	actual = set(record.values)
	if actual != expected:
		missing = sorted(expected - actual)
		extra = sorted(actual - expected)
		raise ValueError(
			f"{source} record field mismatch: missing={missing!r}, extra={extra!r}"
		)

	for item in fields:
		value = record.values[item.name]
		if value is None:
			if item.required:
				raise ValueError(f"{source} record is missing required {item.name!r}")
			continue
		if type(value) is not item.value_type:
			raise TypeError(
				f"{source} field {item.name!r} must be {item.value_type.__name__}; "
				f"got {type(value).__name__}"
			)


def read_records(kind: str, source: str, *args, **kwargs) -> tuple[Record, ...]:
	"""Dispatch, validate canonical values, and return deterministic records."""
	reader = reader_for(kind, source)
	records = tuple(reader(*args, **kwargs))
	for record in records:
		_validate_record(kind, source, record)
	return tuple(sorted(records, key=lambda item: item.source_locations))
