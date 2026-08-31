"""Versioned canonical export profiles over the existing tabular destinations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from mctutil.parse import tabular_log

from .model import CanonicalValue, ImportRecord, RecordKind
from .schema import (
	RECONSTRUCTION_SCHEMA_V1,
	SCAN_SCHEMA_V1,
	CanonicalSchema,
)


@dataclass(frozen=True)
class ExportColumn:
	field_key: str
	header: str

	def __post_init__(self) -> None:
		if not self.field_key.strip() or not self.header.strip():
			raise ValueError("export field key and header must be non-empty")


@dataclass(frozen=True)
class ExportProfile:
	"""A fixed tabular contract independent from schema version numbering."""

	profile_id: str
	header_version: str
	schema: CanonicalSchema
	columns: tuple[ExportColumn, ...]

	def __post_init__(self) -> None:
		if not self.profile_id.strip() or not self.header_version.strip():
			raise ValueError("profile_id and header_version must be non-empty")
		object.__setattr__(self, "columns", tuple(self.columns))
		keys = [column.field_key for column in self.columns]
		headers = [column.header for column in self.columns]
		if not keys:
			raise ValueError("an export profile cannot be empty")
		if len(keys) != len(set(keys)):
			raise ValueError("export profile has duplicate canonical keys")
		if len(headers) != len(set(headers)):
			raise ValueError("export profile has duplicate headers")
		for key in keys:
			try:
				self.schema.field(key)
			except KeyError as exc:
				raise ValueError(
					f"export profile names unknown canonical field: {key}"
				) from exc

	@property
	def kind(self) -> RecordKind:
		return self.schema.kind  # type: ignore[return-value]

	@property
	def schema_version(self) -> str:
		return self.schema.identifier

	@property
	def field_keys(self) -> tuple[str, ...]:
		return tuple(column.field_key for column in self.columns)

	@property
	def headers(self) -> tuple[str, ...]:
		return tuple(column.header for column in self.columns)


@dataclass(frozen=True)
class ExportProfileRegistry:
	profiles: tuple[ExportProfile, ...]

	def __post_init__(self) -> None:
		object.__setattr__(self, "profiles", tuple(self.profiles))
		keys = [(profile.kind, profile.profile_id) for profile in self.profiles]
		if len(keys) != len(set(keys)):
			raise ValueError("duplicate export profile registration")

	@property
	def by_key(self) -> Mapping[tuple[RecordKind, str], ExportProfile]:
		return MappingProxyType(
			{(profile.kind, profile.profile_id): profile for profile in self.profiles}
		)

	def get(self, kind: RecordKind, profile_id: str) -> ExportProfile:
		try:
			return self.by_key[(kind, profile_id)]
		except KeyError as exc:
			raise KeyError(f"unknown {kind} export profile: {profile_id}") from exc


def _stringify(value: object | None) -> str:
	if value is None:
		return ""
	if isinstance(value, bool):
		return "true" if value else "false"
	if isinstance(value, datetime):
		return value.isoformat()
	if isinstance(value, float):
		if not math.isfinite(value):
			raise ValueError("cannot export a non-finite canonical float")
		return str(value)
	if isinstance(value, (list, tuple)):
		return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
	return str(value)


def _record_value(record: ImportRecord, field_key: str) -> CanonicalValue | None:
	return record.values.get(field_key)


def materialize_row(
	record: ImportRecord, profile: ExportProfile
) -> Mapping[str, str]:
	"""Derive one immutable destination row exclusively from canonical values."""

	if record.kind != profile.kind:
		raise ValueError("record kind does not match export profile")
	if record.schema_version != profile.schema_version:
		raise ValueError("record schema version does not match export profile")
	row = {}
	for column in profile.columns:
		canonical_value = _record_value(record, column.field_key)
		row[column.header] = _stringify(
			None if canonical_value is None else canonical_value.value
		)
	return MappingProxyType(row)


def materialize_rows(
	records: Iterable[ImportRecord], profile: ExportProfile
) -> tuple[Mapping[str, str], ...]:
	return tuple(materialize_row(record, profile) for record in records)


def write_records_csv(
	output: Path,
	records: Iterable[ImportRecord],
	profile: ExportProfile,
	*,
	force: bool = False,
) -> None:
	rows = materialize_rows(records, profile)
	tabular_log.write_csv(output, profile.headers, rows, force=force)


def append_records(
	service,
	spreadsheet: str,
	sheet: str,
	records: Iterable[ImportRecord],
	profile: ExportProfile,
	*,
	verify: bool = True,
	strict_header_order: bool = False,
) -> dict:
	rows = materialize_rows(records, profile)
	return tabular_log.append_rows(
		service,
		spreadsheet,
		sheet,
		profile.headers,
		rows,
		verify=verify,
		strict_header_order=strict_header_order,
	)


def _profile(
	profile_id: str,
	header_version: str,
	schema: CanonicalSchema,
	field_keys: tuple[str, ...],
) -> ExportProfile:
	return ExportProfile(
		profile_id,
		header_version,
		schema,
		tuple(
			ExportColumn(field_key, schema.field(field_key).label)
			for field_key in field_keys
		),
	)


SCAN_PROFILE_V1 = _profile(
	"scan-v1",
	"scan-header/1",
	SCAN_SCHEMA_V1,
	tuple(field_spec.key for field_spec in SCAN_SCHEMA_V1.fields),
)
RECONSTRUCTION_PROFILE_V1 = _profile(
	"reconstruction-v1",
	"reconstruction-header/1",
	RECONSTRUCTION_SCHEMA_V1,
	tuple(field_spec.key for field_spec in RECONSTRUCTION_SCHEMA_V1.fields),
)
EXPORT_PROFILE_REGISTRY = ExportProfileRegistry(
	(SCAN_PROFILE_V1, RECONSTRUCTION_PROFILE_V1)
)
