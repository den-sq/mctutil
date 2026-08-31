from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mctutil.parse.importers.model import (
	CanonicalValue,
	FieldSpec,
	FieldType,
	ImportRecord,
	SourceEvidence,
)
from mctutil.parse.importers.schema import (
	SCAN_SCHEMA_V1,
	ScanSchema,
	SchemaRegistry,
	SchemaVersion,
	evolve_minor,
	promote_extension,
)


def _evidence(raw_value: object = "scan-1") -> SourceEvidence:
	return SourceEvidence(
		artifact=Path("scan.h5"),
		artifact_role="primary",
		locator="/exchange/scan_id",
		raw_value=raw_value,
		source_unit=None,
		state="observed",
	)


def _value(value: object) -> CanonicalValue:
	return CanonicalValue(value, "observed", (_evidence(value),))


def _spec(
	key: str = "field_name",
	value_type: FieldType = FieldType.STRING,
	*,
	unit: str | None = None,
	nullable: bool = True,
) -> FieldSpec:
	return FieldSpec(key, value_type, unit, nullable, "Field", "A test field.")


def test_schema_rejects_duplicate_and_unknown_keys() -> None:
	duplicate = _spec()
	with pytest.raises(ValueError, match="duplicate canonical"):
		ScanSchema(SchemaVersion.parse("scan/1.0"), (duplicate, duplicate))

	with pytest.raises(KeyError, match="unknown canonical field"):
		SCAN_SCHEMA_V1.field("not_a_field")


@pytest.mark.parametrize(
	("spec", "value", "error"),
	[
		(_spec(value_type=FieldType.INTEGER), True, TypeError),
		(_spec(value_type=FieldType.FLOAT), 1, TypeError),
		(_spec(value_type=FieldType.FLOAT), float("inf"), ValueError),
		(_spec(value_type=FieldType.DATETIME), datetime(2026, 1, 1), ValueError),
		(_spec(nullable=False), None, ValueError),
		(
			FieldSpec(
				"pair",
				FieldType.FLOAT_TUPLE,
				"px",
				False,
				"Pair",
				"Two finite coordinates.",
				min_items=2,
				max_items=2,
			),
			(1.0,),
			ValueError,
		),
	],
)
def test_field_boundary_enforces_type_null_finite_and_timezone(
	spec: FieldSpec, value: object, error: type[Exception]
) -> None:
	with pytest.raises(error):
		spec.validate(value)

	aware = _spec(value_type=FieldType.DATETIME)
	aware.validate(datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_field_unit_is_declared_and_immutable() -> None:
	spec = _spec(value_type=FieldType.FLOAT, unit="um")
	assert spec.unit == "um"
	with pytest.raises(FrozenInstanceError):
		spec.unit = "mm"  # type: ignore[misc]
	with pytest.raises(ValueError, match="unit"):
		_spec(unit="")


def test_record_owns_immutable_canonical_mappings() -> None:
	source_values = {"scan_id": _value("scan-1")}
	record = ImportRecord("scan", "scan/1.0", "fixture", source_values)
	source_values["sample_id"] = _value("sample")

	assert tuple(record.values) == ("scan_id",)
	with pytest.raises(TypeError):
		record.values["sample_id"] = _value("sample")  # type: ignore[index]
	SCAN_SCHEMA_V1.validate_record(record)


def test_null_canonical_value_uses_missing_state() -> None:
	assert CanonicalValue(None, "missing").value is None
	with pytest.raises(ValueError, match="null canonical"):
		CanonicalValue(None, "observed")
	with pytest.raises(ValueError, match="requires source evidence"):
		CanonicalValue("value", "observed")


def test_extension_promotion_is_one_minor_alias_and_history_stays_valid() -> None:
	old_schema = ScanSchema(
		SchemaVersion.parse("scan/1.0"),
		(_spec("scan_id", nullable=False),),
	)
	promoted_field = _spec(
		"capture_delay_us", FieldType.FLOAT, unit="us", nullable=True
	)
	promoted = promote_extension(
		old_schema,
		extension_key="chenglab_camera.capture_delay_raw",
		field_spec=promoted_field,
	)
	next_schema = evolve_minor(promoted)
	registry = SchemaRegistry("scan", (old_schema, promoted, next_schema))

	assert promoted.identifier == "scan/1.1"
	assert (
		promoted.resolve_key("chenglab_camera.capture_delay_raw")
		== "capture_delay_us"
	)
	with pytest.raises(KeyError):
		next_schema.resolve_key("chenglab_camera.capture_delay_raw")

	# A historical audit-like value mapping is always interpreted using its
	# recorded version; registering a promotion never rewrites it.
	registry.validate_versioned_values("scan/1.0", {"scan_id": _value("old")})
