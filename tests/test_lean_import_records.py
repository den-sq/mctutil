from __future__ import annotations

from datetime import datetime
from numbers import Real

from mctutil.parse.import_mappings import (
	ALS832_RECONSTRUCTION_MAPPING,
	ALS832_SCAN_MAPPING,
	CHENGLAB_CAMERA_SCAN_MAPPING,
	SEVEN_BM_RECONSTRUCTION_MAPPING,
	SEVEN_BM_SCAN_MAPPING,
	SIGRAY_SCAN_MAPPING,
	XAID_RECONSTRUCTION_MAPPING,
)
from mctutil.parse.import_records import (
	RECONSTRUCTION_FIELDS,
	RECONSTRUCTION_HEADERS,
	SCAN_FIELDS,
	SCAN_HEADERS,
	WARNING_HEADER,
	ReconstructionRecord,
	ScanRecord,
)


def _by_name(fields):
	return {item.name: item for item in fields}


def test_field_definitions_are_ordered_unique_and_own_headers():
	assert len(SCAN_FIELDS) == 70
	assert len(RECONSTRUCTION_FIELDS) == 74

	for fields, headers in (
		(SCAN_FIELDS, SCAN_HEADERS),
		(RECONSTRUCTION_FIELDS, RECONSTRUCTION_HEADERS),
	):
		names = [item.name for item in fields]
		field_headers = [item.header for item in fields]

		assert len(names) == len(set(names))
		assert len(field_headers) == len(set(field_headers))
		assert headers == tuple(field_headers) + (WARNING_HEADER,)
		assert all(item.value_type in {str, int, float, bool, datetime, tuple} for item in fields)

	assert _by_name(SCAN_FIELDS)["scan_id"].required
	assert _by_name(RECONSTRUCTION_FIELDS)["reconstruction_id"].required


def test_shared_fields_have_the_same_type_and_unit():
	scan = _by_name(SCAN_FIELDS)
	reconstruction = _by_name(RECONSTRUCTION_FIELDS)

	for name in set(scan) & set(reconstruction):
		assert reconstruction[name].value_type is scan[name].value_type
		assert reconstruction[name].unit == scan[name].unit


def test_planned_and_actual_scan_facts_are_independent_fields():
	fields = _by_name(SCAN_FIELDS)

	assert fields["projection_count_planned"] is not fields["projection_count_acquired"]
	assert fields["rotation_stop_planned_deg"] is not fields["rotation_stop_actual_deg"]
	assert fields["sample_stage_y_stop_planned_mm"] is not fields["sample_stage_y_stop_actual_mm"]


def test_records_are_plain_typed_containers_and_missing_is_none():
	scan_values = {item.name: None for item in SCAN_FIELDS}
	scan_values["scan_id"] = "scan-1"
	reconstruction_values = {item.name: None for item in RECONSTRUCTION_FIELDS}
	reconstruction_values["reconstruction_id"] = "recon-1"

	scan = ScanRecord("sigray", ("scan.h5", "scan_FLAT_1.h5"), scan_values)
	reconstruction = ReconstructionRecord(
		"seven-bm",
		("rec_line.txt", "recon_params.json", "rot_cen.json"),
		reconstruction_values,
	)

	assert scan.values["project_id"] is None
	assert scan.source_locations == ("scan.h5", "scan_FLAT_1.h5")
	assert reconstruction.values["output_path"] is None
	assert reconstruction.source_locations == (
		"rec_line.txt",
		"recon_params.json",
		"rot_cen.json",
	)
	assert scan.warnings == []
	assert reconstruction.warnings == []

	scan.warnings.append("scan warning")
	assert reconstruction.warnings == []


def test_direct_mapping_tables_have_only_the_allowed_shape():
	scan_fields = _by_name(SCAN_FIELDS)
	reconstruction_fields = _by_name(RECONSTRUCTION_FIELDS)
	tables = (
		(ALS832_SCAN_MAPPING, scan_fields),
		(SIGRAY_SCAN_MAPPING, scan_fields),
		(SEVEN_BM_SCAN_MAPPING, scan_fields),
		(CHENGLAB_CAMERA_SCAN_MAPPING, scan_fields),
		(ALS832_RECONSTRUCTION_MAPPING, reconstruction_fields),
		(XAID_RECONSTRUCTION_MAPPING, reconstruction_fields),
		(SEVEN_BM_RECONSTRUCTION_MAPPING, reconstruction_fields),
	)
	allowed = {"locator", "source_unit", "canonical_unit", "multiplier"}

	for table, canonical_fields in tables:
		assert table
		assert set(table) <= set(canonical_fields)
		for field_name, entry in table.items():
			assert set(entry) <= allowed
			assert "locator" in entry
			assert isinstance(entry["locator"], (str, tuple))
			if isinstance(entry["locator"], tuple):
				assert len(entry["locator"]) == 2
				assert all(isinstance(part, str) and part for part in entry["locator"])
			if "canonical_unit" in entry:
				assert entry["canonical_unit"] == canonical_fields[field_name].unit
			assert ("source_unit" in entry) == ("canonical_unit" in entry)
			if "multiplier" in entry:
				assert isinstance(entry["multiplier"], Real)
				assert not isinstance(entry["multiplier"], bool)
				assert "source_unit" in entry
				assert "canonical_unit" in entry


def test_mappings_do_not_encode_source_logic_or_unknown_camera_units():
	for forbidden in {
		"condition",
		"expression",
		"formula",
		"precedence",
		"reader",
		"selector",
		"transform",
	}:
		assert all(forbidden not in entry for entry in CHENGLAB_CAMERA_SCAN_MAPPING.values())

	assert "effective_pixel_size_mm" not in CHENGLAB_CAMERA_SCAN_MAPPING
	assert "crop_offset_x_px" not in CHENGLAB_CAMERA_SCAN_MAPPING
	assert "crop_offset_y_px" not in CHENGLAB_CAMERA_SCAN_MAPPING
	assert "sample_detector_distance_mm" not in CHENGLAB_CAMERA_SCAN_MAPPING


def test_direct_mappings_preserve_reviewed_locators_and_unit_conversions():
	assert ALS832_SCAN_MAPPING["exposure_us"] == {
		"locator": "/measurement/instrument/detector/exposure_time",
		"source_unit": "s",
		"canonical_unit": "us",
		"multiplier": 1_000_000.0,
	}
	assert SIGRAY_SCAN_MAPPING["detector_pixel_pitch_mm"] == {
		"locator": "/measurement/instrument/detector/physical_pixel_size",
		"source_unit": "um",
		"canonical_unit": "mm",
		"multiplier": 0.001,
	}
	assert CHENGLAB_CAMERA_SCAN_MAPPING["projection_count_planned"]["locator"] == (
		"Projection Total"
	)
	assert CHENGLAB_CAMERA_SCAN_MAPPING["projection_count_acquired"]["locator"] == (
		"Projection Total Result"
	)
	assert XAID_RECONSTRUCTION_MAPPING["software_version"]["locator"] == (
		"General_Info",
		"software_version",
	)
	assert SEVEN_BM_RECONSTRUCTION_MAPPING["rotation_axis_coordinate_px"][
		"locator"
	] == "--rotation-axis"
