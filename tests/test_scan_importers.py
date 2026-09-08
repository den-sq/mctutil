from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from mctutil.parse import scan_importers, sigray_scan_log


def _put(handle, path: str, value, **kwargs):
	parent, name = path.rsplit("/", 1)
	return handle.require_group(parent).create_dataset(name, data=value, **kwargs)


def test_als832_reads_exchange_data_angles_and_embedded_references(tmp_path):
	path = tmp_path / "als-scan.h5"
	with h5py.File(path, "w") as handle:
		_put(handle, "/exchange/data", np.zeros((3, 4, 5), dtype=np.uint16))
		_put(handle, "/exchange/theta", (0.0, 90.0, 180.0))
		_put(handle, "/exchange/data_white", np.zeros((2, 4, 5)))
		_put(handle, "/exchange/data_dark", np.zeros((1, 4, 5)))
		_put(handle, "/defaults/file_attrs/facility", np.bytes_("ALS"))
		_put(handle, "/measurement/sample/experiment/proposal", np.bytes_("P-1"))
		_put(handle, "/measurement/sample/file_name", np.bytes_("scan-1"))
		_put(handle, "/process/acquisition/name", np.bytes_("continuous"))
		_put(handle, "/process/acquisition/rotation/num_angles", 4)
		_put(handle, "/measurement/instrument/detector/exposure_time", 0.1)
		_put(handle, "/process/acquisition/flat_fields/flat_field_exposure", 0.2)
		_put(handle, "/process/acquisition/start_date", np.bytes_("2026-01-02T03:04:05Z"))

	record, = scan_importers.read_als832_scans((path,))
	values = record.values

	assert record.source_locations == (str(path),)
	assert values["scan_id"] == "scan-1"
	assert values["projection_count_planned"] == 4
	assert values["projection_count_acquired"] == 3
	assert values["rotation_stop_actual_deg"] == 180.0
	assert values["angular_step_deg"] == 90.0
	assert values["image_width_px"] == 5
	assert values["image_height_px"] == 4
	assert values["stored_dtype"] == "uint16"
	assert values["flat_frame_count"] == 2
	assert values["dark_frame_count"] == 1
	assert values["exposure_us"] == 100_000.0
	assert values["scan_start"].isoformat() == "2026-01-02T03:04:05+00:00"


def test_sigray_reader_reuses_existing_discovery_and_extraction(tmp_path, monkeypatch):
	path = tmp_path / "group_000.h5"
	path.touch()
	row = {field: "" for field in sigray_scan_log.SCAN_LOG_FIELDS}
	row.update(
		{
			"Scan": "group_000",
			"Projection Count": "3",
			"Rotational Start (°)": "0",
			"Rotational Stop (°)": "180",
			"Angular Step (°)": "60",
			"Width (pixels)": "5",
			"Height (pixels)": "4",
			"Stored Bit Depth": "16",
			"Metadata Warnings": "legacy warning",
		}
	)
	calls = []
	monkeypatch.setattr(
		sigray_scan_log,
		"discover_projection_files",
		lambda inputs: calls.append(("discover", inputs)) or (path,),
	)
	monkeypatch.setattr(
		sigray_scan_log,
		"extract_scan_rows",
		lambda paths: calls.append(("extract", paths)) or (row,),
	)
	monkeypatch.setattr(sigray_scan_log, "read_planned_projection_count", lambda _path: 4)

	record, = scan_importers.read_sigray_scans((tmp_path,))

	assert [call[0] for call in calls] == ["discover", "extract"]
	assert record.values["projection_count_planned"] == 4
	assert record.values["projection_count_acquired"] == 3
	assert record.values["rotation_stop_planned_deg"] == 180.0
	assert record.values["rotation_stop_actual_deg"] == 180.0
	assert record.warnings == ["legacy warning"]


def _put_epics_times(handle, times):
	seconds = np.floor(times).astype(np.uint32)
	nanoseconds = np.rint((np.asarray(times) - seconds) * 1_000_000_000).astype(np.uint32)
	_put(handle, "/defaults/NDArrayEpicsTSSec", seconds)
	_put(handle, "/defaults/NDArrayEpicsTSnSec", nanoseconds)


@pytest.mark.parametrize("pixel_unit", ("µm", "μm"))
def test_aps_7bm_populates_real_layout_metadata_from_one_route_authority(
	tmp_path,
	pixel_unit,
):
	path = tmp_path / "seven-bm.h5"
	with h5py.File(path, "w") as handle:
		_put(handle, "/exchange/data", np.zeros((4, 4, 5), dtype=np.uint16))
		_put(handle, "/exchange/theta", (0.0, 10.0, 20.0, 30.0))
		_put(handle, "/exchange/data_white", np.zeros((2, 4, 5)))
		_put(handle, "/exchange/data_dark", np.zeros((1, 4, 5)))
		_put(handle, "/exchange/data_gains", np.zeros((2, 4, 5)))
		routes = (
			"/exchange/data_dark",
			"/exchange/data_white",
			"/exchange/data_gains",
			"/exchange/data",
			"/exchange/data",
			"/exchange/data",
			"/exchange/data",
			"/exchange/data_white",
			"/exchange/data_gains",
		)
		_put(handle, "/defaults/HDF5FrameLocation", routes)
		_put(handle, "/defaults/NDArrayUniqueId", np.arange(10, 19))
		_put_epics_times(handle, (100, 100.01, 100.02, 100.1, 100.208, 100.316, 100.424, 100.5, 100.51))
		_put(handle, "/measurement/instrument/source/name", np.bytes_("Advanced Photon Source"))
		_put(handle, "/measurement/instrument/source/beamline", np.bytes_("7-BM"))
		_put(handle, "/measurement/sample/experiment/proposal", np.bytes_("P-1"))
		_put(handle, "/measurement/sample/name", np.bytes_(""))
		_put(handle, "/measurement/sample/file/name", np.bytes_("sample-group"))
		_put(handle, "/measurement/sample/experimenter/name", np.bytes_("Operator"))
		_put(handle, "/measurement/sample/description_1", np.bytes_(""))
		_put(handle, "/measurement/sample/description_2", np.bytes_("note one"))
		_put(handle, "/measurement/sample/description_3", np.bytes_("note two"))
		_put(handle, "/measurement/instrument/detector/exposure_time", 0.09)
		_put(handle, "/measurement/instrument/detector/binning_x", 2)
		_put(handle, "/measurement/instrument/detector/binning_y", 2)
		_put(handle, "/measurement/instrument/detector/gain", 0.0)
		_put(handle, "/measurement/instrument/detector/roi/min_x", 0)
		_put(handle, "/measurement/instrument/detector/roi/min_y", 0)
		_put(handle, "/measurement/instrument/detector/roi/size_x", 5)
		_put(handle, "/measurement/instrument/detector/roi/size_y", 4)
		_put(handle, "/measurement/instrument/detector/max_size_x", 5)
		_put(handle, "/measurement/instrument/detector/max_size_y", 4)
		_put(handle, "/measurement/instrument/sample_motor_stack/setup/x", 1.0)
		_put(handle, "/measurement/instrument/sample_motor_stack/setup/y", 2.0)
		_put(handle, "/measurement/instrument/sample_motor_stack/setup/z", 3.0)
		_put(handle, "/measurement/instrument/detector_motor_stack/setup/z", 325.0)
		_put(handle, "/measurement/instrument/sample_motor_stack/detector_distance", 325.0)
		pitch = _put(handle, "/measurement/instrument/detector/pixel_size", 6.9)
		pitch.attrs["units"] = pixel_unit
		resolution = _put(
			handle,
			"/measurement/instrument/detection_system/objective/resolution",
			2.0,
		)
		resolution.attrs["units"] = "um"
		_put(handle, "/process/acquisition/scan_type", np.bytes_("Single"))
		_put(handle, "/process/acquisition/rotation/num_angles", 4)
		_put(handle, "/process/acquisition/rotation/start", 0.0)
		_put(handle, "/process/acquisition/rotation/step", 10.0)
		_put(handle, "/process/acquisition/rotation/speed", 100.0)
		_put(handle, "/process/acquisition/pixels_y_per_360_deg", 0.0)
		_put(handle, "/process/acquisition/flat_fields/mode", np.bytes_("Both"))
		_put(handle, "/process/acquisition/flat_fields/number", 1)
		_put(handle, "/process/acquisition/flat_fields/flat_exposure_time", 0.09)
		_put(handle, "/process/acquisition/dark_fields/mode", np.bytes_("Start"))
		_put(handle, "/process/acquisition/dark_fields/number", 1)

	record, = scan_importers.read_aps_7bm_scans((path,))
	values = record.values

	assert values["facility_name"] == "Advanced Photon Source"
	assert values["acquisition_system_id"] == "7-BM"
	assert values["project_id"] == "P-1"
	assert values["sample_id"] is None
	assert values["acquisition_group"] == "sample-group"
	assert values["operator"] == "Operator"
	assert values["notes"] == "note one; note two"
	assert values["projection_count_acquired"] == 4
	assert values["rotation_start_deg"] == 0.0
	assert values["rotation_stop_planned_deg"] == 30.0
	assert values["rotation_stop_actual_deg"] == 30.0
	assert values["angular_step_deg"] == 10.0
	assert values["fov_count"] == 1
	assert values["helical"] is False
	assert values["exposure_us"] == 90_000.0
	assert values["flat_exposure_us"] == 90_000.0
	assert values["trigger_period_us"] == pytest.approx(108_000.0)
	assert values["trigger_overhead_us"] == pytest.approx(18_000.0)
	assert values["scan_start"].isoformat() == "1990-01-01T00:01:40+00:00"
	assert values["scan_stop"].isoformat() == "1990-01-01T00:01:40.510000+00:00"
	assert values["scan_duration_s"] == pytest.approx(0.51)
	assert values["binning_x"] == values["binning_y"] == 2
	assert values["crop_enabled"] is False
	assert values["crop_offset_x_px"] == values["crop_offset_y_px"] == 0
	assert values["detector_gain"] == 0.0
	assert values["sample_stage_x_mm"] == 1.0
	assert values["sample_stage_y_mm"] == 2.0
	assert values["sample_stage_z_mm"] == 3.0
	assert values["detector_stage_z_mm"] == 325.0
	assert values["sample_detector_distance_mm"] == 325.0
	assert values["flat_frame_count"] == 2
	assert values["flat_frame_count_pre"] == 1
	assert values["flat_frame_count_post"] == 1
	assert values["dark_frame_count"] == 1
	assert values["post_reference_files"] == (f"{path}:/exchange/data_white",)
	assert values["dropped_frames"] == 0
	assert values["detector_pixel_pitch_mm"] == pytest.approx(0.0069)
	assert values["effective_pixel_size_mm"] == pytest.approx(0.002)
	assert values["physical_fov_width_mm"] == pytest.approx(0.01)
	assert values["physical_fov_height_mm"] == pytest.approx(0.008)
	assert values["projection_frame_size_mb"] == 0.00004
	assert values["source_energy_kev"] is None
	assert values["source_power"] is None
	assert values["source_target"] is None
	assert values["acquisition_status"] == "Complete"
	assert record.warnings == []


def test_aps_7bm_partial_scan_reports_routed_count_and_uid_gap(tmp_path):
	path = tmp_path / "seven-bm-partial.h5"
	with h5py.File(path, "w") as handle:
		_put(handle, "/exchange/data", np.zeros((3, 4, 5), dtype=np.uint16))
		_put(handle, "/exchange/theta", (0.0, 1.0, 2.0))
		_put(handle, "/defaults/HDF5FrameLocation", ("/exchange/data",) * 3)
		_put(handle, "/defaults/NDArrayUniqueId", (10, 12, 13))
		_put_epics_times(handle, (100.0, 100.1, 100.2))
		_put(handle, "/process/acquisition/rotation/num_angles", 4)

	record, = scan_importers.read_aps_7bm_scans((path,))

	assert record.values["projection_count_acquired"] == 3
	assert record.values["dropped_frames"] == 1
	assert record.values["incomplete_frames"] is None
	assert record.values["acquisition_status"] == "Warnings"
	assert "detector unique IDs indicate 1 dropped projection frame(s)" in record.warnings
	assert "planned projection count 4 does not match routed acquisition count 3" in record.warnings


def test_aps_7bm_zero_route_placeholder_overrides_dataset_shapes(tmp_path):
	path = tmp_path / "seven-bm-placeholder.h5"
	with h5py.File(path, "w") as handle:
		_put(handle, "/exchange/data", np.zeros((1, 4, 5), dtype=np.uint16))
		_put(handle, "/exchange/data_white", np.zeros((1, 4, 5), dtype=np.uint16))
		_put(handle, "/exchange/data_dark", np.zeros((1, 4, 5), dtype=np.uint16))
		_put(handle, "/defaults/HDF5FrameLocation", ("/exchange/data_dark",))
		_put(handle, "/defaults/NDArrayUniqueId", (10,))
		_put_epics_times(handle, (100.0,))
		_put(handle, "/process/acquisition/rotation/num_angles", 1)
		_put(handle, "/process/acquisition/flat_fields/mode", np.bytes_("Both"))
		_put(handle, "/process/acquisition/flat_fields/number", 1)

	record, = scan_importers.read_aps_7bm_scans((path,))

	assert record.values["projection_count_acquired"] == 0
	assert record.values["flat_frame_count"] == 0
	assert record.values["dark_frame_count"] == 1
	assert record.values["flat_reference_files"] is None
	assert record.values["incomplete_frames"] is None
	assert record.values["acquisition_status"] == "Warnings"
	assert any("routes 0 frames to /exchange/data but the dataset stores 1" in item for item in record.warnings)
	assert any("routes 0 frames to /exchange/data_white but the dataset stores 1" in item for item in record.warnings)
	assert "missing /exchange/theta" in record.warnings
	assert "configured flat fields expect 1 pre/1 post frames but routes contain 0 pre/0 post" in record.warnings


def test_aps_7bm_pre_only_crop_and_configured_timing_fallback(tmp_path):
	path = tmp_path / "seven-bm-pre-only.h5"
	with h5py.File(path, "w") as handle:
		_put(handle, "/exchange/data", np.zeros((2, 4, 5), dtype=np.uint16))
		_put(handle, "/exchange/theta", (0.0, 1.0))
		_put(handle, "/exchange/data_white", np.zeros((1, 4, 5), dtype=np.uint16))
		_put(
			handle,
			"/defaults/HDF5FrameLocation",
			("/exchange/data_white", "/exchange/data", "/exchange/data"),
		)
		_put(handle, "/defaults/NDArrayUniqueId", (10, 11, 12))
		_put(handle, "/process/acquisition/start_date", np.bytes_("June 18, 2026 19:00:00"))
		_put(handle, "/process/acquisition/end_date", np.bytes_("June 18, 2026 19:01:00"))
		_put(handle, "/process/acquisition/rotation/start", 0.0)
		_put(handle, "/process/acquisition/rotation/step", 1.0)
		_put(handle, "/process/acquisition/rotation/speed", 10.0)
		_put(handle, "/process/acquisition/rotation/num_angles", 2)
		_put(handle, "/process/acquisition/pixels_y_per_360_deg", 2.0)
		_put(handle, "/process/acquisition/flat_fields/mode", np.bytes_("Start"))
		_put(handle, "/process/acquisition/flat_fields/number", 1)
		_put(handle, "/measurement/instrument/detector/exposure_time", 0.09)
		_put(handle, "/measurement/instrument/detector/roi/min_x", 1)
		_put(handle, "/measurement/instrument/detector/roi/min_y", 2)
		_put(handle, "/measurement/instrument/detector/roi/size_x", 5)
		_put(handle, "/measurement/instrument/detector/roi/size_y", 4)
		_put(handle, "/measurement/instrument/detector/max_size_x", 8)
		_put(handle, "/measurement/instrument/detector/max_size_y", 8)

	record, = scan_importers.read_aps_7bm_scans((path,))
	values = record.values

	assert values["flat_frame_count_pre"] == 1
	assert values["flat_frame_count_post"] == 0
	assert values["post_reference_files"] is None
	assert values["crop_enabled"] is True
	assert values["crop_offset_x_px"] == 1
	assert values["crop_offset_y_px"] == 2
	assert values["helical"] is True
	assert values["fov_count"] is None
	assert values["scan_start"].isoformat() == "2026-06-19T00:00:00+00:00"
	assert values["scan_stop"].isoformat() == "2026-06-19T00:01:00+00:00"
	assert values["scan_duration_s"] == 60.0
	assert values["trigger_period_us"] == pytest.approx(100_000.0)
	assert values["trigger_overhead_us"] == pytest.approx(10_000.0)
	assert "EPICS frame timestamps are missing" in record.warnings
	assert "process acquisition timestamps are timezone-less; inferred America/Chicago" in record.warnings
	assert any("trigger period uses configured rotation step/speed" in item for item in record.warnings)


class _Request:
	def __init__(self, response):
		self.response = response

	def execute(self):
		return self.response


class _ReadonlyValues:
	def __init__(self, rows):
		self.rows = rows
		self.get_calls = []

	def get(self, **kwargs):
		self.get_calls.append(kwargs)
		return _Request({"values": self.rows})


class _ReadonlyService:
	def __init__(self, rows):
		self.values_service = _ReadonlyValues(rows)

	def values(self):
		return self.values_service


def test_chenglab_camera_reads_real_rows_with_only_values_get():
	assert scan_importers.CHENGLAB_READONLY_SCOPES == (
		"https://www.googleapis.com/auth/spreadsheets.readonly",
	)
	headers = [
		"Scan ID",
		"Projection Total",
		"Projection Total Result",
		"Exposure (us)",
		"Trigger (us)",
		"Pre Gains",
		"Post Gains",
		"Scan Start",
		"Scan Stop",
		"Resolution",
	]
	rows = [
		headers,
		["scan-1", 101, 99, 50, 75, 2, 3, 45_000, 45_000 + 1 / 1440, "2 um"],
		["", 100],
	]
	service = _ReadonlyService(rows)

	record, = scan_importers.read_chenglab_camera_scans(
		"sheet-id",
		service=service,
	)
	values = record.values

	assert record.source_locations == ("gsheets://sheet-id/ScanLog!2",)
	assert values["projection_count_planned"] == 101
	assert values["projection_count_acquired"] == 99
	assert values["dropped_frames"] == 2
	assert values["trigger_overhead_us"] == 25.0
	assert values["flat_frame_count"] == 5
	assert values["scan_duration_s"] == pytest.approx(60.0)
	assert values["scan_start"] is None
	assert values["effective_pixel_size_mm"] is None
	assert any("Resolution" in warning for warning in record.warnings)
	assert service.values_service.get_calls == [
		{
			"spreadsheetId": "sheet-id",
			"range": "'ScanLog'!A:BW",
			"valueRenderOption": "UNFORMATTED_VALUE",
			"dateTimeRenderOption": "FORMATTED_STRING",
		}
	]


def test_chenglab_auth_rejects_broad_legacy_token_and_creates_readonly_token(
	tmp_path,
	monkeypatch,
):
	class RefreshError(Exception):
		pass

	class LegacyCredentials:
		valid = True

		def refresh(self, _request):
			raise RefreshError

	class ReadonlyCredentials:
		valid = True
		granted_scopes = scan_importers.CHENGLAB_READONLY_SCOPES
		scopes = scan_importers.CHENGLAB_READONLY_SCOPES

		def to_json(self):
			return "{}"

	requested_scopes = []
	credentials_module = SimpleNamespace(
		Credentials=SimpleNamespace(
			from_authorized_user_file=lambda _path, scopes: (
				requested_scopes.append(tuple(scopes)) or LegacyCredentials()
			)
		)
	)
	flow_module = SimpleNamespace(
		InstalledAppFlow=SimpleNamespace(
			from_client_secrets_file=lambda _path, scopes: (
				requested_scopes.append(tuple(scopes))
				or SimpleNamespace(run_local_server=lambda **_kwargs: ReadonlyCredentials())
			)
		)
	)
	api = object()
	modules = (
		SimpleNamespace(RefreshError=RefreshError),
		SimpleNamespace(Request=object),
		credentials_module,
		flow_module,
		SimpleNamespace(
			build=lambda *_args, **_kwargs: SimpleNamespace(spreadsheets=lambda: api)
		),
	)
	monkeypatch.setattr(scan_importers, "require", lambda *_args, **_kwargs: modules)
	(tmp_path / "gsheets_token.json").write_text("{}", encoding="utf-8")
	(tmp_path / "gsheets_credentials.json").write_text("{}", encoding="utf-8")

	service = scan_importers.build_chenglab_sheets_service(tmp_path)

	assert service is api
	assert requested_scopes == [
		scan_importers.CHENGLAB_READONLY_SCOPES,
		scan_importers.CHENGLAB_READONLY_SCOPES,
	]
	assert (tmp_path / "gsheets_readonly_token.json").read_text(encoding="utf-8") == "{}"


def test_chenglab_scope_check_rejects_credentials_with_extra_authority():
	readonly = scan_importers.CHENGLAB_READONLY_SCOPES[0]
	assert scan_importers._has_exact_readonly_scope(
		SimpleNamespace(granted_scopes=(readonly,), scopes=None)
	)
	assert not scan_importers._has_exact_readonly_scope(
		SimpleNamespace(
			granted_scopes=(readonly, "https://www.googleapis.com/auth/spreadsheets"),
			scopes=None,
		)
	)
