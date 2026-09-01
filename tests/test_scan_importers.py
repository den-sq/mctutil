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


def test_aps_7bm_filters_frames_before_angles_and_integrity(tmp_path):
	path = tmp_path / "seven-bm.h5"
	with h5py.File(path, "w") as handle:
		_put(handle, "/exchange/data", np.zeros((4, 4, 5), dtype=np.uint16))
		_put(handle, "/exchange/theta", (0.0, 10.0, 20.0, 30.0))
		_put(handle, "/exchange/HDF5FrameLocation", (1, 0, 1, 1))
		_put(handle, "/measurement/defaults/NDArrayUniqueId", (10, 11, 13, 15))
		pitch = _put(handle, "/measurement/instrument/detector/physical_pixel_size", 6.5)
		pitch.attrs["units"] = "um"
		resolution = _put(handle, "/measurement/instrument/objective/resolution", 2.0)
		resolution.attrs["units"] = "um"
		_put(handle, "/process/acquisition/scan_type", np.bytes_("fly"))
		_put(handle, "/process/acquisition/rotation/num_angles", 4)
		_put(handle, "/exchange/data_white", np.zeros((2, 4, 5)))
		_put(handle, "/exchange/data_dark", np.zeros((1, 4, 5)))

	record, = scan_importers.read_aps_7bm_scans((path,))
	values = record.values

	assert values["projection_count_acquired"] == 3
	assert values["rotation_start_deg"] == 0.0
	assert values["rotation_stop_actual_deg"] == 30.0
	assert values["angular_step_deg"] == 15.0
	assert values["dropped_frames"] == 3
	assert values["detector_pixel_pitch_mm"] == pytest.approx(0.0065)
	assert values["effective_pixel_size_mm"] == pytest.approx(0.002)
	assert values["physical_fov_width_mm"] == pytest.approx(0.01)
	assert values["physical_fov_height_mm"] == pytest.approx(0.008)


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
