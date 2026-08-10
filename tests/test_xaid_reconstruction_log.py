from __future__ import annotations

import csv
from pathlib import Path

from click.testing import CliRunner
import pytest


SOURCE_PATH = (
	"Z:/DemoScans/PSU/20260515_PSU-WaterFlee-319-40-120-UV-MX4145/"
	"20260515_PSU-WaterFlee-319-40-120-UV-MX4145_000.h5"
)


XAID_CONFIG = f"""a[General_Info]
software_version = 2026.5.1
path_to_data = {SOURCE_PATH}

[Detektor_Info]
pixel_size = 0.009130000000000001
num_px_u = 4096
num_px_v = 4096

[ROI]
roi_posx = 0
roi_posy = 0
roi_posz = 800
roi_sizex = 4095
roi_sizey = 4095
roi_sizez = 2463

[VolumeData]
volume_rotation_y = 0.0
volume_rotation_x = 0.0
volume_rotation_z = 0.0
voxel_size = 0.0004652572631042871
nat_vx_size = 0.000465257263104

[Proj_Filters]
median_filter_size = 2
gauss_filter_size = 0.5
outlier_size = 3
outlier_delta = 0.01

[Ringfilter_Settings]
ring_partial_delta = 0.0
ring_partial_size = 2
ring_a_size = 2
ring_b_size = 0

[Reconstruction_Settings]
sod = 5.640559192545288
sdd = 110.68979560880115
freeray = None
reco_type = FDK
reg_type = None
iterations = 0
reg_threshold = 0.0
reg_lambda = 0.0
apply_roundmask = 1
fdk_filter = hamming
roi_filter = 1
is_short_scan = 0
is_offset_scan = 0
is_detector_offset = 0
offset_shift = 0
img_binning = 1.000000000000617
gui_img_binning = 1
is_compute_opt_oblique_slice_y = 0
is_opt_oblique_geom = 0
is_opt_proj_offset_v = 0
is_opt_proj_slant_v = 0
is_opt_oblique_angle = 0
is_opt_stage_shifts_triangle = 0
is_opt_stage_shifts = 0
is_opt_proj_shifts_2 = 0
is_align_axis = 0
is_opt_triangle = 0
is_opt_jitter = 0
is_jitter_smooth = 1
jitter_iterations = 3
shift_u_max = 10
shift_v_max = 10
oblique_opt_slice_y = 0

[BHC_Values]
bhc_cupping_value = 0.0
bhc_streak_value = 0.0
bhc_streak_thl_value = 0.0
starvation_value = 0.0
scatter_basic_value = 0.0
dual_energy_factor = 0.0

[GC_Values]
rotation_axis_offset = [-12.218087352435388]
cs_y_slice = 2048
rotation_axis_tilt = 0.0
tilt_y_slice_min = None
tilt_y_slice_max = None
drift_x = 0.0
drift_y = 0.0
drift_z = 0.0

[TScan]
is_opt_tscan = 0
tscan_max_shift = 140

[Global_Reco_Settings]
gcf = None
rotation_axis_tilt = 0.0
rotation_axis_slant = 0.0
det_rot_eta = 0.0
det_rot_theta = 0.0
det_rot_phi = 0.0
det_offset_u = 0.0
det_offset_v = 0.0

[Post_Filters]

[Final_Image_Settings]
save_path = Z:\\DemoScans\\PSU\\Tail_%04d.tif
min = -1.1867414
max = 7.5952454
export_type = tif16
export_order = xzy
recolocation = Z:\\DemoScans\\PSU\\Tail_%04d.tif
location_string = |w!|u16|xzy|scaleMinMax
"""


def _config_file(tmp_path: Path) -> Path:
	path = tmp_path / "config.txt"
	path.write_text(XAID_CONFIG, encoding="utf-8")
	return path


class FakeRequest:
	def __init__(self, response):
		self.response = response

	def execute(self):
		return self.response


class FakeValuesService:
	def __init__(self, header=None):
		self.header = list(header or [])
		self.get_calls = []
		self.update_calls = []
		self.append_calls = []

	def get(self, **kwargs):
		self.get_calls.append(kwargs)
		values = [self.header] if self.header else []
		return FakeRequest({"values": values})

	def update(self, **kwargs):
		self.update_calls.append(kwargs)
		self.header = list(kwargs["body"]["values"][0])
		return FakeRequest({"updatedRows": 1})

	def append(self, **kwargs):
		self.append_calls.append(kwargs)
		return FakeRequest(
			{
				"updates": {
					"updatedRange": "Reconstructions!A24:CJ24",
					"updatedRows": 1,
				}
			}
		)


class FakeSheetsService:
	def __init__(self, header=None, sheet_titles=()):
		self.values_service = FakeValuesService(header)
		self.sheet_titles = list(sheet_titles)
		self.get_calls = []
		self.batch_update_calls = []

	def values(self):
		return self.values_service

	def get(self, **kwargs):
		self.get_calls.append(kwargs)
		return FakeRequest(
			{
				"sheets": [
					{"properties": {"title": title}}
					for title in self.sheet_titles
				]
			}
		)

	def batchUpdate(self, **kwargs):
		self.batch_update_calls.append(kwargs)
		properties = kwargs["body"]["requests"][0]["addSheet"]["properties"]
		self.sheet_titles.append(properties["title"])
		return FakeRequest({"replies": [{"addSheet": {"properties": properties}}]})


def test_build_row_maps_xaid_reconstruction_fields(load_module, tmp_path):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	parsed = module.parse_xaid_config(_config_file(tmp_path))
	row = module.build_reconstruction_log_row(parsed.config)

	assert parsed.repaired_first_header is True
	assert tuple(row) == module.RECONSTRUCTION_LOG_FIELDS
	assert len(row) == 88
	assert row["Software Release Version"] == "2026.5.1"
	assert row["Input Projection Data File"] == SOURCE_PATH
	assert row["Detector Pixel Pitch (mm)"] == "0.009130000000000001"
	assert row["Sub-Volume Start, Z"] == "800"
	assert row["Native Voxel Size at Magnification (mm)"] == "0.000465257263104"
	assert row["Projection Gaussian Smoothing Sigma (px)"] == "0.5"
	assert row["Partial-Ring Filter Strength"] == "0.0"
	assert row["Air-Normalization Region"] == "None"
	assert row["Reconstruction Algorithm"] == "FDK"
	assert row["Auto-Optimize: Per-Projection Shifts (Variant 2) ⚠"] == "0"
	assert row["Beam-Hardening Cupping Correction Strength"] == "0.0"
	assert row["Center-of-Rotation Shift (px)"] == "[-12.218087352435388]"
	assert row["Translation-Scan Max Shift Search (px)"] == "140"
	assert row["Detector Rotation η (°)"] == "0.0"
	assert row["Internal Export Descriptor ⚠"] == "|w!|u16|xzy|scaleMinMax"


def test_mapping_field_names_and_sheet_range_are_complete(load_module):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")

	assert len(module.XAID_CONFIG_FIELD_MAPPING) == 88
	assert len(set(module.RECONSTRUCTION_LOG_FIELDS)) == 88
	assert module.RECONSTRUCTION_LOG_FIELDS[:3] == (
		"Software Release Version",
		"Input Projection Data File",
		"Detector Pixel Pitch (mm)",
	)
	assert module.RECONSTRUCTION_LOG_FIELDS[-3:] == (
		"Output Axis Ordering",
		"Reconstruction Output Location",
		"Internal Export Descriptor ⚠",
	)
	assert module.RECONSTRUCTION_LOG_LAST_COLUMN == "CJ"
	assert module.RECONSTRUCTION_LOG_COLUMN_RANGE == "A:CJ"
	assert module.RECONSTRUCTION_LOG_HEADER_RANGE == "A1:CJ1"


def test_missing_mapped_config_fields_are_blank(load_module, tmp_path):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	path = tmp_path / "minimal-config.txt"
	path.write_text("[General_Info]\nsoftware_version = 2026.5.1\n", encoding="utf-8")

	config = module.parse_xaid_config(path).config
	row = module.build_reconstruction_log_row(config)

	assert row["Software Release Version"] == "2026.5.1"
	assert row["Input Projection Data File"] == ""
	assert row["Internal Export Descriptor ⚠"] == ""


def test_cli_writes_mapped_schema_and_values(load_module, tmp_path):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config_path = _config_file(tmp_path)
	output = tmp_path / "converted.csv"
	result = CliRunner().invoke(
		module.xaid_log,
		[
			str(config_path),
			"--output",
			str(output),
		],
	)

	assert result.exit_code == 0, result.output
	assert "malformed first section header" in result.output
	with output.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle))
	assert tuple(rows[0]) == module.RECONSTRUCTION_LOG_FIELDS
	assert rows[0]["Software Release Version"] == "2026.5.1"
	assert rows[0]["Reconstruction Algorithm"] == "FDK"
	assert rows[0]["Center-of-Rotation Shift (px)"] == "[-12.218087352435388]"
	assert rows[0]["Output File Format"] == "tif16"


def test_cli_refuses_to_replace_output_without_force(load_module, tmp_path):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config_path = _config_file(tmp_path)
	output = tmp_path / "existing.csv"
	output.write_text("keep me", encoding="utf-8")

	result = CliRunner().invoke(
		module.xaid_log,
		[str(config_path), "--output", str(output)],
	)
	assert result.exit_code != 0
	assert "pass --force" in result.output
	assert output.read_text(encoding="utf-8") == "keep me"


def test_append_google_sheet_verifies_header_and_uses_raw_insert(load_module, tmp_path):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config = module.parse_xaid_config(_config_file(tmp_path)).config
	row = module.build_reconstruction_log_row(config)
	service = FakeSheetsService(module.RECONSTRUCTION_LOG_FIELDS)

	response = module.append_reconstruction_log_row(
		service,
		"spreadsheet-id",
		"Recon's",
		row,
	)

	assert response["updates"]["updatedRows"] == 1
	assert service.values_service.get_calls == [
		{"spreadsheetId": "spreadsheet-id", "range": "'Recon''s'!A1:CJ1"}
	]
	append_call = service.values_service.append_calls[0]
	assert append_call["spreadsheetId"] == "spreadsheet-id"
	assert append_call["range"] == "'Recon''s'!A:CJ"
	assert append_call["valueInputOption"] == "RAW"
	assert append_call["insertDataOption"] == "INSERT_ROWS"
	assert append_call["body"] == {
		"majorDimension": "ROWS",
		"values": [[row[field] for field in module.RECONSTRUCTION_LOG_FIELDS]],
	}


def test_append_google_sheet_rejects_wrong_header(load_module, tmp_path):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config = module.parse_xaid_config(_config_file(tmp_path)).config
	row = module.build_reconstruction_log_row(config)
	service = FakeSheetsService(["wrong", "header"])

	with pytest.raises(ValueError, match="header mismatch"):
		module.append_reconstruction_log_row(
			service,
			"spreadsheet-id",
			"Scans",
			row,
		)
	assert service.values_service.append_calls == []


def test_append_google_sheet_can_explicitly_skip_header_check(load_module, tmp_path):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config = module.parse_xaid_config(_config_file(tmp_path)).config
	row = module.build_reconstruction_log_row(config)
	service = FakeSheetsService()

	module.append_reconstruction_log_row(
		service,
		"spreadsheet-id",
		"Reconstructions",
		row,
		verify_header=False,
	)
	assert service.values_service.get_calls == []
	assert len(service.values_service.append_calls) == 1


def test_cli_create_tab_writes_header_then_verifies_and_appends(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config_path = _config_file(tmp_path)
	service = FakeSheetsService(sheet_titles=["Existing"])
	monkeypatch.setattr(module, "build_google_sheets_service", lambda path: service)

	result = CliRunner().invoke(
		module.xaid_log,
		[
			str(config_path),
			"--upload",
			"--spreadsheet",
			"spreadsheet-id",
			"--sheet",
			"New Recon",
			"--create-tab",
		],
	)

	assert result.exit_code == 0, result.output
	assert "Created Google Sheets tab with reconstruction header: New Recon" in result.output
	assert service.get_calls == [
		{
			"spreadsheetId": "spreadsheet-id",
			"fields": "sheets.properties.title",
		}
	]
	assert service.batch_update_calls == [
		{
			"spreadsheetId": "spreadsheet-id",
			"body": {
				"requests": [
					{
						"addSheet": {
							"properties": {"title": "New Recon"},
						}
					}
				]
			},
		}
	]
	assert service.values_service.update_calls == [
		{
			"spreadsheetId": "spreadsheet-id",
			"range": "'New Recon'!A1:CJ1",
			"valueInputOption": "RAW",
			"body": {
				"majorDimension": "ROWS",
				"values": [list(module.RECONSTRUCTION_LOG_FIELDS)],
			},
		}
	]
	assert service.values_service.get_calls == [
		{"spreadsheetId": "spreadsheet-id", "range": "'New Recon'!A1:CJ1"}
	]
	assert len(service.values_service.append_calls) == 1


def test_cli_create_tab_preserves_existing_empty_tab_and_header_error(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config_path = _config_file(tmp_path)
	service = FakeSheetsService(sheet_titles=["Reconstructions"])
	monkeypatch.setattr(module, "build_google_sheets_service", lambda path: service)

	result = CliRunner().invoke(
		module.xaid_log,
		[
			str(config_path),
			"--upload",
			"--spreadsheet",
			"spreadsheet-id",
			"--sheet",
			"Reconstructions",
			"--create-tab",
		],
	)

	assert result.exit_code != 0
	assert "header mismatch" in result.output
	assert service.batch_update_calls == []
	assert service.values_service.update_calls == []
	assert service.values_service.append_calls == []


def test_cli_uploads_without_creating_local_csv(load_module, tmp_path, monkeypatch):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config_path = _config_file(tmp_path)
	service = FakeSheetsService(module.RECONSTRUCTION_LOG_FIELDS)
	google_conf = tmp_path / "google-conf"
	monkeypatch.setattr(module, "build_google_sheets_service", lambda path: service)

	result = CliRunner().invoke(
		module.xaid_log,
		[
			str(config_path),
			"--upload",
			"--spreadsheet",
			"spreadsheet-id",
			"--sheet",
			"Reconstructions",
			"--google-conf",
			str(google_conf),
		],
	)

	assert result.exit_code == 0, result.output
	assert "Appended reconstruction log row: Reconstructions!A24:CJ24" in result.output
	assert not (tmp_path / "config_reconstruction_log.csv").exists()
	assert len(service.values_service.append_calls) == 1


def test_cli_upload_requires_destination_and_excludes_local_output(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config_path = _config_file(tmp_path)
	monkeypatch.delenv("MCTUTIL_GSHEET_ID", raising=False)
	monkeypatch.delenv("MCTUTIL_GSHEET_SHEET", raising=False)

	missing = CliRunner().invoke(module.xaid_log, [str(config_path), "--upload"])
	assert missing.exit_code != 0
	assert "--spreadsheet and --sheet are required" in missing.output

	conflicting = CliRunner().invoke(
		module.xaid_log,
		[
			str(config_path),
			"--upload",
			"--spreadsheet",
			"spreadsheet-id",
			"--sheet",
			"Reconstructions",
			"--output",
			str(tmp_path / "local.csv"),
		],
	)
	assert conflicting.exit_code != 0
	assert "--output cannot be combined with --upload" in conflicting.output

	create_without_upload = CliRunner().invoke(
		module.xaid_log,
		[str(config_path), "--create-tab"],
	)
	assert create_without_upload.exit_code != 0
	assert "--create-tab requires --upload" in create_without_upload.output
