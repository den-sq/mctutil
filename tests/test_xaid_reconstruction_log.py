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
reco_type = FDK
apply_roundmask = 1
fdk_filter = hamming
roi_filter = 1
img_binning = 1.000000000000617
is_opt_stage_shifts_triangle = 0
is_opt_stage_shifts = 0
is_opt_proj_shifts_2 = 0
is_opt_jitter = 0

[GC_Values]
rotation_axis_offset = [-12.218087352435388]
drift_x = 0.0
drift_y = 0.0
drift_z = 0.0

[Final_Image_Settings]
save_path = Z:\\DemoScans\\PSU\\Tail_%04d.tif
min = -1.1867414
max = 7.5952454
export_type = tif16
export_order = xzy
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
		self.append_calls = []

	def get(self, **kwargs):
		self.get_calls.append(kwargs)
		values = [self.header] if self.header else []
		return FakeRequest({"values": values})

	def append(self, **kwargs):
		self.append_calls.append(kwargs)
		return FakeRequest(
			{
				"updates": {
					"updatedRange": "Reconstructions!A24:R24",
					"updatedRows": 1,
				}
			}
		)


class FakeSheetsService:
	def __init__(self, header=None):
		self.values_service = FakeValuesService(header)

	def values(self):
		return self.values_service


def test_build_row_maps_xaid_reconstruction_fields(load_module, tmp_path):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	parsed = module.parse_xaid_config(_config_file(tmp_path))
	row = module.build_reconstruction_log_row(parsed.config)

	assert parsed.repaired_first_header is True
	assert row["Type"] == "PSU"
	assert row["Sample ID"] == "WaterFlee-319-40-120-UV-MX4145"
	assert row["center*"] == "X-AID rotation-axis offset: -12.218087352435388"
	assert row["Do Movement Correction?"] == "0"
	assert "median=2" in row["proj_filter (BF settings)"]
	assert "FDK" in row["hoto_tomo_algo (BAC settings)"]
	assert row["final_recon_crop"] == "X-AID ROI pos=(0,0,800); size=(4095,4095,2463)"
	assert row["imageJ W"] == "8.7819868"
	assert row["imageJ L"] == "3.204252"
	assert row["recon offset angle"] == "0.0"
	assert "not present in config: scan number, SSD, stain, energy" in row["Notes"]


def test_center_conventions_are_explicit(load_module, tmp_path):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config = module.parse_xaid_config(_config_file(tmp_path)).config

	width_half = module.build_reconstruction_log_row(
		config,
		center_convention="width-half",
	)
	pixel_center = module.build_reconstruction_log_row(
		config,
		center_convention="pixel-center",
	)
	assert width_half["center*"] == "2035.781912647564612"
	assert pixel_center["center*"] == "2035.281912647564612"


def test_cli_writes_target_schema_and_accepts_metadata_overrides(load_module, tmp_path):
	module = load_module("mctutil/parse/xaid_reconstruction_log.py")
	config_path = _config_file(tmp_path)
	output = tmp_path / "converted.csv"
	result = CliRunner().invoke(
		module.xaid_log,
		[
			str(config_path),
			"--output",
			str(output),
			"--scan-number",
			"147",
			"--stain",
			"unstained",
			"--energy",
			"40 kV",
			"--ssd",
			"105 mm",
			"--center",
			"2035.75",
			"--notes",
			"operator verified",
		],
	)

	assert result.exit_code == 0, result.output
	assert "malformed first section header" in result.output
	with output.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle))
	assert tuple(rows[0]) == module.RECONSTRUCTION_LOG_FIELDS
	assert rows[0]["o"] == "147"
	assert rows[0]["SSD"] == "105 mm"
	assert rows[0]["Stain"] == "unstained"
	assert rows[0]["Energy"] == "40 kV"
	assert rows[0]["center*"] == "2035.75"
	assert "operator verified" in rows[0]["Notes"]
	assert "not present in config: SIFT ZDP, preview, tight crop" in rows[0]["Notes"]
	assert "center preserved as the X-AID offset" not in rows[0]["Notes"]


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
		{"spreadsheetId": "spreadsheet-id", "range": "'Recon''s'!A1:R1"}
	]
	append_call = service.values_service.append_calls[0]
	assert append_call["spreadsheetId"] == "spreadsheet-id"
	assert append_call["range"] == "'Recon''s'!A:R"
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
	assert "Appended reconstruction log row: Reconstructions!A24:R24" in result.output
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
