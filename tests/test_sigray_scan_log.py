from __future__ import annotations

import csv
from pathlib import Path

from click.testing import CliRunner
import h5py
import numpy as np

from mctutil.parse import tabular_log
from mctutil.parse import sigray_scan_log as module


class _Request:
	def __init__(self, response):
		self.response = response

	def execute(self):
		return self.response


class _ValuesService:
	def __init__(self, header):
		self.header = list(header)
		self.append_calls = []

	def get(self, **_kwargs):
		return _Request({"values": [self.header] if self.header else []})

	def update(self, **kwargs):
		self.header = list(kwargs["body"]["values"][0])
		return _Request({"updatedRows": 1})

	def append(self, **kwargs):
		self.append_calls.append(kwargs)
		return _Request(
			{"updates": {"updatedRange": "Scans!A2:AP3", "updatedRows": 2}}
		)


class _SheetsService:
	def __init__(self, header):
		self.values_service = _ValuesService(header)

	def values(self):
		return self.values_service


def _put(handle, path: str, value, **kwargs):
	parent, name = path.rsplit("/", 1)
	group = handle.require_group(parent)
	return group.create_dataset(name, data=value, **kwargs)


def _projection_file(
	directory: Path,
	index: str,
	*,
	frame_averaging: int = 1,
	unique_ids: tuple[int, ...] = (10, 11, 12),
	complete: tuple[int, ...] = (1, 1, 1),
	include_optional_source: bool = True,
	stage_y_um: float | None = None,
) -> Path:
	prefix = directory.name
	path = directory / f"{prefix}_{index}.h5"
	theta = np.asarray((-180.0, 0.0, 180.0))
	with h5py.File(path, "w") as handle:
		_put(handle, module.DATA_PATH, np.zeros((3, 4, 5), dtype=np.uint16))
		_put(handle, module.THETA_PATH, theta)
		_put(handle, module.SAMPLE_NUMBER_PATH, np.bytes_("17"))
		_put(handle, module.SYSTEM_SERIAL_PATH, np.bytes_("MX-TEST"))
		_put(handle, module.SCAN_TYPE_PATH, np.bytes_("Vertical Stitching"))
		_put(handle, module.EXPECTED_PROJECTIONS_PATH, np.full(3, 3))
		_put(handle, module.CONFIGURED_STEP_PATH, np.full(3, 180.0))
		_put(handle, module.EXPOSURE_PATH, np.full(3, 0.25))
		_put(handle, module.COMPLETE_PATH, np.asarray(complete))
		_put(handle, module.UID_PATH, np.asarray(unique_ids))
		_put(handle, module.EPICS_SECONDS_PATH, np.asarray((100, 101, 102)))
		_put(handle, module.EPICS_NANOSECONDS_PATH, np.asarray((0, 500_000_000, 0)))
		_put(handle, module.FRAME_AVERAGING_PATH, np.full(3, frame_averaging))
		_put(handle, module.BINNING_X_PATH, np.full(3, 2))
		_put(handle, module.BINNING_Y_PATH, np.full(3, 2))
		_put(handle, module.DIMENSION_X_PATH, np.full(3, 5))
		_put(handle, module.DIMENSION_Y_PATH, np.full(3, 4))
		_put(handle, module.PIXEL_PITCH_PATH, np.full(3, 9.13))
		_put(handle, module.EFFECTIVE_PIXEL_X_PATH, np.full(3, 0.5))
		_put(handle, module.EFFECTIVE_PIXEL_Y_PATH, np.full(3, 0.5))
		_put(handle, module.MAGNIFICATION_PATH, np.full(3, 18.26))
		_put(handle, module.SAMPLE_X_PATH, np.asarray((1000.0, 1001.0, 1000.0)))
		_put(
			handle,
			module.SAMPLE_Y_PATH,
			np.full(3, stage_y_um if stage_y_um is not None else 2000.0 + int(index)),
		)
		_put(handle, module.SAMPLE_Z_PATH, np.full(3, 3000.0))
		_put(handle, module.DETECTOR_Z_PATH, np.full(3, 100_000.0))
		_put(handle, module.SOURCE_Z_PATH, np.full(3, -5000.0))
		_put(handle, module.SOURCE_TARGET_PATH, np.full(3, np.bytes_("Cu")))
		if include_optional_source:
			_put(handle, module.N3_ENERGY_PATH, np.bytes_("40.00"))
			_put(handle, module.N3_POWER_PATH, np.bytes_("1.25"))
	return path


def _reference_file(
	directory: Path,
	kind: str,
	index: str,
	*,
	frame_count: int,
	stage_y_um: float,
) -> Path:
	prefix = directory.name
	path = directory / f"{prefix}_{kind}_{index}.h5"
	with h5py.File(path, "w") as handle:
		_put(
			handle,
			module.DATA_PATH,
			np.zeros((frame_count, 4, 5), dtype=np.uint16),
		)
		_put(handle, module.SAMPLE_X_PATH, np.full(frame_count, 1000.0))
		_put(handle, module.SAMPLE_Y_PATH, np.full(frame_count, stage_y_um))
		_put(handle, module.SAMPLE_Z_PATH, np.full(frame_count, 3000.0))
	return path


def test_extracts_exact_sigray_paths_and_matches_references_by_stage(tmp_path):
	group = tmp_path / "20260813_sample"
	group.mkdir()
	projection = _projection_file(group, "000")
	flat = _reference_file(group, "FLAT", "004", frame_count=10, stage_y_um=9000)
	dark = _reference_file(group, "DARK", "006", frame_count=7, stage_y_um=9000)
	post = _reference_file(group, "POST", "123", frame_count=5, stage_y_um=2000)

	row = module.extract_scan_row(projection)

	assert tuple(row) == module.SCAN_LOG_FIELDS
	assert len(row) == 42
	assert row["Scan"] == "20260813_sample_000"
	assert row["Acquisition Group"] == group.name
	assert row["FOV Index"] == "000"
	assert row["Projection Data File"] == projection.name
	assert row["Projection Count"] == "3"
	assert row["Rotational Start (°)"] == "-180"
	assert row["Rotational Stop (°)"] == "180"
	assert row["Angular Step (°)"] == "180"
	assert row["Exposure (µs)"] == "250000"
	assert row["Scan Start (UTC)"] == "1990-01-01T00:01:40.000000000Z"
	assert row["Scan Stop (UTC)"] == "1990-01-01T00:01:42.000000000Z"
	assert row["Scan Duration (s)"] == "2"
	assert row["Width (pixels)"] == "5"
	assert row["Height (pixels)"] == "4"
	assert row["Stored Bit Depth"] == "16"
	assert row["Detector Binning"] == "2"
	assert row["Detector Pixel Pitch (mm)"] == "0.00913"
	assert row["Effective Pixel Size (mm)"] == "0.0005"
	assert row["Sample Stage Y (mm)"] == "2"
	assert row["Source Energy Readback"] == "40"
	assert row["Source Power Readback"] == "1.25"
	assert row["Flat Reference Files"] == flat.name
	assert row["Flat Frames"] == "10"
	assert row["Dark Reference Files"] == dark.name
	assert row["Dark Frames"] == "7"
	assert row["Post Reference Files"] == post.name
	assert row["Post Frames"] == "5"
	assert row["Dropped Frames"] == "0"
	assert row["Incomplete Frames"] == "0"
	assert row["Acquisition Status"] == "Complete"
	assert row["Metadata Warnings"] == ""


def test_directory_produces_one_row_per_projection_and_excludes_references(tmp_path):
	group = tmp_path / "scan_group"
	group.mkdir()
	first = _projection_file(group, "000")
	second = _projection_file(group, "001", stage_y_um=2020)
	_reference_file(group, "FLAT", "000", frame_count=2, stage_y_um=5000)
	_reference_file(group, "DARK", "000", frame_count=2, stage_y_um=5000)
	_reference_file(group, "POST", "000", frame_count=2, stage_y_um=2000)
	_reference_file(group, "POST", "001", frame_count=2, stage_y_um=2020)

	discovered = module.discover_projection_files((group, first))
	rows = module.extract_scan_rows(discovered)

	assert discovered == (first, second)
	assert [row["FOV Index"] for row in rows] == ["000", "001"]
	assert all("_POST_" not in row["Scan"] for row in rows)
	assert rows[0]["Post Reference Files"].endswith("_POST_000.h5")
	assert rows[1]["Post Reference Files"].endswith("_POST_001.h5")


def test_frame_averaging_adjusts_dropped_count_and_incomplete_status(tmp_path):
	group = tmp_path / "averaged"
	group.mkdir()
	projection = _projection_file(
		group,
		"000",
		frame_averaging=2,
		unique_ids=(10, 12, 16),
		complete=(1, 0, 1),
	)

	row = module.extract_scan_row(projection, references=())

	assert row["Exposures per Projection"] == "2"
	assert row["Dropped Frames"] == "2"
	assert row["Incomplete Frames"] == "1"
	assert row["Acquisition Status"] == "Issues detected"
	assert "2 dropped frame(s)" in row["Metadata Warnings"]
	assert "1 frame(s) marked incomplete" in row["Metadata Warnings"]


def test_missing_optional_source_values_stay_blank_instead_of_using_siblings(tmp_path):
	group = tmp_path / "missing_source"
	group.mkdir()
	first = _projection_file(group, "000", include_optional_source=False)
	second = _projection_file(group, "001", include_optional_source=True)

	rows = module.extract_scan_rows((first, second))

	assert rows[0]["Source Energy Readback"] == ""
	assert rows[0]["Source Power Readback"] == ""
	assert "unavailable in this acquisition" in rows[0]["Metadata Warnings"]
	assert rows[0]["Acquisition Status"] == "Complete"
	assert rows[1]["Source Energy Readback"] == "40"


def test_exact_data_exchange_path_is_required(tmp_path):
	path = tmp_path / "sample_000.h5"
	with h5py.File(path, "w") as handle:
		_put(handle, "/other/data", np.zeros((3, 4, 5), dtype=np.uint16))

	row = module.extract_scan_row(path, references=())

	assert row["Acquisition Status"] == "Invalid"
	assert row["Metadata Warnings"] == "missing three-dimensional /exchange/data"


def test_cli_writes_all_rows_through_shared_tabular_destination(tmp_path, monkeypatch):
	group = tmp_path / "cli_group"
	group.mkdir()
	_projection_file(group, "000")
	_projection_file(group, "001")
	output = tmp_path / "scan_log.csv"
	shared_calls = []
	real_write_csv = tabular_log.write_csv

	def record_write_csv(path, fields, rows, *, force=False):
		materialized_rows = list(rows)
		shared_calls.append((path, fields, materialized_rows, force))
		return real_write_csv(path, fields, materialized_rows, force=force)

	monkeypatch.setattr(tabular_log, "write_csv", record_write_csv)
	result = CliRunner().invoke(
		module.sigray_scan_log,
		[str(group), "--output", str(output)],
	)

	assert result.exit_code == 0, result.output
	assert len(shared_calls) == 1
	assert shared_calls[0][0] == output
	assert shared_calls[0][1] is module.SCAN_LOG_FIELDS
	with output.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle))
	assert [row["FOV Index"] for row in rows] == ["000", "001"]


def test_cli_uploads_all_rows_by_live_header_name(tmp_path, monkeypatch):
	group = tmp_path / "upload_group"
	group.mkdir()
	_projection_file(group, "000")
	_projection_file(group, "001")
	header = ["Operator", *reversed(module.SCAN_LOG_FIELDS), "Notes"]
	service = _SheetsService(header)
	monkeypatch.setattr(
		tabular_log,
		"build_google_sheets_service",
		lambda _path: service,
	)

	result = CliRunner().invoke(
		module.sigray_scan_log,
		[
			str(group),
			"--upload",
			"--spreadsheet",
			"spreadsheet-id",
			"--sheet",
			"Scans",
		],
	)

	assert result.exit_code == 0, result.output
	assert "Appended 2 scan log row(s): Scans!A2:AP3" in result.output
	append_call = service.values_service.append_calls[0]
	assert append_call["range"] == "'Scans'!A:AR"
	assert len(append_call["body"]["values"]) == 2
	assert append_call["body"]["values"][0][0] is None
	assert append_call["body"]["values"][0][-1] is None
	assert append_call["body"]["values"][0][1:-1] == [
		module.extract_scan_rows(module.discover_projection_files((group,)))[0][field]
		for field in reversed(module.SCAN_LOG_FIELDS)
	]


def test_xaid_and_sigray_use_same_google_and_csv_destination_module():
	from mctutil.parse import xaid_reconstruction_log

	assert xaid_reconstruction_log.tabular_log is module.tabular_log
	assert xaid_reconstruction_log._column_label is tabular_log.column_label
