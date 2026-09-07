from __future__ import annotations

import csv
from pathlib import Path

from click.testing import CliRunner
import h5py
import numpy as np
import pytest

from mctutil.cli import main
from mctutil.parse import import_commands, tabular_log
from mctutil.parse.import_records import (
	RECONSTRUCTION_HEADERS,
	SCAN_HEADERS,
	ScanRecord,
	empty_scan_values,
)


def _put(handle, path: str, value):
	parent, name = path.rsplit("/", 1)
	return handle.require_group(parent).create_dataset(name, data=value)


def _als_scan(tmp_path: Path) -> Path:
	path = tmp_path / "als-scan.h5"
	with h5py.File(path, "w") as handle:
		_put(handle, "/exchange/data", np.zeros((3, 4, 5), dtype=np.uint16))
		_put(handle, "/exchange/theta", (0.0, 90.0, 180.0))
		_put(handle, "/exchange/data_white", np.zeros((2, 4, 5)))
		_put(handle, "/measurement/sample/file_name", np.bytes_("scan-1"))
		_put(handle, "/process/acquisition/start_date", np.bytes_("2026-01-02T03:04:05Z"))
	return path


def _xaid_config(tmp_path: Path) -> Path:
	path = tmp_path / "xaid.txt"
	path.write_text(
		"""a[General_Info]
software_version = 2026.5
path_to_data = /data/scan-7.h5

[VolumeData]
nat_vx_size = 0.002
voxel_size = 0.004

[Reconstruction_Settings]
reco_type = FDK
""",
		encoding="utf-8",
	)
	return path


def _read_csv(path: Path):
	with path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		return reader.fieldnames, list(reader)


def _stub_sheet_destination(monkeypatch, *, created=False):
	calls = {}
	service = object()

	def build(google_conf):
		calls["google_conf"] = google_conf
		return service

	def create(received_service, spreadsheet, sheet, headers):
		calls["create"] = (received_service, spreadsheet, sheet, headers)
		return created

	def append(
		received_service,
		spreadsheet,
		sheet,
		headers,
		rows,
		*,
		verify,
		strict_header_order,
	):
		calls["append"] = (
			received_service,
			spreadsheet,
			sheet,
			headers,
			list(rows),
			verify,
			strict_header_order,
		)
		return {"updates": {"updatedRange": f"{sheet}!A2"}}

	monkeypatch.setattr(tabular_log, "build_google_sheets_service", build)
	monkeypatch.setattr(tabular_log, "create_tab_if_missing", create)
	monkeypatch.setattr(tabular_log, "append_rows", append)
	monkeypatch.setattr(
		tabular_log,
		"require_google_sheets_dependencies",
		lambda **_kwargs: (),
	)
	return service, calls


def test_scan_log_end_to_end_uses_canonical_headers_and_csv_writer(tmp_path, monkeypatch):
	input_path = _als_scan(tmp_path)
	output = tmp_path / "scans.csv"
	calls = []
	real_write_csv = tabular_log.write_csv

	def recording_writer(path, fields, rows, *, force=False):
		materialized = list(rows)
		calls.append((path, fields, materialized, force))
		return real_write_csv(path, fields, materialized, force=force)

	monkeypatch.setattr(tabular_log, "write_csv", recording_writer)
	result = CliRunner().invoke(
		main,
		[
			"parse",
			"scan-log",
			str(input_path),
			"--source",
			"als832",
			"--output",
			str(output),
		],
	)

	assert result.exit_code == 0, result.output
	assert result.output == f"Wrote 1 scan record(s) to {output}\n"
	assert calls[0][0] == output
	assert calls[0][1] is SCAN_HEADERS
	fieldnames, rows = _read_csv(output)
	assert fieldnames == list(SCAN_HEADERS)
	assert rows[0]["Scan ID"] == "scan-1"
	assert rows[0]["Scan Start (UTC)"] == "2026-01-02T03:04:05+00:00"
	assert rows[0]["Flat Reference Files"] == f"{input_path}:/exchange/data_white"


def test_reconstruction_log_end_to_end_prints_contextual_warning(tmp_path):
	input_path = _xaid_config(tmp_path)
	output = tmp_path / "reconstructions.csv"
	result = CliRunner().invoke(
		main,
		[
			"parse",
			"reconstruction-log",
			str(input_path),
			"--source",
			"xaid",
			"--output",
			str(output),
		],
	)

	assert result.exit_code == 0, result.output
	assert f"Warning [{input_path}]: repaired malformed first section header" in result.output
	fieldnames, rows = _read_csv(output)
	assert fieldnames == list(RECONSTRUCTION_HEADERS)
	assert rows[0]["Reconstruction ID"] == "xaid"
	assert rows[0]["Scan ID"] == "scan-7"
	assert rows[0]["Metadata Warnings"] == "repaired malformed first section header"


def test_scan_log_uploads_canonical_rows_and_can_create_tab(tmp_path, monkeypatch):
	input_path = _als_scan(tmp_path)
	google_conf = tmp_path / "google"
	service, calls = _stub_sheet_destination(monkeypatch, created=True)

	result = CliRunner().invoke(
		import_commands.scan_log,
		[
			str(input_path),
			"--source",
			"als832",
			"--upload",
			"--spreadsheet",
			"spreadsheet-id",
			"--sheet",
			"Scans",
			"--create-tab",
			"--google-conf",
			str(google_conf),
		],
	)

	assert result.exit_code == 0, result.output
	assert result.output == (
		"Created Google Sheets tab with scan header: Scans\n"
		"Appended 1 scan record(s): Scans!A2\n"
	)
	assert calls["google_conf"] == google_conf
	assert calls["create"] == (service, "spreadsheet-id", "Scans", SCAN_HEADERS)
	append = calls["append"]
	assert append[:4] == (service, "spreadsheet-id", "Scans", SCAN_HEADERS)
	assert append[4][0]["Scan ID"] == "scan-1"
	assert append[4][0]["Scan Start (UTC)"] == "2026-01-02T03:04:05+00:00"
	assert append[5:] == (True, False)


def test_reconstruction_log_upload_uses_environment_destination(tmp_path, monkeypatch):
	input_path = _xaid_config(tmp_path)
	google_conf = Path("~/.creds/mctutil-test").expanduser()
	service, calls = _stub_sheet_destination(monkeypatch)
	monkeypatch.setenv("MCTUTIL_GSHEET_ID", "environment-spreadsheet")
	monkeypatch.setenv("MCTUTIL_GSHEET_SHEET", "Reconstructions")
	monkeypatch.setenv("MCTUTIL_GOOGLE_CONF", "~/.creds/mctutil-test")

	result = CliRunner().invoke(
		import_commands.reconstruction_log,
		[
			str(input_path),
			"--source",
			"xaid",
			"--upload",
			"--strict-header-order",
		],
	)

	assert result.exit_code == 0, result.output
	assert "Appended 1 reconstruction record(s): Reconstructions!A2" in result.output
	assert calls["google_conf"] == google_conf
	assert "create" not in calls
	append = calls["append"]
	assert append[:4] == (
		service,
		"environment-spreadsheet",
		"Reconstructions",
		RECONSTRUCTION_HEADERS,
	)
	assert append[4][0]["Reconstruction ID"] == "xaid"
	assert append[4][0]["Metadata Warnings"] == "repaired malformed first section header"
	assert append[5:] == (True, True)


def test_upload_dependency_failure_precedes_source_dispatch(tmp_path, monkeypatch):
	input_path = tmp_path / "input.h5"
	input_path.touch()

	def missing_dependencies(*, error_type=RuntimeError):
		raise error_type("Google Sheets upload dependencies are unavailable")

	monkeypatch.setattr(
		tabular_log,
		"require_google_sheets_dependencies",
		missing_dependencies,
	)
	monkeypatch.setattr(
		import_commands.import_pipeline,
		"read_records",
		lambda *_args, **_kwargs: pytest.fail("dispatcher must not run"),
	)

	result = CliRunner().invoke(
		import_commands.scan_log,
		[
			str(input_path),
			"--source",
			"als832",
			"--upload",
			"--spreadsheet",
			"spreadsheet-id",
			"--sheet",
			"Scans",
		],
	)

	assert result.exit_code == 1
	assert "Google Sheets upload dependencies are unavailable" in result.output
	assert not isinstance(result.exception, AssertionError)


@pytest.mark.parametrize(
	"command",
	(import_commands.scan_log, import_commands.reconstruction_log),
)
def test_destination_help_declares_google_conf_environment_and_fallback(command):
	result = CliRunner().invoke(command, ["--help"])

	assert result.exit_code == 0, result.output
	assert "env var: MCTUTIL_GOOGLE_CONF" in result.output
	assert f"default: {tabular_log.DEFAULT_GOOGLE_CONF}" in result.output


@pytest.mark.parametrize(
	"destination_arguments, message",
	(
		((), "--output is required unless --upload is used"),
		(("--upload",), "--spreadsheet and --sheet are required"),
		(
			(
				"--upload",
				"--spreadsheet",
				"spreadsheet-id",
				"--sheet",
				"Scans",
				"--output",
				"output.csv",
			),
			"--output cannot be combined with --upload",
		),
		(
			(
				"--upload",
				"--spreadsheet",
				"spreadsheet-id",
				"--sheet",
				"Scans",
				"--force",
			),
			"--force applies only to local CSV output",
		),
		(
			(
				"--upload",
				"--spreadsheet",
				"spreadsheet-id",
				"--sheet",
				"Scans",
				"--strict-header-order",
				"--no-verify-header",
			),
			"--strict-header-order cannot be combined with --no-verify-header",
		),
		(("--create-tab", "--output", "output.csv"), "--create-tab requires --upload"),
	),
)
def test_destination_contract_fails_before_dispatch(
	tmp_path,
	monkeypatch,
	destination_arguments,
	message,
):
	input_path = tmp_path / "input.h5"
	input_path.touch()
	monkeypatch.delenv("MCTUTIL_GSHEET_ID", raising=False)
	monkeypatch.delenv("MCTUTIL_GSHEET_SHEET", raising=False)
	monkeypatch.setattr(
		import_commands.import_pipeline,
		"read_records",
		lambda *_args, **_kwargs: pytest.fail("dispatcher must not run"),
	)

	result = CliRunner().invoke(
		import_commands.scan_log,
		[
			str(input_path),
			"--source",
			"als832",
			*destination_arguments,
		],
	)

	assert result.exit_code == 2
	assert message in result.output


@pytest.mark.parametrize(
	"arguments, message",
	(
		(
			("--source", "als832"),
			"als832 requires one or more input paths",
		),
		(
			("--source", "als832", "--input-spreadsheet", "sheet-id"),
			"Sheets input options are valid only for chenglab-camera",
		),
		(
			("--source", "chenglab-camera"),
			"chenglab-camera requires --input-spreadsheet",
		),
	),
)
def test_scan_log_rejects_invalid_input_contract_before_dispatch(
	tmp_path,
	monkeypatch,
	arguments,
	message,
):
	def unexpected_dispatch(*_args, **_kwargs):
		raise AssertionError("dispatcher must not run")

	monkeypatch.setattr(import_commands.import_pipeline, "read_records", unexpected_dispatch)
	positional = []
	if "--input-spreadsheet" in arguments and "als832" in arguments:
		input_path = tmp_path / "input.h5"
		input_path.touch()
		positional.append(str(input_path))
	result = CliRunner().invoke(
		import_commands.scan_log,
		[*positional, *arguments, "--output", str(tmp_path / "output.csv")],
	)

	assert result.exit_code == 2
	assert message in result.output
	assert not isinstance(result.exception, AssertionError)


def test_chenglab_contract_dispatches_without_positional_input_and_uses_defaults(
	tmp_path,
	monkeypatch,
):
	values = empty_scan_values()
	values["scan_id"] = "scan-1"
	record = ScanRecord(
		"chenglab-camera",
		("gsheets://sheet-id/ScanLog!2",),
		values,
		["timezone is unconfirmed"],
	)
	calls = []

	def recording_dispatch(kind, source, *args, **kwargs):
		calls.append((kind, source, args, kwargs))
		return (record,)

	monkeypatch.setattr(import_commands.import_pipeline, "read_records", recording_dispatch)
	output = tmp_path / "camera.csv"
	result = CliRunner().invoke(
		import_commands.scan_log,
		[
			"--source",
			"chenglab-camera",
			"--input-spreadsheet",
			"sheet-id",
			"--output",
			str(output),
		],
	)

	assert result.exit_code == 0, result.output
	assert calls == [
		(
			"scan",
			"chenglab-camera",
			("sheet-id",),
			{
				"sheet": "ScanLog",
				"google_conf": Path("~/.creds/gsheets").expanduser(),
			},
		)
	]
	assert "Warning [gsheets://sheet-id/ScanLog!2]: timezone is unconfirmed" in result.output


def test_scan_log_rejects_mixed_chenglab_path_before_dispatch(tmp_path, monkeypatch):
	input_path = tmp_path / "input.h5"
	input_path.touch()
	monkeypatch.setattr(
		import_commands.import_pipeline,
		"read_records",
		lambda *_args, **_kwargs: pytest.fail("dispatcher must not run"),
	)
	result = CliRunner().invoke(
		import_commands.scan_log,
		[
			str(input_path),
			"--source",
			"chenglab-camera",
			"--input-spreadsheet",
			"sheet-id",
			"--output",
			str(tmp_path / "output.csv"),
		],
	)

	assert result.exit_code == 2
	assert "chenglab-camera accepts no positional input paths" in result.output


def test_existing_output_is_fatal_without_force(tmp_path):
	input_path = _als_scan(tmp_path)
	output = tmp_path / "scans.csv"
	output.write_text("existing\n", encoding="utf-8")
	result = CliRunner().invoke(
		import_commands.scan_log,
		[
			str(input_path),
			"--source",
			"als832",
			"--output",
			str(output),
		],
	)

	assert result.exit_code == 1
	assert "pass --force to replace it" in result.output
	assert output.read_text(encoding="utf-8") == "existing\n"

	forced = CliRunner().invoke(
		import_commands.scan_log,
		[
			str(input_path),
			"--source",
			"als832",
			"--output",
			str(output),
			"--force",
		],
	)
	assert forced.exit_code == 0, forced.output
	assert _read_csv(output)[0] == list(SCAN_HEADERS)


def test_fatal_dispatch_validation_exits_nonzero(tmp_path, monkeypatch):
	input_path = tmp_path / "input.h5"
	input_path.touch()

	def fail_dispatch(*_args, **_kwargs):
		raise ValueError("record is missing required 'scan_id'")

	monkeypatch.setattr(
		import_commands.import_pipeline,
		"read_records",
		fail_dispatch,
	)
	result = CliRunner().invoke(
		import_commands.scan_log,
		[
			str(input_path),
			"--source",
			"als832",
			"--output",
			str(tmp_path / "output.csv"),
		],
	)

	assert result.exit_code == 1
	assert "Error: record is missing required 'scan_id'" in result.output
