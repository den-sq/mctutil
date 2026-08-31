from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from mctutil.cli import main as root_cli
from mctutil.parse import main as parse_group
from mctutil.parse import tabular_log
from mctutil.parse.importers.cli import reconstruction_log, scan_log


def _fixture(path: Path, kind: str, **values: object) -> Path:
	path.write_text(
		json.dumps({"mctutil_fixture_kind": kind, **values}),
		encoding="utf-8",
	)
	return path


@pytest.mark.parametrize("command", (scan_log, reconstruction_log))
def test_generic_command_help_has_shared_options(command) -> None:
	result = CliRunner().invoke(command, ["--help"])

	assert result.exit_code == 0, result.output
	for option in (
		"--source",
		"--mapping",
		"--profile",
		"--output",
		"--upload",
		"--verify-header",
		"--strict-header-order",
		"--strict",
		"--validate-only",
		"--audit-output",
		"--force",
	):
		assert option in result.output


def test_lazy_parse_group_exposes_both_generic_commands() -> None:
	result = CliRunner().invoke(parse_group, ["--help"])

	assert result.exit_code == 0, result.output
	assert "scan-log" in result.output
	assert "reconstruction-log" in result.output


def test_scan_command_runs_engine_and_writes_optional_csv(tmp_path: Path) -> None:
	source = _fixture(
		tmp_path / "scan.json",
		"scan",
		scan_id="scan-1",
		projection_count=1200,
	)
	output = tmp_path / "scans.csv"
	result = CliRunner().invoke(scan_log, [str(source), "--output", str(output)])

	assert result.exit_code == 0, result.output
	with output.open(newline="", encoding="utf-8") as handle:
		row = next(csv.DictReader(handle))
	assert row["Scan ID"] == "scan-1"
	assert row["Acquired Projections"] == "1200"


def test_full_mctutil_parse_command_path_runs_lazy_command(tmp_path: Path) -> None:
	source = _fixture(
		tmp_path / "scan.json",
		"scan",
		scan_id="scan-root",
		projection_count=12,
	)
	output = tmp_path / "root.csv"
	result = CliRunner().invoke(
		root_cli, ["parse", "scan-log", str(source), "--output", str(output)]
	)

	assert result.exit_code == 0, result.output
	assert output.exists()


def test_reconstruction_command_runs_engine_and_audit_output(tmp_path: Path) -> None:
	source = _fixture(
		tmp_path / "reconstruction.json",
		"reconstruction",
		reconstruction_id="recon-1",
		algorithm="gridrec",
	)
	output = tmp_path / "recon.csv"
	audit = tmp_path / "recon.audit.jsonl"
	result = CliRunner().invoke(
		reconstruction_log,
		[str(source), "--output", str(output), "--audit-output", str(audit)],
	)

	assert result.exit_code == 0, result.output
	document = json.loads(audit.read_text(encoding="utf-8"))
	assert document["source"] == {
		"id": "fixture-reconstruction",
		"version": "fixture/1",
	}
	assert document["values"]["reconstruction_id"]["value"] == "recon-1"
	assert document["values"]["reconstruction_algorithm"]["value"] == "gridrec"


def test_validate_only_runs_full_import_without_any_output(tmp_path: Path) -> None:
	source = _fixture(
		tmp_path / "scan.json", "scan", scan_id="scan-1", projection_count=10
	)
	output = tmp_path / "scans.csv"
	audit = tmp_path / "scans.audit.jsonl"
	output.write_text("existing", encoding="utf-8")
	audit.write_text("existing", encoding="utf-8")
	result = CliRunner().invoke(
		scan_log,
		[
			str(source),
			"--output",
			str(output),
			"--audit-output",
			str(audit),
			"--validate-only",
		],
	)

	assert result.exit_code == 0, result.output
	assert "no output written" in result.output
	assert output.read_text(encoding="utf-8") == "existing"
	assert audit.read_text(encoding="utf-8") == "existing"


def test_strict_blocks_record_error_but_default_writes(tmp_path: Path) -> None:
	source = _fixture(tmp_path / "missing-id.json", "scan", projection_count=10)
	default_output = tmp_path / "default.csv"
	default_result = CliRunner().invoke(
		scan_log,
		[
			str(source),
			"--source",
			"fixture-scan",
			"--output",
			str(default_output),
		],
	)
	strict_output = tmp_path / "strict.csv"
	strict_result = CliRunner().invoke(
		scan_log,
		[
			str(source),
			"--source",
			"fixture-scan",
			"--output",
			str(strict_output),
			"--strict",
		],
	)

	assert default_result.exit_code == 0, default_result.output
	assert default_output.exists()
	assert strict_result.exit_code == 1
	assert "Strict validation failed" in strict_result.output
	assert not strict_output.exists()


def test_auto_detection_fails_closed_before_output(tmp_path: Path) -> None:
	source = _fixture(tmp_path / "unknown.json", "other", scan_id="scan-1")
	output = tmp_path / "out.csv"
	result = CliRunner().invoke(scan_log, [str(source), "--output", str(output)])

	assert result.exit_code == 1
	assert "no_high_confidence_match" in result.output
	assert not output.exists()


def test_output_and_audit_paths_must_be_distinct(tmp_path: Path) -> None:
	source = _fixture(
		tmp_path / "scan.json", "scan", scan_id="scan-1", projection_count=10
	)
	output = tmp_path / "same.out"
	result = CliRunner().invoke(
		scan_log,
		[str(source), "--output", str(output), "--audit-output", str(output)],
	)

	assert result.exit_code == 2
	assert "must be different paths" in result.output


def test_upload_routes_profile_rows_through_tabular_log(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	source = _fixture(
		tmp_path / "scan.json", "scan", scan_id="scan-1", projection_count=10
	)
	service = object()
	calls = {}
	monkeypatch.setattr(
		tabular_log, "build_google_sheets_service", lambda google_conf: service
	)

	def fake_create(actual_service, spreadsheet, sheet, fields):
		calls["create"] = (actual_service, spreadsheet, sheet, fields)
		return True

	def fake_append(
		actual_service,
		spreadsheet,
		sheet,
		fields,
		rows,
		*,
		verify=True,
		strict_header_order=False,
	):
		calls["append"] = (
			actual_service,
			spreadsheet,
			sheet,
			fields,
			tuple(dict(row) for row in rows),
			verify,
			strict_header_order,
		)
		return {"updates": {"updatedRange": "ScanLog!A2:O2"}}

	monkeypatch.setattr(tabular_log, "create_tab_if_missing", fake_create)
	monkeypatch.setattr(tabular_log, "append_rows", fake_append)
	result = CliRunner().invoke(
		scan_log,
		[
			str(source),
			"--upload",
			"--spreadsheet",
			"spreadsheet-id",
			"--sheet",
			"ScanLog",
			"--create-tab",
			"--strict-header-order",
		],
	)

	assert result.exit_code == 0, result.output
	assert calls["create"][:3] == (service, "spreadsheet-id", "ScanLog")
	assert calls["append"][:3] == (service, "spreadsheet-id", "ScanLog")
	assert calls["append"][4][0]["Scan ID"] == "scan-1"
	assert calls["append"][5:] == (True, True)
