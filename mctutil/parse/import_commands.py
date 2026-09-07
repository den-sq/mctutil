"""CSV and Google Sheets commands for the canonical metadata readers."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import click

from mctutil.parse import import_pipeline, tabular_log
from mctutil.parse.import_records import (
	RECONSTRUCTION_FIELDS,
	RECONSTRUCTION_HEADERS,
	SCAN_FIELDS,
	SCAN_HEADERS,
	WARNING_HEADER,
)


_DEFAULT_CHENGLAB_SHEET = "ScanLog"
_DEFAULT_GOOGLE_CONF = Path("~/.creds/gsheets").expanduser()


def _google_conf_default() -> Path:
	return Path(
		os.environ.get("MCTUTIL_GOOGLE_CONF", str(_DEFAULT_GOOGLE_CONF))
	).expanduser()


def _destination_options(function):
	"""Declare the established tabular-log destination contract once."""
	options = (
		click.option(
			"--output",
			type=click.Path(dir_okay=False, path_type=Path),
			help="Destination CSV; required unless --upload is used.",
		),
		click.option(
			"--upload",
			is_flag=True,
			help="Append records to Google Sheets instead of writing a local CSV.",
		),
		click.option(
			"--spreadsheet",
			default=lambda: os.environ.get("MCTUTIL_GSHEET_ID"),
			help="Destination spreadsheet ID; may use MCTUTIL_GSHEET_ID.",
		),
		click.option(
			"--sheet",
			default=lambda: os.environ.get("MCTUTIL_GSHEET_SHEET"),
			help="Destination tab; may use MCTUTIL_GSHEET_SHEET.",
		),
		click.option(
			"--create-tab",
			is_flag=True,
			help="Create a missing destination tab with the canonical header.",
		),
		click.option(
			"--google-conf",
			type=click.Path(file_okay=False, path_type=Path),
			default=_google_conf_default,
			show_default="~/.creds/gsheets",
			help="Directory containing Google Sheets credentials and tokens.",
		),
		click.option(
			"--verify-header/--no-verify-header",
			default=True,
			show_default=True,
			help=(
				"Match destination columns by exact header name. Disabling this uses "
				"canonical positional order."
			),
		),
		click.option(
			"--strict-header-order",
			is_flag=True,
			help="Require canonical header order instead of matching by name.",
		),
		click.option("--force", is_flag=True, help="Replace an existing output CSV."),
	)
	for option in reversed(options):
		function = option(function)
	return function


def _csv_value(value):
	if isinstance(value, datetime):
		return value.isoformat()
	if isinstance(value, tuple):
		return "; ".join(str(_csv_value(item)) for item in value)
	return value


def _rows(records, fields):
	for record in records:
		row = {
			field.header: _csv_value(record.values[field.name])
			for field in fields
		}
		row[WARNING_HEADER] = "; ".join(record.warnings)
		yield row


def _emit_warnings(records) -> None:
	for record in records:
		context = ", ".join(record.source_locations)
		for warning in record.warnings:
			click.echo(f"Warning [{context}]: {warning}", err=True)


def _validate_destination(
	*,
	output: Path | None,
	upload: bool,
	spreadsheet: str | None,
	sheet: str | None,
	create_tab: bool,
	force: bool,
	verify_header: bool,
	strict_header_order: bool,
) -> None:
	tabular_log.validate_destination_options(
		upload=upload,
		create_tab=create_tab,
		output=output,
		force=force,
		spreadsheet=spreadsheet,
		sheet=sheet,
		verify=verify_header,
		strict_header_order=strict_header_order,
	)
	if not upload and output is None:
		raise click.UsageError("--output is required unless --upload is used.")


def _upload_rows(
	kind: str,
	rows: list[dict[str, object]],
	headers: tuple[str, ...],
	google_conf: Path,
	spreadsheet: str,
	sheet: str,
	create_tab: bool,
	verify_header: bool,
	strict_header_order: bool,
) -> None:
	try:
		service = tabular_log.build_google_sheets_service(google_conf)
		created_tab = create_tab and tabular_log.create_tab_if_missing(
			service,
			spreadsheet,
			sheet,
			headers,
		)
		response = tabular_log.append_rows(
			service,
			spreadsheet,
			sheet,
			headers,
			rows,
			verify=verify_header,
			strict_header_order=strict_header_order,
		)
	except Exception as exc:
		raise click.ClickException(f"Google Sheets upload failed: {exc}") from exc
	if created_tab:
		click.echo(f"Created Google Sheets tab with {kind} header: {sheet}")
	updated_range = response.get("updates", {}).get("updatedRange", "unknown range")
	click.echo(f"Appended {len(rows)} {kind} record(s): {updated_range}")


def _import_records(
	kind: str,
	source: str,
	reader_args: tuple,
	reader_kwargs: dict,
	output: Path | None,
	force: bool,
	fields,
	headers: tuple[str, ...],
	*,
	upload: bool,
	spreadsheet: str | None,
	sheet: str | None,
	create_tab: bool,
	google_conf: Path,
	verify_header: bool,
	strict_header_order: bool,
) -> None:
	try:
		records = import_pipeline.read_records(
			kind,
			source,
			*reader_args,
			**reader_kwargs,
		)
		if not records:
			raise ValueError(f"{source} produced no {kind} records")
		_emit_warnings(records)
		rows = list(_rows(records, fields))
		if upload:
			_upload_rows(
				kind,
				rows,
				headers,
				google_conf,
				spreadsheet,
				sheet,
				create_tab,
				verify_header,
				strict_header_order,
			)
			return
		tabular_log.write_csv(output, headers, rows, force=force)
	except click.ClickException:
		raise
	except (OSError, RuntimeError, TypeError, ValueError) as exc:
		raise click.ClickException(str(exc)) from exc
	click.echo(f"Wrote {len(records)} {kind} record(s) to {output}")


@click.command("scan-log")
@click.argument(
	"inputs",
	nargs=-1,
	type=click.Path(exists=True, readable=True, path_type=Path),
)
@click.option(
	"--source",
	type=click.Choice(import_pipeline.SCAN_SOURCES, case_sensitive=False),
	required=True,
	help="Explicit scan source; automatic detection is not supported.",
)
@click.option(
	"--input-spreadsheet",
	help="Google spreadsheet ID; valid only for chenglab-camera.",
)
@click.option(
	"--input-sheet",
	help="ChengLab input tab (default: ScanLog).",
)
@_destination_options
def scan_log(
	inputs: tuple[Path, ...],
	source: str,
	input_spreadsheet: str | None,
	input_sheet: str | None,
	output: Path | None,
	upload: bool,
	spreadsheet: str | None,
	sheet: str | None,
	create_tab: bool,
	google_conf: Path,
	verify_header: bool,
	strict_header_order: bool,
	force: bool,
) -> None:
	"""Import canonical scan records from one explicit homogeneous source."""
	_validate_destination(
		output=output,
		upload=upload,
		spreadsheet=spreadsheet,
		sheet=sheet,
		create_tab=create_tab,
		force=force,
		verify_header=verify_header,
		strict_header_order=strict_header_order,
	)
	if source == "chenglab-camera":
		if inputs:
			raise click.UsageError(
				"chenglab-camera accepts no positional input paths"
			)
		if not input_spreadsheet:
			raise click.UsageError(
				"chenglab-camera requires --input-spreadsheet"
			)
		reader_args = (input_spreadsheet,)
		reader_kwargs = {
			"sheet": input_sheet or _DEFAULT_CHENGLAB_SHEET,
			"google_conf": google_conf,
		}
	else:
		if not inputs:
			raise click.UsageError(f"{source} requires one or more input paths")
		if input_spreadsheet is not None or input_sheet is not None:
			raise click.UsageError(
				"Sheets input options are valid only for chenglab-camera"
			)
		reader_args = (inputs,)
		reader_kwargs = {}

	_import_records(
		"scan",
		source,
		reader_args,
		reader_kwargs,
		output,
		force,
		SCAN_FIELDS,
		SCAN_HEADERS,
		upload=upload,
		spreadsheet=spreadsheet,
		sheet=sheet,
		create_tab=create_tab,
		google_conf=google_conf,
		verify_header=verify_header,
		strict_header_order=strict_header_order,
	)


@click.command("reconstruction-log")
@click.argument(
	"inputs",
	nargs=-1,
	required=True,
	type=click.Path(exists=True, readable=True, path_type=Path),
)
@click.option(
	"--source",
	type=click.Choice(import_pipeline.RECONSTRUCTION_SOURCES, case_sensitive=False),
	required=True,
	help="Explicit reconstruction source; automatic detection is not supported.",
)
@_destination_options
def reconstruction_log(
	inputs: tuple[Path, ...],
	source: str,
	output: Path | None,
	upload: bool,
	spreadsheet: str | None,
	sheet: str | None,
	create_tab: bool,
	google_conf: Path,
	verify_header: bool,
	strict_header_order: bool,
	force: bool,
) -> None:
	"""Import canonical reconstruction records from one explicit source."""
	_validate_destination(
		output=output,
		upload=upload,
		spreadsheet=spreadsheet,
		sheet=sheet,
		create_tab=create_tab,
		force=force,
		verify_header=verify_header,
		strict_header_order=strict_header_order,
	)
	_import_records(
		"reconstruction",
		source,
		(inputs,),
		{},
		output,
		force,
		RECONSTRUCTION_FIELDS,
		RECONSTRUCTION_HEADERS,
		upload=upload,
		spreadsheet=spreadsheet,
		sheet=sheet,
		create_tab=create_tab,
		google_conf=google_conf,
		verify_header=verify_header,
		strict_header_order=strict_header_order,
	)
