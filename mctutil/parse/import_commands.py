"""CSV commands for the canonical scan and reconstruction readers."""

from __future__ import annotations

from datetime import datetime
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


def _import_to_csv(
	kind: str,
	source: str,
	reader_args: tuple,
	reader_kwargs: dict,
	output: Path,
	force: bool,
	fields,
	headers: tuple[str, ...],
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
		tabular_log.write_csv(
			output,
			headers,
			_rows(records, fields),
			force=force,
		)
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
@click.option(
	"--google-conf",
	type=click.Path(file_okay=False, path_type=Path),
	help="ChengLab credential directory (default: ~/.creds/gsheets).",
)
@click.option(
	"--output",
	type=click.Path(dir_okay=False, path_type=Path),
	required=True,
	help="Destination CSV.",
)
@click.option("--force", is_flag=True, help="Replace an existing output CSV.")
def scan_log(
	inputs: tuple[Path, ...],
	source: str,
	input_spreadsheet: str | None,
	input_sheet: str | None,
	google_conf: Path | None,
	output: Path,
	force: bool,
) -> None:
	"""Import canonical scan records from one explicit homogeneous source."""
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
			"google_conf": google_conf or _DEFAULT_GOOGLE_CONF,
		}
	else:
		if not inputs:
			raise click.UsageError(f"{source} requires one or more input paths")
		if input_spreadsheet is not None or input_sheet is not None or google_conf is not None:
			raise click.UsageError(
				"Sheets input options are valid only for chenglab-camera"
			)
		reader_args = (inputs,)
		reader_kwargs = {}

	_import_to_csv(
		"scan",
		source,
		reader_args,
		reader_kwargs,
		output,
		force,
		SCAN_FIELDS,
		SCAN_HEADERS,
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
@click.option(
	"--output",
	type=click.Path(dir_okay=False, path_type=Path),
	required=True,
	help="Destination CSV.",
)
@click.option("--force", is_flag=True, help="Replace an existing output CSV.")
def reconstruction_log(
	inputs: tuple[Path, ...],
	source: str,
	output: Path,
	force: bool,
) -> None:
	"""Import canonical reconstruction records from one explicit source."""
	_import_to_csv(
		"reconstruction",
		source,
		(inputs,),
		{},
		output,
		force,
		RECONSTRUCTION_FIELDS,
		RECONSTRUCTION_HEADERS,
	)
