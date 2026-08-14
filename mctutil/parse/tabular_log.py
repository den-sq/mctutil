"""Shared local CSV and Google Sheets destinations for tabular logs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Mapping

import click

from mctutil.shared.deps import require


GOOGLE_SHEETS_SCOPES = ("https://www.googleapis.com/auth/spreadsheets",)


def column_label(column_number: int) -> str:
	"""Return the A1 column label for a one-based column number."""
	if column_number < 1:
		raise ValueError("column number must be positive")
	label = ""
	while column_number:
		column_number, remainder = divmod(column_number - 1, 26)
		label = chr(ord("A") + remainder) + label
	return label


def column_range(fields: tuple[str, ...]) -> str:
	"""Return the whole-column A1 range for a canonical field sequence."""
	return f"A:{column_label(len(fields))}"


def header_range(fields: tuple[str, ...]) -> str:
	"""Return the first-row A1 range for a canonical field sequence."""
	return f"A1:{column_label(len(fields))}1"


def write_csv(
	output: Path,
	fields: tuple[str, ...],
	rows: Iterable[Mapping[str, object]],
	*,
	force: bool = False,
) -> None:
	"""Atomically write tabular rows to a CSV in canonical field order."""
	if output.exists() and not force:
		raise FileExistsError(f"Output already exists: {output} (pass --force to replace it)")
	output.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = None
	try:
		with NamedTemporaryFile(
			"w",
			encoding="utf-8",
			newline="",
			dir=output.parent,
			prefix=f".{output.name}.",
			delete=False,
		) as handle:
			temporary_path = Path(handle.name)
			writer = csv.DictWriter(handle, fieldnames=fields)
			writer.writeheader()
			writer.writerows(rows)
		os.replace(temporary_path, output)
	except OSError:
		if temporary_path is not None:
			temporary_path.unlink(missing_ok=True)
		raise


def build_google_sheets_service(google_conf: Path):
	"""Authenticate and return the Google Sheets v4 spreadsheets service."""
	(
		google_auth_requests,
		google_oauth2_credentials,
		google_auth_oauthlib_flow,
		googleapiclient_discovery,
	) = require(
		(
			"google.auth.transport.requests",
			"google.oauth2.credentials",
			"google_auth_oauthlib.flow",
			"googleapiclient.discovery",
		),
		"google-sheets",
		purpose="Google Sheets upload dependencies are unavailable",
	)
	Request = google_auth_requests.Request
	Credentials = google_oauth2_credentials.Credentials
	InstalledAppFlow = google_auth_oauthlib_flow.InstalledAppFlow
	build = googleapiclient_discovery.build

	token_path = google_conf / "gsheets_token.json"
	credentials_path = google_conf / "gsheets_credentials.json"
	credentials = None
	if token_path.exists():
		credentials = Credentials.from_authorized_user_file(
			token_path,
			GOOGLE_SHEETS_SCOPES,
		)

	if credentials is None or not credentials.valid:
		if credentials and credentials.expired and credentials.refresh_token:
			credentials.refresh(Request())
		else:
			if not credentials_path.exists():
				raise FileNotFoundError(
					f"Google OAuth client credentials not found: {credentials_path}"
				)
			flow = InstalledAppFlow.from_client_secrets_file(
				credentials_path,
				GOOGLE_SHEETS_SCOPES,
			)
			credentials = flow.run_local_server(port=0)
		token_path.write_text(credentials.to_json(), encoding="utf-8")

	return build("sheets", "v4", credentials=credentials).spreadsheets()


def sheet_range(sheet: str, cells: str) -> str:
	"""Return quoted A1 notation, including support for apostrophes in names."""
	quoted_sheet = sheet.replace("'", "''")
	return f"'{quoted_sheet}'!{cells}"


def create_tab_if_missing(
	service,
	spreadsheet: str,
	sheet: str,
	fields: tuple[str, ...],
) -> bool:
	"""Create a missing tab with the canonical header; preserve existing tabs."""
	response = service.get(
		spreadsheetId=spreadsheet,
		fields="sheets.properties.title",
	).execute()
	titles = {
		item.get("properties", {}).get("title")
		for item in response.get("sheets", [])
	}
	if sheet in titles:
		return False

	service.batchUpdate(
		spreadsheetId=spreadsheet,
		body={
			"requests": [
				{
					"addSheet": {
						"properties": {"title": sheet},
					}
				}
			],
		},
	).execute()
	service.values().update(
		spreadsheetId=spreadsheet,
		range=sheet_range(sheet, header_range(fields)),
		valueInputOption="RAW",
		body={
			"majorDimension": "ROWS",
			"values": [list(fields)],
		},
	).execute()
	return True


def verify_header(
	service,
	spreadsheet: str,
	sheet: str,
	fields: tuple[str, ...],
) -> None:
	"""Require the destination header to match the canonical field order."""
	requested_range = sheet_range(sheet, header_range(fields))
	response = service.values().get(
		spreadsheetId=spreadsheet,
		range=requested_range,
	).execute()
	values = response.get("values", [])
	actual = tuple(values[0]) if values else ()
	if actual != fields:
		raise ValueError(
			f"Google Sheet header mismatch in {requested_range}. "
			f"Expected {list(fields)!r}; got {list(actual)!r}. "
			"Select the intended tab, use the default name-matching mode, or pass "
			"--no-verify-header for an intentionally positional upload."
		)


def match_header(
	service,
	spreadsheet: str,
	sheet: str,
	fields: tuple[str, ...],
	rows: Iterable[Mapping[str, object]],
) -> tuple[str, list[list[object | None]]]:
	"""Lay out rows according to exact field names in the live Sheet header."""
	requested_range = sheet_range(sheet, "1:1")
	response = service.values().get(
		spreadsheetId=spreadsheet,
		range=requested_range,
	).execute()
	live_rows = response.get("values", [])
	header = list(live_rows[0]) if live_rows else []
	required = set(fields)
	positions: dict[str, list[int]] = {}
	for index, field in enumerate(header):
		if field in required:
			positions.setdefault(field, []).append(index)

	missing = [field for field in fields if field not in positions]
	duplicates = [field for field in fields if len(positions.get(field, ())) > 1]
	if missing or duplicates:
		problems = []
		if missing:
			problems.append(f"missing required fields {missing!r}")
		if duplicates:
			problems.append(f"duplicate required fields {duplicates!r}")
		raise ValueError(
			f"Google Sheet header mismatch in {requested_range}: "
			f"{'; '.join(problems)}. Header names must match exactly."
		)

	first_column_index = next(
		index for index, field in enumerate(header) if field not in (None, "")
	)
	last_column_index = len(header) - 1
	matched_rows = []
	for row in rows:
		values: list[object | None] = [
			None for _ in range(last_column_index - first_column_index + 1)
		]
		for field, (column_index,) in positions.items():
			values[column_index - first_column_index] = row[field]
		matched_rows.append(values)

	matched_range = (
		f"{column_label(first_column_index + 1)}:"
		f"{column_label(last_column_index + 1)}"
	)
	return matched_range, matched_rows


def append_rows(
	service,
	spreadsheet: str,
	sheet: str,
	fields: tuple[str, ...],
	rows: Iterable[Mapping[str, object]],
	*,
	verify: bool = True,
	strict_header_order: bool = False,
) -> dict:
	"""Append tabular rows to Google Sheets and return the API response."""
	if strict_header_order and not verify:
		raise ValueError(
			"strict header ordering cannot be combined with disabled header verification"
		)
	materialized_rows = list(rows)
	if strict_header_order:
		verify_header(service, spreadsheet, sheet, fields)
		append_range = column_range(fields)
		values = [[row[field] for field in fields] for row in materialized_rows]
	elif verify:
		append_range, values = match_header(
			service,
			spreadsheet,
			sheet,
			fields,
			materialized_rows,
		)
	else:
		append_range = column_range(fields)
		values = [[row[field] for field in fields] for row in materialized_rows]
	return service.values().append(
		spreadsheetId=spreadsheet,
		range=sheet_range(sheet, append_range),
		valueInputOption="RAW",
		insertDataOption="INSERT_ROWS",
		body={"majorDimension": "ROWS", "values": values},
	).execute()


def validate_destination_options(
	*,
	upload: bool,
	create_tab: bool,
	output: Path | None,
	force: bool,
	spreadsheet: str | None,
	sheet: str | None,
	verify: bool,
	strict_header_order: bool,
) -> None:
	"""Validate the common local-CSV versus Google-Sheets CLI contract."""
	if create_tab and not upload:
		raise click.UsageError("--create-tab requires --upload.")
	if upload and output is not None:
		raise click.UsageError("--output cannot be combined with --upload.")
	if upload and force:
		raise click.UsageError("--force applies only to local CSV output.")
	if upload and (not spreadsheet or not sheet):
		raise click.UsageError(
			"--spreadsheet and --sheet are required with --upload "
			"(or set MCTUTIL_GSHEET_ID and MCTUTIL_GSHEET_SHEET)."
		)
	if strict_header_order and not verify:
		raise click.UsageError(
			"--strict-header-order cannot be combined with --no-verify-header."
		)
