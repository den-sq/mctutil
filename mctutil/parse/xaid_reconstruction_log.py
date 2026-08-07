"""Convert a MITOS X-AID reconstruction config to the lab CSV schema."""

from __future__ import annotations

import ast
import configparser
import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path, PurePosixPath
import re
from tempfile import NamedTemporaryFile

import click

from mctutil.shared.deps import require


RECONSTRUCTION_LOG_FIELDS = (
	"o",
	"Type",
	"Sample ID",
	"SSD",
	"Stain",
	"Energy",
	"center*",
	"SIFT ZDP (x,y)",
	"Do Movement Correction?",
	"proj_filter (BF settings)",
	"hoto_tomo_algo (BAC settings)",
	"final_recon_crop",
	"imageJ W",
	"imageJ L",
	"recon offset angle",
	"preview",
	"tight crop",
	"Notes",
)
GOOGLE_SHEETS_SCOPES = ("https://www.googleapis.com/auth/spreadsheets",)


@dataclass(frozen=True)
class ParsedXAidConfig:
	"""Parsed X-AID config plus non-destructive repairs applied while reading."""

	config: configparser.ConfigParser
	repaired_first_header: bool = False


@dataclass(frozen=True)
class SourceIdentity:
	"""Metadata inferred from the X-AID source-data filename convention."""

	date: str = ""
	sample_type: str = ""
	sample_id: str = ""


def _repair_first_section_header(text: str) -> tuple[str, bool]:
	"""Repair X-AID's observed ``a[General_Info]`` first-line corruption."""
	lines = text.splitlines(keepends=True)
	for index, line in enumerate(lines):
		stripped = line.strip()
		if not stripped:
			continue
		if stripped.startswith("["):
			return text, False
		match = re.fullmatch(r"[^\[]+(\[[^\[\]]+\])", stripped)
		if match is None:
			return text, False
		newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
		lines[index] = f"{match.group(1)}{newline}"
		return "".join(lines), True
	return text, False


def parse_xaid_config(path: Path) -> ParsedXAidConfig:
	"""Read an X-AID INI-like config, repairing the known first-line defect."""
	try:
		text = path.read_text(encoding="utf-8-sig")
	except (OSError, UnicodeError) as exc:
		raise ValueError(f"Could not read X-AID config {path}: {exc}") from exc

	text, repaired = _repair_first_section_header(text)
	config = configparser.ConfigParser(interpolation=None)
	try:
		config.read_string(text, source=str(path))
	except configparser.Error as exc:
		raise ValueError(f"Could not parse X-AID config {path}: {exc}") from exc
	return ParsedXAidConfig(config=config, repaired_first_header=repaired)


def _section(config: configparser.ConfigParser, name: str) -> str | None:
	for candidate in config.sections():
		if candidate.casefold() == name.casefold():
			return candidate
	return None


def _get(config: configparser.ConfigParser, section: str, key: str) -> str:
	section_name = _section(config, section)
	if section_name is None:
		return ""
	return config.get(section_name, key, fallback="").strip()


def _decimal(value: str) -> Decimal | None:
	try:
		return Decimal(value)
	except (InvalidOperation, ValueError):
		return None


def _format_decimal(value: Decimal) -> str:
	formatted = format(value, "f")
	if "." in formatted:
		formatted = formatted.rstrip("0").rstrip(".")
	return formatted or "0"


def _list_first(value: str) -> str:
	try:
		parsed = ast.literal_eval(value)
	except (SyntaxError, ValueError):
		return value
	if isinstance(parsed, (list, tuple)) and parsed:
		return str(parsed[0])
	return str(parsed)


def infer_source_identity(path_to_data: str) -> SourceIdentity:
	"""Infer date, type, and sample ID from the observed X-AID path pattern."""
	if not path_to_data:
		return SourceIdentity()
	filename = PurePosixPath(path_to_data.replace("\\", "/")).stem
	filename = re.sub(r"_\d{3,}$", "", filename)
	match = re.fullmatch(r"(?P<date>\d{8})_(?P<body>.+)", filename)
	if match is None:
		return SourceIdentity(sample_id=filename)

	date = match.group("date")
	try:
		date = datetime.strptime(date, "%Y%m%d").date().isoformat()
	except ValueError:
		pass
	body = match.group("body")
	sample_type, separator, sample_id = body.partition("-")
	if not separator:
		return SourceIdentity(date=date, sample_id=body)
	return SourceIdentity(date=date, sample_type=sample_type, sample_id=sample_id)


def _movement_correction(config: configparser.ConfigParser) -> str:
	values = []
	for key in (
		"is_opt_stage_shifts_triangle",
		"is_opt_stage_shifts",
		"is_opt_proj_shifts_2",
		"is_opt_jitter",
	):
		value = _get(config, "Reconstruction_Settings", key)
		if value:
			values.append(value)
	for key in ("drift_x", "drift_y", "drift_z"):
		value = _get(config, "GC_Values", key)
		if value:
			values.append(value)
	if not values:
		return ""

	def is_enabled(value: str) -> bool:
		if value.casefold() in {"true", "yes", "on"}:
			return True
		if value.casefold() in {"false", "no", "off", "none"}:
			return False
		parsed = _decimal(value)
		return parsed is not None and parsed != 0

	return "1" if any(is_enabled(value) for value in values) else "0"


def _projection_filters(config: configparser.ConfigParser) -> str:
	settings = (
		("Proj_Filters", "median_filter_size", "median"),
		("Proj_Filters", "gauss_filter_size", "gaussian"),
		("Proj_Filters", "outlier_size", "outlier size"),
		("Proj_Filters", "outlier_delta", "outlier delta"),
		("Ringfilter_Settings", "ring_partial_delta", "ring partial delta"),
		("Ringfilter_Settings", "ring_partial_size", "ring partial size"),
		("Ringfilter_Settings", "ring_a_size", "ring a"),
		("Ringfilter_Settings", "ring_b_size", "ring b"),
	)
	parts = [
		f"{label}={value}"
		for section, key, label in settings
		if (value := _get(config, section, key))
	]
	return f"X-AID {'; '.join(parts)}" if parts else ""


def _reconstruction_algorithm(config: configparser.ConfigParser) -> str:
	settings = (
		("reco_type", ""),
		("fdk_filter", "filter"),
		("roi_filter", "ROI filter"),
		("apply_roundmask", "round mask"),
		("img_binning", "image binning"),
	)
	parts = []
	for key, label in settings:
		value = _get(config, "Reconstruction_Settings", key)
		if not value:
			continue
		parts.append(value if not label else f"{label}={value}")
	return f"X-AID {'; '.join(parts)}" if parts else ""


def _reconstruction_roi(config: configparser.ConfigParser) -> str:
	position = tuple(_get(config, "ROI", f"roi_pos{axis}") for axis in "xyz")
	size = tuple(_get(config, "ROI", f"roi_size{axis}") for axis in "xyz")
	if not any(position + size):
		return ""
	return f"X-AID ROI pos=({','.join(position)}); size=({','.join(size)})"


def _window_level(config: configparser.ConfigParser) -> tuple[str, str]:
	minimum = _decimal(_get(config, "Final_Image_Settings", "min"))
	maximum = _decimal(_get(config, "Final_Image_Settings", "max"))
	if minimum is None or maximum is None:
		return "", ""
	return (
		_format_decimal(maximum - minimum),
		_format_decimal((maximum + minimum) / Decimal(2)),
	)


def _center_value(
	config: configparser.ConfigParser,
	convention: str,
) -> str:
	if convention not in {"offset", "width-half", "pixel-center"}:
		raise ValueError(f"Unknown center convention: {convention}")
	offset_text = _list_first(_get(config, "GC_Values", "rotation_axis_offset"))
	if not offset_text:
		return ""
	if convention == "offset":
		return f"X-AID rotation-axis offset: {offset_text}"

	offset = _decimal(offset_text)
	width = _decimal(_get(config, "Detektor_Info", "num_px_u"))
	if offset is None or width is None:
		raise ValueError(
			f"Center convention {convention!r} requires numeric rotation_axis_offset and num_px_u."
		)
	base = width / Decimal(2)
	if convention == "pixel-center":
		base = (width - Decimal(1)) / Decimal(2)
	return _format_decimal(base + offset)


def _geometry_magnification(config: configparser.ConfigParser) -> str:
	sod = _decimal(_get(config, "Reconstruction_Settings", "sod"))
	sdd = _decimal(_get(config, "Reconstruction_Settings", "sdd"))
	if sod is None or sdd is None or sod == 0:
		return ""
	return _format_decimal(sdd / sod)


def _source_notes(
	config: configparser.ConfigParser,
	identity: SourceIdentity,
) -> list[str]:
	parts = []
	software_version = _get(config, "General_Info", "software_version")
	if software_version:
		parts.append(f"X-AID {software_version}")
	if identity.date:
		parts.append(f"source date={identity.date}")
	path_to_data = _get(config, "General_Info", "path_to_data")
	if path_to_data:
		parts.append(f"source={path_to_data}")
	save_path = _get(config, "Final_Image_Settings", "save_path")
	if save_path:
		parts.append(f"output={save_path}")
	return parts


def _detector_notes(config: configparser.ConfigParser) -> list[str]:
	parts = []
	detector_size = "x".join(
		_get(config, "Detektor_Info", key) for key in ("num_px_u", "num_px_v")
	)
	pixel_size = _get(config, "Detektor_Info", "pixel_size")
	if detector_size != "x" and pixel_size:
		parts.append(f"detector={detector_size}; pixel size={pixel_size}")
	elif detector_size != "x":
		parts.append(f"detector={detector_size}")

	voxel_size = _get(config, "VolumeData", "voxel_size")
	if voxel_size:
		parts.append(f"reconstruction voxel={voxel_size}")
	return parts


def _geometry_notes(config: configparser.ConfigParser) -> list[str]:
	parts = []
	sod = _get(config, "Reconstruction_Settings", "sod")
	sdd = _get(config, "Reconstruction_Settings", "sdd")
	if sod or sdd:
		parts.append(f"SOD={sod or 'unknown'}; SDD={sdd or 'unknown'} (X-AID config units)")
	magnification = _geometry_magnification(config)
	if magnification:
		parts.append(f"geometric magnification={magnification}")
	return parts


def _output_notes(config: configparser.ConfigParser) -> list[str]:
	parts = []
	export_type = _get(config, "Final_Image_Settings", "export_type")
	export_order = _get(config, "Final_Image_Settings", "export_order")
	if export_type or export_order:
		parts.append(f"export={export_type or 'unknown'}/{export_order or 'unknown'}")
	minimum = _get(config, "Final_Image_Settings", "min")
	maximum = _get(config, "Final_Image_Settings", "max")
	if minimum or maximum:
		parts.append(f"window min={minimum or 'unknown'} max={maximum or 'unknown'}")
	return parts


def _notes(
	config: configparser.ConfigParser,
	identity: SourceIdentity,
	center_is_raw_offset: bool,
	missing_fields: tuple[str, ...],
	extra_notes: str,
) -> str:
	parts = _source_notes(config, identity)
	parts.extend(_detector_notes(config))
	parts.extend(_geometry_notes(config))
	parts.extend(_output_notes(config))
	if center_is_raw_offset:
		parts.append(
			"center preserved as the X-AID offset; select an absolute center convention if verified"
		)
	if missing_fields:
		parts.append(f"not present in config: {', '.join(missing_fields)}")
	if extra_notes:
		parts.append(extra_notes.strip())
	return "; ".join(parts)


def build_reconstruction_log_row(
	config: configparser.ConfigParser,
	*,
	scan_number: str = "",
	sample_type: str | None = None,
	sample_id: str | None = None,
	ssd: str = "",
	stain: str = "",
	energy: str = "",
	center: str | None = None,
	center_convention: str = "offset",
	sift_zdp: str = "",
	preview: str = "",
	tight_crop: str = "",
	extra_notes: str = "",
) -> dict[str, str]:
	"""Build one row compatible with columns A-R of the reconstruction log."""
	path_to_data = _get(config, "General_Info", "path_to_data")
	identity = infer_source_identity(path_to_data)
	resolved_type = identity.sample_type if sample_type is None else sample_type
	resolved_sample = identity.sample_id if sample_id is None else sample_id
	window_width, window_level = _window_level(config)
	center_value = center if center is not None else _center_value(config, center_convention)

	missing_values = {
		"scan number": scan_number,
		"SSD": ssd,
		"stain": stain,
		"energy": energy,
		"SIFT ZDP": sift_zdp,
		"preview": preview,
		"tight crop": tight_crop,
	}
	missing_fields = tuple(name for name, value in missing_values.items() if not value)
	row = {
		"o": scan_number,
		"Type": resolved_type,
		"Sample ID": resolved_sample,
		"SSD": ssd,
		"Stain": stain,
		"Energy": energy,
		"center*": center_value,
		"SIFT ZDP (x,y)": sift_zdp,
		"Do Movement Correction?": _movement_correction(config),
		"proj_filter (BF settings)": _projection_filters(config),
		"hoto_tomo_algo (BAC settings)": _reconstruction_algorithm(config),
		"final_recon_crop": _reconstruction_roi(config),
		"imageJ W": window_width,
		"imageJ L": window_level,
		"recon offset angle": _get(config, "VolumeData", "volume_rotation_z"),
		"preview": preview,
		"tight crop": tight_crop,
		"Notes": _notes(
			config,
			identity,
			center is None and center_convention == "offset",
			missing_fields,
			extra_notes,
		),
	}
	return row


def write_reconstruction_log_csv(
	output: Path,
	row: dict[str, str],
	*,
	force: bool = False,
) -> None:
	"""Atomically write a one-row reconstruction-log CSV."""
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
			writer = csv.DictWriter(handle, fieldnames=RECONSTRUCTION_LOG_FIELDS)
			writer.writeheader()
			writer.writerow(row)
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


def _sheet_range(sheet: str, cells: str) -> str:
	"""Return quoted A1 notation, including support for apostrophes in tab names."""
	quoted_sheet = sheet.replace("'", "''")
	return f"'{quoted_sheet}'!{cells}"


def verify_reconstruction_log_header(service, spreadsheet: str, sheet: str) -> None:
	"""Require columns A-R to match the converter's target schema."""
	header_range = _sheet_range(sheet, "A1:R1")
	response = service.values().get(
		spreadsheetId=spreadsheet,
		range=header_range,
	).execute()
	values = response.get("values", [])
	actual = tuple(values[0]) if values else ()
	if actual != RECONSTRUCTION_LOG_FIELDS:
		raise ValueError(
			f"Google Sheet header mismatch in {header_range}. "
			f"Expected {list(RECONSTRUCTION_LOG_FIELDS)!r}; got {list(actual)!r}. "
			"Select the Reconstructions tab or pass --no-verify-header to override."
		)


def append_reconstruction_log_row(
	service,
	spreadsheet: str,
	sheet: str,
	row: dict[str, str],
	*,
	verify_header: bool = True,
) -> dict:
	"""Append one reconstruction row to a Google Sheet and return its response."""
	if verify_header:
		verify_reconstruction_log_header(service, spreadsheet, sheet)
	values = [row[field] for field in RECONSTRUCTION_LOG_FIELDS]
	return service.values().append(
		spreadsheetId=spreadsheet,
		range=_sheet_range(sheet, "A:R"),
		valueInputOption="RAW",
		insertDataOption="INSERT_ROWS",
		body={"majorDimension": "ROWS", "values": [values]},
	).execute()


def _validate_destination_options(
	*,
	upload: bool,
	output: Path | None,
	force: bool,
	spreadsheet: str | None,
	sheet: str | None,
) -> None:
	if upload and output is not None:
		raise click.UsageError("--output cannot be combined with --upload.")
	if upload and force:
		raise click.UsageError("--force applies only to local CSV output.")
	if upload and (not spreadsheet or not sheet):
		raise click.UsageError(
			"--spreadsheet and --sheet are required with --upload "
			"(or set MCTUTIL_GSHEET_ID and MCTUTIL_GSHEET_SHEET)."
		)


def _upload_reconstruction_log(
	google_conf: Path,
	spreadsheet: str,
	sheet: str,
	row: dict[str, str],
	verify_header: bool,
) -> str:
	try:
		service = build_google_sheets_service(google_conf)
		response = append_reconstruction_log_row(
			service,
			spreadsheet,
			sheet,
			row,
			verify_header=verify_header,
		)
	except Exception as exc:
		raise click.ClickException(f"Google Sheets upload failed: {exc}") from exc
	return response.get("updates", {}).get("updatedRange", "unknown range")


def _write_local_reconstruction_log(
	config_path: Path,
	output: Path | None,
	row: dict[str, str],
	force: bool,
) -> Path:
	if output is None:
		output = config_path.with_name(f"{config_path.stem}_reconstruction_log.csv")
	try:
		write_reconstruction_log_csv(output, row, force=force)
	except OSError as exc:
		raise click.ClickException(str(exc)) from exc
	return output


@click.command("xaid-log")
@click.argument(
	"config_path",
	type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
	"--output",
	"-o",
	type=click.Path(dir_okay=False, path_type=Path),
	help="Output CSV. Defaults to CONFIG_reconstruction_log.csv beside CONFIG.",
)
@click.option(
	"--upload",
	is_flag=True,
	help="Append to Google Sheets instead of writing a local CSV.",
)
@click.option(
	"--spreadsheet",
	default=lambda: os.environ.get("MCTUTIL_GSHEET_ID"),
	help="Destination spreadsheet ID for --upload; may use MCTUTIL_GSHEET_ID.",
)
@click.option(
	"--sheet",
	default=lambda: os.environ.get("MCTUTIL_GSHEET_SHEET"),
	help="Destination tab for --upload; may use MCTUTIL_GSHEET_SHEET.",
)
@click.option(
	"--google-conf",
	type=click.Path(path_type=Path),
	default=lambda: Path(os.environ.get("MCTUTIL_GOOGLE_CONF", "conf")),
	show_default="conf",
	help="Directory containing gsheets_credentials.json and gsheets_token.json.",
)
@click.option(
	"--verify-header/--no-verify-header",
	default=True,
	show_default=True,
	help="Verify destination columns A-R before uploading.",
)
@click.option("--scan-number", default="", help="Trip scan/log row number for column o.")
@click.option("--sample-type", help="Override Type inferred from the source filename.")
@click.option("--sample-id", help="Override Sample ID inferred from the source filename.")
@click.option("--ssd", default="", help="Sample-to-scintillator distance; not inferred from SOD/SDD.")
@click.option("--stain", default="", help="Sample stain metadata absent from X-AID configs.")
@click.option("--energy", default="", help="Acquisition energy metadata absent from X-AID configs.")
@click.option("--center", help="Explicit center value, overriding --center-convention.")
@click.option(
	"--center-convention",
	type=click.Choice(("offset", "width-half", "pixel-center"), case_sensitive=False),
	default="offset",
	show_default=True,
	help=(
		"offset preserves X-AID's raw offset; width-half adds it to width/2; "
		"pixel-center adds it to (width-1)/2."
	),
)
@click.option("--sift-zdp", default="", help="Optional SIFT ZDP value for the target log.")
@click.option("--preview", default="", help="Optional preview value for the target log.")
@click.option("--tight-crop", default="", help="Optional tight-crop value for the target log.")
@click.option("--notes", "extra_notes", default="", help="Notes appended to generated provenance.")
@click.option("--force", is_flag=True, help="Replace OUTPUT if it already exists.")
def xaid_log(
	config_path: Path,
	output: Path | None,
	upload: bool,
	spreadsheet: str | None,
	sheet: str | None,
	google_conf: Path,
	verify_header: bool,
	scan_number: str,
	sample_type: str | None,
	sample_id: str | None,
	ssd: str,
	stain: str,
	energy: str,
	center: str | None,
	center_convention: str,
	sift_zdp: str,
	preview: str,
	tight_crop: str,
	extra_notes: str,
	force: bool,
) -> None:
	"""Convert CONFIG_PATH to a local or Google Sheets reconstruction log."""
	_validate_destination_options(
		upload=upload,
		output=output,
		force=force,
		spreadsheet=spreadsheet,
		sheet=sheet,
	)
	try:
		parsed = parse_xaid_config(config_path)
		row = build_reconstruction_log_row(
			parsed.config,
			scan_number=scan_number,
			sample_type=sample_type,
			sample_id=sample_id,
			ssd=ssd,
			stain=stain,
			energy=energy,
			center=center,
			center_convention=center_convention,
			sift_zdp=sift_zdp,
			preview=preview,
			tight_crop=tight_crop,
			extra_notes=extra_notes,
		)
	except (OSError, ValueError) as exc:
		raise click.ClickException(str(exc)) from exc

	if parsed.repaired_first_header:
		click.echo(
			"Warning: interpreted the malformed first section header as [General_Info].",
			err=True,
		)

	if upload:
		updated_range = _upload_reconstruction_log(
			google_conf,
			spreadsheet,
			sheet,
			row,
			verify_header,
		)
		click.echo(f"Appended reconstruction log row: {updated_range}")
		return

	output = _write_local_reconstruction_log(config_path, output, row, force)
	click.echo(f"Wrote reconstruction log CSV: {output}")


if __name__ == "__main__":
	xaid_log()
