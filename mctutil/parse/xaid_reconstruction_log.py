"""Convert a MITOS X-AID reconstruction config to clear-name log fields."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
import os
from pathlib import Path
import re

import click

from mctutil.parse import tabular_log


XAID_CONFIG_FIELD_MAPPING = (
	("General_Info", "software_version", "Software Release Version"),
	("General_Info", "path_to_data", "Input Projection Data File"),
	("Detektor_Info", "pixel_size", "Detector Pixel Pitch (mm)"),
	("Detektor_Info", "num_px_u", "Detector Width (pixels)"),
	("Detektor_Info", "num_px_v", "Detector Height (pixels)"),
	("ROI", "roi_posx", "Sub-Volume Start, X"),
	("ROI", "roi_posy", "Sub-Volume Start, Y"),
	("ROI", "roi_posz", "Sub-Volume Start, Z"),
	("ROI", "roi_sizex", "Sub-Volume Extent, X"),
	("ROI", "roi_sizey", "Sub-Volume Extent, Y"),
	("ROI", "roi_sizez", "Sub-Volume Extent, Z"),
	("VolumeData", "volume_rotation_x", "Output Volume Rotation, X (°)"),
	("VolumeData", "volume_rotation_y", "Output Volume Rotation, Y (°)"),
	("VolumeData", "volume_rotation_z", "Output Volume Rotation, Z (°)"),
	("VolumeData", "voxel_size", "Reconstructed Voxel Size (mm)"),
	("VolumeData", "nat_vx_size", "Native Voxel Size at Magnification (mm)"),
	("Proj_Filters", "median_filter_size", "Projection Median-Filter Kernel (px)"),
	("Proj_Filters", "gauss_filter_size", "Projection Gaussian Smoothing Sigma (px)"),
	("Proj_Filters", "outlier_size", "Hot-Pixel Removal Kernel (px)"),
	("Proj_Filters", "outlier_delta", "Hot-Pixel Detection Threshold"),
	("Ringfilter_Settings", "ring_partial_delta", "Partial-Ring Filter Strength"),
	("Ringfilter_Settings", "ring_partial_size", "Partial-Ring Filter Kernel (px)"),
	("Ringfilter_Settings", "ring_a_size", "Ring Filter Stage A Kernel (px)"),
	("Ringfilter_Settings", "ring_b_size", "Ring Filter Stage B Kernel (px)"),
	("Reconstruction_Settings", "sod", "Source-to-Object Distance (mm)"),
	("Reconstruction_Settings", "sdd", "Source-to-Detector Distance (mm)"),
	("Reconstruction_Settings", "freeray", "Air-Normalization Region"),
	("Reconstruction_Settings", "reco_type", "Reconstruction Algorithm"),
	("Reconstruction_Settings", "reg_type", "Iterative Regularization Method"),
	("Reconstruction_Settings", "iterations", "Iterative Reconstruction Iteration Count"),
	("Reconstruction_Settings", "reg_threshold", "Regularization Threshold"),
	("Reconstruction_Settings", "reg_lambda", "Regularization Weight"),
	("Reconstruction_Settings", "apply_roundmask", "Cylindrical Field-of-View Mask"),
	("Reconstruction_Settings", "fdk_filter", "Ramp-Filter Apodization Window"),
	("Reconstruction_Settings", "roi_filter", "Truncation-Aware ROI Filtering ⚠"),
	("Reconstruction_Settings", "is_short_scan", "Short-Scan Mode (<360°)"),
	("Reconstruction_Settings", "is_offset_scan", "Extended-FOV Offset Scan"),
	("Reconstruction_Settings", "is_detector_offset", "Detector Lateral Offset Flag"),
	("Reconstruction_Settings", "offset_shift", "Offset-Scan Shift Amount (px)"),
	("Reconstruction_Settings", "img_binning", "Effective Projection Binning Factor"),
	("Reconstruction_Settings", "gui_img_binning", "User-Selected Binning Factor"),
	("Reconstruction_Settings", "is_compute_opt_oblique_slice_y", "Auto-Optimize: Oblique Slice Computation"),
	("Reconstruction_Settings", "is_opt_oblique_geom", "Auto-Optimize: Oblique Geometry"),
	("Reconstruction_Settings", "is_opt_proj_offset_v", "Auto-Optimize: Vertical Projection Offset"),
	("Reconstruction_Settings", "is_opt_proj_slant_v", "Auto-Optimize: Vertical Projection Slant"),
	("Reconstruction_Settings", "is_opt_oblique_angle", "Auto-Optimize: Oblique Angle"),
	("Reconstruction_Settings", "is_opt_stage_shifts_triangle", "Auto-Optimize: Stage Shifts (Triangulation) ⚠"),
	("Reconstruction_Settings", "is_opt_stage_shifts", "Auto-Optimize: Stage Shifts"),
	("Reconstruction_Settings", "is_opt_proj_shifts_2", "Auto-Optimize: Per-Projection Shifts (Variant 2) ⚠"),
	("Reconstruction_Settings", "is_align_axis", "Auto-Optimize: Rotation-Axis Alignment"),
	("Reconstruction_Settings", "is_opt_triangle", "Auto-Optimize: Triangulation Alignment ⚠"),
	("Reconstruction_Settings", "is_opt_jitter", "Auto-Optimize: Jitter Correction"),
	("Reconstruction_Settings", "is_jitter_smooth", "Jitter-Correction Smoothing"),
	("Reconstruction_Settings", "jitter_iterations", "Jitter-Correction Iteration Count"),
	("Reconstruction_Settings", "shift_u_max", "Shift Search Range, Horizontal (px)"),
	("Reconstruction_Settings", "shift_v_max", "Shift Search Range, Vertical (px)"),
	("Reconstruction_Settings", "oblique_opt_slice_y", "Oblique-Optimization Reference Slice"),
	("BHC_Values", "bhc_cupping_value", "Beam-Hardening Cupping Correction Strength"),
	("BHC_Values", "bhc_streak_value", "Beam-Hardening Streak Correction Strength"),
	("BHC_Values", "bhc_streak_thl_value", "Streak Correction Density Threshold"),
	("BHC_Values", "starvation_value", "Photon-Starvation Correction Strength"),
	("BHC_Values", "scatter_basic_value", "Basic Scatter Correction Strength"),
	("BHC_Values", "dual_energy_factor", "Dual-Energy Blending Factor"),
	("GC_Values", "rotation_axis_offset", "Center-of-Rotation Shift (px)"),
	("GC_Values", "cs_y_slice", "Center-Shift Reference Detector Row"),
	("GC_Values", "rotation_axis_tilt", "Rotation-Axis In-Plane Tilt (°)"),
	("GC_Values", "tilt_y_slice_min", "Tilt-Determination Lower Slice"),
	("GC_Values", "tilt_y_slice_max", "Tilt-Determination Upper Slice"),
	("GC_Values", "drift_x", "Stage/Source Drift Correction, X"),
	("GC_Values", "drift_y", "Stage/Source Drift Correction, Y"),
	("GC_Values", "drift_z", "Stage/Source Drift Correction, Z"),
	("TScan", "is_opt_tscan", "Translation-Scan Optimization"),
	("TScan", "tscan_max_shift", "Translation-Scan Max Shift Search (px)"),
	("Global_Reco_Settings", "gcf", "Geometry Correction File/Factor ⚠"),
	("Global_Reco_Settings", "rotation_axis_tilt", "Global Rotation-Axis Tilt (°)"),
	("Global_Reco_Settings", "rotation_axis_slant", "Rotation-Axis Slant (°)"),
	("Global_Reco_Settings", "det_rot_eta", "Detector Rotation η (°)"),
	("Global_Reco_Settings", "det_rot_theta", "Detector Rotation θ (°)"),
	("Global_Reco_Settings", "det_rot_phi", "Detector Rotation φ (°)"),
	("Global_Reco_Settings", "det_offset_u", "Detector Offset, Horizontal (px)"),
	("Global_Reco_Settings", "det_offset_v", "Detector Offset, Vertical (px)"),
	("Final_Image_Settings", "save_path", "Output Slice-Stack Path Template"),
	("Final_Image_Settings", "min", "Grayscale Mapping: Black Point"),
	("Final_Image_Settings", "max", "Grayscale Mapping: White Point"),
	("Final_Image_Settings", "export_type", "Output File Format"),
	("Final_Image_Settings", "export_order", "Output Axis Ordering"),
	("Final_Image_Settings", "recolocation", "Reconstruction Output Location"),
	("Final_Image_Settings", "location_string", "Internal Export Descriptor ⚠"),
)


_column_label = tabular_log.column_label


RECONSTRUCTION_LOG_FIELDS = tuple(item[2] for item in XAID_CONFIG_FIELD_MAPPING)
RECONSTRUCTION_LOG_LAST_COLUMN = _column_label(len(RECONSTRUCTION_LOG_FIELDS))
RECONSTRUCTION_LOG_COLUMN_RANGE = f"A:{RECONSTRUCTION_LOG_LAST_COLUMN}"
RECONSTRUCTION_LOG_HEADER_RANGE = f"A1:{RECONSTRUCTION_LOG_LAST_COLUMN}1"
GOOGLE_SHEETS_SCOPES = tabular_log.GOOGLE_SHEETS_SCOPES


@dataclass(frozen=True)
class ParsedXAidConfig:
	"""Parsed X-AID config plus non-destructive repairs applied while reading."""

	config: configparser.ConfigParser
	repaired_first_header: bool = False


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


def build_reconstruction_log_row(
	config: configparser.ConfigParser,
) -> dict[str, str]:
	"""Build one row using the authoritative X-AID clear-name field mapping."""
	return {
		output_name: _get(config, section, key)
		for section, key, output_name in XAID_CONFIG_FIELD_MAPPING
	}


def write_reconstruction_log_csv(
	output: Path,
	row: dict[str, str],
	*,
	force: bool = False,
) -> None:
	"""Atomically write a one-row reconstruction-log CSV."""
	tabular_log.write_csv(
		output,
		RECONSTRUCTION_LOG_FIELDS,
		[row],
		force=force,
	)


def build_google_sheets_service(google_conf: Path):
	"""Authenticate and return the Google Sheets v4 spreadsheets service."""
	return tabular_log.build_google_sheets_service(google_conf)


def _sheet_range(sheet: str, cells: str) -> str:
	"""Return quoted A1 notation, including support for apostrophes in tab names."""
	return tabular_log.sheet_range(sheet, cells)


def create_reconstruction_log_tab_if_missing(
	service,
	spreadsheet: str,
	sheet: str,
) -> bool:
	"""Create a missing tab with the reconstruction header; preserve existing tabs."""
	return tabular_log.create_tab_if_missing(
		service,
		spreadsheet,
		sheet,
		RECONSTRUCTION_LOG_FIELDS,
	)


def verify_reconstruction_log_header(service, spreadsheet: str, sheet: str) -> None:
	"""Require the mapped columns to match the converter's canonical order."""
	tabular_log.verify_header(
		service,
		spreadsheet,
		sheet,
		RECONSTRUCTION_LOG_FIELDS,
	)


def _match_reconstruction_log_header(
	service,
	spreadsheet: str,
	sheet: str,
	row: dict[str, str],
) -> tuple[str, list[str | None]]:
	"""Lay out one row according to the live Sheet header names."""
	matched_range, values = tabular_log.match_header(
		service,
		spreadsheet,
		sheet,
		RECONSTRUCTION_LOG_FIELDS,
		[row],
	)
	return matched_range, values[0]


def append_reconstruction_log_row(
	service,
	spreadsheet: str,
	sheet: str,
	row: dict[str, str],
	*,
	verify_header: bool = True,
	strict_header_order: bool = False,
) -> dict:
	"""Append one reconstruction row to a Google Sheet and return its response."""
	return tabular_log.append_rows(
		service,
		spreadsheet,
		sheet,
		RECONSTRUCTION_LOG_FIELDS,
		[row],
		verify=verify_header,
		strict_header_order=strict_header_order,
	)


def _validate_destination_options(
	*,
	upload: bool,
	create_tab: bool,
	output: Path | None,
	force: bool,
	spreadsheet: str | None,
	sheet: str | None,
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


def _upload_reconstruction_log(
	google_conf: Path,
	spreadsheet: str,
	sheet: str,
	row: dict[str, str],
	verify_header: bool,
	strict_header_order: bool,
	create_tab: bool,
) -> tuple[str, bool]:
	try:
		service = build_google_sheets_service(google_conf)
		created_tab = False
		if create_tab:
			created_tab = create_reconstruction_log_tab_if_missing(
				service,
				spreadsheet,
				sheet,
			)
		response = append_reconstruction_log_row(
			service,
			spreadsheet,
			sheet,
			row,
			verify_header=verify_header,
			strict_header_order=strict_header_order,
		)
	except Exception as exc:
		raise click.ClickException(f"Google Sheets upload failed: {exc}") from exc
	updated_range = response.get("updates", {}).get("updatedRange", "unknown range")
	return updated_range, created_tab


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
	"--create-tab",
	is_flag=True,
	help="Create a missing destination tab with the expected mapped header.",
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
	help=(
		"Match destination columns by exact header name before uploading. "
		"Disabling this uses the legacy positional order."
	),
)
@click.option(
	"--strict-header-order",
	is_flag=True,
	help="Require the canonical A:CJ header order instead of matching by name.",
)
@click.option("--force", is_flag=True, help="Replace OUTPUT if it already exists.")
def xaid_log(
	config_path: Path,
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
	"""Convert CONFIG_PATH to a local or Google Sheets reconstruction log."""
	_validate_destination_options(
		upload=upload,
		create_tab=create_tab,
		output=output,
		force=force,
		spreadsheet=spreadsheet,
		sheet=sheet,
		verify_header=verify_header,
		strict_header_order=strict_header_order,
	)
	try:
		parsed = parse_xaid_config(config_path)
		row = build_reconstruction_log_row(parsed.config)
	except (OSError, ValueError) as exc:
		raise click.ClickException(str(exc)) from exc

	if parsed.repaired_first_header:
		click.echo(
			"Warning: interpreted the malformed first section header as [General_Info].",
			err=True,
		)

	if upload:
		updated_range, created_tab = _upload_reconstruction_log(
			google_conf,
			spreadsheet,
			sheet,
			row,
			verify_header,
			strict_header_order,
			create_tab,
		)
		if created_tab:
			click.echo(f"Created Google Sheets tab with reconstruction header: {sheet}")
		click.echo(f"Appended reconstruction log row: {updated_range}")
		return

	output = _write_local_reconstruction_log(config_path, output, row, force)
	click.echo(f"Wrote reconstruction log CSV: {output}")


if __name__ == "__main__":
	xaid_log()
