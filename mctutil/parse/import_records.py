"""Plain canonical records and ordered field definitions for log imports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import re

import numpy as np


@dataclass(frozen=True)
class FieldDefinition:
	"""One canonical value's name, runtime type, unit, and CSV header."""

	name: str
	value_type: type
	header: str
	unit: str | None = None
	required: bool = False


@dataclass
class ScanRecord:
	"""Canonical values returned by one scan source reader."""

	source_id: str
	source_locations: tuple[str, ...]
	values: dict[str, object | None]
	warnings: list[str] = field(default_factory=list)


@dataclass
class ReconstructionRecord:
	"""Canonical values returned by one reconstruction source reader."""

	source_id: str
	source_locations: tuple[str, ...]
	values: dict[str, object | None]
	warnings: list[str] = field(default_factory=list)


def decode_scalar(value) -> str:
	"""Decode one source scalar to stripped text."""
	if isinstance(value, (bytes, np.bytes_)):
		return bytes(value).decode("utf-8", errors="replace").strip()
	return str(value).strip()


def parse_bool(value) -> bool | None:
	"""Parse the boolean spellings observed across importer sources."""
	if type(value) is bool:
		return value
	if isinstance(value, (int, float)) and not isinstance(value, bool):
		return bool(value)
	text = decode_scalar(value).casefold()
	if text in {"1", "true", "yes", "y", "on", "x"}:
		return True
	if text in {"0", "false", "no", "n", "off", ""}:
		return False
	return None


def parse_float(value) -> float | None:
	"""Parse one finite numeric source scalar."""
	if value is None or type(value) is bool:
		return None
	if isinstance(value, (int, float, np.number)):
		result = float(value)
	else:
		match = re.search(
			r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
			decode_scalar(value),
		)
		if match is None:
			return None
		result = float(match.group(0))
	return result if math.isfinite(result) else None


def parse_int(value) -> int | None:
	"""Parse one integer-valued source scalar."""
	number = parse_float(value)
	if number is None or not math.isclose(number, round(number), abs_tol=1e-6):
		return None
	return int(round(number))


def parse_datetime(value) -> datetime | None:
	"""Parse one timezone-declared source timestamp and normalize it to UTC."""
	if isinstance(value, datetime):
		result = value
	else:
		text = decode_scalar(value).replace("Z", "+00:00")
		try:
			result = datetime.fromisoformat(text)
		except ValueError:
			return None
	if result.tzinfo is None:
		return None
	return result.astimezone(timezone.utc)


def coerce_value(value, value_type: type):
	"""Coerce one source scalar to its declared canonical field type."""
	if value is None:
		return None
	if isinstance(value, str) and value.strip().casefold() in {"", "none", "null", "nan"}:
		return None
	if value_type is str:
		return decode_scalar(value)
	if value_type is int:
		return parse_int(value)
	if value_type is float:
		return parse_float(value)
	if value_type is bool:
		return parse_bool(value)
	if value_type is datetime:
		return parse_datetime(value)
	if value_type is tuple and isinstance(value, (list, tuple)):
		return tuple(value)
	return None


def apply_direct_mapping(
	mapping: Mapping[str, Mapping[str, object]],
	field_by_name: Mapping[str, FieldDefinition],
	read_value: Callable[[object], object],
	values: dict[str, object | None],
) -> None:
	"""Apply only direct locators, numeric multipliers, and declared types."""
	for field_name, entry in mapping.items():
		raw_value = read_value(entry["locator"])
		if raw_value is None:
			continue
		if "multiplier" in entry:
			number = parse_float(raw_value)
			raw_value = number * entry["multiplier"] if number is not None else None
		values[field_name] = coerce_value(raw_value, field_by_name[field_name].value_type)


def _f(
	name: str,
	value_type: type,
	header: str,
	unit: str | None = None,
	*,
	required: bool = False,
) -> FieldDefinition:
	return FieldDefinition(name, value_type, header, unit, required)


SCAN_FIELDS = (
	_f("facility_name", str, "Facility"),
	_f("acquisition_system_id", str, "Acquisition System"),
	_f("project_id", str, "Project ID"),
	_f("sample_id", str, "Sample ID"),
	_f("scan_id", str, "Scan ID", required=True),
	_f("acquisition_group", str, "Acquisition Group"),
	_f("fov_index", str, "FOV Index"),
	_f("scan_method", str, "Scan Method"),
	_f("projection_data_file", str, "Projection Data File"),
	_f("source_metadata_file", str, "Source Metadata File"),
	_f("scan_log_file", str, "Scan Log File"),
	_f("operator", str, "Operator"),
	_f("sample_stain", str, "Sample Stain"),
	_f("notes", str, "Notes"),
	_f("projection_count_planned", int, "Planned Projections", "count"),
	_f("projection_count_acquired", int, "Acquired Projections", "count"),
	_f("rotation_start_deg", float, "Rotation Start (deg)", "deg"),
	_f("rotation_stop_planned_deg", float, "Planned Rotation Stop (deg)", "deg"),
	_f("rotation_stop_actual_deg", float, "Actual Rotation Stop (deg)", "deg"),
	_f("angular_step_deg", float, "Angular Step (deg)", "deg"),
	_f("angular_range_deg", float, "Angular Range (deg)", "deg"),
	_f("fov_count", int, "FOV Count", "count"),
	_f("helical", bool, "Helical"),
	_f("exposure_us", float, "Exposure (us)", "us"),
	_f("flat_exposure_us", float, "Flat Exposure (us)", "us"),
	_f("exposures_per_projection", int, "Exposures per Projection", "count"),
	_f("trigger_period_us", float, "Trigger Period (us)", "us"),
	_f("trigger_overhead_us", float, "Trigger Overhead (us)", "us"),
	_f("hardware_trigger_enabled", bool, "Hardware Trigger Enabled"),
	_f("scan_start", datetime, "Scan Start (UTC)"),
	_f("scan_stop", datetime, "Scan Stop (UTC)"),
	_f("scan_duration_s", float, "Scan Duration (s)", "s"),
	_f("image_width_px", int, "Image Width (px)", "px"),
	_f("image_height_px", int, "Image Height (px)", "px"),
	_f("stored_dtype", str, "Stored Data Type"),
	_f("stored_bit_depth", int, "Stored Bit Depth", "bit"),
	_f("binning_x", int, "Binning X"),
	_f("binning_y", int, "Binning Y"),
	_f("detector_pixel_pitch_mm", float, "Detector Pixel Pitch (mm)", "mm"),
	_f("effective_pixel_size_mm", float, "Effective Pixel Size (mm)", "mm"),
	_f("geometric_magnification", float, "Geometric Magnification"),
	_f("physical_fov_width_mm", float, "Physical FOV Width (mm)", "mm"),
	_f("physical_fov_height_mm", float, "Physical FOV Height (mm)", "mm"),
	_f("crop_enabled", bool, "Crop Enabled"),
	_f("crop_offset_x_px", int, "Crop Offset X (px)", "px"),
	_f("crop_offset_y_px", int, "Crop Offset Y (px)", "px"),
	_f("detector_gain", float, "Detector Gain"),
	_f("sample_stage_x_mm", float, "Sample Stage X (mm)", "mm"),
	_f("sample_stage_y_mm", float, "Sample Stage Y (mm)", "mm"),
	_f("sample_stage_z_mm", float, "Sample Stage Z (mm)", "mm"),
	_f("sample_stage_y_stop_planned_mm", float, "Planned Sample Stage Y Stop (mm)", "mm"),
	_f("sample_stage_y_stop_actual_mm", float, "Actual Sample Stage Y Stop (mm)", "mm"),
	_f("detector_stage_z_mm", float, "Detector Stage Z (mm)", "mm"),
	_f("source_stage_z_mm", float, "Source Stage Z (mm)", "mm"),
	_f("sample_detector_distance_mm", float, "Sample-Detector Distance (mm)", "mm"),
	_f("source_energy_kev", float, "Source Energy (keV)", "keV"),
	_f("source_power", str, "Source Power"),
	_f("source_target", str, "Source Target"),
	_f("flat_reference_files", tuple, "Flat Reference Files"),
	_f("flat_frame_count", int, "Flat Frames", "count"),
	_f("flat_frame_count_pre", int, "Pre-Scan Flat Frames", "count"),
	_f("flat_frame_count_post", int, "Post-Scan Flat Frames", "count"),
	_f("dark_reference_files", tuple, "Dark Reference Files"),
	_f("dark_frame_count", int, "Dark Frames", "count"),
	_f("post_reference_files", tuple, "Post Reference Files"),
	_f("projection_frame_size_mb", float, "Projection Frame Size (MB)", "MB"),
	_f("scan_file_size_gb", float, "Scan File Size (GB)", "GB"),
	_f("dropped_frames", int, "Dropped Frames", "count"),
	_f("incomplete_frames", int, "Incomplete Frames", "count"),
	_f("acquisition_status", str, "Acquisition Status"),
)


RECONSTRUCTION_FIELDS = (
	_f("reconstruction_id", str, "Reconstruction ID", required=True),
	_f("project_id", str, "Project ID"),
	_f("sample_id", str, "Sample ID"),
	_f("scan_id", str, "Scan ID"),
	_f("input_projection_file", str, "Input Projection File"),
	_f("input_data_path", str, "Input Data Path"),
	_f("software_name", str, "Software"),
	_f("software_version", str, "Software Version"),
	_f("configuration_state", str, "Configuration State"),
	_f("effective_command", str, "Effective Command"),
	_f("reconstruction_status", str, "Reconstruction Status"),
	_f("source_metadata_files", tuple, "Source Metadata Files"),
	_f("rotation_axis_coordinate_px", float, "Rotation Axis Coordinate (px)", "px"),
	_f("rotation_axis_offset_px", float, "Rotation Axis Offset (px)", "px"),
	_f("rotation_axis_auto_method", str, "Rotation Axis Auto Method"),
	_f("rotation_axis_tilt_deg", float, "Rotation Axis Tilt (deg)", "deg"),
	_f("rotation_axis_slant_deg", float, "Rotation Axis Slant (deg)", "deg"),
	_f("detector_rotation_eta_deg", float, "Detector Rotation Eta (deg)", "deg"),
	_f("detector_rotation_theta_deg", float, "Detector Rotation Theta (deg)", "deg"),
	_f("detector_rotation_phi_deg", float, "Detector Rotation Phi (deg)", "deg"),
	_f("detector_offset_u_px", float, "Detector Offset U (px)", "px"),
	_f("detector_offset_v_px", float, "Detector Offset V (px)", "px"),
	_f("source_object_distance_mm", float, "Source-Object Distance (mm)", "mm"),
	_f("source_detector_distance_mm", float, "Source-Detector Distance (mm)", "mm"),
	_f("propagation_distance_mm", float, "Propagation Distance (mm)", "mm"),
	_f("projection_binning", int, "Projection Binning"),
	_f("native_voxel_size_mm", float, "Native Voxel Size (mm)", "mm"),
	_f("output_voxel_size_mm", float, "Output Voxel Size (mm)", "mm"),
	_f("projection_range", tuple, "Projection Range"),
	_f("sinogram_range", tuple, "Sinogram Range"),
	_f("volume_roi_start_xyz", tuple, "Volume ROI Start XYZ"),
	_f("volume_roi_extent_xyz", tuple, "Volume ROI Extent XYZ"),
	_f("output_rotation_xyz_deg", tuple, "Output Rotation XYZ (deg)", "deg"),
	_f("cylindrical_mask_enabled", bool, "Cylindrical Mask Enabled"),
	_f("normalization_method", str, "Normalization Method"),
	_f("normalization_roi", tuple, "Normalization ROI"),
	_f("bright_ratio", float, "Bright Ratio"),
	_f("flat_linear", bool, "Flat Linearization"),
	_f("minus_log_enabled", bool, "Minus Log Enabled"),
	_f("median_filter_size_px", int, "Median Filter Size (px)", "px"),
	_f("gaussian_sigma_px", float, "Gaussian Sigma (px)", "px"),
	_f("outlier_removal_enabled", bool, "Outlier Removal Enabled"),
	_f("outlier_kernel_px", int, "Outlier Kernel (px)", "px"),
	_f("outlier_threshold", float, "Outlier Threshold"),
	_f("ring_stripe_method", str, "Ring/Stripe Method"),
	_f("ring_stripe_parameters", str, "Ring/Stripe Parameters"),
	_f("phase_retrieval_enabled", bool, "Phase Retrieval Enabled"),
	_f("phase_retrieval_method", str, "Phase Retrieval Method"),
	_f("phase_beta", float, "Phase Beta"),
	_f("phase_delta", float, "Phase Delta"),
	_f("beam_hardening_parameters", str, "Beam Hardening Parameters"),
	_f("photon_starvation_strength", float, "Photon Starvation Strength"),
	_f("scatter_correction_strength", float, "Scatter Correction Strength"),
	_f("dual_energy_factor", float, "Dual Energy Factor"),
	_f("drift_correction_xyz", tuple, "Drift Correction XYZ"),
	_f("jitter_correction", str, "Jitter Correction"),
	_f("reconstruction_type", str, "Reconstruction Type"),
	_f("reconstruction_algorithm", str, "Reconstruction Algorithm"),
	_f("reconstruction_filter", str, "Reconstruction Filter"),
	_f("iterative_regularization_method", str, "Iterative Regularization Method"),
	_f("iteration_count", int, "Iteration Count", "count"),
	_f("regularization_threshold", float, "Regularization Threshold"),
	_f("regularization_weight", float, "Regularization Weight"),
	_f("output_path", str, "Output Path"),
	_f("output_format", str, "Output Format"),
	_f("output_dtype", str, "Output Data Type"),
	_f("output_axis_order", str, "Output Axis Order"),
	_f("output_min", float, "Output Minimum"),
	_f("output_max", float, "Output Maximum"),
	_f("overwrite_policy", str, "Overwrite Policy"),
	_f("performance_parameters", str, "Performance Parameters"),
	_f("output_file_count", int, "Output File Count", "count"),
	_f("output_image_shape", tuple, "Output Image Shape (px)", "px"),
	_f("output_dtype_observed", str, "Observed Output Data Type"),
)


SCAN_FIELD_BY_NAME = {item.name: item for item in SCAN_FIELDS}
RECONSTRUCTION_FIELD_BY_NAME = {
	item.name: item
	for item in RECONSTRUCTION_FIELDS
}


WARNING_HEADER = "Metadata Warnings"
SCAN_HEADERS = tuple(item.header for item in SCAN_FIELDS) + (WARNING_HEADER,)
RECONSTRUCTION_HEADERS = tuple(item.header for item in RECONSTRUCTION_FIELDS) + (
	WARNING_HEADER,
)


def empty_scan_values() -> dict[str, object | None]:
	"""Return one missing-value slot for every canonical scan field."""
	return {item.name: None for item in SCAN_FIELDS}


def empty_reconstruction_values() -> dict[str, object | None]:
	"""Return one missing-value slot for every canonical reconstruction field."""
	return {item.name: None for item in RECONSTRUCTION_FIELDS}
