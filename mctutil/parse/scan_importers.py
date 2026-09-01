"""Direct readers for the four supported canonical scan sources."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np

from mctutil.parse import sigray_scan_log, tabular_log
from mctutil.parse.import_mappings import (
	ALS832_SCAN_MAPPING,
	CHENGLAB_CAMERA_SCAN_MAPPING,
	SEVEN_BM_SCAN_MAPPING,
)
from mctutil.parse.import_records import (
	SCAN_FIELD_BY_NAME,
	ScanRecord,
	apply_direct_mapping,
	coerce_value,
	decode_scalar,
	empty_scan_values,
	parse_datetime,
	parse_float,
	parse_int,
)
from mctutil.shared.deps import require


CHENGLAB_READONLY_SCOPES = (
	"https://www.googleapis.com/auth/spreadsheets.readonly",
)
_H5_SUFFIXES = (".h5", ".hdf5")
_FRAME_LOCATION_PATHS = (
	"/exchange/HDF5FrameLocation",
	"/measurement/defaults/HDF5FrameLocation",
	"/measurement/instrument/detector/HDF5FrameLocation",
)


def _require_h5py():
	return require(
		"h5py",
		"als832",
		purpose="scan HDF5 import dependencies are unavailable",
	)


def _discover_h5_files(inputs: Iterable[Path]) -> tuple[Path, ...]:
	paths: set[Path] = set()
	for raw_path in inputs:
		path = Path(raw_path)
		if path.is_dir():
			paths.update(
				item
				for item in path.iterdir()
				if item.is_file() and item.suffix.casefold() in _H5_SUFFIXES
			)
		elif path.is_file() and path.suffix.casefold() in _H5_SUFFIXES:
			paths.add(path)
	if not paths:
		raise ValueError("no .h5 or .hdf5 scan inputs found")
	return tuple(sorted(paths, key=lambda item: str(item)))


def _dataset(handle, path: str):
	item = handle.get(path)
	return item if item is not None and hasattr(item, "shape") else None


def _representative_value(handle, path: str):
	dataset = _dataset(handle, path)
	if dataset is None:
		return None
	try:
		values = np.asarray(dataset[()]).reshape(-1)
	except OSError:
		return None
	if not values.size:
		return None
	if values.dtype.kind in "iufb":
		finite = values[np.isfinite(values.astype(float, copy=False))]
		return float(np.median(finite)) if finite.size else None
	for value in values:
		decoded = decode_scalar(value)
		if decoded:
			return decoded
	return None


def _apply_h5_mapping(handle, mapping, values: dict[str, object | None]) -> None:
	apply_direct_mapping(
		mapping,
		SCAN_FIELD_BY_NAME,
		lambda locator: _representative_value(handle, locator),
		values,
	)


def _reference_count(dataset) -> int:
	return int(dataset.shape[0]) if len(dataset.shape) >= 3 else 1


def _set_reference_values(handle, path: Path, values: dict[str, object | None]) -> None:
	for dataset_path, files_field, count_field in (
		("/exchange/data_white", "flat_reference_files", "flat_frame_count"),
		("/exchange/data_dark", "dark_reference_files", "dark_frame_count"),
	):
		dataset = _dataset(handle, dataset_path)
		if dataset is None:
			continue
		values[files_field] = (f"{path}:{dataset_path}",)
		values[count_field] = _reference_count(dataset)


def _set_image_values(data, values: dict[str, object | None]) -> None:
	values["image_height_px"] = int(data.shape[-2])
	values["image_width_px"] = int(data.shape[-1])
	values["stored_dtype"] = str(data.dtype)
	values["stored_bit_depth"] = int(data.dtype.itemsize * 8)


def _valid_angles(
	handle,
	projection_count: int,
	mask: np.ndarray | None,
	warnings: list[str],
) -> np.ndarray:
	dataset = _dataset(handle, "/exchange/theta")
	if dataset is None:
		warnings.append("missing /exchange/theta")
		return np.asarray([], dtype=float)
	theta = np.asarray(dataset[()]).reshape(-1).astype(float, copy=False)
	if mask is not None and theta.size == mask.size:
		theta = theta[mask]
	if theta.size != projection_count:
		warnings.append(
			f"theta length {theta.size} does not match valid projection count {projection_count}"
		)
		theta = theta[:projection_count]
	return theta[np.isfinite(theta)]


def _set_angle_values(theta: np.ndarray, values: dict[str, object | None], warnings: list[str]) -> None:
	if not theta.size:
		return
	values["rotation_start_deg"] = float(theta[0])
	values["rotation_stop_actual_deg"] = float(theta[-1])
	values["angular_range_deg"] = float(theta[-1] - theta[0])
	if theta.size < 2:
		return
	differences = np.diff(theta)
	step = float(np.median(differences))
	values["angular_step_deg"] = step
	if not (np.all(differences > 0) or np.all(differences < 0)):
		warnings.append("theta is not monotonic")


def _set_physical_fov(values: dict[str, object | None]) -> None:
	pixel_size = values["effective_pixel_size_mm"]
	width = values["image_width_px"]
	height = values["image_height_px"]
	if type(pixel_size) is float and type(width) is int:
		values["physical_fov_width_mm"] = float(pixel_size * width)
	if type(pixel_size) is float and type(height) is int:
		values["physical_fov_height_mm"] = float(pixel_size * height)


def _finish_scan(values: dict[str, object | None], warnings: list[str]) -> None:
	if values["acquisition_status"] is None:
		values["acquisition_status"] = "Warnings" if warnings else "Complete"
	_set_physical_fov(values)


def _als832_record(path: Path) -> ScanRecord:
	h5py = _require_h5py()
	values = empty_scan_values()
	warnings: list[str] = []
	with h5py.File(path, "r") as handle:
		data = _dataset(handle, "/exchange/data")
		if data is None or len(data.shape) != 3:
			raise ValueError(f"{path}: missing three-dimensional /exchange/data")
		_apply_h5_mapping(handle, ALS832_SCAN_MAPPING, values)
		values["scan_id"] = values["scan_id"] or path.stem
		values["projection_data_file"] = str(path)
		values["source_metadata_file"] = str(path)
		values["projection_count_acquired"] = int(data.shape[0])
		_set_image_values(data, values)
		theta = _valid_angles(handle, int(data.shape[0]), None, warnings)
		_set_angle_values(theta, values, warnings)
		_set_reference_values(handle, path, values)
	values["scan_file_size_gb"] = float(path.stat().st_size / 1e9)
	_finish_scan(values, warnings)
	return ScanRecord("als832", (str(path),), values, warnings)


def read_als832_scans(inputs: Iterable[Path]) -> tuple[ScanRecord, ...]:
	"""Read ALS 8.3.2 Data Exchange scans and embedded references."""
	return tuple(_als832_record(path) for path in _discover_h5_files(inputs))


def _split_references(value: str, parent: Path) -> tuple[str, ...]:
	return tuple(
		str(parent / name.strip())
		for name in value.split(";")
		if name.strip()
	)


def _legacy_datetime(value: str) -> datetime | None:
	return parse_datetime(value) if value else None


def _populate_sigray_row(
	path: Path,
	row: dict[str, str],
	values: dict[str, object | None],
) -> tuple[str, ...]:
	text_fields = {
		"scan_id": "Scan",
		"acquisition_group": "Acquisition Group",
		"fov_index": "FOV Index",
		"sample_id": "Recorded Sample Number",
		"acquisition_system_id": "System Serial Number",
		"scan_method": "Scan Method",
		"source_target": "Source Target",
		"acquisition_status": "Acquisition Status",
	}
	numeric_fields = {
		"projection_count_acquired": "Projection Count",
		"rotation_start_deg": "Rotational Start (°)",
		"rotation_stop_actual_deg": "Rotational Stop (°)",
		"angular_step_deg": "Angular Step (°)",
		"exposure_us": "Exposure (µs)",
		"exposures_per_projection": "Exposures per Projection",
		"scan_duration_s": "Scan Duration (s)",
		"image_width_px": "Width (pixels)",
		"image_height_px": "Height (pixels)",
		"stored_bit_depth": "Stored Bit Depth",
		"detector_pixel_pitch_mm": "Detector Pixel Pitch (mm)",
		"effective_pixel_size_mm": "Effective Pixel Size (mm)",
		"geometric_magnification": "Geometric Magnification",
		"sample_stage_x_mm": "Sample Stage X (mm)",
		"sample_stage_y_mm": "Sample Stage Y (mm)",
		"sample_stage_z_mm": "Sample Stage Z (mm)",
		"detector_stage_z_mm": "Detector Stage Z (mm)",
		"source_stage_z_mm": "Source Stage Z (mm)",
		"source_energy_kev": "Source Energy Readback",
		"flat_frame_count": "Flat Frames",
		"dark_frame_count": "Dark Frames",
		"dropped_frames": "Dropped Frames",
		"incomplete_frames": "Incomplete Frames",
	}
	for field_name, header in text_fields.items():
		values[field_name] = row[header] or None
	for field_name, header in numeric_fields.items():
		values[field_name] = coerce_value(
			row[header],
			SCAN_FIELD_BY_NAME[field_name].value_type,
		)

	values["projection_data_file"] = str(path)
	values["source_metadata_file"] = str(path)
	values["stored_dtype"] = None
	values["scan_start"] = _legacy_datetime(row["Scan Start (UTC)"])
	values["scan_stop"] = _legacy_datetime(row["Scan Stop (UTC)"])
	values["source_power"] = row["Source Power Readback"] or None
	values["scan_file_size_gb"] = float(path.stat().st_size / 1e9)

	binning = tuple(part.strip() for part in row["Detector Binning"].split("x"))
	if binning and binning[0]:
		values["binning_x"] = parse_int(binning[0])
		values["binning_y"] = parse_int(binning[-1])

	references: list[str] = []
	for header, field_name in (
		("Flat Reference Files", "flat_reference_files"),
		("Dark Reference Files", "dark_reference_files"),
		("Post Reference Files", "post_reference_files"),
	):
		paths = _split_references(row[header], path.parent)
		values[field_name] = paths or None
		references.extend(paths)
	return tuple(references)


def _sigray_record(path: Path, row: dict[str, str]) -> ScanRecord:
	values = empty_scan_values()
	references = _populate_sigray_row(path, row, values)
	warnings = [item.strip() for item in row["Metadata Warnings"].split(";") if item.strip()]
	values["projection_count_planned"] = sigray_scan_log.read_planned_projection_count(path)
	start = values["rotation_start_deg"]
	step = values["angular_step_deg"]
	planned = values["projection_count_planned"]
	if type(start) is float and type(step) is float and type(planned) is int:
		values["rotation_stop_planned_deg"] = start + step * max(0, planned - 1)
	actual_stop = values["rotation_stop_actual_deg"]
	if type(start) is float and type(actual_stop) is float:
		values["angular_range_deg"] = actual_stop - start
	_finish_scan(values, warnings)
	return ScanRecord("sigray", (str(path),) + references, values, warnings)


def read_sigray_scans(inputs: Iterable[Path]) -> tuple[ScanRecord, ...]:
	"""Reuse the existing Sigray discovery, association, checks, and warnings."""
	paths = sigray_scan_log.discover_projection_files(tuple(Path(item) for item in inputs))
	if not paths:
		raise ValueError("no numbered Sigray projection HDF5 inputs found")
	rows = sigray_scan_log.extract_scan_rows(paths)
	return tuple(_sigray_record(path, row) for path, row in zip(paths, rows))


def _find_frame_location(handle):
	for path in _FRAME_LOCATION_PATHS:
		dataset = _dataset(handle, path)
		if dataset is not None:
			return dataset, path
	matches = []

	def visitor(name, item):
		if hasattr(item, "shape") and name.rsplit("/", 1)[-1].casefold() == "hdf5framelocation":
			matches.append((item, f"/{name}"))

	handle.visititems(visitor)
	return matches[0] if len(matches) == 1 else (None, None)


def _frame_mask(dataset) -> np.ndarray:
	values = np.asarray(dataset[()]).reshape(-1)
	if values.dtype.kind in "iufb":
		numeric = values.astype(float, copy=False)
		if np.all(np.isin(numeric, (0, 1))):
			return numeric == 1
		return numeric >= 0
	bad = {"", "none", "null", "missing", "invalid", "-1"}
	return np.asarray([decode_scalar(value).casefold() not in bad for value in values])


def _seven_bm_mask(handle, frame_count: int, warnings: list[str]) -> np.ndarray:
	dataset, path = _find_frame_location(handle)
	if dataset is None:
		warnings.append("HDF5FrameLocation is missing; no frames were accepted")
		return np.zeros(frame_count, dtype=bool)
	mask = _frame_mask(dataset)
	if mask.size != frame_count:
		warnings.append(
			f"{path} length {mask.size} does not match stored frame count {frame_count}"
		)
		return np.zeros(frame_count, dtype=bool)
	return mask


def _dataset_unit(dataset) -> str | None:
	for key in ("units", "unit", "NDAttrUnits"):
		if key in dataset.attrs:
			return decode_scalar(dataset.attrs[key]).casefold()
	for key in dataset.attrs:
		if decode_scalar(key).casefold().endswith("_units"):
			return decode_scalar(dataset.attrs[key]).casefold()
	return None


def _length_mm(handle, paths: tuple[str, ...], label: str, warnings: list[str]) -> float | None:
	for path in paths:
		dataset = _dataset(handle, path)
		if dataset is None:
			continue
		value = parse_float(_representative_value(handle, path))
		unit = _dataset_unit(dataset)
		if value is None:
			return None
		if unit in {"mm", "millimeter", "millimetre"}:
			return value
		if unit in {"um", "µm", "micron", "microns", "micrometer", "micrometre"}:
			return value * 0.001
		warnings.append(f"{label} unit is unavailable or unsupported at {path}")
		return None
	return None


def _seven_bm_integrity(
	handle,
	mask: np.ndarray,
	values: dict[str, object | None],
	warnings: list[str],
) -> None:
	for path in (
		"/measurement/defaults/NDArrayUniqueId",
		"/measurement/instrument/detector/NDArrayUniqueId",
	):
		dataset = _dataset(handle, path)
		if dataset is None:
			continue
		identifiers = np.asarray(dataset[()]).reshape(-1)
		if identifiers.size != mask.size:
			warnings.append("detector unique-ID length does not match frame locations")
			return
		differences = np.diff(identifiers[mask].astype(int, copy=False))
		values["dropped_frames"] = int(np.sum(np.maximum(differences - 1, 0)))
		if np.any(differences <= 0):
			warnings.append("detector unique IDs repeat or run backward")
		return


def _aps_7bm_record(path: Path) -> ScanRecord:
	h5py = _require_h5py()
	values = empty_scan_values()
	warnings: list[str] = []
	with h5py.File(path, "r") as handle:
		data = _dataset(handle, "/exchange/data")
		if data is None or len(data.shape) != 3:
			raise ValueError(f"{path}: missing three-dimensional /exchange/data")
		_apply_h5_mapping(handle, SEVEN_BM_SCAN_MAPPING, values)
		mask = _seven_bm_mask(handle, int(data.shape[0]), warnings)
		projection_count = int(np.count_nonzero(mask))
		values.update(
			{
				"facility_name": "APS 7-BM",
				"acquisition_system_id": "APS 7-BM",
				"scan_id": path.stem,
				"projection_data_file": str(path),
				"source_metadata_file": str(path),
				"projection_count_acquired": projection_count,
			}
		)
		_set_image_values(data, values)
		theta = _valid_angles(handle, projection_count, mask, warnings)
		_set_angle_values(theta, values, warnings)
		_set_reference_values(handle, path, values)
		values["detector_pixel_pitch_mm"] = _length_mm(
			handle,
			(
				"/measurement/instrument/detector/physical_pixel_size",
				"/measurement/instrument/detector/pixel_size",
			),
			"detector pixel pitch",
			warnings,
		)
		values["effective_pixel_size_mm"] = _length_mm(
			handle,
			(
				"/measurement/instrument/objective/resolution",
				"/measurement/instrument/detector/resolution",
				"/measurement/instrument/detector/actual_pixel_size",
			),
			"sample-plane resolution",
			warnings,
		)
		_seven_bm_integrity(handle, mask, values, warnings)
	values["scan_file_size_gb"] = float(path.stat().st_size / 1e9)
	_finish_scan(values, warnings)
	return ScanRecord("aps-7bm", (str(path),), values, warnings)


def read_aps_7bm_scans(inputs: Iterable[Path]) -> tuple[ScanRecord, ...]:
	"""Read APS 7-BM scans after frame-location filtering."""
	return tuple(_aps_7bm_record(path) for path in _discover_h5_files(inputs))


def _has_exact_readonly_scope(credentials) -> bool:
	granted = set(credentials.granted_scopes or credentials.scopes or ())
	return granted == set(CHENGLAB_READONLY_SCOPES)


def build_chenglab_sheets_service(google_conf: Path):
	"""Build a Sheets values service authorized with the read-only scope."""
	(
		google_auth_exceptions,
		google_auth_requests,
		google_oauth2_credentials,
		google_auth_oauthlib_flow,
		googleapiclient_discovery,
	) = require(
		(
			"google.auth.exceptions",
			"google.auth.transport.requests",
			"google.oauth2.credentials",
			"google_auth_oauthlib.flow",
			"googleapiclient.discovery",
		),
		"google-sheets",
		purpose="ChengLab Camera read-only Sheets dependencies are unavailable",
	)
	Request = google_auth_requests.Request
	Credentials = google_oauth2_credentials.Credentials
	InstalledAppFlow = google_auth_oauthlib_flow.InstalledAppFlow
	build = googleapiclient_discovery.build

	google_conf = Path(google_conf)
	readonly_token = google_conf / "gsheets_readonly_token.json"
	legacy_token = google_conf / "gsheets_token.json"
	credentials_path = google_conf / "gsheets_credentials.json"
	credentials = None
	for token_path in (readonly_token, legacy_token):
		if not token_path.exists():
			continue
		candidate = Credentials.from_authorized_user_file(
			token_path,
			CHENGLAB_READONLY_SCOPES,
		)
		if token_path == legacy_token or not candidate.valid:
			try:
				candidate.refresh(Request())
			except google_auth_exceptions.RefreshError:
				continue
		if _has_exact_readonly_scope(candidate):
			credentials = candidate
			break

	if credentials is None:
		if not credentials_path.exists():
			raise FileNotFoundError(f"Google OAuth client credentials not found: {credentials_path}")
		flow = InstalledAppFlow.from_client_secrets_file(
			credentials_path,
			CHENGLAB_READONLY_SCOPES,
		)
		credentials = flow.run_local_server(port=0)
		if not _has_exact_readonly_scope(credentials):
			raise ValueError("Google did not grant exactly the requested read-only Sheets scope")
		readonly_token.write_text(credentials.to_json(), encoding="utf-8")

	return build("sheets", "v4", credentials=credentials).spreadsheets()


def _camera_local_datetime(value) -> datetime | None:
	if value in (None, ""):
		return None
	if isinstance(value, (int, float)) and not isinstance(value, bool):
		return datetime(1899, 12, 30) + timedelta(days=float(value))
	text = decode_scalar(value).replace("Z", "+00:00")
	try:
		return datetime.fromisoformat(text)
	except ValueError:
		return None


def _camera_value(row: dict[str, object], header: str):
	value = row.get(header)
	return None if value in (None, "") else value


def _set_camera_derived_values(
	values: dict[str, object | None],
	start: datetime | None,
	stop: datetime | None,
) -> None:
	if start is not None and stop is not None:
		values["scan_duration_s"] = float((stop - start).total_seconds())
	exposure = values["exposure_us"]
	trigger = values["trigger_period_us"]
	if type(exposure) is float and type(trigger) is float:
		values["trigger_overhead_us"] = trigger - exposure
	planned = values["projection_count_planned"]
	acquired = values["projection_count_acquired"]
	if type(planned) is int and type(acquired) is int:
		values["dropped_frames"] = max(0, planned - acquired)
	pre = values["flat_frame_count_pre"]
	post = values["flat_frame_count_post"]
	if type(pre) is int or type(post) is int:
		values["flat_frame_count"] = (pre or 0) + (post or 0)
	rotation_start = values["rotation_start_deg"]
	rotation_stop = values["rotation_stop_actual_deg"]
	if type(rotation_start) is float and type(rotation_stop) is float:
		values["angular_range_deg"] = rotation_stop - rotation_start


def _warn_camera_ambiguities(row: dict[str, object], warnings: list[str]) -> None:
	for header in (
		"Resolution",
		"Offset X",
		"Offset Y",
		"SSD (cm)",
		"Vertical Start",
		"Vertical Stop",
	):
		if _camera_value(row, header) is not None:
			warnings.append(
				f"{header} was not normalized because its unit or semantics are unconfirmed"
			)


def _camera_record(
	spreadsheet: str,
	sheet: str,
	row_number: int,
	row: dict[str, object],
) -> ScanRecord:
	values = empty_scan_values()
	warnings: list[str] = []
	for field_name, entry in CHENGLAB_CAMERA_SCAN_MAPPING.items():
		if field_name in {"scan_start", "scan_stop"}:
			continue
		raw_value = _camera_value(row, entry["locator"])
		values[field_name] = coerce_value(
			raw_value,
			SCAN_FIELD_BY_NAME[field_name].value_type,
		)

	values["facility_name"] = "ChengLab Camera"
	values["acquisition_system_id"] = "ChengLab Camera"
	start = _camera_local_datetime(_camera_value(row, "Scan Start"))
	stop = _camera_local_datetime(_camera_value(row, "Scan Stop"))
	if start is not None or stop is not None:
		warnings.append("scan timestamps have no declared timezone and were not labeled UTC")
	_set_camera_derived_values(values, start, stop)
	_warn_camera_ambiguities(row, warnings)
	_finish_scan(values, warnings)
	location = f"gsheets://{spreadsheet}/{sheet}!{row_number}"
	return ScanRecord("chenglab-camera", (location,), values, warnings)


def read_chenglab_camera_scans(
	spreadsheet: str,
	*,
	sheet: str = "ScanLog",
	google_conf: Path = Path("~/.creds/gsheets").expanduser(),
	service=None,
) -> tuple[ScanRecord, ...]:
	"""Read real ScanLog rows through the Google Sheets read-only values API."""
	if not spreadsheet:
		raise ValueError("ChengLab Camera requires an input spreadsheet ID")
	service = service or build_chenglab_sheets_service(google_conf)
	response = service.values().get(
		spreadsheetId=spreadsheet,
		range=tabular_log.sheet_range(sheet, "A:BW"),
		valueRenderOption="UNFORMATTED_VALUE",
		dateTimeRenderOption="FORMATTED_STRING",
	).execute()
	rows = response.get("values", [])
	if not rows:
		raise ValueError(f"{sheet!r} contains no header row")
	headers = [decode_scalar(value) for value in rows[0]]
	if "Scan ID" not in headers:
		raise ValueError(f"{sheet!r} is missing required 'Scan ID' header")

	records = []
	for row_number, raw_row in enumerate(rows[1:], start=2):
		row = dict(zip(headers, raw_row))
		scan_id = decode_scalar(row.get("Scan ID", ""))
		if not scan_id or scan_id.casefold() in {"scan id", "setup", "default"}:
			continue
		records.append(_camera_record(spreadsheet, sheet, row_number, row))
	return tuple(records)
