"""Extract one scan-log row per Sigray projection HDF5 acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import re
from typing import Iterable

import click
from natsort import natsorted
import numpy as np

from mctutil.parse import tabular_log
from mctutil.shared.deps import require


SCAN_LOG_FIELDS = (
	"Scan",
	"Acquisition Group",
	"FOV Index",
	"Recorded Sample Number",
	"System Serial Number",
	"Scan Method",
	"Projection Data File",
	"Projection Count",
	"Rotational Start (°)",
	"Rotational Stop (°)",
	"Angular Step (°)",
	"Exposure (µs)",
	"Exposures per Projection",
	"Scan Start (UTC)",
	"Scan Stop (UTC)",
	"Scan Duration (s)",
	"Width (pixels)",
	"Height (pixels)",
	"Stored Bit Depth",
	"Detector Binning",
	"Detector Pixel Pitch (mm)",
	"Effective Pixel Size (mm)",
	"Geometric Magnification",
	"Sample Stage X (mm)",
	"Sample Stage Y (mm)",
	"Sample Stage Z (mm)",
	"Detector Stage Z (mm)",
	"Source Stage Z (mm)",
	"Source Energy Readback",
	"Source Power Readback",
	"Source Target",
	"Flat Reference Files",
	"Flat Frames",
	"Dark Reference Files",
	"Dark Frames",
	"Post Reference Files",
	"Post Frames",
	"Dropped Frames",
	"Incomplete Frames",
	"Acquisition Status",
	"File Size (GB)",
	"Metadata Warnings",
)

H5_SUFFIXES = (".h5", ".hdf5")
REFERENCE_NAME_RE = re.compile(
	r"^(?P<prefix>.+)_(?P<kind>FLAT|DARK|POST)_(?P<index>\d+)$",
	re.IGNORECASE,
)
PROJECTION_NAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d+)$")
EPICS_EPOCH = datetime(1990, 1, 1, tzinfo=timezone.utc)
STAGE_MOVEMENT_WARNING_UM = 10.0
POST_STAGE_MATCH_UM = 10.0

DATA_PATH = "/exchange/data"
THETA_PATH = "/exchange/theta"
SAMPLE_NUMBER_PATH = "/exchange/sample_number"
SYSTEM_SERIAL_PATH = "/exchange/system_serial_number"
SCAN_TYPE_PATH = "/exchange/scan_type"
EXPECTED_PROJECTIONS_PATH = (
	"/measurement/process/acquisition/setup/number_of_projections"
)
CONFIGURED_STEP_PATH = "/measurement/process/acquisition/setup/angular_step"
EXPOSURE_PATH = "/measurement/process/acquisition/image_exposure_time"
COMPLETE_PATH = "/measurement/process/acquisition/image_is_complete"
UID_PATH = "/measurement/defaults/NDArrayUniqueId"
EPICS_SECONDS_PATH = "/measurement/defaults/NDArrayEpicsTSSec"
EPICS_NANOSECONDS_PATH = "/measurement/defaults/NDArrayEpicsTSnSec"
FRAME_AVERAGING_PATH = "/measurement/instrument/detector/frame_averaging"
BINNING_X_PATH = "/measurement/instrument/detector/binning_x"
BINNING_Y_PATH = "/measurement/instrument/detector/binning_y"
DIMENSION_X_PATH = "/measurement/instrument/detector/dimension_x"
DIMENSION_Y_PATH = "/measurement/instrument/detector/dimension_y"
PIXEL_PITCH_PATH = "/measurement/instrument/detector/physical_pixel_size"
EFFECTIVE_PIXEL_X_PATH = "/measurement/instrument/detector/actual_pixel_size_x"
EFFECTIVE_PIXEL_Y_PATH = "/measurement/instrument/detector/actual_pixel_size_y"
MAGNIFICATION_PATH = "/measurement/instrument/detector/geometric_magnification"
SAMPLE_X_PATH = "/measurement/instrument/sample/setup/sample_x"
SAMPLE_Y_PATH = "/measurement/instrument/sample/setup/sample_y"
SAMPLE_Z_PATH = "/measurement/instrument/sample/setup/sample_z"
DETECTOR_Z_PATH = "/measurement/instrument/detector/setup/detector_z"
SOURCE_Z_PATH = "/measurement/instrument/source/setup/source_z"
N3_ENERGY_PATH = "/measurement/instrument/source/n3_energy_RBV"
N3_POWER_PATH = "/measurement/instrument/source/n3_power_RBV"
SRPS_ENERGY_PATH = "/measurement/instrument/source/srps_voltage"
SRPS_POWER_PATH = "/measurement/instrument/source/srps_power"
SOURCE_TARGET_PATH = "/measurement/instrument/source/srps_target"


@dataclass(frozen=True)
class ProjectionIdentity:
	prefix: str
	fov_index: str


@dataclass(frozen=True)
class ReferenceFile:
	path: Path
	kind: str
	prefix: str
	frame_count: int
	image_shape: tuple[int, int]
	stage_um: tuple[float, float, float] | None


@dataclass
class ExtractionState:
	warnings: list[str]
	problems: list[str]
	metadata_incomplete: bool = False

	def missing(self, warning: str) -> None:
		self.metadata_incomplete = True
		self.warnings.append(warning)

	def finish(self, row: dict[str, str]) -> None:
		if self.problems:
			row["Acquisition Status"] = "Issues detected"
			self.warnings.extend(self.problems)
		elif self.metadata_incomplete:
			row["Acquisition Status"] = "Metadata incomplete"
		else:
			row["Acquisition Status"] = "Complete"
		row["Metadata Warnings"] = "; ".join(dict.fromkeys(self.warnings))


def _require_h5py():
	return require(
		"h5py",
		"sigray",
		purpose="h5py is required for Sigray scan-log extraction",
		error_type=click.ClickException,
	)


def _projection_identity(path: Path) -> ProjectionIdentity | None:
	if path.suffix.casefold() not in H5_SUFFIXES:
		return None
	if REFERENCE_NAME_RE.fullmatch(path.stem):
		return None
	match = PROJECTION_NAME_RE.fullmatch(path.stem)
	if match is None:
		return None
	return ProjectionIdentity(match.group("prefix"), match.group("index"))


def _reference_identity(path: Path) -> tuple[str, str] | None:
	if path.suffix.casefold() not in H5_SUFFIXES:
		return None
	match = REFERENCE_NAME_RE.fullmatch(path.stem)
	if match is None:
		return None
	return match.group("prefix"), match.group("kind").upper()


def discover_projection_files(inputs: Iterable[Path]) -> tuple[Path, ...]:
	"""Return deterministic, de-duplicated numbered projection HDF5 inputs."""
	candidates: list[Path] = []
	for input_path in inputs:
		if input_path.is_dir():
			for candidate in input_path.iterdir():
				if candidate.is_file() and _projection_identity(candidate) is not None:
					candidates.append(candidate)
		elif _projection_identity(input_path) is not None:
			candidates.append(input_path)

	unique: dict[Path, Path] = {}
	for candidate in candidates:
		unique.setdefault(candidate.resolve(), candidate)
	return tuple(natsorted(unique.values(), key=lambda item: str(item)))


def _dataset(handle, path: str):
	obj = handle.get(path)
	return obj if obj is not None and hasattr(obj, "shape") else None


def _numeric_values(handle, path: str) -> np.ndarray:
	dataset = _dataset(handle, path)
	if dataset is None:
		return np.asarray([], dtype=np.float64)
	try:
		values = np.asarray(dataset[()], dtype=np.float64).reshape(-1)
	except (TypeError, ValueError, OSError):
		return np.asarray([], dtype=np.float64)
	return values[np.isfinite(values)]


def _median(handle, path: str) -> float | None:
	values = _numeric_values(handle, path)
	return float(np.median(values)) if values.size else None


def _decode(value) -> str:
	if isinstance(value, (bytes, bytearray, np.bytes_)):
		return bytes(value).decode("utf-8", errors="replace").strip()
	return str(value).strip()


def _string_value(handle, path: str) -> str:
	dataset = _dataset(handle, path)
	if dataset is None:
		return ""
	try:
		values = np.asarray(dataset[()]).reshape(-1)
	except OSError:
		return ""
	for value in values:
		decoded = _decode(value)
		if decoded:
			return decoded
	return ""


def _number(value: float | int | None) -> str:
	if value is None or not math.isfinite(float(value)):
		return ""
	value = float(value)
	if value.is_integer():
		return str(int(value))
	return format(value, ".12g")


def _positive_readback(handle, preferred_path: str, fallback_path: str) -> str:
	raw = _string_value(handle, preferred_path)
	if raw:
		try:
			value = float(raw)
		except ValueError:
			return raw
		if math.isfinite(value) and value > 0:
			return _number(value)
	fallback = _median(handle, fallback_path)
	return _number(fallback) if fallback is not None and fallback > 0 else ""


def _median_with_movement_warning(
	handle,
	path: str,
	label: str,
	warnings: list[str],
) -> float | None:
	values = _numeric_values(handle, path)
	if not values.size:
		return None
	span = float(np.ptp(values))
	if span > STAGE_MOVEMENT_WARNING_UM:
		warnings.append(f"{label} moved {format(span / 1000, '.6g')} mm during scan")
	return float(np.median(values))


def _stage_um(handle, warnings: list[str] | None = None) -> tuple[float, float, float] | None:
	labels_and_paths = (
		("sample X", SAMPLE_X_PATH),
		("sample Y", SAMPLE_Y_PATH),
		("sample Z", SAMPLE_Z_PATH),
	)
	values = []
	for label, path in labels_and_paths:
		if warnings is None:
			value = _median(handle, path)
		else:
			value = _median_with_movement_warning(handle, path, label, warnings)
		if value is None:
			return None
		values.append(value)
	return tuple(values)


def _format_epics_timestamp(seconds: int, nanoseconds: int) -> str:
	stamp = EPICS_EPOCH + timedelta(seconds=seconds)
	return f"{stamp.strftime('%Y-%m-%dT%H:%M:%S')}.{nanoseconds:09d}Z"


def _timestamps(handle, warnings: list[str]) -> tuple[str, str, str] | None:
	seconds = _numeric_values(handle, EPICS_SECONDS_PATH)
	nanoseconds = _numeric_values(handle, EPICS_NANOSECONDS_PATH)
	if not seconds.size or seconds.size != nanoseconds.size:
		warnings.append("EPICS timestamps are missing or length-mismatched")
		return None
	start_seconds = int(seconds[0])
	stop_seconds = int(seconds[-1])
	start_nanoseconds = int(nanoseconds[0])
	stop_nanoseconds = int(nanoseconds[-1])
	duration = (
		stop_seconds
		+ stop_nanoseconds / 1e9
		- start_seconds
		- start_nanoseconds / 1e9
	)
	if duration < 0:
		warnings.append("EPICS timestamps run backward")
	return (
		_format_epics_timestamp(start_seconds, start_nanoseconds),
		_format_epics_timestamp(stop_seconds, stop_nanoseconds),
		_number(duration),
	)


def _dropped_frames(unique_ids: np.ndarray, frame_averaging: int) -> tuple[int, bool]:
	"""Count raw detector frames missing beyond the normal averaging stride."""
	if unique_ids.size < 2:
		return 0, False
	differences = np.diff(unique_ids.astype(np.int64))
	invalid_order = bool(np.any(differences <= 0))
	positive_gaps = differences[differences > frame_averaging]
	dropped = int(sum(int(gap) - frame_averaging for gap in positive_gaps))
	return dropped, invalid_order


def _load_reference_file(path: Path, prefix: str, kind: str) -> ReferenceFile:
	h5py = _require_h5py()
	with h5py.File(path, "r") as handle:
		data = _dataset(handle, DATA_PATH)
		if data is None or len(data.shape) != 3:
			raise ValueError(f"missing three-dimensional {DATA_PATH}")
		return ReferenceFile(
			path=path,
			kind=kind,
			prefix=prefix,
			frame_count=int(data.shape[0]),
			image_shape=(int(data.shape[1]), int(data.shape[2])),
			stage_um=_stage_um(handle),
		)


def load_reference_files(directory: Path, prefix: str) -> tuple[tuple[ReferenceFile, ...], tuple[str, ...]]:
	"""Read sibling reference metadata without reading any image arrays."""
	references = []
	warnings = []
	for path in natsorted(directory.iterdir(), key=lambda item: item.name):
		if not path.is_file():
			continue
		identity = _reference_identity(path)
		if identity is None or identity[0] != prefix:
			continue
		try:
			references.append(_load_reference_file(path, *identity))
		except (OSError, ValueError) as exc:
			warnings.append(f"could not inspect reference {path.name}: {exc}")
	return tuple(references), tuple(warnings)


def _reference_values(references: Iterable[ReferenceFile]) -> tuple[str, str]:
	references = tuple(natsorted(references, key=lambda item: item.path.name))
	return (
		"; ".join(reference.path.name for reference in references),
		str(sum(reference.frame_count for reference in references)),
	)


def _associate_shared_reference(
	references: tuple[ReferenceFile, ...],
	kind: str,
	image_shape: tuple[int, int],
	warnings: list[str],
) -> tuple[str, str]:
	matches = tuple(
		reference
		for reference in references
		if reference.kind == kind and reference.image_shape == image_shape
	)
	if len(matches) == 1:
		return _reference_values(matches)
	if len(matches) > 1:
		warnings.append(
			f"multiple {kind} references match detector shape; association is ambiguous"
		)
	return "", ""


def _associate_post_reference(
	references: tuple[ReferenceFile, ...],
	image_shape: tuple[int, int],
	stage_um: tuple[float, float, float] | None,
	warnings: list[str],
) -> tuple[str, str]:
	if stage_um is None:
		warnings.append("POST reference cannot be stage-matched because sample position is missing")
		return "", ""
	matches = []
	for reference in references:
		if reference.kind != "POST" or reference.image_shape != image_shape:
			continue
		if reference.stage_um is None:
			continue
		distance = math.dist(stage_um, reference.stage_um)
		if distance <= POST_STAGE_MATCH_UM:
			matches.append(reference)
	if len(matches) == 1:
		return _reference_values(matches)
	if len(matches) > 1:
		warnings.append("multiple POST references match sample stage; association is ambiguous")
	return "", ""


def _embedded_reference(
	handle,
	projection_path: Path,
	dataset_path: str,
	image_shape: tuple[int, int],
) -> str:
	dataset = _dataset(handle, dataset_path)
	if dataset is None or tuple(dataset.shape[-2:]) != image_shape:
		return ""
	try:
		if dataset.id.get_storage_size() <= 0:
			return ""
	except AttributeError:
		pass
	return f"{projection_path.name}:{dataset_path}"


def _base_row(path: Path, identity: ProjectionIdentity) -> dict[str, str]:
	row = {field: "" for field in SCAN_LOG_FIELDS}
	row.update(
		{
			"Scan": path.stem,
			"Acquisition Group": path.parent.name,
			"FOV Index": identity.fov_index,
			"Projection Data File": path.name,
			"File Size (GB)": _number(path.stat().st_size / 1e9),
		}
	)
	return row


def _populate_core_fields(handle, data, row: dict[str, str]) -> tuple[int, tuple[int, int]]:
	projection_count = int(data.shape[0])
	image_shape = (int(data.shape[1]), int(data.shape[2]))
	row.update(
		{
			"Projection Count": str(projection_count),
			"Width (pixels)": str(data.shape[2]),
			"Height (pixels)": str(data.shape[1]),
			"Stored Bit Depth": str(data.dtype.itemsize * 8),
			"Recorded Sample Number": _string_value(handle, SAMPLE_NUMBER_PATH),
			"System Serial Number": _string_value(handle, SYSTEM_SERIAL_PATH),
			"Scan Method": _string_value(handle, SCAN_TYPE_PATH),
		}
	)
	return projection_count, image_shape


def _populate_theta(
	handle,
	projection_count: int,
	row: dict[str, str],
	state: ExtractionState,
) -> None:
	theta = _numeric_values(handle, THETA_PATH)
	if theta.size != projection_count:
		state.missing(
			f"theta length {theta.size} does not match projection count {projection_count}"
		)
		return
	row["Rotational Start (°)"] = _number(theta[0])
	row["Rotational Stop (°)"] = _number(theta[-1])
	differences = np.diff(theta)
	if not differences.size:
		return
	step = float(np.median(differences))
	row["Angular Step (°)"] = _number(step)
	if not np.all(differences > 0) and not np.all(differences < 0):
		state.problems.append("theta is not monotonic")
	deviation = float(np.max(np.abs(differences - step)))
	if deviation > max(1e-6, abs(step) * 0.01):
		state.warnings.append("angular spacing varies by more than 1%")
	configured_step = _median(handle, CONFIGURED_STEP_PATH)
	if configured_step is not None and not math.isclose(
		step,
		configured_step,
		rel_tol=0.01,
		abs_tol=1e-6,
	):
		state.warnings.append("actual angular step differs from configured step")


def _populate_acquisition(
	handle,
	projection_count: int,
	row: dict[str, str],
	state: ExtractionState,
) -> int:
	expected = _median(handle, EXPECTED_PROJECTIONS_PATH)
	if expected is None:
		state.missing("expected projection count is missing")
	elif int(round(expected)) != projection_count:
		state.problems.append(
			f"expected {int(round(expected))} projections but stored {projection_count}"
		)
	_populate_theta(handle, projection_count, row, state)

	exposure_seconds = _median(handle, EXPOSURE_PATH)
	if exposure_seconds is not None:
		row["Exposure (µs)"] = _number(exposure_seconds * 1e6)
	frame_averaging_value = _median(handle, FRAME_AVERAGING_PATH)
	frame_averaging = max(1, int(round(frame_averaging_value or 1)))
	row["Exposures per Projection"] = str(frame_averaging)
	return frame_averaging


def _populate_integrity(
	handle,
	projection_count: int,
	frame_averaging: int,
	row: dict[str, str],
	state: ExtractionState,
) -> None:
	complete = _numeric_values(handle, COMPLETE_PATH)
	if complete.size != projection_count:
		state.missing("completion flags are missing or length-mismatched")
	else:
		incomplete_count = int(np.count_nonzero(complete != 1))
		row["Incomplete Frames"] = str(incomplete_count)
		if incomplete_count:
			state.problems.append(f"{incomplete_count} frame(s) marked incomplete")

	unique_ids = _numeric_values(handle, UID_PATH)
	if unique_ids.size != projection_count:
		state.missing("detector IDs are missing or length-mismatched")
		return
	dropped, invalid_order = _dropped_frames(unique_ids, frame_averaging)
	row["Dropped Frames"] = str(dropped)
	if dropped:
		state.problems.append(
			f"{dropped} dropped frame(s) inferred from detector IDs"
		)
	if invalid_order:
		state.problems.append("detector IDs are repeated or run backward")


def _populate_timestamps(
	handle,
	row: dict[str, str],
	state: ExtractionState,
) -> None:
	values = _timestamps(handle, state.warnings)
	if values is None:
		state.metadata_incomplete = True
		return
	row["Scan Start (UTC)"], row["Scan Stop (UTC)"], row["Scan Duration (s)"] = values


def _populate_detector(
	handle,
	data,
	row: dict[str, str],
	warnings: list[str],
) -> None:
	binning_x = _median(handle, BINNING_X_PATH)
	binning_y = _median(handle, BINNING_Y_PATH)
	if binning_x is not None and binning_y is not None:
		row["Detector Binning"] = (
			_number(binning_x)
			if math.isclose(binning_x, binning_y)
			else f"{_number(binning_x)} x {_number(binning_y)}"
		)

	for dataset_path, actual_dimension, label in (
		(DIMENSION_X_PATH, data.shape[2], "detector width"),
		(DIMENSION_Y_PATH, data.shape[1], "detector height"),
	):
		recorded_dimension = _median(handle, dataset_path)
		if recorded_dimension is not None and int(round(recorded_dimension)) != actual_dimension:
			warnings.append(f"recorded {label} differs from stored data shape")

	pixel_pitch_um = _median(handle, PIXEL_PITCH_PATH)
	if pixel_pitch_um is not None:
		row["Detector Pixel Pitch (mm)"] = _number(pixel_pitch_um / 1000)
	effective_x_um = _median(handle, EFFECTIVE_PIXEL_X_PATH)
	effective_y_um = _median(handle, EFFECTIVE_PIXEL_Y_PATH)
	if effective_x_um is not None:
		row["Effective Pixel Size (mm)"] = _number(effective_x_um / 1000)
	if (
		effective_x_um is not None
		and effective_y_um is not None
		and not math.isclose(effective_x_um, effective_y_um, rel_tol=1e-4)
	):
		warnings.append("effective X and Y pixel sizes differ")
	row["Geometric Magnification"] = _number(_median(handle, MAGNIFICATION_PATH))


def _populate_stage(
	handle,
	row: dict[str, str],
	warnings: list[str],
) -> tuple[float, float, float] | None:
	stage_um = _stage_um(handle, warnings)
	if stage_um is not None:
		fields = (
			"Sample Stage X (mm)",
			"Sample Stage Y (mm)",
			"Sample Stage Z (mm)",
		)
		for field, value in zip(fields, stage_um):
			row[field] = _number(value / 1000)
	detector_z_um = _median_with_movement_warning(
		handle,
		DETECTOR_Z_PATH,
		"detector Z",
		warnings,
	)
	source_z_um = _median_with_movement_warning(
		handle,
		SOURCE_Z_PATH,
		"source Z",
		warnings,
	)
	if detector_z_um is not None:
		row["Detector Stage Z (mm)"] = _number(detector_z_um / 1000)
	if source_z_um is not None:
		row["Source Stage Z (mm)"] = _number(source_z_um / 1000)
	return stage_um


def _populate_source(handle, row: dict[str, str], warnings: list[str]) -> None:
	row["Source Energy Readback"] = _positive_readback(
		handle,
		N3_ENERGY_PATH,
		SRPS_ENERGY_PATH,
	)
	row["Source Power Readback"] = _positive_readback(
		handle,
		N3_POWER_PATH,
		SRPS_POWER_PATH,
	)
	if not row["Source Energy Readback"]:
		warnings.append("source energy readback is unavailable in this acquisition")
	if not row["Source Power Readback"]:
		warnings.append("source power readback is unavailable in this acquisition")
	row["Source Target"] = _string_value(handle, SOURCE_TARGET_PATH)


def _flat_or_dark_reference(
	handle,
	path: Path,
	references: tuple[ReferenceFile, ...],
	kind: str,
	image_shape: tuple[int, int],
	warnings: list[str],
) -> tuple[str, str]:
	files, frames = _associate_shared_reference(
		references,
		kind,
		image_shape,
		warnings,
	)
	if files:
		return files, frames
	dataset_path = "/exchange/data_white" if kind == "FLAT" else "/exchange/data_dark"
	files = _embedded_reference(handle, path, dataset_path, image_shape)
	if files:
		warnings.append(
			f"{kind.casefold()} frame count is unavailable for embedded processed reference"
		)
	else:
		warnings.append(f"no {kind} reference matches this acquisition")
	return files, frames


def _post_reference(
	handle,
	path: Path,
	references: tuple[ReferenceFile, ...],
	image_shape: tuple[int, int],
	stage_um: tuple[float, float, float] | None,
	warnings: list[str],
) -> tuple[str, str]:
	files, frames = _associate_post_reference(
		references,
		image_shape,
		stage_um,
		warnings,
	)
	if files:
		return files, frames
	post_dataset = _dataset(handle, "/post/exchange/data")
	files = _embedded_reference(
		handle,
		path,
		"/post/exchange/data",
		image_shape,
	)
	if files and post_dataset is not None:
		return files, str(post_dataset.shape[0])
	warnings.append("no POST reference matches detector shape and sample stage")
	return "", ""


def _populate_references(
	handle,
	path: Path,
	references: tuple[ReferenceFile, ...],
	image_shape: tuple[int, int],
	stage_um: tuple[float, float, float] | None,
	row: dict[str, str],
	warnings: list[str],
) -> None:
	flat_files, flat_frames = _flat_or_dark_reference(
		handle,
		path,
		references,
		"FLAT",
		image_shape,
		warnings,
	)
	dark_files, dark_frames = _flat_or_dark_reference(
		handle,
		path,
		references,
		"DARK",
		image_shape,
		warnings,
	)
	post_files, post_frames = _post_reference(
		handle,
		path,
		references,
		image_shape,
		stage_um,
		warnings,
	)
	row.update(
		{
			"Flat Reference Files": flat_files,
			"Flat Frames": flat_frames,
			"Dark Reference Files": dark_files,
			"Dark Frames": dark_frames,
			"Post Reference Files": post_files,
			"Post Frames": post_frames,
		}
	)


def extract_scan_row(
	path: Path,
	*,
	references: tuple[ReferenceFile, ...] | None = None,
	reference_warnings: tuple[str, ...] = (),
) -> dict[str, str]:
	"""Extract one canonical scan row from one numbered projection HDF5 file."""
	identity = _projection_identity(path)
	if identity is None:
		raise ValueError(f"Not a numbered projection HDF5 filename: {path.name}")
	row = _base_row(path, identity)
	state = ExtractionState(list(reference_warnings), [])
	h5py = _require_h5py()

	try:
		handle = h5py.File(path, "r")
	except Exception as exc:
		row["Acquisition Status"] = "Unreadable"
		row["Metadata Warnings"] = f"could not open HDF5: {exc}"
		return row

	with handle:
		data = _dataset(handle, DATA_PATH)
		if data is None or len(data.shape) != 3:
			row["Acquisition Status"] = "Invalid"
			row["Metadata Warnings"] = f"missing three-dimensional {DATA_PATH}"
			return row

		projection_count, image_shape = _populate_core_fields(handle, data, row)
		frame_averaging = _populate_acquisition(
			handle,
			projection_count,
			row,
			state,
		)
		_populate_integrity(handle, projection_count, frame_averaging, row, state)
		_populate_timestamps(handle, row, state)
		_populate_detector(handle, data, row, state.warnings)
		stage_um = _populate_stage(handle, row, state.warnings)
		_populate_source(handle, row, state.warnings)
		if references is None:
			references, discovered_warnings = load_reference_files(path.parent, identity.prefix)
			state.warnings.extend(discovered_warnings)
		_populate_references(
			handle,
			path,
			references,
			image_shape,
			stage_um,
			row,
			state.warnings,
		)

	state.finish(row)
	return row


def extract_scan_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
	"""Extract rows while sharing sibling-reference inspection within each group."""
	reference_cache: dict[tuple[Path, str], tuple[tuple[ReferenceFile, ...], tuple[str, ...]]] = {}
	rows = []
	for path in paths:
		identity = _projection_identity(path)
		if identity is None:
			continue
		cache_key = (path.parent.resolve(), identity.prefix)
		if cache_key not in reference_cache:
			reference_cache[cache_key] = load_reference_files(path.parent, identity.prefix)
		references, warnings = reference_cache[cache_key]
		rows.append(
			extract_scan_row(
				path,
				references=references,
				reference_warnings=warnings,
			)
		)
	return rows


def _default_output(inputs: tuple[Path, ...]) -> Path:
	if len(inputs) == 1:
		input_path = inputs[0]
		if input_path.is_dir():
			return input_path / f"{input_path.name}_scan_log.csv"
		return input_path.with_name(f"{input_path.stem}_scan_log.csv")
	return Path("sigray_scan_log.csv")


def _emit_row_warnings(rows: Iterable[dict[str, str]]) -> None:
	for row in rows:
		if row["Metadata Warnings"]:
			click.echo(
				f"Warning: {row['Scan']}: {row['Metadata Warnings']}",
				err=True,
			)


def _upload_rows(
	rows: list[dict[str, str]],
	google_conf: Path,
	spreadsheet: str,
	sheet: str,
	create_tab: bool,
	verify_header: bool,
	strict_header_order: bool,
) -> None:
	try:
		service = tabular_log.build_google_sheets_service(google_conf)
		created_tab = False
		if create_tab:
			created_tab = tabular_log.create_tab_if_missing(
				service,
				spreadsheet,
				sheet,
				SCAN_LOG_FIELDS,
			)
		response = tabular_log.append_rows(
			service,
			spreadsheet,
			sheet,
			SCAN_LOG_FIELDS,
			rows,
			verify=verify_header,
			strict_header_order=strict_header_order,
		)
	except Exception as exc:
		raise click.ClickException(f"Google Sheets upload failed: {exc}") from exc
	if created_tab:
		click.echo(f"Created Google Sheets tab with scan header: {sheet}")
	updated_range = response.get("updates", {}).get("updatedRange", "unknown range")
	click.echo(f"Appended {len(rows)} scan log row(s): {updated_range}")


def _write_rows(output: Path, rows: list[dict[str, str]], force: bool) -> None:
	try:
		tabular_log.write_csv(output, SCAN_LOG_FIELDS, rows, force=force)
	except OSError as exc:
		raise click.ClickException(str(exc)) from exc
	click.echo(f"Wrote {len(rows)} scan log row(s): {output}")


@click.command("sigray-scan-log")
@click.argument(
	"inputs",
	nargs=-1,
	type=click.Path(exists=True, path_type=Path),
)
@click.option(
	"--output",
	"-o",
	type=click.Path(dir_okay=False, path_type=Path),
	help="Output CSV. Defaults beside a single input, or to sigray_scan_log.csv.",
)
@click.option(
	"--upload",
	is_flag=True,
	help="Append all extracted rows to Google Sheets instead of writing a local CSV.",
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
	help="Create a missing destination tab with the canonical scan header.",
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
		"Disabling this uses canonical positional order."
	),
)
@click.option(
	"--strict-header-order",
	is_flag=True,
	help="Require the canonical scan header order instead of matching by name.",
)
@click.option("--force", is_flag=True, help="Replace OUTPUT if it already exists.")
def sigray_scan_log(
	inputs: tuple[Path, ...],
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
	"""Extract one scan-log row per numbered projection HDF5 acquisition."""
	if not inputs:
		raise click.UsageError("At least one HDF5 file or directory is required.")
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
	paths = discover_projection_files(inputs)
	if not paths:
		raise click.ClickException(
			"No numbered projection .h5 or .hdf5 files found; reference-only "
			"FLAT, DARK, and POST files do not produce scan rows."
		)

	try:
		rows = extract_scan_rows(paths)
	except (OSError, ValueError) as exc:
		raise click.ClickException(str(exc)) from exc

	_emit_row_warnings(rows)

	if upload:
		_upload_rows(
			rows,
			google_conf,
			spreadsheet,
			sheet,
			create_tab,
			verify_header,
			strict_header_order,
		)
		return

	output = output or _default_output(inputs)
	_write_rows(output, rows, force)


if __name__ == "__main__":
	sigray_scan_log()
