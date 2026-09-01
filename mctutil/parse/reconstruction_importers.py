"""Direct readers for the three supported reconstruction metadata sources."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shlex
from typing import Iterable

import numpy as np

from mctutil.parse import xaid_reconstruction_log
from mctutil.parse.import_mappings import (
	ALS832_RECONSTRUCTION_MAPPING,
	SEVEN_BM_RECONSTRUCTION_MAPPING,
	XAID_RECONSTRUCTION_MAPPING,
)
from mctutil.parse.import_records import (
	RECONSTRUCTION_FIELDS,
	ReconstructionRecord,
	empty_reconstruction_values,
)
from mctutil.shared.deps import require


_FIELD_BY_NAME = {item.name: item for item in RECONSTRUCTION_FIELDS}
_H5_SUFFIXES = (".h5", ".hdf5")


def _require_h5py():
	return require(
		"h5py",
		"als832",
		purpose="reconstruction HDF5 import dependencies are unavailable",
	)


def _decode(value) -> str:
	if isinstance(value, (bytes, np.bytes_)):
		return bytes(value).decode("utf-8", errors="replace").strip()
	return str(value).strip()


def _parse_float(value) -> float | None:
	if value is None or type(value) is bool:
		return None
	if isinstance(value, (int, float, np.number)):
		result = float(value)
	else:
		match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", _decode(value))
		if match is None:
			return None
		result = float(match.group(0))
	return result if math.isfinite(result) else None


def _parse_int(value) -> int | None:
	number = _parse_float(value)
	if number is None or not math.isclose(number, round(number), abs_tol=1e-6):
		return None
	return int(round(number))


def _parse_bool(value) -> bool | None:
	if type(value) is bool:
		return value
	if isinstance(value, (int, float)) and not isinstance(value, bool):
		return bool(value)
	text = _decode(value).casefold()
	if text in {"1", "true", "yes", "y", "on"}:
		return True
	if text in {"0", "false", "no", "n", "off"}:
		return False
	return None


def _coerce(value, field_name: str):
	if value is None:
		return None
	if isinstance(value, str) and value.strip().casefold() in {"", "none", "null", "nan"}:
		return None
	value_type = _FIELD_BY_NAME[field_name].value_type
	if value_type is str:
		return _decode(value)
	if value_type is int:
		return _parse_int(value)
	if value_type is float:
		return _parse_float(value)
	if value_type is bool:
		return _parse_bool(value)
	if value_type is tuple and isinstance(value, (list, tuple)):
		return tuple(value)
	return None


def _discover_files(inputs: Iterable[Path], suffixes: tuple[str, ...]) -> tuple[Path, ...]:
	paths: set[Path] = set()
	for raw_path in inputs:
		path = Path(raw_path)
		if path.is_dir():
			paths.update(
				item
				for item in path.iterdir()
				if item.is_file() and item.suffix.casefold() in suffixes
			)
		elif path.is_file() and path.suffix.casefold() in suffixes:
			paths.add(path)
	if not paths:
		raise ValueError("no matching reconstruction metadata inputs found")
	return tuple(sorted(paths, key=lambda item: str(item)))


def _h5_dataset(handle, path: str):
	item = handle.get(path)
	return item if item is not None and hasattr(item, "shape") else None


def _h5_value(handle, path: str):
	dataset = _h5_dataset(handle, path)
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
		decoded = _decode(value)
		if decoded:
			return decoded
	return None


def _apply_h5_mapping(handle, mapping, values: dict[str, object | None]) -> None:
	for field_name, entry in mapping.items():
		raw_value = _h5_value(handle, entry["locator"])
		if raw_value is None:
			continue
		if "multiplier" in entry:
			number = _parse_float(raw_value)
			raw_value = number * entry["multiplier"] if number is not None else None
		values[field_name] = _coerce(raw_value, field_name)


def _tuple_or_none(items: Iterable[object | None]) -> tuple | None:
	values = tuple(items)
	return values if values and all(value is not None for value in values) else None


def _als832_reconstruction(path: Path) -> ReconstructionRecord:
	h5py = _require_h5py()
	values = empty_reconstruction_values()
	warnings: list[str] = []
	with h5py.File(path, "r") as handle:
		group = handle.get("/process/tomo_rec/setup/algorithm")
		if group is None:
			raise ValueError(f"{path}: missing /process/tomo_rec/setup/algorithm")
		_apply_h5_mapping(handle, ALS832_RECONSTRUCTION_MAPPING, values)
		scan_id = _h5_value(handle, "/measurement/sample/file_name") or path.stem
		project_id = _h5_value(handle, "/measurement/sample/experiment/proposal")
		values.update(
			{
				"reconstruction_id": f"{path.stem}:tomo_rec_setup",
				"project_id": _coerce(project_id, "project_id"),
				"scan_id": _decode(scan_id),
				"input_projection_file": str(path),
				"input_data_path": "/exchange/data",
				"software_name": "ALS832 tomography setup",
				"configuration_state": "configured",
				"reconstruction_status": "configured",
				"source_metadata_files": (str(path),),
			}
		)
		normalize_by_roi = _parse_bool(
			_h5_value(handle, "/process/tomo_rec/setup/algorithm/normalize_by_ROI")
		)
		if normalize_by_roi is not None:
			values["normalization_method"] = "roi" if normalize_by_roi else "flat-dark"
		values["normalization_roi"] = _tuple_or_none(
			_parse_int(_h5_value(handle, f"/process/tomo_rec/setup/algorithm/{name}"))
			for name in (
				"normalization_ROI_left",
				"normalization_ROI_right",
				"normalization_ROI_top",
				"normalization_ROI_bottom",
			)
		)
	for field_name in ("reconstruction_type", "ring_stripe_method", "output_format"):
		if values[field_name] == "hdf5":
			warnings.append(f"configured {field_name} value 'hdf5' appears mis-serialized")
	return ReconstructionRecord("als832", (str(path),), values, warnings)


def read_als832_reconstructions(inputs: Iterable[Path]) -> tuple[ReconstructionRecord, ...]:
	"""Read configured ALS832 embedded reconstruction setup values."""
	paths = _discover_files(inputs, _H5_SUFFIXES)
	return tuple(_als832_reconstruction(path) for path in paths)


def _ini_value(config, section: str, key: str) -> str | None:
	value = xaid_reconstruction_log.get_xaid_value(config, section, key)
	return value or None


def _apply_ini_mapping(config, values: dict[str, object | None]) -> None:
	for field_name, entry in XAID_RECONSTRUCTION_MAPPING.items():
		section, key = entry["locator"]
		raw_value = _ini_value(config, section, key)
		if raw_value is None:
			continue
		if "multiplier" in entry:
			number = _parse_float(raw_value)
			raw_value = number * entry["multiplier"] if number is not None else None
		values[field_name] = _coerce(raw_value, field_name)


def _ini_tuple(config, section: str, keys: tuple[str, ...], parser) -> tuple | None:
	return _tuple_or_none(parser(_ini_value(config, section, key)) for key in keys)


def _named_ini_values(config, section: str, keys: tuple[str, ...]) -> str | None:
	items = []
	for key in keys:
		value = _ini_value(config, section, key)
		if value is not None:
			items.append(f"{key}={value}")
	return "; ".join(items) or None


def _set_xaid_composites(config, values: dict[str, object | None]) -> None:
	values["volume_roi_start_xyz"] = _ini_tuple(
		config,
		"ROI",
		("roi_posx", "roi_posy", "roi_posz"),
		_parse_int,
	)
	values["volume_roi_extent_xyz"] = _ini_tuple(
		config,
		"ROI",
		("roi_sizex", "roi_sizey", "roi_sizez"),
		_parse_int,
	)
	values["output_rotation_xyz_deg"] = _ini_tuple(
		config,
		"VolumeData",
		("volume_rotation_x", "volume_rotation_y", "volume_rotation_z"),
		_parse_float,
	)
	values["drift_correction_xyz"] = _ini_tuple(
		config,
		"GC_Values",
		("drift_x", "drift_y", "drift_z"),
		_parse_float,
	)
	values["ring_stripe_parameters"] = _named_ini_values(
		config,
		"Ringfilter_Settings",
		("ring_partial_delta", "ring_partial_size", "ring_a_size", "ring_b_size"),
	)
	values["beam_hardening_parameters"] = _named_ini_values(
		config,
		"BHC_Values",
		("bhc_cupping_value", "bhc_streak_value", "bhc_streak_thl_value"),
	)
	values["jitter_correction"] = _named_ini_values(
		config,
		"Reconstruction_Settings",
		("is_opt_jitter", "is_jitter_smooth", "jitter_iterations", "shift_u_max", "shift_v_max"),
	)


def _xaid_reconstruction(path: Path) -> ReconstructionRecord:
	parsed = xaid_reconstruction_log.parse_xaid_config(path)
	config = parsed.config
	values = empty_reconstruction_values()
	warnings = ["repaired malformed first section header"] if parsed.repaired_first_header else []
	_apply_ini_mapping(config, values)
	_set_xaid_composites(config, values)
	input_path = values["input_projection_file"]
	values.update(
		{
			"reconstruction_id": path.stem,
			"scan_id": _source_stem(input_path) if type(input_path) is str else None,
			"software_name": "MITOS X-AID",
			"configuration_state": "configured",
			"reconstruction_status": "configured",
			"source_metadata_files": (str(path),),
		}
	)
	kernel = values["outlier_kernel_px"]
	if type(kernel) is int:
		values["outlier_removal_enabled"] = kernel > 0
	return ReconstructionRecord("xaid", (str(path),), values, warnings)


def read_xaid_reconstructions(inputs: Iterable[Path]) -> tuple[ReconstructionRecord, ...]:
	"""Wrap the existing X-AID parser and normalize reviewed configuration fields."""
	paths = _discover_files(inputs, (".cfg", ".ini", ".txt"))
	return tuple(_xaid_reconstruction(path) for path in paths)


def _source_stem(value: str) -> str:
	name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
	return Path(name).stem


def _parse_cli_options(command: str) -> dict[str, str]:
	tokens = shlex.split(command, posix=True)
	options: dict[str, str] = {}
	index = 0
	while index < len(tokens):
		token = tokens[index]
		if not token.startswith("--"):
			index += 1
			continue
		option = token[2:]
		if "=" in option:
			option, value = option.split("=", 1)
		elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
			index += 1
			value = tokens[index]
		else:
			value = "true"
		options[option.casefold().replace("_", "-")] = value
		index += 1
	return options


def _normal_key(value: str) -> str:
	return value.casefold().replace("_", "-")


def _dict_matches_scan(value: dict, scan_id: str) -> bool:
	identity_values = (
		child
		for key, child in value.items()
		if _normal_key(str(key)) in {"file-name", "filename", "scan-id", "scan"}
	)
	return any(_source_stem(str(candidate)) == scan_id for candidate in identity_values)


def _walk_dicts(node):
	if isinstance(node, dict):
		yield node
		for child in node.values():
			yield from _walk_dicts(child)
	elif isinstance(node, list):
		for child in node:
			yield from _walk_dicts(child)


def _matching_json_records(node, scan_id: str) -> tuple[dict, ...]:
	matches: list[dict] = []
	for value in _walk_dicts(node):
		for key, child in value.items():
			if isinstance(child, dict) and _source_stem(str(key)) == scan_id:
				matches.append(child)
		if _dict_matches_scan(value, scan_id):
			matches.append(value)
	unique = []
	seen = set()
	for match in matches:
		if id(match) not in seen:
			seen.add(id(match))
			unique.append(match)
	return tuple(unique)


def _flatten_options(node: dict, warnings: list[str]) -> dict[str, object]:
	options: dict[str, object] = {}
	conflicts: set[str] = set()

	def visit(value):
		if not isinstance(value, dict):
			return
		for key, child in value.items():
			if isinstance(child, dict):
				visit(child)
				continue
			if isinstance(child, list) and child and isinstance(child[0], dict):
				for item in child:
					visit(item)
				continue
			normalized = _normal_key(str(key))
			if normalized in options and options[normalized] != child:
				conflicts.add(normalized)
			else:
				options[normalized] = child

	visit(node)
	for key in sorted(conflicts):
		options.pop(key, None)
		warnings.append(f"configured JSON contains conflicting {key!r} values")
	return options


def _load_matching_config(path: Path | None, scan_id: str, warnings: list[str]) -> tuple[dict, Path | None]:
	if path is None:
		return {}, None
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, UnicodeError, json.JSONDecodeError) as exc:
		raise ValueError(f"could not read {path}: {exc}") from exc
	matches = _matching_json_records(data, scan_id)
	if not matches:
		warnings.append(f"{path.name} has no exact record for scan {scan_id!r}; ignored")
		return {}, None
	if len(matches) > 1:
		raise ValueError(f"{path}: multiple JSON records match scan {scan_id!r}")
	return _flatten_options(matches[0], warnings), path


def _named_json_numbers(node, accepted_keys: set[str]) -> tuple[float, ...]:
	numbers: list[float] = []
	if isinstance(node, dict):
		for key, value in node.items():
			if _normal_key(str(key)) in accepted_keys:
				number = _parse_float(value)
				if number is not None:
					numbers.append(number)
			numbers.extend(_named_json_numbers(value, accepted_keys))
	elif isinstance(node, list):
		for value in node:
			numbers.extend(_named_json_numbers(value, accepted_keys))
	return tuple(dict.fromkeys(numbers))


def _scalar_number(value) -> float | None:
	if isinstance(value, (str, bytes, int, float, np.number)) and not isinstance(value, bool):
		return _parse_float(value)
	return None


def _one_number(values: Iterable[float | None]) -> float | None:
	numbers = tuple(dict.fromkeys(value for value in values if value is not None))
	return numbers[0] if len(numbers) == 1 else None


def _rotation_center_for_scan(data, scan_id: str) -> float | None:
	accepted_keys = {"rotation-axis", "rotation-center", "rot-cen", "center"}
	direct = _scalar_number(data)
	if direct is not None:
		return direct
	if not isinstance(data, dict):
		return None
	matching: list[float | None] = []
	for key, value in data.items():
		if _source_stem(str(key)) == scan_id:
			direct = _scalar_number(value)
			matching.append(
				direct if direct is not None else _one_number(
					_named_json_numbers(value, accepted_keys)
				)
			)
	for value in _walk_dicts(data):
		if _dict_matches_scan(value, scan_id):
			matching.append(_one_number(_named_json_numbers(value, accepted_keys)))
	matched = _one_number(matching)
	if matched is not None:
		return matched
	return _one_number(
		_scalar_number(value)
		for key, value in data.items()
		if _normal_key(str(key)) in accepted_keys
	)


def _read_rotation_center(path: Path | None, scan_id: str) -> float | None:
	if path is None:
		return None
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, UnicodeError, json.JSONDecodeError) as exc:
		raise ValueError(f"could not read {path}: {exc}") from exc
	return _rotation_center_for_scan(data, scan_id)


def _effective_option(
	name: str,
	command_options: dict[str, str],
	configured_options: dict[str, object],
):
	return command_options.get(name, configured_options.get(name))


def _apply_tomocupy_mapping(
	command_options: dict[str, str],
	configured_options: dict[str, object],
	values: dict[str, object | None],
) -> None:
	for field_name, entry in SEVEN_BM_RECONSTRUCTION_MAPPING.items():
		option = entry["locator"].removeprefix("--")
		raw_value = _effective_option(option, command_options, configured_options)
		if raw_value is None:
			continue
		if "multiplier" in entry:
			number = _parse_float(raw_value)
			raw_value = number * entry["multiplier"] if number is not None else None
		values[field_name] = _coerce(raw_value, field_name)


def _tomocupy_ranges(
	command_options: dict[str, str],
	configured_options: dict[str, object],
	values: dict[str, object | None],
) -> None:
	values["projection_range"] = _tuple_or_none(
		_parse_int(_effective_option(name, command_options, configured_options))
		for name in ("start-proj", "end-proj")
	)
	values["sinogram_range"] = _tuple_or_none(
		_parse_int(_effective_option(name, command_options, configured_options))
		for name in ("start-row", "end-row")
	)


def _tomocupy_derived(
	command_options: dict[str, str],
	configured_options: dict[str, object],
	values: dict[str, object | None],
) -> None:
	exponent = _parse_int(_effective_option("binning", command_options, configured_options))
	if exponent is not None and exponent >= 0:
		values["projection_binning"] = 2 ** exponent
		native = values["native_voxel_size_mm"]
		if type(native) is float:
			values["output_voxel_size_mm"] = native * (2 ** exponent)
	method = values["phase_retrieval_method"]
	if type(method) is str:
		values["phase_retrieval_enabled"] = method.casefold() != "none"
	kernel = values["outlier_kernel_px"]
	if type(kernel) is int:
		values["outlier_removal_enabled"] = kernel > 0
	if values["rotation_axis_auto_method"] is None:
		center_method = configured_options.get("cor-method-full")
		if center_method is not None:
			values["rotation_axis_auto_method"] = (
				"manual" if _decode(center_method).casefold() == "manual" else "auto"
			)
	_tomocupy_ranges(command_options, configured_options, values)


def _tomocupy_rec_lines(inputs: Iterable[Path]) -> tuple[Path, ...]:
	paths: set[Path] = set()
	for raw_path in inputs:
		path = Path(raw_path)
		if path.is_file() and path.name == "rec_line.txt":
			paths.add(path)
		elif path.is_dir():
			direct = path / "rec_line.txt"
			if direct.is_file():
				paths.add(direct)
			for child in path.iterdir():
				candidate = child / "rec_line.txt" if child.is_dir() else None
				if candidate is not None and candidate.is_file():
					paths.add(candidate)
	if not paths:
		raise ValueError("no TomoCuPy rec_line.txt inputs found")
	return tuple(sorted(paths, key=lambda item: str(item)))


def _explicit_artifact(inputs: Iterable[Path], name: str) -> Path | None:
	matches = tuple(
		Path(item)
		for item in inputs
		if Path(item).is_file() and Path(item).name == name
	)
	if len(matches) > 1:
		raise ValueError(f"multiple explicit {name} inputs")
	return matches[0] if matches else None


def _nearby_artifact(rec_line: Path, explicit: Path | None, name: str) -> Path | None:
	if explicit is not None:
		return explicit
	for directory in (rec_line.parent, rec_line.parent.parent):
		candidate = directory / name
		if candidate.is_file():
			return candidate
	return None


def _output_directory(rec_line: Path, output_path: object | None) -> Path:
	if type(output_path) is str and output_path.casefold() not in {"", "none", "null"}:
		path = Path(output_path)
		return path if path.is_absolute() else rec_line.parent / path
	return rec_line.parent


def _inspect_tiff_outputs(
	output_directory: Path,
	values: dict[str, object | None],
	warnings: list[str],
) -> None:
	if not output_directory.is_dir():
		return
	paths = tuple(
		sorted(
			(
				path
				for path in output_directory.iterdir()
				if path.is_file() and path.suffix.casefold() in {".tif", ".tiff"}
			),
			key=lambda item: item.name,
		)
	)
	if not paths:
		return
	tifffile = require(
		"tifffile",
		"als832",
		purpose="TomoCuPy TIFF metadata inspection is unavailable",
	)
	metadata = []
	for path in dict.fromkeys((paths[0], paths[-1])):
		with tifffile.TiffFile(path) as handle:
			metadata.append((tuple(int(item) for item in handle.pages[0].shape), str(handle.pages[0].dtype)))
	if len(set(metadata)) > 1:
		warnings.append("first and last reconstructed TIFF metadata differ")
	values["output_file_count"] = len(paths)
	values["output_image_shape"] = metadata[0][0]
	values["output_dtype_observed"] = metadata[0][1]


def _tomocupy_reconstruction(
	rec_line: Path,
	recon_params: Path | None,
	rot_cen: Path | None,
) -> ReconstructionRecord:
	try:
		command = rec_line.read_text(encoding="utf-8").strip()
	except (OSError, UnicodeError) as exc:
		raise ValueError(f"could not read {rec_line}: {exc}") from exc
	command_options = _parse_cli_options(command)
	input_path = command_options.get("file-name")
	if not input_path:
		raise ValueError(f"{rec_line}: effective command has no --file-name")
	scan_id = _source_stem(input_path)
	warnings: list[str] = []
	configured_options, matched_config = _load_matching_config(recon_params, scan_id, warnings)
	values = empty_reconstruction_values()
	_apply_tomocupy_mapping(command_options, configured_options, values)
	_tomocupy_derived(command_options, configured_options, values)

	center_file_value = _read_rotation_center(rot_cen, scan_id)
	if rot_cen is not None and center_file_value is None:
		warnings.append(f"{rot_cen.name} has no exact center for scan {scan_id!r}; ignored")
	command_center = _parse_float(command_options.get("rotation-axis"))
	if command_center is not None:
		values["rotation_axis_coordinate_px"] = command_center
		if center_file_value is not None and not math.isclose(
			command_center,
			center_file_value,
			rel_tol=1e-9,
			abs_tol=1e-6,
		):
			warnings.append("rot_cen.json disagrees with effective --rotation-axis")
	elif center_file_value is not None:
		values["rotation_axis_coordinate_px"] = center_file_value

	values.update(
		{
			"reconstruction_id": rec_line.parent.name,
			"scan_id": scan_id,
			"input_projection_file": input_path,
			"software_name": "TomoCuPy",
			"configuration_state": "effective",
			"effective_command": command,
		}
	)
	output_directory = _output_directory(rec_line, values["output_path"])
	_inspect_tiff_outputs(output_directory, values, warnings)
	values["reconstruction_status"] = (
		"completed" if values["output_file_count"] is not None else "started"
	)
	metadata_files = tuple(
		str(path)
		for path in (matched_config, rec_line, rot_cen)
		if path is not None
	)
	values["source_metadata_files"] = metadata_files
	locations = metadata_files + (str(output_directory),)
	return ReconstructionRecord("tomocupy-7bm", locations, values, warnings)


def read_tomocupy_7bm_reconstructions(
	inputs: Iterable[Path],
) -> tuple[ReconstructionRecord, ...]:
	"""Read matching 7-BM command/JSON/center artifacts with direct precedence."""
	inputs = tuple(Path(item) for item in inputs)
	rec_lines = _tomocupy_rec_lines(inputs)
	explicit_config = _explicit_artifact(inputs, "recon_params.json")
	explicit_center = _explicit_artifact(inputs, "rot_cen.json")
	return tuple(
		_tomocupy_reconstruction(
			rec_line,
			_nearby_artifact(rec_line, explicit_config, "recon_params.json"),
			_nearby_artifact(rec_line, explicit_center, "rot_cen.json"),
		)
		for rec_line in rec_lines
	)
