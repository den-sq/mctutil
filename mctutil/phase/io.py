"""JSON and NumPy adapters for phase command inputs and result records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_parameters(config_path: Path) -> tuple[dict, dict]:
	"""Load function arguments, resolving array paths beside the JSON config."""

	parameters = json.loads(config_path.read_text(encoding="utf-8"))
	if not isinstance(parameters, dict):
		raise ValueError("phase configuration must contain a JSON object")
	arguments = dict(parameters)
	for name in ("projections", "B_init", "phi_init", "envelope", "angles_deg"):
		value = arguments.get(name)
		if value is None:
			continue
		if isinstance(value, str):
			path = Path(value)
			if path.suffix.lower() != ".npy":
				raise ValueError(f"{name} must reference a .npy array: {value}")
			arguments[name] = np.load(config_path.parent / path, allow_pickle=False)
		else:
			arguments[name] = np.asarray(value)
	return parameters, arguments


def _numpy_json(value):
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	raise TypeError(f"cannot serialize {type(value).__name__} as phase result JSON")


def json_text(value) -> str:
	"""Serialize a phase record or export through the same NumPy boundary."""

	return json.dumps(value, default=_numpy_json, allow_nan=False, indent=2) + "\n"


def result_json(producer: str, parameters: dict, config_path: Path, result) -> str:
	"""Serialize one result before opening its output, retaining input provenance."""

	return json_text({
		"schema_version": 1,
		"producer": producer,
		"parameters": parameters,
		"config_path": str(config_path.resolve()),
		"result": result,
	})
