"""Single-source optional dependency checks and installation hints."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable
from types import ModuleType


EXTRA_MODULES: dict[str, tuple[str, ...]] = {
	"google-sheets": (
		"google.auth.transport.requests",
		"google.oauth2.credentials",
		"google_auth_oauthlib.flow",
		"googleapiclient.discovery",
	),
	"als832": ("h5py", "tifffile"),
	"flats": ("scipy", "tifffile"),
	"transform": ("h5py", "tifffile", "zarr"),
	"ng": (
		"cloudvolume",
		"cloudfiles",
		"neuroglancer_scripts",
		"tifffile",
		"zarr",
	),
	"serve": (
		"flask",
		"flask_cors",
		"neuroglancer",
		"qrcode",
		"RangeHTTPServer",
	),
	"sino": ("skimage", "tifffile"),
	"mesh": ("cloudvolume", "igneous.task_creation", "taskqueue"),
	"aws": ("boto3",),
	"dragonfly": (),
}


def _as_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
	return (value,) if isinstance(value, str) else tuple(value)


def install_command(extras: str | Iterable[str]) -> str:
	"""Return the canonical editable-install command for one or more extras."""
	extra_names = _as_tuple(extras)
	unknown = tuple(extra for extra in extra_names if extra not in EXTRA_MODULES)
	if unknown:
		raise ValueError(f"unknown optional dependency extra(s): {', '.join(unknown)}")
	return f"pip install -e '.[{','.join(extra_names)}]'"


def _declared_for(module_name: str, extras: tuple[str, ...]) -> bool:
	return any(
		module_name == declared
		or module_name.startswith(f"{declared}.")
		for extra in extras
		for declared in EXTRA_MODULES[extra]
	)


def require(
	modules: str | Iterable[str],
	extra: str | Iterable[str],
	*,
	purpose: str | None = None,
	error_type: type[Exception] = RuntimeError,
) -> ModuleType | tuple[ModuleType, ...]:
	"""Import declared optional modules or raise one canonical install hint.

	``purpose`` describes the feature in the error prefix. ``error_type`` lets
	Click-facing commands preserve ``ClickException`` while library helpers use
	``RuntimeError``.
	"""
	module_names = _as_tuple(modules)
	extras = _as_tuple(extra)
	# Validate extras before indexing the shared table.
	install = install_command(extras)
	undeclared = tuple(
		module_name
		for module_name in module_names
		if not _declared_for(module_name, extras)
	)
	if undeclared:
		raise ValueError(
			f"module(s) {', '.join(undeclared)} are not declared by "
			f"extra(s) {', '.join(extras)}"
		)

	loaded = []
	try:
		for module_name in module_names:
			loaded.append(importlib.import_module(module_name))
	except ImportError as exc:
		feature = purpose or ", ".join(module_names)
		raise error_type(f"{feature}; install with {install}") from exc

	return loaded[0] if isinstance(modules, str) else tuple(loaded)


def module_available(module_name: str) -> bool:
	"""Return whether an optional module can be resolved without importing it."""
	try:
		return importlib.util.find_spec(module_name) is not None
	except (ImportError, ModuleNotFoundError, ValueError):
		return False


def missing_dependencies(
	extras: Iterable[str],
) -> dict[str, tuple[str, ...]]:
	"""Return missing import modules grouped by canonical optional extra."""
	extra_names = _as_tuple(extras)
	# Reuse canonical validation and preserve the caller's requested order.
	install_command(extra_names)
	return {
		extra: tuple(
			module
			for module in EXTRA_MODULES[extra]
			if not module_available(module)
		)
		for extra in extra_names
		if any(
			not module_available(module)
			for module in EXTRA_MODULES[extra]
		)
	}
