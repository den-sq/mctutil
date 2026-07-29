"""Compatibility shim for historical CloudFiles null-interval failures."""

from __future__ import annotations

from functools import wraps
from importlib import import_module


_PATCH_MARKER = "_mctutil_null_interval_safe"


def patch_cloudfiles_monitoring() -> bool:
	"""Patch old CloudFiles monitoring releases once; newer releases are a no-op."""
	try:
		monitoring = import_module("cloudfiles.monitoring")
		monitor = monitoring.TransmissionMonitor
	except (AttributeError, ImportError):
		return False

	original = monitor.end_io
	if getattr(original, _PATCH_MARKER, False):
		return True

	@wraps(original)
	def safe_end_io(self, flight_id, num_bytes):
		if num_bytes is None or num_bytes <= 0:
			return None
		try:
			return original(self, flight_id, num_bytes)
		except ValueError as exc:
			if "Null Interval objects not allowed" in str(exc):
				return None
			raise

	setattr(safe_end_io, _PATCH_MARKER, True)
	monitor.end_io = safe_end_io
	return True
