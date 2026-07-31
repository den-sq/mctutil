"""Normalize synchronous stdout emitted by Igneous task factories."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from functools import wraps
import getpass
from io import StringIO
import os
import re

from mctutil.shared.log import log, LOG


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_NORMALIZER = ContextVar("mctutil_igneous_output_normalizer", default=None)
_PROVENANCE_CONTACT_FALLBACK = (
	'Unable to determine provenance contact email. Set "git config '
	'user.email". Using unix $USER instead.'
)
_PROVENANCE_CONTACT_KEY = "provenance-contact-email"


class IgneousOutputNormalizer:
	"""Classify and deduplicate routine Igneous task-creation output."""

	def __init__(self):
		self._seen = set()

	def emit(self, text: str, *, stderr: bool = False) -> None:
		for raw_line in text.splitlines():
			line = _ANSI_ESCAPE.sub("", raw_line).strip()
			if not line:
				continue
			log_level, key, statement = self._classify(line, stderr)
			if key is not None:
				if key in self._seen:
					if key == _PROVENANCE_CONTACT_KEY:
						log_level = LOG.DEBUG
					else:
						continue
				else:
					self._seen.add(key)
			log.write("Igneous", statement, log_level=log_level)

	@staticmethod
	def _classify(line: str, stderr: bool) -> tuple[LOG, str | None, str]:
		if line.startswith("Volume Bounds:"):
			return LOG.INFO, f"bounds:{line}", line
		if line.startswith("Selected ROI:"):
			return LOG.INFO, f"roi:{line}", line
		if line == _PROVENANCE_CONTACT_FALLBACK:
			user = os.environ.get("USER") or getpass.getuser()
			return (
				LOG.WARN,
				_PROVENANCE_CONTACT_KEY,
				(
					"Provenance contact email is not configured; "
					f"using Unix user {user!r}. Set "
					"`git config user.email <email>` to record a stable "
					"provenance contact."
				),
			)
		if line == "WARNING: No scales generated.":
			return LOG.INFO, "no-scales-generated", "No additional scales generated."
		if line.startswith("No factors generated."):
			return LOG.INFO, f"no-factors:{line}", line
		return LOG.WARN, None, line if not stderr else f"stderr: {line}"


@contextmanager
def igneous_output_session():
	"""Deduplicate Igneous messages for one outer CLI command."""
	current = _NORMALIZER.get()
	if current is not None:
		yield current
		return
	normalizer = IgneousOutputNormalizer()
	token = _NORMALIZER.set(normalizer)
	try:
		yield normalizer
	finally:
		_NORMALIZER.reset(token)


def igneous_output_command(function):
	"""Run a command callback within one Igneous output session."""
	@wraps(function)
	def wrapped(*args, **kwargs):
		with igneous_output_session():
			return function(*args, **kwargs)

	return wrapped


def _capture_output(normalizer, function):
	stdout_buffer = StringIO()
	stderr_buffer = StringIO()
	try:
		with (
			redirect_stdout(stdout_buffer),
			redirect_stderr(stderr_buffer),
		):
			return function()
	finally:
		normalizer.emit(stdout_buffer.getvalue())
		normalizer.emit(stderr_buffer.getvalue(), stderr=True)


class _CapturedIterator:
	"""Capture output emitted while a lazy task factory is consumed."""

	def __init__(self, iterator, normalizer):
		self._iterator = iterator
		self._normalizer = normalizer

	def __iter__(self):
		return self

	def __next__(self):
		return _capture_output(
			self._normalizer,
			lambda: next(self._iterator),
		)


class _CapturedIterable:
	"""Proxy a reusable lazy iterable without dropping its sequence protocol."""

	def __init__(self, iterable, normalizer):
		self._iterable = iterable
		self._normalizer = normalizer

	def __iter__(self):
		iterator = _capture_output(
			self._normalizer,
			lambda: iter(self._iterable),
		)
		return _CapturedIterator(iterator, self._normalizer)

	def __len__(self):
		return _capture_output(
			self._normalizer,
			lambda: len(self._iterable),
		)

	def __getitem__(self, index):
		value = _capture_output(
			self._normalizer,
			lambda: self._iterable[index],
		)
		if isinstance(index, slice):
			return _CapturedIterable(value, self._normalizer)
		return value

	def __getattr__(self, name):
		return getattr(self._iterable, name)


def capture_igneous_call(function, *args, **kwargs):
	"""Call an Igneous factory and normalize eager or lazy output."""
	with igneous_output_session() as normalizer:
		result = _capture_output(
			normalizer,
			lambda: function(*args, **kwargs),
		)
		if isinstance(result, Iterator):
			return _CapturedIterator(result, normalizer)
		if (
			isinstance(result, Iterable)
			and not isinstance(
				result,
				(str, bytes, bytearray, dict, list, tuple, set, frozenset),
			)
		):
			return _CapturedIterable(result, normalizer)
		return result
