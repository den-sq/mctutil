"""Bitmask-filtered structured logger for mctutil.

Ported from den-sq/lftomo's `lftomo/util/log.py` to align with the authoring
conventions of the lab's other CLI tool, with three real bug-fixes vs. that
source: the default file-log mask uses bitwise `|` (not `or`) so it actually
combines the LOG flags it intends to; the IntFlag's redundant `__init__` is
dropped (colors live in a property dict); and the default file-log path is
allocated lazily via `set_log_file()` instead of being constructed at module
import time relative to whatever cwd Python started in.

Adds two surface tweaks over lftomo: `progress()` (mctutil's
`click.progressbar` wrapper, was `log_progress` pre-rename) and `start()`
(convenience for header + Script Start line, matches the pre-rename ergonomic).
Public `set_screen()` / `set_threshold()` setters let the top-level `mctutil`
CLI group wire `--log-level` / `--quiet` / `--verbose` without monkey-patching.

Module-level singleton: `log = Logger()`. Idiomatic import shape is
`from mctutil.shared.log import log, LOG`.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntFlag
from functools import reduce
from operator import ior
from pathlib import Path
from sys import stdout, exc_info
from typing import Callable, Iterable, TextIO

import click
import psutil


_DEFAULT_PROGRESS_MESSAGE = object()


class LOG(IntFlag):
	SILENT = 0
	ERROR = 1
	STATUS = 2
	TIME = 4
	WARN = 8
	INFO = 16
	DEBUG = 32

	@property
	def color(self):
		return {
			LOG.SILENT: "black",
			LOG.ERROR: "red",
			LOG.STATUS: "green",
			LOG.TIME: "cyan",
			LOG.WARN: "yellow",
			LOG.INFO: "white",
			LOG.DEBUG: "magenta",
		}[self]


# Convenience masks for the top-level CLI plumbing. Each is the set of
# levels that get routed to stdout at that verbosity.
LOG_MASK_QUIET = LOG.ERROR
LOG_MASK_DEFAULT = LOG.ERROR | LOG.STATUS | LOG.TIME | LOG.WARN
LOG_MASK_VERBOSE = LOG.ERROR | LOG.STATUS | LOG.TIME | LOG.WARN | LOG.INFO
LOG_MASK_ALL = LOG.ERROR | LOG.STATUS | LOG.TIME | LOG.WARN | LOG.INFO | LOG.DEBUG


class ProgressHandle:
	"""Context-managed progress that supports iteration and manual updates."""

	def __init__(
		self,
		logger,
		step: str,
		items: Iterable | None,
		length: int | None,
		disp,
		out: TextIO,
		log_level: LOG,
		initial: int,
		start_message: str | Callable | None | object,
		final_message: str | Callable | None | object,
	):
		if items is None and length is None:
			raise TypeError("items or length is required")
		self.logger = logger
		self.step = step
		self.items = items
		self.length = length
		self.log_level = log_level
		self.initial = initial
		self.position = initial
		self.start_message = start_message
		self.final_message = final_message
		self._entered = False
		self._enabled = logger.screen_enabled(log_level)
		self._interactive = (
			self._enabled
			and bool(getattr(out, "isatty", lambda: False)())
		)
		self._bar = click.progressbar(
			items,
			length=length,
			item_show_func=disp,
			file=out,
			show_eta=True,
			show_pos=True,
			label=logger.progress_label(step, log_level),
			info_sep="|",
			width=39,
			bar_template="%(label)s|%(bar)s|%(info)s|",
			hidden=not self._interactive,
		)
		if self.length is None:
			self.length = self._bar.length

	def _message(self, value, default: str) -> str | None:
		if value is _DEFAULT_PROGRESS_MESSAGE:
			return default
		if callable(value):
			return value(self)
		return value

	def __enter__(self):
		start = self._message(
			self.start_message,
			f"Started; total={self.length}.",
		)
		if start is not None:
			self.logger.write(self.step, start, log_level=self.log_level)
		if self.initial:
			self._bar.pos = self.initial
		self._bar.__enter__()
		self._entered = True
		return self

	def __iter__(self):
		if not self._entered:
			raise RuntimeError("progress handle must be entered before iteration")
		if not self._interactive:
			for item in self._bar.iter:
				yield item
				self.update(1)
			return
		for item in self._bar:
			yield item
			self.position = self._bar.pos
		self.position = self._bar.pos

	def update(self, count: int) -> None:
		"""Advance manually by ``count`` completed units."""
		if not self._entered:
			raise RuntimeError("progress handle must be entered before update")
		if count < 0:
			raise ValueError("progress updates must be non-negative")
		self.position += count
		self._bar.update(count)

	def __exit__(self, exc_type, exc_value, traceback):
		try:
			return self._bar.__exit__(exc_type, exc_value, traceback)
		finally:
			self._entered = False
			self.position = self._bar.pos
			if exc_type is None:
				final = self._message(
					self.final_message,
					f"Completed {self.position}/{self.length}.",
				)
				if final is not None:
					self.logger.write(
						self.step,
						final,
						log_level=self.log_level,
					)


class Logger:
	"""Bitmask-filtered structured logger.

	Each line has shape `TYPE|STEP|TIMESTAMP|MEM_USE|MEM_FREE|STATEMENT` and is
	dispatched via per-destination masks: `__log_screen` (a `{stdout, stderr}`
	dict, each value a LOG mask) and `__logs` (a `{name: (Path, LOG)}` dict for
	named file destinations). A level emits to a destination iff its bit is set
	in that destination's mask.
	"""

	def __init__(self, log_screen=None, log_files=None):
		self.script_start = datetime.now()
		self.__attached_funcs = []
		self.__log_screen = log_screen if log_screen is not None else {
			"stdout": LOG_MASK_DEFAULT,
			"stderr": LOG.ERROR,
		}
		# Bug-fix vs. lftomo: default to no file logs; opt in via
		# ``set_log_file()`` so we don't allocate a path at import time.
		self.__logs = log_files if log_files is not None else {}
		self.__pid = psutil.Process().pid

	def set_screen(self, stdout_mask=None, stderr_mask=None):
		"""Replace per-stream LOG masks. ``None`` leaves a stream unchanged."""
		if stdout_mask is not None:
			self.__log_screen["stdout"] = stdout_mask
		if stderr_mask is not None:
			self.__log_screen["stderr"] = stderr_mask

	def set_threshold(self, mask):
		"""Convenience: set stdout mask to ``mask`` and stderr mask to ``LOG.ERROR``."""
		self.set_screen(stdout_mask=mask, stderr_mask=LOG.ERROR)

	def screen_enabled(self, log_level: LOG) -> bool:
		"""Return whether ``log_level`` is enabled on the progress stream."""
		return bool(self.__log_screen.get("stdout", LOG.SILENT) & log_level)

	def set_log_file(self, name, path, mask=None):
		"""Add or replace a named file destination.

		Default mask is everything except ``INFO`` and ``DEBUG`` (matches
		the intent of lftomo's default; bug-fixed to use bitwise ``|``).
		"""
		if mask is None:
			mask = LOG_MASK_ALL & ~(LOG.INFO | LOG.DEBUG)
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		self.__logs[name] = (path, mask)

	def attach_func(self, func):
		"""Run ``func(step, pid)`` for every ``write()`` call (and ``confirm`` / ``prompt``)."""
		if func not in self.__attached_funcs:
			self.__attached_funcs.append(func)

	def header(self, out=None):
		"""Write a column-header line to all configured screens and files."""
		header_message = f'{"TYPE":6}|{"STEP":20}|   TIMESTAMP   |MEM USAGE|MEM FREE | STATEMENT '
		self.__special_write(header_message, out)

	def footer(self, out=None, error=None):
		"""Write a COMPLETED or ERRORED bookend line."""
		step = "COMPLETED" if error is None else "ERRORED"
		statement = "" if error is None else str(error)
		log_level = LOG.STATUS if error is None else LOG.ERROR
		self.__special_write(self.__log_message(step, statement, log_level), out)

	def start(self, out=None):
		"""Convenience: column header followed by a ``Script Start`` STATUS line."""
		self.header(out=out)
		self.write("Script Start", str(self.script_start), log_level=LOG.STATUS, out=out)

	def write(self, step, statement='', log_level=LOG.TIME, out=None, write_tb=False):
		"""Emit a formatted log line to all destinations whose mask matches."""
		for func in self.__attached_funcs:
			func(step, self.__pid)

		if write_tb:
			_, _, last_traceback = exc_info()
			statement = f'{statement}-tb-{last_traceback}'

		message = self.__log_message(step, statement, log_level)

		if out is not None:
			click.echo(message, file=out, err=(log_level == LOG.ERROR))
		else:
			self.__multi_write(message, log_level)

	def confirm(self, step, statement='', log_level=LOG.TIME, out: TextIO = stdout):
		for func in self.__attached_funcs:
			func(step, self.__pid)
		return click.confirm(self.__log_message(step, statement, log_level),
								err=(log_level == LOG.ERROR))

	def prompt(self, step, statement='', log_level=LOG.TIME, out: TextIO = stdout, default=None):
		for func in self.__attached_funcs:
			func(step, self.__pid)
		return click.prompt(self.__log_message(step, statement, log_level),
								err=(log_level == LOG.ERROR), default=default)

	def progress_label(self, step: str, log_level: LOG = LOG.STATUS) -> str:
		"""Return the structured label used by interactive progress bars."""
		styled_type = click.style(f'{log_level.name:6}', log_level.color)
		return f'{styled_type}|{step[:20]:20}'

	def progress(
		self,
		step,
		items=None,
		length=None,
		disp=None,
		out: TextIO | None = None,
		log_level=LOG.STATUS,
		initial=0,
		start_message=_DEFAULT_PROGRESS_MESSAGE,
		final_message=_DEFAULT_PROGRESS_MESSAGE,
	):
		"""Create a progress handle for iteration or manual ``update()`` calls.

		Interactive bars are restricted to TTY output. Durable start/final
		records still flow through ``write()`` in captured output, and both the
		bar and records honor the configured log threshold.
		"""
		out = out or click.get_text_stream("stdout")
		return ProgressHandle(
			self,
			step,
			items,
			length,
			disp,
			out,
			log_level,
			initial,
			start_message,
			final_message,
		)

	def dump(self, statement, log_level=LOG.INFO, out=None, use_color=False):
		"""Write a raw statement without the TYPE/STEP/TIMESTAMP/MEM preamble. Attached funcs aren't run."""
		if use_color:
			statement = click.style(statement, log_level.color)
		if out is not None:
			click.echo(statement, file=out, err=(log_level == LOG.ERROR))
		else:
			self.__multi_write(statement, log_level)

	def __log_message(self, step, statement='', log_level=LOG.TIME):
		styled_type = click.style(f'{log_level.name:6}', log_level.color)
		return (f'{styled_type}'
				f'|{step[:20]:20}'
				f'|{str(datetime.now() - self.script_start).zfill(15)}'
				f'|{psutil.Process().memory_info().vms // 1024 ** 2:09.2f}MB'
				f'|{psutil.virtual_memory().available // 1024 ** 2:09.2f}MB'
				f'|"{statement}"')

	def __special_write(self, message, out=None):
		# Bug-fix vs. lftomo: write to both stdout and stderr if both are
		# configured (lftomo's elif chain skipped stderr when stdout was set).
		if out is not None:
			click.echo(message, file=out)
			return
		if self.__log_screen.get("stdout"):
			click.echo(message)
		if self.__log_screen.get("stderr"):
			click.echo(message, err=True)
		for path, _flag in self.__logs.values():
			with open(path, "a") as handle:
				click.echo(message, file=handle)

	def __multi_write(self, message, log_level):
		stream_mask = reduce(ior, self.__log_screen.values(), LOG.SILENT)
		if log_level & stream_mask:
			stderr_mask = self.__log_screen.get("stderr", LOG.SILENT)
			err = bool(log_level & stderr_mask)
			click.echo(message, err=err)

		for path, log_flag in self.__logs.values():
			if log_level & log_flag:
				with open(path, "a") as handle:
					click.echo(message, file=handle)


log = Logger()
