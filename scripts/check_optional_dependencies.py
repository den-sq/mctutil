"""Import the lazy dependency set promised by one or more project extras."""

from __future__ import annotations

import argparse
import importlib

from mctutil.shared.deps import EXTRA_MODULES


def check_extras(extras: tuple[str, ...]) -> tuple[tuple[str, str, str], ...]:
	"""Return ``(extra, module, error)`` records for failed imports."""
	failures = []
	for extra in extras:
		for module_name in EXTRA_MODULES[extra]:
			try:
				importlib.import_module(module_name)
			except Exception as exc:  # Import-time binary/API errors matter here.
				failures.append((extra, module_name, str(exc)))
			else:
				print(f"{extra}: imported {module_name}")
	return tuple(failures)


def parse_args(arguments=None):
	parser = argparse.ArgumentParser(
		description="Smoke-test modules installed by mctutil optional extras.",
	)
	selection = parser.add_mutually_exclusive_group(required=True)
	selection.add_argument(
		"--extra",
		action="append",
		choices=tuple(EXTRA_MODULES),
		help="Extra to check; repeat for multiple extras.",
	)
	selection.add_argument(
		"--all",
		action="store_true",
		help="Check every declared extra.",
	)
	return parser.parse_args(arguments)


def main(arguments=None) -> int:
	options = parse_args(arguments)
	extras = tuple(EXTRA_MODULES) if options.all else tuple(options.extra)
	failures = check_extras(extras)
	for extra, module_name, error in failures:
		print(f"{extra}: failed to import {module_name}: {error}")
	return 1 if failures else 0


if __name__ == "__main__":
	raise SystemExit(main())
