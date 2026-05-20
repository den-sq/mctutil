#!/usr/bin/env python3
"""Reject Python files whose leading indentation starts with spaces."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

LEADING_WS = re.compile(r'^(?P<ws>[ \t]+)(?=\S)')


def iter_python_files(paths: list[str]) -> list[Path]:
	if paths:
		return [Path(path) for path in paths if path.endswith('.py')]

	return sorted(Path('.').rglob('*.py'))


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description='Fail when Python files use spaces for leading indentation.',
	)
	parser.add_argument('paths', nargs='*')
	args = parser.parse_args(argv)

	violations: list[str] = []
	for path in iter_python_files(args.paths):
		if not path.is_file():
			continue

		for line_number, line in enumerate(path.read_text().splitlines(), start=1):
			match = LEADING_WS.match(line)
			if not match:
				continue

			whitespace = match.group('ws')
			if whitespace.startswith(' ') or ' \t' in whitespace:
				violations.append(f'{path}:{line_number}: leading indentation must use tabs')

	if violations:
		print('\n'.join(violations), file=sys.stderr)
		return 1

	return 0


if __name__ == '__main__':
	raise SystemExit(main())
