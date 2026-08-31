"""One atomic text-file output boundary shared by local writers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, TextIO


@contextmanager
def atomic_text_output(
	output: Path,
	*,
	force: bool = False,
	newline: str | None = None,
) -> Iterator[TextIO]:
	"""Yield a sibling temporary text file and atomically replace *output*."""

	if output.exists() and not force:
		raise FileExistsError(
			f"Output already exists: {output} (pass --force to replace it)"
		)
	output.parent.mkdir(parents=True, exist_ok=True)
	temporary_path: Path | None = None
	try:
		with NamedTemporaryFile(
			"w",
			encoding="utf-8",
			newline=newline,
			dir=output.parent,
			prefix=f".{output.name}.",
			delete=False,
		) as handle:
			temporary_path = Path(handle.name)
			yield handle
		os.replace(temporary_path, output)
	except BaseException:
		if temporary_path is not None:
			temporary_path.unlink(missing_ok=True)
		raise
