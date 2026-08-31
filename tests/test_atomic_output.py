from pathlib import Path

import pytest

from mctutil.shared.atomic_output import atomic_text_output


def test_atomic_text_output_replaces_only_after_success(tmp_path: Path) -> None:
	output = tmp_path / "output.txt"
	with atomic_text_output(output) as handle:
		handle.write("complete")
		assert not output.exists()
	assert output.read_text(encoding="utf-8") == "complete"


def test_atomic_text_output_cleans_failed_temporary_file(tmp_path: Path) -> None:
	output = tmp_path / "output.txt"
	with pytest.raises(RuntimeError, match="stop"):
		with atomic_text_output(output) as handle:
			handle.write("partial")
			raise RuntimeError("stop")

	assert not output.exists()
	assert not list(tmp_path.glob(".output.txt.*"))
