from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import tifffile
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SerialPool:
	def __init__(self, *_args, **_kwargs):
		pass

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		return False

	def starmap(self, func, iterable):
		return [func(*args) for args in iterable]


def test_trim_crops_xy_and_z_ranges(load_module, tmp_path, monkeypatch):
	module = load_module("transform/trim.py")
	monkeypatch.setattr(module, "Pool", SerialPool)
	monkeypatch.setattr(module.log, "start", lambda: None)
	monkeypatch.setattr(module.log, "log", lambda *_args, **_kwargs: None)

	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()

	for index in range(3):
		data = np.arange(16, dtype=np.uint8).reshape(4, 4) + index * 20
		tifffile.imwrite(input_dir / f"slice_{index}.tif", data)

	result = CliRunner().invoke(
		module.trim,
		[
			"-d", str(input_dir),
			"-o", str(output_dir),
			"-v", "1,1",
			"-h", "1,1",
			"-z", "1,0",
		],
	)

	assert result.exit_code == 0, result.output
	written = sorted(output_dir.glob("*.tif"))
	assert [path.name for path in written] == ["slice_1.tif", "slice_2.tif"]
	assert tifffile.imread(written[0]).tolist() == [[25, 26], [29, 30]]
	assert tifffile.imread(written[1]).tolist() == [[45, 46], [49, 50]]


def test_normalize_handles_partial_final_batch(load_module, tmp_path, monkeypatch):
	module = load_module("transform/normalize.py")
	monkeypatch.setattr(module, "Pool", SerialPool)
	monkeypatch.setattr(module.log, "start", lambda: None)
	monkeypatch.setattr(module.log, "log", lambda *_args, **_kwargs: None)

	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	output_dir.mkdir()
	data = np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32)
	tifffile.imwrite(input_dir / "slice_0.tif", data)

	result = CliRunner().invoke(
		module.norm,
		[
			"-n", "0,100",
			"-d", str(input_dir),
			"-o", str(output_dir),
			"-p", "2",
		],
	)

	assert result.exit_code == 0, result.output
	written = tifffile.imread(output_dir / "slice_0.tif")
	assert written.dtype == np.float32
	assert np.allclose(written, np.array([[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]], dtype=np.float32))


def test_sinogram_preproc_normalizes_small_real_input(load_module, tmp_path, monkeypatch):
	module = load_module("transform/sinogram.py")
	monkeypatch.setattr(module, "Pool", SerialPool)
	monkeypatch.setattr(module.log, "log", lambda *_args, **_kwargs: None)
	monkeypatch.setattr(module, "estimate_sigma", lambda *_args, **_kwargs: 0.0)
	monkeypatch.setattr(module, "denoise_nl_means", lambda data, **_kwargs: data)

	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	output_dir.mkdir()
	tifffile.imwrite(input_dir / "sino_0.tif", np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))

	result = CliRunner().invoke(
		module.sino_convert,
		[
			"--mode", "preproc",
			"-i", str(input_dir),
			"-o", str(output_dir),
			"-p", "1",
			"--min-val", "2",
			"--max-val", "8",
		],
	)

	assert result.exit_code == 0, result.output
	written = tifffile.imread(output_dir / "sino_0.tif")
	assert np.allclose(written, np.array([[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]], dtype=np.float32))
