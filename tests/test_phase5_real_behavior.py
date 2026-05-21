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


def fake_distribute_read(target_mem, _pj, window, _int_window, image_order, **_kwargs):
	with target_mem as target:
		for projection_index, image_path in image_order:
			image = tifffile.imread(image_path)
			target[projection_index, :len(window), :] = image[list(window), :]


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


def test_trim_dry_run_writes_nothing(load_module, tmp_path, monkeypatch):
	module = load_module("transform/trim.py")
	monkeypatch.setattr(module, "Pool", SerialPool)
	monkeypatch.setattr(module.log, "start", lambda: None)
	monkeypatch.setattr(module.log, "log", lambda *_args, **_kwargs: None)

	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	tifffile.imwrite(input_dir / "slice_0.tif", np.arange(16, dtype=np.uint8).reshape(4, 4))

	result = CliRunner().invoke(
		module.trim,
		["-d", str(input_dir), "-o", str(output_dir), "--dry-run"],
	)

	assert result.exit_code == 0, result.output
	assert not output_dir.exists()


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


def test_normalize_dry_run_writes_nothing(load_module, tmp_path, monkeypatch):
	module = load_module("transform/normalize.py")
	monkeypatch.setattr(module, "Pool", SerialPool)
	monkeypatch.setattr(module.log, "start", lambda: None)
	monkeypatch.setattr(module.log, "log", lambda *_args, **_kwargs: None)

	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	tifffile.imwrite(input_dir / "slice_0.tif", np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32))

	result = CliRunner().invoke(
		module.norm,
		[
			"-n", "0,100",
			"-d", str(input_dir),
			"-o", str(output_dir),
			"-p", "1",
			"--dry-run",
		],
	)

	assert result.exit_code == 0, result.output
	assert not output_dir.exists()


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


def test_sinogram_full_mode_normalizes_small_real_input(load_module, tmp_path, monkeypatch):
	module = load_module("transform/sinogram.py")
	monkeypatch.setattr(module, "Pool", SerialPool)
	monkeypatch.setattr(module.log, "log", lambda *_args, **_kwargs: None)
	monkeypatch.setattr(module, "distribute_read", fake_distribute_read)

	input_dir = tmp_path / "input"
	flat_dir = tmp_path / "flats"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	flat_dir.mkdir()
	output_dir.mkdir()

	tifffile.imwrite(input_dir / "proj_0.tif", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
	for name, data in {
		"pregain_median.tiff": np.ones((2, 2), dtype=np.float32),
		"postgain_median.tiff": np.ones((2, 2), dtype=np.float32),
		"predark_median.tiff": np.zeros((2, 2), dtype=np.float32),
		"postdark_median.tiff": np.zeros((2, 2), dtype=np.float32),
	}.items():
		tifffile.imwrite(flat_dir / name, data)

	result = CliRunner().invoke(
		module.sino_convert,
		[
			"--mode", "full",
			"-i", str(input_dir),
			"-o", str(output_dir),
			"-f", str(flat_dir),
			"-p", "1",
		],
	)

	assert result.exit_code == 0, result.output
	assert tifffile.imread(output_dir / "sino_00000.tiff").tolist() == [[1.0, 2.0]]
	assert tifffile.imread(output_dir / "sino_00001.tiff").tolist() == [[3.0, 4.0]]


def test_sinogram_dry_run_writes_nothing(load_module, tmp_path, monkeypatch):
	module = load_module("transform/sinogram.py")
	monkeypatch.setattr(module, "Pool", SerialPool)
	monkeypatch.setattr(module.log, "log", lambda *_args, **_kwargs: None)
	monkeypatch.setattr(module, "estimate_sigma", lambda *_args, **_kwargs: 0.0)
	monkeypatch.setattr(module, "denoise_nl_means", lambda data, **_kwargs: data)

	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
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
			"--dry-run",
		],
	)

	assert result.exit_code == 0, result.output
	assert not output_dir.exists()
