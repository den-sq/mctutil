from __future__ import annotations

import numpy as np
import tifffile
from click.testing import CliRunner


def test_memmap_prep_writes_and_verifies_contiguous_stack(load_module, tmp_path):
	module = load_module("mctutil/transform/memmap_prep.py")
	input_path = tmp_path / "input.tif"
	output_path = tmp_path / "output.tif"
	source = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
	tifffile.imwrite(input_path, source, photometric="minisblack")

	result = CliRunner().invoke(
		module.memmap_prep,
		[str(input_path), str(output_path), "--verify"],
	)

	assert result.exit_code == 0, result.output
	mapped = tifffile.memmap(output_path)
	assert mapped.shape == source.shape
	assert mapped.dtype == source.dtype
	assert np.array_equal(mapped, source)


def test_memmap_prep_normalizes_multiple_outputs(load_module, tmp_path):
	module = load_module("mctutil/transform/memmap_prep.py")
	input_path = tmp_path / "input.tif"
	output_dir = tmp_path / "outputs"
	source = np.array(
		[
			[[0.0, 1.0], [2.0, 3.0]],
			[[4.0, 5.0], [6.0, 10.0]],
		],
		dtype=np.float32,
	)
	tifffile.imwrite(input_path, source, photometric="minisblack")

	result = CliRunner().invoke(
		module.memmap_prep,
		[
			str(input_path),
			"--output-dir", str(output_dir),
			"--out-dtypes", "original,uint16",
			"--normalize", "manual",
			"--norm-min", "0",
			"--norm-max", "10",
			"--verify",
		],
	)

	assert result.exit_code == 0, result.output
	float_output = tifffile.memmap(output_dir / "input_MEMMAP_original.tif")
	uint_output = tifffile.memmap(output_dir / "input_MEMMAP_uint16.tif")
	assert float_output.dtype == np.float32
	assert np.allclose(float_output, source / 10.0)
	assert uint_output.dtype == np.uint16
	assert uint_output[0, 0, 0] == 0
	assert uint_output[-1, -1, -1] == np.iinfo(np.uint16).max


def test_memmap_prep_dry_run_writes_nothing(load_module, tmp_path):
	module = load_module("mctutil/transform/memmap_prep.py")
	input_path = tmp_path / "input.tif"
	output_dir = tmp_path / "outputs"
	tifffile.imwrite(
		input_path,
		np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4),
		photometric="minisblack",
	)

	result = CliRunner().invoke(
		module.memmap_prep,
		[
			str(input_path),
			"--output-dir", str(output_dir),
			"--out-dtypes", "original,uint16",
			"--dry-run",
		],
	)

	assert result.exit_code == 0, result.output
	assert "BigTIFF=" in result.output
	assert not output_dir.exists()


def test_memmap_prep_rejects_existing_output_without_overwrite(load_module, tmp_path):
	module = load_module("mctutil/transform/memmap_prep.py")
	input_path = tmp_path / "input.tif"
	output_path = tmp_path / "output.tif"
	tifffile.imwrite(
		input_path,
		np.arange(2 * 2 * 2, dtype=np.uint8).reshape(2, 2, 2),
		photometric="minisblack",
	)
	output_path.write_bytes(b"keep")

	result = CliRunner().invoke(module.memmap_prep, [str(input_path), str(output_path)])

	assert result.exit_code != 0
	assert "use --overwrite" in result.output
	assert output_path.read_bytes() == b"keep"
