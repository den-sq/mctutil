from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import numpy as np
import tifffile

from mctutil.transform import stitch_reconstructions as module


def write_stack(
	directory: Path,
	entries: list[tuple[str, np.ndarray]],
	**write_options,
) -> None:
	directory.mkdir()
	for name, image in entries:
		tifffile.imwrite(directory / name, image, **write_options)


def scalar_image(value, dtype=np.uint16, shape=(2, 3)) -> np.ndarray:
	return np.full(shape, value, dtype=dtype)


def invoke_stitch(
	stack_a: Path,
	stack_b: Path,
	output: Path,
	*options: str,
):
	return CliRunner().invoke(
		module.stitch_reconstructions,
		[
			str(stack_a),
			str(stack_b),
			str(output),
			"--a-stop",
			"2",
			"--b-start",
			"1",
			*options,
		],
	)


def output_values(output: Path) -> list[int]:
	return [
		int(tifffile.imread(path)[0, 0])
		for path in sorted(output.glob("slice_*.tif"))
	]


def test_exact_boundaries_natural_order_and_parallel_names(tmp_path):
	stack_a = tmp_path / "a"
	stack_b = tmp_path / "b"
	write_stack(
		stack_a,
		[
			("slice_10.tif", scalar_image(20)),
			("slice_1.tif", scalar_image(11)),
			("slice_2.tif", scalar_image(12)),
		],
	)
	write_stack(
		stack_b,
		[
			("slice_10.tif", scalar_image(110)),
			("slice_1.tif", scalar_image(101)),
			("slice_2.tif", scalar_image(102)),
		],
	)
	serial = tmp_path / "serial"
	parallel = tmp_path / "parallel"

	serial_result = invoke_stitch(stack_a, stack_b, serial, "--workers", "1")
	parallel_result = invoke_stitch(stack_a, stack_b, parallel, "--workers", "3")

	assert serial_result.exit_code == 0, serial_result.output
	assert parallel_result.exit_code == 0, parallel_result.output
	expected_names = [f"slice_{index:05d}.tif" for index in range(4)]
	assert [path.name for path in sorted(serial.glob("*.tif"))] == expected_names
	assert [path.name for path in sorted(parallel.glob("*.tif"))] == expected_names
	assert output_values(serial) == [11, 12, 102, 110]
	assert output_values(parallel) == [11, 12, 102, 110]

	manifest = json.loads(
		(serial / module.MANIFEST_NAME).read_text(encoding="utf-8")
	)
	assert manifest["inputs"][0]["retained"] == {"start": 0, "stop": 2}
	assert manifest["inputs"][1]["retained"] == {"start": 1, "stop": 3}
	assert manifest["output"]["count"] == 4
	assert manifest["output"]["dtype"] == "uint16"


def test_dry_run_validates_and_writes_nothing(tmp_path):
	stack_a = tmp_path / "a"
	stack_b = tmp_path / "b"
	write_stack(
		stack_a,
		[("a_0.tif", scalar_image(1)), ("a_1.tif", scalar_image(2))],
	)
	write_stack(
		stack_b,
		[("b_0.tif", scalar_image(3)), ("b_1.tif", scalar_image(4))],
	)
	output = tmp_path / "output"

	result = invoke_stitch(stack_a, stack_b, output, "--dry-run")

	assert result.exit_code == 0, result.output
	assert "Retain A: [0:2) (2 slices)" in result.output
	assert "Retain B: [1:2) (1 slices)" in result.output
	assert "Output stack: 3 slices; shape=(2, 3); axes=YX; dtype=uint16" in result.output
	assert not output.exists()
	assert not list(tmp_path.glob(".output.stitching-*"))


def test_incompatible_shape_fails_before_output_is_created(tmp_path):
	stack_a = tmp_path / "a"
	stack_b = tmp_path / "b"
	write_stack(
		stack_a,
		[("a_0.tif", scalar_image(1)), ("a_1.tif", scalar_image(2))],
	)
	write_stack(
		stack_b,
		[
			("b_0.tif", scalar_image(3)),
			("b_1.tif", scalar_image(4, shape=(3, 3))),
		],
	)
	output = tmp_path / "output"

	result = invoke_stitch(stack_a, stack_b, output)

	assert result.exit_code != 0
	assert "incompatible slice layout" in result.output
	assert not output.exists()


def test_mixed_dtype_requires_explicit_conversion(tmp_path):
	stack_a = tmp_path / "a"
	stack_b = tmp_path / "b"
	write_stack(
		stack_a,
		[
			("a_0.tif", scalar_image(1, dtype=np.uint8)),
			("a_1.tif", scalar_image(2, dtype=np.uint8)),
		],
	)
	write_stack(
		stack_b,
		[
			("b_0.tif", scalar_image(3, dtype=np.uint16)),
			("b_1.tif", scalar_image(4, dtype=np.uint16)),
		],
	)
	output = tmp_path / "output"

	result = invoke_stitch(stack_a, stack_b, output)

	assert result.exit_code != 0
	assert "mixed dtypes (uint8, uint16)" in result.output
	assert not output.exists()


def test_unsupported_channel_layout_fails_before_output_is_created(tmp_path):
	stack_a = tmp_path / "a"
	stack_b = tmp_path / "b"
	write_stack(
		stack_a,
		[("a_0.tif", scalar_image(1)), ("a_1.tif", scalar_image(2))],
	)
	write_stack(
		stack_b,
		[
			("b_0.tif", scalar_image(3)),
			("b_1.tif", np.zeros((2, 3, 2), dtype=np.uint16)),
		],
	)
	output = tmp_path / "output"

	result = invoke_stitch(stack_a, stack_b, output)

	assert result.exit_code != 0
	assert "unsupported slice layout" in result.output
	assert not output.exists()


def test_integer_conversion_clips_then_casts_without_scaling(tmp_path):
	stack_a = tmp_path / "a"
	stack_b = tmp_path / "b"
	image = np.array([[-2.0, 1.9], [255.9, 300.0]], dtype=np.float32)
	write_stack(stack_a, [("a_0.tif", image), ("a_1.tif", image)])
	write_stack(stack_b, [("b_0.tif", image), ("b_1.tif", image)])
	output = tmp_path / "output"

	result = invoke_stitch(
		stack_a,
		stack_b,
		output,
		"--dtype",
		"uint8",
		"--workers",
		"1",
	)

	assert result.exit_code == 0, result.output
	written = tifffile.imread(output / "slice_00000.tif")
	assert written.dtype == np.uint8
	assert written.tolist() == [[0, 1], [255, 255]]
	manifest = json.loads(
		(output / module.MANIFEST_NAME).read_text(encoding="utf-8")
	)
	assert "no rescaling" in manifest["conversion"]


def test_float_to_uint64_saturates_without_wraparound():
	image = np.array([[-1.0, 1.0e30]], dtype=np.float64)

	converted = module._convert_image(
		image,
		np.dtype("uint64"),
		Path("source.tif"),
	)

	assert converted.tolist() == [[0, np.iinfo(np.uint64).max]]


def test_ambiguous_natural_order_is_rejected(tmp_path):
	stack_a = tmp_path / "a"
	stack_b = tmp_path / "b"
	write_stack(
		stack_a,
		[
			("slice_1.tif", scalar_image(1)),
			("slice_01.tiff", scalar_image(2)),
		],
	)
	write_stack(
		stack_b,
		[("b_0.tif", scalar_image(3)), ("b_1.tif", scalar_image(4))],
	)
	output = tmp_path / "output"

	result = invoke_stitch(stack_a, stack_b, output)

	assert result.exit_code != 0
	assert "ambiguous natural ordering" in result.output
	assert not output.exists()


def test_existing_output_requires_overwrite_and_is_replaced_transactionally(
	tmp_path,
):
	stack_a = tmp_path / "a"
	stack_b = tmp_path / "b"
	write_stack(
		stack_a,
		[("a_0.tif", scalar_image(1)), ("a_1.tif", scalar_image(2))],
	)
	write_stack(
		stack_b,
		[("b_0.tif", scalar_image(3)), ("b_1.tif", scalar_image(4))],
	)
	output = tmp_path / "output"
	output.mkdir()
	sentinel = output / "old.txt"
	sentinel.write_text("old", encoding="utf-8")

	refused = invoke_stitch(stack_a, stack_b, output)

	assert refused.exit_code != 0
	assert "--overwrite" in refused.output
	assert sentinel.read_text(encoding="utf-8") == "old"

	replaced = invoke_stitch(stack_a, stack_b, output, "--overwrite")

	assert replaced.exit_code == 0, replaced.output
	assert not sentinel.exists()
	assert output_values(output) == [1, 2, 4]
	assert not list(tmp_path.glob(".output.backup-*"))


def test_partial_failure_leaves_existing_output_untouched(
	tmp_path,
	monkeypatch,
):
	stack_a = tmp_path / "a"
	stack_b = tmp_path / "b"
	write_stack(
		stack_a,
		[("a_0.tif", scalar_image(1)), ("a_1.tif", scalar_image(2))],
	)
	write_stack(
		stack_b,
		[("b_0.tif", scalar_image(3)), ("b_1.tif", scalar_image(4))],
	)
	output = tmp_path / "output"
	output.mkdir()
	sentinel = output / "old.txt"
	sentinel.write_text("old", encoding="utf-8")
	original = module.write_planned_slice

	def fail_second(spec, staging_dir, output_dtype):
		if spec.output_index == 1:
			raise RuntimeError("synthetic write failure")
		return original(spec, staging_dir, output_dtype)

	monkeypatch.setattr(module, "write_planned_slice", fail_second)

	result = invoke_stitch(
		stack_a,
		stack_b,
		output,
		"--overwrite",
		"--workers",
		"1",
	)

	assert result.exit_code != 0
	assert "synthetic write failure" in result.output
	assert sentinel.read_text(encoding="utf-8") == "old"
	assert list(output.iterdir()) == [sentinel]
	assert not list(tmp_path.glob(".output.stitching-*"))
