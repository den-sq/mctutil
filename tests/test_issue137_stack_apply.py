from pathlib import Path

from click.testing import CliRunner
import numpy as np
import tifffile

from mctutil.shared.stack_apply import (
	apply_array,
	apply_image_stack,
	batched,
	plan_stack_map,
	tiff_paths,
)


def test_stack_listing_mapping_and_partial_batches(tmp_path):
	input_dir = tmp_path / "input"
	input_dir.mkdir()
	for name in ("slice_10.tiff", "ignore.txt", "slice_02.tif"):
		(input_dir / name).write_bytes(b"data")

	paths = tiff_paths(input_dir)
	assert [path.name for path in paths] == ["slice_02.tif", "slice_10.tiff"]
	items = plan_stack_map(reversed(paths), tmp_path / "output")
	assert [(item.source.name, item.target.name) for item in items] == [
		("slice_10.tiff", "slice_10.tiff"),
		("slice_02.tif", "slice_02.tif"),
	]
	assert batched(paths, 3) == (paths,)


def test_apply_array_is_pure_and_stack_dry_run_is_non_mutating(tmp_path):
	image = np.arange(4, dtype=np.uint8).reshape(2, 2)
	result = apply_array(image, lambda value: np.flip(value, axis=1))
	assert result.tolist() == [[1, 0], [3, 2]]
	assert image.tolist() == [[0, 1], [2, 3]]

	source = tmp_path / "source.tif"
	target = tmp_path / "output" / "target.tif"
	tifffile.imwrite(source, image)
	apply_image_stack(
		plan_stack_map((source,), target.parent, target_names=(target.name,)),
		lambda _image: (_ for _ in ()).throw(AssertionError("operation ran")),
		dry_run=True,
	)
	assert not target.parent.exists()


def test_convert_dry_run_uses_shared_stack_plan(
	load_module,
	tmp_path,
	verbose_logging,
):
	module = load_module("mctutil/transform/convert.py")
	input_dir = tmp_path / "input"
	input_dir.mkdir()
	tifffile.imwrite(input_dir / "slice.tif", np.ones((2, 2), dtype=np.uint8))
	output_dir = tmp_path / "output"

	result = CliRunner().invoke(
		module.convert,
		[
			"--output-type", "uint8",
			"--preserve-names",
			"--dry-run",
			str(input_dir),
			str(output_dir),
		],
	)

	assert result.exit_code == 0, result.output
	assert "Would write" in result.output
	assert not output_dir.exists()


def test_transform_commands_use_shared_stack_scaffolding_and_xyz():
	for name in ("flip.py", "reslice.py"):
		source = Path("mctutil/transform", name).read_text(encoding="utf-8")
		assert "def tiff_paths(" not in source

	reslice = Path("mctutil/transform/reslice.py").read_text(encoding="utf-8")
	assert "type=XYZ" in reslice
	assert "class Coordinates" not in reslice
	assert "from mctutil.shared.stack_apply import" in reslice

	for name in ("convert.py", "normalize.py", "flip.py"):
		source = Path("mctutil/transform", name).read_text(encoding="utf-8")
		assert "from mctutil.shared.stack_apply import" in source
