from pathlib import Path

from click.testing import CliRunner
import numpy as np
import tifffile

from mctutil.cli import main
from mctutil.transform import pipeline as pipeline_module
from mctutil.transform.convert import converted_image
from mctutil.transform.normalize import normalization_bounds, normalized_image
from mctutil.transform.ops import (
	circular_mask,
	maximum_intensity_projection,
	spatial_bin,
)
from mctutil.transform.trim import cropped_image


def test_restored_volume_ops_cover_all_mip_axes_mask_and_bin_power():
	volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
	for axis in range(3):
		actual = maximum_intensity_projection(volume, 2, axis)
		expected = np.stack(
			[
				np.max(
					np.take(volume, (index, index + 1), axis=axis),
					axis=axis,
				)
				for index in range(volume.shape[axis] - 1)
			],
			axis=axis,
		)
		assert np.array_equal(actual, expected)

	masked = circular_mask(np.ones((2, 5, 5), dtype=np.float32), 0.8, value=-1)
	assert np.all(masked[:, 2, 2] == 1)
	assert np.all(masked[:, 0, 0] == -1)

	to_bin = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
	binned = spatial_bin(to_bin, 2)
	expected_bin = to_bin.reshape(1, 2, 4, 2, 4).mean(axis=(2, 4))
	assert binned.shape == (1, 2, 2)
	assert np.array_equal(binned, expected_bin)
	integer_bin = spatial_bin(np.arange(16, dtype=np.uint8).reshape(1, 4, 4), 1)
	assert integer_bin.tolist() == [[[3, 5], [11, 13]]]


def test_fused_array_result_matches_the_same_shared_cores_in_sequence():
	volume = np.arange(4 * 6 * 6, dtype=np.float32).reshape(4, 6, 6)
	floor, ceiling = normalization_bounds(volume, 5, 95)
	expected = normalized_image(volume, floor, ceiling)
	expected = cropped_image(expected, (1, 1), (1, 1))
	expected = maximum_intensity_projection(expected, 2, axis=0)
	expected = circular_mask(expected, 1.0, axis=0, value=0)
	expected = np.stack(
		tuple(converted_image(image, np.uint8) for image in expected),
		axis=0,
	)
	expected = spatial_bin(expected, 1)

	actual = pipeline_module.apply_transform_pipeline(
		volume,
		normalize_range=(5, 95),
		vertical_trim=(1, 1),
		horizontal_trim=(1, 1),
		mip_width=2,
		mip_axis=0,
		circ_mask_ratio=1.0,
		out_dtype=np.uint8,
		bin_power=1,
	)

	assert actual.dtype == np.uint8
	assert actual.shape == (3, 2, 2)
	assert np.array_equal(actual, expected)


def test_pipeline_cli_reads_and_writes_each_plane_once(
	tmp_path,
	monkeypatch,
):
	monkeypatch.setattr(pipeline_module.log, "start", lambda: None)
	monkeypatch.setattr(pipeline_module.log, "write", lambda *_args, **_kwargs: None)
	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	volume = np.stack(
		[
			np.arange(16, dtype=np.float32).reshape(4, 4) + index * 3
			for index in range(3)
		]
	)
	for index, image in enumerate(volume):
		tifffile.imwrite(input_dir / f"slice_{index}.tif", image)

	read_paths = []
	real_imread = pipeline_module.tf.imread

	def counting_imread(path):
		read_paths.append(Path(path))
		return real_imread(path)

	monkeypatch.setattr(pipeline_module.tf, "imread", counting_imread)
	result = CliRunner().invoke(
		pipeline_module.pipeline,
		[
			"--data-dir", str(input_dir),
			"--output-dir", str(output_dir),
			"--normalize-over", "0,100",
			"--mips", "2",
			"--mips-axis", "z",
			"--circ-mask-ratio", "1",
			"--out-dtype", "uint8",
			"--bin-power", "1",
			"--processes", "1",
		],
	)

	assert result.exit_code == 0, result.output
	assert sorted(path.name for path in read_paths) == [
		"slice_0.tif",
		"slice_1.tif",
		"slice_2.tif",
	]
	written = sorted(output_dir.glob("*.tif"))
	assert [path.name for path in written] == ["slice_1.tif", "slice_2.tif"]
	expected = pipeline_module.apply_transform_pipeline(
		volume,
		normalize_range=(0, 100),
		mip_width=2,
		mip_axis=0,
		circ_mask_ratio=1,
		out_dtype=np.uint8,
		bin_power=1,
	)
	assert np.array_equal(
		np.stack([tifffile.imread(path) for path in written]),
		expected,
	)


def test_pipeline_dry_run_does_not_read_or_create_output(
	tmp_path,
	monkeypatch,
	verbose_logging,
):
	monkeypatch.setattr(pipeline_module.log, "start", lambda: None)
	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	tifffile.imwrite(input_dir / "slice_0.tif", np.ones((2, 2), dtype=np.uint8))
	monkeypatch.setattr(
		pipeline_module.tf,
		"imread",
		lambda *_args, **_kwargs: (_ for _ in ()).throw(
			AssertionError("dry run read image data")
		),
	)

	result = CliRunner().invoke(
		pipeline_module.pipeline,
		[
			"--data-dir", str(input_dir),
			"--output-dir", str(output_dir),
			"--dry-run",
		],
	)

	assert result.exit_code == 0, result.output
	assert "Would write" in result.output
	assert not output_dir.exists()


def test_pipeline_is_live_and_the_orphaned_command_is_removed():
	result = CliRunner().invoke(main, ["transform", "--help"])
	assert result.exit_code == 0, result.output
	assert "pipeline" in result.output
	assert not Path("mctutil/transform/transform.py").exists()
