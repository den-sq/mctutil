from pathlib import Path

from click.testing import CliRunner
import numpy as np
import tifffile

from mctutil.transform import pipeline as pipeline_module
from mctutil.transform.flip import flipped_image, flipped_volume
from mctutil.transform.simple_noise import (
	denoised_volume,
	flat_denoised_center,
	threshold_denoised_center,
)


class SerialPool:
	def __init__(self, *_args, **_kwargs):
		pass

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		return False

	def starmap(self, function, arguments):
		return [function(*args) for args in arguments]


def test_denoise_cores_are_symmetric_and_define_boundary_behavior():
	volume = np.array([10, 100, 10, 100, 10], dtype=np.int16).reshape(5, 1, 1)
	denoised = denoised_volume(volume, "threshold", 0.5)

	assert denoised.dtype == volume.dtype
	assert denoised[:, 0, 0].tolist() == [10, 10, 100, 10, 10]
	assert volume[:, 0, 0].tolist() == [10, 100, 10, 100, 10]
	assert denoised_volume(
		volume,
		"threshold",
		0.5,
		boundary="drop",
	)[:, 0, 0].tolist() == [10, 100, 10]

	negative_spike = np.array([100, 0, 100], dtype=np.int16).reshape(3, 1, 1)
	assert threshold_denoised_center(negative_spike, 0.5).item() == 100
	flat = np.array([5, 99, 5], dtype=np.uint16).reshape(3, 1, 1)
	assert flat_denoised_center(flat, 10).item() == 0


def test_flip_cores_cover_volume_and_standalone_axis_conventions():
	volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
	for axis in range(3):
		assert np.array_equal(flipped_volume(volume, axis), np.flip(volume, axis))
	assert np.array_equal(flipped_image(volume[0], 1), np.flip(volume[0], 0))
	assert np.array_equal(flipped_image(volume[0], 2), np.flip(volume[0], 1))


def test_standalone_denoise_leaf_uses_the_canonical_center_core(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transform/simple_noise.py")
	monkeypatch.setattr(module, "Pool", SerialPool)
	monkeypatch.setattr(module.log, "write", lambda *_args, **_kwargs: None)
	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	for index, value in enumerate((10, 100, 10)):
		tifffile.imwrite(
			input_dir / f"slice_{index}.tif",
			np.full((2, 2), value, dtype=np.uint16),
		)

	core_calls = []
	real_core = module.threshold_denoised_center

	def recording_core(images, threshold):
		core_calls.append((np.asarray(images).shape, threshold))
		return real_core(images, threshold)

	monkeypatch.setattr(module, "threshold_denoised_center", recording_core)
	result = CliRunner().invoke(
		module.simple_denoise,
		[
			"--threshold", "0.5",
			"--num-processes", "1",
			str(input_dir),
			str(output_dir),
		],
	)

	assert result.exit_code == 0, result.output
	assert core_calls == [((3, 2, 2), 0.5)]
	assert [path.name for path in output_dir.glob("*.tif")] == ["slice_1.tif"]
	assert np.all(tifffile.imread(output_dir / "slice_1.tif") == 10)


def test_fused_denoise_and_flip_match_canonical_core_sequence():
	volume = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
	volume[3, 1, 1] = 1000
	expected = volume[1:-1]
	expected = pipeline_module.maximum_intensity_projection(expected, 2, 0)
	expected = denoised_volume(expected, "threshold", 0.25)
	expected = flipped_volume(expected, 0)

	actual = pipeline_module.apply_transform_pipeline(
		volume,
		z_trim=(1, 1),
		mip_width=2,
		mip_axis=0,
		denoise_mode="threshold",
		denoise_threshold=0.25,
		flip_axis=0,
		out_dtype=None,
	)

	assert np.array_equal(actual, expected)


def test_z_flip_planning_tracks_trimmed_mip_sources_and_target_names(tmp_path):
	inputs = tuple(Path(f"slice_{index}.tif") for index in range(6))
	items = pipeline_module.plan_pipeline_outputs(
		inputs,
		tmp_path,
		z_trim=(1, 1),
		mip_width=2,
		mip_axis=0,
		flip_axis=0,
	)

	assert [item.source.name for item in items] == [
		"slice_4.tif",
		"slice_3.tif",
		"slice_2.tif",
	]
	assert [item.target.name for item in items] == [
		"slice_2.tif",
		"slice_3.tif",
		"slice_4.tif",
	]


def test_pipeline_cli_denoises_and_flips_with_one_read_per_input(
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
			for index in range(5)
		]
	)
	volume[2, 1, 1] = 100
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
			"--denoise-mode", "threshold",
			"--denoise-threshold", "0.25",
			"--flip-axis", "z",
			"--out-dtype", "uint16",
			"--processes", "1",
		],
	)

	assert result.exit_code == 0, result.output
	assert [path.name for path in read_paths] == [
		f"slice_{index}.tif" for index in range(5)
	]
	written = sorted(output_dir.glob("*.tif"))
	assert [path.name for path in written] == [
		f"slice_{index}.tif" for index in range(5)
	]
	expected = pipeline_module.apply_transform_pipeline(
		volume,
		denoise_mode="threshold",
		denoise_threshold=0.25,
		flip_axis=0,
		out_dtype=np.uint16,
	)
	assert np.array_equal(
		np.stack([tifffile.imread(path) for path in written]),
		expected,
	)


def test_pipeline_requires_complete_denoise_configuration(tmp_path):
	input_dir = tmp_path / "input"
	input_dir.mkdir()
	tifffile.imwrite(input_dir / "slice.tif", np.ones((2, 2), dtype=np.uint8))
	result = CliRunner().invoke(
		pipeline_module.pipeline,
		[
			"--data-dir", str(input_dir),
			"--output-dir", str(tmp_path / "output"),
			"--denoise-mode", "threshold",
			"--dry-run",
		],
	)

	assert result.exit_code == 2
	assert "denoise mode and threshold must be supplied together" in result.output
