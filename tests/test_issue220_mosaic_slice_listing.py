from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import pytest
from click.testing import CliRunner

from mctutil.mosaic import common
from mctutil.shared import stack_apply

profile_module = importlib.import_module("mctutil.mosaic.profile")
window_module = importlib.import_module("mctutil.mosaic.window")
wiener_module = importlib.import_module("mctutil.mosaic.wiener")
stitch_module = importlib.import_module("mctutil.transform.stitch_reconstructions")

pytest.importorskip("cv2")
pytest.importorskip("scipy.ndimage")
pytest.importorskip("scipy.fft")
tifffile = pytest.importorskip("tifffile")

NAMES = {
	"numeric": ("00000.tif", "00001.tif", "00002.tif"),
	"prefixed": ("slice_000.tif", "slice_001.tif", "slice_002.tif"),
	"mixed": ("slice_0.TIF", "slice_2.tiff", "slice_10.tif"),
}


@pytest.fixture
def volume():
	images = 1 + np.arange(3 * 32 * 32, dtype=np.float32).reshape(3, 32, 32) / 1000
	images[:, 0, 0] = 0
	return images


def write_named_stack(path, names, volume):
	path.mkdir()
	for name, image in reversed(list(zip(names, volume))):
		tifffile.imwrite(path / name, image)
	return path


def identity_filter(source, shape, depth):
	# Unit XY gains isolate the input-row mapping while exercising the real FFT/blend path.
	bank = SimpleNamespace(nu=np.array([1.0]), xy=[np.ones((shape[0], shape[1] // 2 + 1), np.float32)])
	noise = np.zeros((depth, 2, 4, 4), np.float16)
	noise[:, 0] = 1
	options = {"noise_scale": 1.0, "depth": depth, "shape": shape, "fft_workers": 1}
	return wiener_module.Filter(source, noise, bank, 1.0, options)


def test_mosaic_consumers_share_the_canonical_listing_facilities():
	assert stitch_module.discover_slices is stack_apply.discover_slices
	assert stitch_module.natural_sort_key is stack_apply.natural_sort_key
	assert common.discover_slices is stack_apply.discover_slices
	assert profile_module.stack_paths is common.stack_paths
	assert window_module.stack_paths is common.stack_paths
	assert wiener_module.RowStack is common.RowStack


@pytest.mark.parametrize("naming", NAMES)
def test_row_stack_reads_the_discovered_paths_in_natural_order(tmp_path, monkeypatch, volume, naming):
	source = write_named_stack(tmp_path / naming, NAMES[naming], volume)
	(source / "notes.txt").write_text("not a slice")
	(source / "nested").mkdir()
	tifffile.imwrite(source / "nested" / "extra.tif", volume[0])
	listed = stack_apply.discover_slices(source)
	calls = []

	def discover(path):
		calls.append(Path(path))
		return listed

	monkeypatch.setattr(common, "discover_slices", discover)
	stack = common.RowStack(source)
	assert stack.paths is listed
	assert tuple(path.name for path in stack.paths) == NAMES[naming]
	assert stack.rows == len(listed) == len(volume)
	for row, expected in enumerate(volume):
		np.testing.assert_array_equal(stack.read(row), expected)
	assert stack.size == volume.shape[1]
	assert calls == [source]


@pytest.mark.parametrize("row", [-1, 3])
def test_row_stack_rejects_rows_outside_its_listing(tmp_path, volume, row):
	source = write_named_stack(tmp_path / "rows", NAMES["prefixed"], volume)
	with pytest.raises(click.ClickException, match="outside the stack of 3 slices"):
		common.RowStack(source).read(row)


@pytest.mark.parametrize("naming", ["prefixed", "mixed"])
def test_profile_window_and_conversion_match_numeric_names(tmp_path, volume, naming):
	runner = CliRunner()
	results = {}
	for key in ("numeric", naming):
		source = write_named_stack(tmp_path / key, NAMES[key], volume)
		profile_path, converted = tmp_path / f"{key}.npy", tmp_path / f"{key}_u16"
		profile = runner.invoke(profile_module.profile, [str(source), str(profile_path),
														"--r-lo", "2", "--r-ref", "8", "--step", "2"])
		window = runner.invoke(window_module.window, [str(source), "--step", "2", "--pixel-step", "1"])
		conversion = runner.invoke(window_module.to_u16_command,
									[str(source), str(converted), "--window", "0", "5"])
		for result in (profile, window, conversion):
			assert result.exit_code == 0, result.output
		assert {path.name for path in converted.iterdir()} == set(NAMES[key])
		results[key] = (np.load(profile_path), json.loads(window.output),
						[tifffile.imread(converted / name) for name in NAMES[key]])
	actual, expected = results[naming], results["numeric"]
	np.testing.assert_array_equal(actual[0], expected[0])
	assert actual[1] == expected[1]
	for actual_image, expected_image in zip(actual[2], expected[2]):
		np.testing.assert_array_equal(actual_image, expected_image)


@pytest.mark.parametrize("naming", NAMES)
def test_wiener_reads_ordinal_rows_and_keeps_numeric_output_names(tmp_path, volume, naming):
	source = write_named_stack(tmp_path / naming, NAMES[naming], volume)
	baseline = write_named_stack(tmp_path / "baseline", NAMES["numeric"], volume)
	flow = identity_filter(source, volume.shape[1:], len(volume))
	reference = identity_filter(baseline, volume.shape[1:], len(volume))
	for requested in (-1, 0, 1, 2, 3):
		row = min(max(requested, 0), len(volume) - 1)
		values, covered = flow.xy_blend(requested)
		expected, expected_coverage = reference.xy_blend(requested)
		np.testing.assert_array_equal(covered, volume[row] != 0)
		np.testing.assert_array_equal(covered, expected_coverage)
		np.testing.assert_array_equal(values, expected)
	target = tmp_path / "identity"
	target.mkdir()
	wiener_module.write_identity(source, target, (0, 5), (0, len(volume)))
	assert {path.name for path in target.iterdir()} == set(NAMES["numeric"])
	for row, name in enumerate(NAMES["numeric"]):
		expected = window_module.to_u16(volume[row], 0, 5, volume[row] != 0)
		np.testing.assert_array_equal(tifffile.imread(target / name), expected)


@pytest.mark.parametrize("consumer", ["row_stack", "profile", "window", "to_u16", "filter", "identity"])
def test_ambiguous_natural_keys_are_rejected_by_each_mosaic_input(tmp_path, volume, consumer):
	source = write_named_stack(tmp_path / "ambiguous", ("slice_2.tif", "slice_02.TIFF"), volume[:2])
	target = tmp_path / "out"
	if consumer in {"row_stack", "filter", "identity"}:
		with pytest.raises(click.ClickException, match="ambiguous natural ordering"):
			if consumer == "row_stack":
				common.RowStack(source)
			elif consumer == "filter":
				identity_filter(source, volume.shape[1:], 2).xy_blend(0)
			else:
				target.mkdir()
				wiener_module.write_identity(source, target, (0, 5), (0, 2))
	else:
		command, arguments = {
			"profile": (profile_module.profile, [str(source), str(target), "--r-lo", "2", "--r-ref", "8"]),
			"window": (window_module.window, [str(source)]),
			"to_u16": (window_module.to_u16_command, [str(source), str(target), "--window", "0", "5"]),
		}[consumer]
		result = CliRunner().invoke(command, arguments)
		assert result.exit_code != 0 and "ambiguous natural ordering" in result.output
