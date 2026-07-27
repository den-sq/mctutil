from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner
import pytest

from mctutil.als832 import h5_tree as h5_tree_module


h5py = pytest.importorskip("h5py")


@pytest.fixture()
def metadata_h5(tmp_path):
	path = tmp_path / "metadata.h5"
	with h5py.File(path, "w") as handle:
		detector = handle.create_group("measurement/instrument/detector")
		pixel_size = detector.create_dataset("actual_pixel_size", data=[0.065, 0.065, 0.065])
		pixel_size.attrs["units"] = "mm"
		setup = detector.create_group("setup")
		setup.create_dataset("detector_x", data=[1.0, 2.0])
		handle.create_dataset("exchange/data", data=[[1, 2], [3, 4]])
	return path


def test_exact_path_reads_only_selected_dataset(metadata_h5):
	result = CliRunner().invoke(
		h5_tree_module.h5_tree,
		[str(metadata_h5), "--path", "measurement/instrument/detector/actual_pixel_size"],
	)

	assert result.exit_code == 0, result.output
	assert "/measurement/instrument/detector/actual_pixel_size" in result.output
	assert "[0.065, 0.065, 0.065]" in result.output
	assert "@units = 'mm'" in result.output
	assert "detector_x" not in result.output
	assert "exchange/data" not in result.output


def test_group_path_recursively_reads_all_descendants(metadata_h5):
	result = CliRunner().invoke(
		h5_tree_module.h5_tree,
		[str(metadata_h5), "--path", "/measurement/instrument/detector/"],
	)

	assert result.exit_code == 0, result.output
	assert "/measurement/instrument/detector/" in result.output
	assert "/measurement/instrument/detector/actual_pixel_size" in result.output
	assert "/measurement/instrument/detector/setup/detector_x" in result.output
	assert "exchange/data" not in result.output


def test_selected_dataset_safety_limit_can_be_disabled(metadata_h5):
	limited = CliRunner().invoke(
		h5_tree_module.h5_tree,
		[
			str(metadata_h5),
			"--path",
			"measurement/instrument/detector/actual_pixel_size",
			"--max-values",
			"2",
		],
	)
	unlimited = CliRunner().invoke(
		h5_tree_module.h5_tree,
		[
			str(metadata_h5),
			"--path",
			"measurement/instrument/detector/actual_pixel_size",
			"--max-values",
			"0",
		],
	)

	assert limited.exit_code == 0, limited.output
	assert "exceeds --max-values 2" in limited.output
	assert unlimited.exit_code == 0, unlimited.output
	assert "[0.065, 0.065, 0.065]" in unlimited.output


def test_missing_path_is_an_error(metadata_h5):
	result = CliRunner().invoke(
		h5_tree_module.h5_tree,
		[str(metadata_h5), "--path", "measurement/instrument/setup"],
	)

	assert result.exit_code != 0
	assert "HDF5 path not found: /measurement/instrument/setup" in result.output


def test_source_file_is_opened_read_only(metadata_h5, monkeypatch):
	modes = []

	def read_only_file(path, mode):
		modes.append(mode)
		return h5py.File(path, mode)

	h5py_proxy = SimpleNamespace(
		File=read_only_file,
		Dataset=h5py.Dataset,
		Group=h5py.Group,
	)
	monkeypatch.setattr(h5_tree_module, "_require_h5py", lambda: h5py_proxy)

	result = CliRunner().invoke(
		h5_tree_module.h5_tree,
		[str(metadata_h5), "--path", "measurement/instrument/detector"],
	)

	assert result.exit_code == 0, result.output
	assert modes == ["r"]
