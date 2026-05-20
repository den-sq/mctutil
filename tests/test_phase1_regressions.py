from __future__ import annotations

from pathlib import Path
import types

import click
import numpy as np
import pytest
import tifffile
from click.testing import CliRunner


class _StopAfterDimensions(RuntimeError):
	pass


class _StopSharedNP:
	def __init__(self, *_args, **_kwargs):
		pass

	def __enter__(self):
		raise _StopAfterDimensions()

	def __exit__(self, exc_type, exc, tb):
		return False


class _FakeMem:
	def __init__(self, array):
		self.array = array

	def __enter__(self):
		return self.array

	def __exit__(self, exc_type, exc, tb):
		return False


def test_scanlog_fetch_creates_logs_dir_and_copies_valid_scanlogs(load_module, monkeypatch, tmp_path):
	module = load_module("parsing/scanlog_fetch.py")
	root_dir = tmp_path / "scan"
	scan_dir = root_dir / "sample"
	scan_dir.mkdir(parents=True)
	(scan_dir / "scanlog.txt").write_text("x" * 94)
	monkeypatch.chdir(tmp_path)

	result = CliRunner().invoke(module.scanlog_fetch, [str(root_dir)])

	assert result.exit_code == 0, result.output
	copied = tmp_path / "logs" / "sample_scanlog.txt"
	assert copied.exists()
	assert copied.read_text() == "x" * 94


def test_normalize_processes_option_is_an_int_param(load_module):
	module = load_module("transform/normalize.py")
	process_option = next(param for param in module.norm.params if param.name == "processes")
	assert isinstance(process_option.type, click.types.IntParamType)


def test_df_write_tiff_uses_source_title_for_output_name(load_module, monkeypatch, tmp_path):
	module = load_module("transform/df_write_tiff.py")
	source_path = tmp_path / "source.ors"
	source_path.write_text("placeholder")
	output_dir = tmp_path / "out"
	output_dir.mkdir()
	written = {}

	class DummyList:
		def loadFromFileFiltered(self, *_args, **_kwargs):
			return None

	class DummySource:
		def getTitle(self):
			return "roi-title"

		def getAsNDArray(self, *_args):
			return np.arange(4, dtype=np.uint8).reshape(2, 2)

	source = DummySource()
	monkeypatch.setattr(module, "List", DummyList)
	monkeypatch.setattr(module, "Managed", types.SimpleNamespace(
		getAllObjectsOfClassAndTitle=staticmethod(lambda *_args, **_kwargs: [source]),
	))
	monkeypatch.setattr(module, "orsObj", lambda *_args, **_kwargs: source)
	monkeypatch.setattr(module.tf, "imwrite", lambda path, data: written.update(path=Path(path), data=data.copy()))

	module.df_write_tiff.callback(source_path, None, None, "dummy-id", output_dir)

	assert written["path"].name == "roi-title.tif"
	assert written["path"].parent == output_dir
	assert written["data"].shape == (2, 2)


def test_hdf_convert_finds_raw_subdir_with_posix_glob(load_module, tmp_path):
	module = load_module("transform/hdf_convert.py")
	folder = tmp_path / "proj"
	raw_dir = folder / "raw"
	raw_dir.mkdir(parents=True)
	(raw_dir / "slice.hdf").write_text("stub")

	paths = module.get_image_paths(folder)

	assert paths == [raw_dir / "slice.hdf"]


def test_s3upload_does_not_require_client_close(load_module, monkeypatch, tmp_path):
	module = load_module("transport/s3upload.py")
	source_dir = tmp_path / "input"
	source_dir.mkdir()
	client_calls = []

	class DummyClient:
		def put_object(self, **kwargs):
			client_calls.append(kwargs)

	class DummySession:
		def client(self, *_args, **_kwargs):
			return DummyClient()

	monkeypatch.setattr(module, "session", DummySession())
	monkeypatch.setattr(module, "upload_folder_to_s3_parallel", lambda *_args, **_kwargs: None)

	module.s3upload.callback(Path("prefix"), "bucket", 4, False, source_dir, Path("target"))

	assert client_calls == [{"Bucket": "bucket", "Key": "prefix/target/"}]


def test_layer_tag_retries_generated_intensity_until_unique(load_module, monkeypatch, tmp_path):
	module = load_module("ng/layer_tag.py")
	source = {
		"layers": [
			{"name": "existing", "type": "annotation", "annotations": []},
			{"name": "fresh", "type": "annotation", "annotations": []},
		],
	}
	src = tmp_path / "input.json"
	dst = tmp_path / "output.json"
	src.write_text(__import__("json").dumps(source))
	monkeypatch.setattr(module.np, "nditer", lambda *_args, **_kwargs: iter([5, 7]))

	result = CliRunner().invoke(
		module.layer_tag,
		[
			"-j", str(src),
			"-r", str(dst),
			"existing:5",
			"fresh",
		],
	)

	assert result.exit_code == 0, result.output
	output = __import__("json").loads(dst.read_text())
	layer_names = {layer["name"] for layer in output["layers"]}
	assert "existing ID5r30" in layer_names
	assert "fresh ID7r30" in layer_names


def test_image_bounds_returns_min_and_max(load_module):
	module = load_module("transform/sino_preproc.py")
	bounds = module.image_bounds(_FakeMem(np.array([[3, 8], [1, 5]], dtype=np.float32)))
	assert np.array_equal(bounds, np.array([1, 8], dtype=np.float32))


def test_multitrim_applies_vertical_to_axis_zero_and_horizontal_to_axis_one(load_module, monkeypatch, tmp_path):
	module = load_module("transform/multitrim.py")
	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	output_dir.mkdir()
	tifffile.imwrite(input_dir / "slice.tif", np.arange(16, dtype=np.uint8).reshape(4, 4))
	captured = {}

	monkeypatch.setattr(module, "SharedNP", _StopSharedNP)
	monkeypatch.setattr(module.log, "start", lambda: None)
	monkeypatch.setattr(module.log, "log", lambda stage, message, *args: captured.setdefault(stage, message))
	monkeypatch.setattr(module.psutil, "cpu_count", lambda: 1)

	with pytest.raises(_StopAfterDimensions):
		module.trim.callback(str(input_dir), str(output_dir), 0.25, 0.5, module.cli.NUMPYTYPE.convert("uint8", None, None))

	assert captured["Dimensions"] == "(4, 4)-(slice(1, 3, None), slice(2, 2, None))"


def test_layer_urlshift_updates_sources_in_place(load_module, tmp_path):
	module = load_module("ng/layer_urlshift.py")
	source = {
		"layers": [
			{"name": "img", "type": "image", "source": "bucket|rest"},
			{"name": "seg", "type": "segmentation", "source": {"url": "bucket-two|rest"}},
		],
	}
	src = tmp_path / "input.json"
	dst = tmp_path / "output.json"
	src.write_text(__import__("json").dumps(source))

	result = CliRunner().invoke(module.layer_urlshift, ["-j", str(src), "-r", str(dst)])

	assert result.exit_code == 0, result.output
	output = __import__("json").loads(dst.read_text())
	assert output["layers"][0]["source"] == "precomputed://bucket"
	assert output["layers"][1]["source"]["url"] == "precomputed://bucket-two"
