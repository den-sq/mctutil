from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
import numpy as np
import tifffile

from mctutil.shared.tiff_stack_writer import (
	SliceNaming,
	write_tiff_stack,
)


def test_writer_dry_run_plans_names_without_reading_or_writing(tmp_path):
	output = tmp_path / "slices"
	paths = write_tiff_stack(
		lambda _index: (_ for _ in ()).throw(AssertionError("decoded")),
		2,
		output,
		mode="slices",
		indices=(3, 7),
		naming=SliceNaming("sample", digits=4, separator="_"),
		dry_run=True,
	)

	assert paths == (
		output / "sample_0003.tif",
		output / "sample_0007.tif",
	)
	assert not output.exists()


def test_writer_preserves_existing_tifffile_image_and_stack_bytes(tmp_path):
	frames = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
	direct_image = tmp_path / "direct-image.tif"
	shared_image = tmp_path / "shared-image.tif"
	tifffile.imwrite(direct_image, frames[0], compression=None, bigtiff=True)
	write_tiff_stack(
		lambda _index: frames[0],
		1,
		shared_image,
		mode="image",
		compression=None,
		bigtiff=True,
	)
	assert direct_image.read_bytes() == shared_image.read_bytes()

	direct_stack = tmp_path / "direct-stack.tif"
	shared_stack = tmp_path / "shared-stack.tif"
	with tifffile.TiffWriter(direct_stack, bigtiff=True) as writer:
		for frame in frames:
			writer.write(frame, compression=None, contiguous=False)
	write_tiff_stack(
		lambda index: frames[index],
		len(frames),
		shared_stack,
		mode="stack",
		compression=None,
		bigtiff=True,
		contiguous=False,
	)
	assert direct_stack.read_bytes() == shared_stack.read_bytes()


def test_raw_convert_and_stack_split_share_per_z_naming(load_module, tmp_path):
	raw_module = load_module("mctutil/transform/raw_convert.py")
	split_module = load_module("mctutil/transform/stack_split.py")
	data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
	raw_path = tmp_path / "sample.raw"
	raw_path.write_bytes(data.tobytes())
	stack = tmp_path / "sample.tif"

	result = CliRunner().invoke(
		raw_module.raw_convert,
		[
			str(raw_path),
			"--output", str(stack),
			"--width", "4",
			"--height", "3",
			"--depth", "2",
			"--dtype", "uint16",
		],
	)
	assert result.exit_code == 0, result.output

	output = tmp_path / "split"
	result = CliRunner().invoke(
		split_module.stack_split,
		[str(stack), "--output-dir", str(output), "--prefix", "plane"],
	)
	assert result.exit_code == 0, result.output
	assert sorted(path.name for path in output.iterdir()) == [
		"plane_z0000.tif",
		"plane_z0001.tif",
	]
	assert np.array_equal(tifffile.imread(output / "plane_z0001.tif"), data[1])


def test_dicom_conversion_preserves_dicom2tiff_decode_head(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transform/dicom_conv.py")
	source_dir = tmp_path / "patient"
	source_dir.mkdir()
	source = source_dir / "scan.dcm"
	source.write_bytes(b"dicom")
	calls = []
	monkeypatch.setattr(
		module.dicom2jpg,
		"dicom2tiff",
		lambda path, target: calls.append((path, target)),
		raising=False,
	)
	monkeypatch.setattr(
		module.dicom2jpg,
		"dicom2img",
		lambda _path: (_ for _ in ()).throw(AssertionError("display decode used")),
		raising=False,
	)
	output = tmp_path / "output"

	result = CliRunner().invoke(module.dicom_conv, [str(source), str(output)])

	assert result.exit_code == 0, result.output
	assert calls == [(source, output / "patient" / "scan.dcm")]


def test_dicom_dry_run_does_not_decode_or_create_output(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transform/dicom_conv.py")
	source = tmp_path / "scan.dcm"
	source.write_bytes(b"dicom")
	monkeypatch.setattr(
		module.dicom2jpg,
		"dicom2tiff",
		lambda *_args: (_ for _ in ()).throw(AssertionError("decoded")),
		raising=False,
	)
	output = tmp_path / "output"

	result = CliRunner().invoke(
		module.dicom_conv,
		[str(source), str(output), "--dry-run"],
	)

	assert result.exit_code == 0, result.output
	assert "Would write" in result.output
	assert not output.exists()


def test_decode_commands_do_not_own_tifffile_write_policy():
	paths = (
		Path("mctutil/als832/extract_projections.py"),
		Path("mctutil/als832/extract_refs.py"),
		Path("mctutil/transform/hdf_convert.py"),
		Path("mctutil/transform/h5_convert.py"),
		Path("mctutil/transform/raw_convert.py"),
		Path("mctutil/transform/stack_split.py"),
	)
	for path in paths:
		source = path.read_text(encoding="utf-8")
		assert "tifffile.imwrite" not in source
		assert "TiffWriter" not in source
		assert "dicom2tiff" not in source
		assert "write_tiff_stack" in source

	dicom_source = Path("mctutil/transform/dicom_conv.py").read_text(
		encoding="utf-8"
	)
	assert "dicom2tiff" in dicom_source
	assert "dicom2img" not in dicom_source
	assert "write_tiff_stack" not in dicom_source
