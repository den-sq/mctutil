from pathlib import Path

from click.testing import CliRunner
import numpy as np
import tifffile


def make_input(root: Path) -> tuple[Path, np.ndarray]:
	input_dir = root / "input"
	input_dir.mkdir()
	data = np.array([[0, 10], [20, 30]], dtype=np.int16)
	tifffile.imwrite(input_dir / "slice.tif", data)
	return input_dir, data


def test_convert_exposes_legacy_dtype_only_behavior(load_module, tmp_path):
	module = load_module("mctutil/transform/convert.py")
	input_dir, data = make_input(tmp_path)
	output_dir = tmp_path / "converted"

	result = CliRunner().invoke(
		module.convert,
		[
			"--output-type", "uint8",
			"--preserve-names",
			"--uncompressed",
			"--workers", "1",
			str(input_dir),
			str(output_dir),
		],
	)

	assert result.exit_code == 0, result.output
	assert (output_dir / "slice.tif").is_file()
	assert tifffile.imread(output_dir / "slice.tif").tolist() == [
		[0, 85],
		[170, 255],
	]
	assert data.dtype == np.int16


def test_downsample_is_deprecated_alias_for_convert_core(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transform/convert.py")
	input_dir, _data = make_input(tmp_path)
	output_dir = tmp_path / "alias"
	calls = []
	monkeypatch.setattr(
		module,
		"convert_stack",
		lambda *args, **kwargs: calls.append((args, kwargs)),
	)

	result = CliRunner().invoke(
		module.downsample,
		[
			"--data-dir", str(input_dir),
			"--output-dir", str(output_dir),
		],
	)

	assert result.exit_code == 0, result.output
	combined_output = result.output
	try:
		combined_output += result.stderr
	except (AttributeError, ValueError):
		pass
	assert "downsample is deprecated" in combined_output
	assert calls[0][0][:2] == (input_dir, output_dir)
	assert np.dtype(calls[0][0][2]) == np.dtype("uint8")
	assert calls[0][1] == {
		"compression": False,
		"preserve_names": True,
		"workers": 1,
	}


def test_preserve_names_rejects_horizontal_sections(load_module, tmp_path):
	module = load_module("mctutil/transform/convert.py")
	input_dir, _data = make_input(tmp_path)

	result = CliRunner().invoke(
		module.convert,
		[
			"--output-type", "uint8",
			"--preserve-names",
			"--horizontal-sections", "2",
			str(input_dir),
			str(tmp_path / "output"),
		],
	)

	assert result.exit_code == 2
	assert "cannot be combined" in result.output


def test_downsample_module_contains_no_dtype_loop():
	source = Path("mctutil/transform/downsample.py").read_text(encoding="utf-8")
	routing = Path("mctutil/transform/__init__.py").read_text(encoding="utf-8")
	docs = Path("mctutil/transform/README.md").read_text(encoding="utf-8")

	assert "for path in" not in source
	assert "mctutil.transform.convert:downsample" in routing
	assert "performs no spatial downsampling" in docs
	assert "#132" in docs
