from click.testing import CliRunner
import numpy as np
import tifffile

from mctutil.cli import main


def test_squeeze_directory_preserves_pixels_and_dtype(tmp_path):
	images = {
		"trailing.tif": np.arange(500 * 500, dtype=np.uint16).reshape(500, 500, 1),
		"multiple.TIFF": np.arange(30, dtype=np.float32).reshape(1, 5, 1, 6, 1),
		"already_2d.tiff": np.arange(30, dtype=np.uint8).reshape(5, 6),
	}
	for name, image in images.items():
		tifffile.imwrite(tmp_path / name, image, photometric="minisblack")
		assert tifffile.imread(tmp_path / name).shape == image.shape
	ignored = tmp_path / "ignore.txt"
	ignored.write_text("untouched")
	nested = tmp_path / "nested"
	nested.mkdir()
	nested_tiff = nested / "slice.tif"
	tifffile.imwrite(nested_tiff, images["trailing.tif"], photometric="minisblack")
	nested_bytes = nested_tiff.read_bytes()

	result = CliRunner().invoke(main, ["transform", "squeeze", str(tmp_path)])

	assert result.exit_code == 0, result.output
	for name, image in images.items():
		actual = tifffile.imread(tmp_path / name)
		np.testing.assert_array_equal(actual, np.squeeze(image), strict=True)
	assert ignored.read_text() == "untouched"
	assert nested_tiff.read_bytes() == nested_bytes


def test_squeeze_dry_run_does_not_rewrite_tiffs(tmp_path):
	path = tmp_path / "slice.tif"
	tifffile.imwrite(path, np.ones((5, 6, 1), dtype=np.uint16), photometric="minisblack")
	original = path.read_bytes()

	result = CliRunner().invoke(main, ["-v", "transform", "squeeze", "--dry-run", str(tmp_path)])

	assert result.exit_code == 0, result.output
	assert "Would write" in result.output
	assert path.read_bytes() == original
