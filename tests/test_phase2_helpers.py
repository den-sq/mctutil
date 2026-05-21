from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mctutil.shared.cli import CROP_NUMBER, DelimitedRecord, crop_val  # noqa: E402
from mctutil.shared.np_convert import np_convert  # noqa: E402


@dataclass(frozen=True)
class _TaggedValue:
	name: str
	level: int
	radius: int


def test_np_convert_normalizes_signed_data_to_uint8():
	result = np_convert(np.uint8, np.array([-1.0, 0.0, 1.0], dtype=np.float32))
	assert np.array_equal(result, np.array([0, 127, 255], dtype=np.uint8))


def test_crop_number_parses_single_and_pair_values():
	assert CROP_NUMBER.convert("0.25", None, None) == [0.25, 0.25]
	assert CROP_NUMBER.convert("5,2", None, None) == [5, 2]
	assert crop_val([0.25, 0.5], 8) == slice(2, 4, None)


def test_delimited_record_supports_optional_fields():
	parser = DelimitedRecord(
		_TaggedValue,
		[str, lambda value: int(value) if len(value) > 0 else -1, lambda value: int(value) if len(value) > 0 else -1],
		defaults=(None, "", ""),
		min_fields=1,
	)

	assert parser.convert("axon", None, None) == _TaggedValue("axon", -1, -1)
	assert parser.convert("axon:9:", None, None) == _TaggedValue("axon", 9, -1)


def test_downsample_uses_shared_np_convert_scaling(load_module, tmp_path):
	module = load_module("mctutil/transform/downsample.py")
	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	output_dir.mkdir()
	tifffile.imwrite(input_dir / "slice.tif", np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32))

	module.downsample.callback(str(input_dir), str(output_dir), module.cli.NUMPYTYPE.convert("uint8", None, None))

	written = tifffile.imread(output_dir / "slice.tif")
	assert written.dtype == np.uint8
	assert np.array_equal(written, np.array([[0, 63], [127, 255]], dtype=np.uint8))
