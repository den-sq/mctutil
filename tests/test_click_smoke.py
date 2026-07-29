from __future__ import annotations

from dataclasses import dataclass

import pytest
from click.testing import CliRunner


@dataclass(frozen=True)
class CommandCase:
	module_path: str
	attr_name: str
	argv: tuple[str, ...] = ("--help",)


CASES = [
	CommandCase("mctutil/hpc/timecheck.py", "timecheck"),
	CommandCase("mctutil/mem/clean.py", "clean"),
	CommandCase("mctutil/mem/clean.py", "mark"),
	CommandCase("mctutil/mem/from_file.py", "from_file"),
	CommandCase("mctutil/mem/from_range.py", "from_range"),
	CommandCase("mctutil/als832/extract_projections.py", "extract_projections"),
	CommandCase("mctutil/als832/extract_refs.py", "extract_refs"),
	CommandCase("mctutil/als832/h5_tree.py", "h5_tree"),
	CommandCase("mctutil/flats/beam_tracking.py", "beam_tracking"),
	CommandCase("mctutil/flats/medianize.py", "medianize"),
	CommandCase("mctutil/flats/series_digest.py", "series_digest"),
	CommandCase("mctutil/ng/change_color.py", "change_color"),
	CommandCase("mctutil/ng/downsample_pyramid.py", "downsample_pyramid"),
	CommandCase("mctutil/ng/layer_copy.py", "layer_copy"),
	CommandCase("mctutil/ng/layer_extract.py", "layer_extract"),
	CommandCase("mctutil/ng/layer_tag.py", "layer_tag"),
	CommandCase("mctutil/ng/layer_urlshift.py", "layer_urlshift"),
	CommandCase("mctutil/ng/point_add.py", "point_add"),
	CommandCase("mctutil/ng/point_merge.py", "point_merge"),
	CommandCase("mctutil/ng/point_shift.py", "point_shift"),
	CommandCase("mctutil/ng/point_sort.py", "point_sort"),
	CommandCase("mctutil/ng/position_copy.py", "position_copy"),
	CommandCase("mctutil/ng/precompute.py", "precompute"),
	CommandCase("mctutil/ng/shift_angle.py", "shift_angle"),
	CommandCase("mctutil/parse/meta_shift.py", "meta_shift"),
	CommandCase("mctutil/parse/pull_config.py", "get_conf"),
	CommandCase("mctutil/parse/scanlog_fetch.py", "scanlog_fetch"),
	CommandCase("mctutil/transform/channelize.py", "channelize"),
	CommandCase("mctutil/transform/convert.py", "convert"),
	CommandCase("mctutil/transform/df_write_tiff.py", "df_write_tiff"),
	CommandCase("mctutil/transform/dicom_conv.py", "dicom_conv"),
	CommandCase("mctutil/transform/downsample.py", "downsample"),
	CommandCase("mctutil/transform/find_bounds.py", "find_bounds"),
	CommandCase("mctutil/transform/fix_name.py", "fix_names"),
	CommandCase("mctutil/transform/flip.py", "flip_stack"),
	CommandCase("mctutil/transform/gz_strip.py", "stripgz"),
	CommandCase("mctutil/transform/hdf_convert.py", "hdf_convert"),
	CommandCase("mctutil/transform/memmap_prep.py", "memmap_prep"),
	CommandCase("mctutil/mesh/build.py", "mesh"),
	CommandCase("mctutil/transform/ng.py", "neuroglance"),
	CommandCase("mctutil/transform/normalize.py", "norm"),
	CommandCase("mctutil/transform/quickgunzip.py", "gunzip"),
	CommandCase("mctutil/transform/reslice.py", "reslice"),
	CommandCase("mctutil/transform/simple_noise.py", "simple_denoise"),
	CommandCase("mctutil/transform/sinogram.py", "sino_convert"),
	CommandCase("mctutil/transform/stitch.py", "stitch"),
	CommandCase("mctutil/transform/transform.py", "norm"),
	CommandCase("mctutil/transform/transpose.py", "transpose_stack"),
	CommandCase("mctutil/transform/trim.py", "trim"),
	CommandCase("mctutil/transform/uncompress.py", "uncompress"),
	CommandCase("mctutil/transport/cv_import.py", "cloudvolume_fetch"),
	CommandCase("mctutil/transport/s3upload.py", "s3upload"),
]


@pytest.mark.parametrize("case", CASES)
def test_click_help_smoke(load_module, case: CommandCase):
	module = load_module(case.module_path)
	command = getattr(module, case.attr_name)
	result = CliRunner().invoke(command, list(case.argv))
	assert result.exit_code == 0, result.output


def test_issue76_modified_command_flags(load_module):
	cases = [
		("mctutil/transform/ng.py", "neuroglance", "--channel-count"),
		("mctutil/transform/normalize.py", "norm", "--hard-cut"),
		("mctutil/transform/normalize.py", "norm", "--relative-cut"),
		("mctutil/transport/cv_import.py", "cloudvolume_fetch", "--out-dtype"),
		("mctutil/transport/cv_import.py", "cloudvolume_fetch", "--transpose-axes"),
		("mctutil/transport/cv_import.py", "cloudvolume_fetch", "--original-axes"),
	]
	for module_path, attr_name, option in cases:
		module = load_module(module_path)
		result = CliRunner().invoke(getattr(module, attr_name), ["--help"])
		assert result.exit_code == 0, result.output
		assert option in result.output
