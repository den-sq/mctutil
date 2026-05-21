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
	CommandCase("hpc_work/timecheck.py", "timecheck"),
	CommandCase("mem/clean.py", "memclean", ("clean", "--help")),
	CommandCase("mem/clean.py", "memclean", ("mark", "--help")),
	CommandCase("ng/change_color.py", "change_color"),
	CommandCase("ng/layer_copy.py", "layer_copy"),
	CommandCase("ng/layer_extract.py", "layer_extract"),
	CommandCase("ng/layer_tag.py", "layer_tag"),
	CommandCase("ng/layer_urlshift.py", "layer_urlshift"),
	CommandCase("ng/point_add.py", "point_add"),
	CommandCase("ng/point_merge.py", "point_merge"),
	CommandCase("ng/point_shift.py", "point_shift"),
	CommandCase("ng/point_sort.py", "point_sort"),
	CommandCase("ng/position_copy.py", "position_copy"),
	CommandCase("ng/shift_angle.py", "shift_angle"),
	CommandCase("parsing/pull_config.py", "get_conf"),
	CommandCase("parsing/scanlog_fetch.py", "scanlog_fetch"),
	CommandCase("transform/channelize.py", "channelize"),
	CommandCase("transform/convert.py", "convert"),
	CommandCase("transform/df_write_tiff.py", "df_write_tiff"),
	CommandCase("transform/dicom_conv.py", "dicom_conv"),
	CommandCase("transform/downsample.py", "downsample"),
	CommandCase("transform/f_transpose.py", "f_transpose"),
	CommandCase("transform/find_bounds.py", "find_bounds"),
	CommandCase("transform/fix_name.py", "fix_names"),
	CommandCase("transform/gz_strip.py", "stripgz"),
	CommandCase("transform/hdf_convert.py", "hdf_convert"),
	CommandCase("transform/mesh.py", "mesh"),
	CommandCase("transform/mesh_ig.py", "mesh_ig"),
	CommandCase("transform/ng.py", "neuroglance"),
	CommandCase("transform/normalize.py", "norm"),
	CommandCase("transform/quickgunzip.py", "gunzip"),
	CommandCase("transform/simple_noise.py", "simple_denoise"),
	CommandCase("transform/sino_preproc.py", "sino_convert"),
	CommandCase("transform/sinogram.py", "sino_convert"),
	CommandCase("transform/stitch.py", "stitch"),
	CommandCase("transform/transform.py", "norm"),
	CommandCase("transform/transpose.py", "transpose_stack"),
	CommandCase("transform/trim.py", "trim"),
	CommandCase("transform/uncompress.py", "uncompress"),
	CommandCase("transform/upload.py", "upload"),
	CommandCase("transport/cv_import.py", "cloudvolume_fetch"),
	CommandCase("transport/s3upload.py", "s3upload"),
]


@pytest.mark.parametrize("case", CASES)
def test_click_help_smoke(load_module, case: CommandCase):
	module = load_module(case.module_path)
	command = getattr(module, case.attr_name)
	result = CliRunner().invoke(command, list(case.argv))
	assert result.exit_code == 0, result.output


def test_upload_short_options_are_unique(load_module):
	module = load_module("transform/upload.py")
	short_opts = [
		option
		for param in module.upload.params
		for option in getattr(param, "opts", [])
		if option.startswith("-") and not option.startswith("--")
	]
	assert len(short_opts) == len(set(short_opts))
