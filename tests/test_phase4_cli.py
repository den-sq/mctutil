from __future__ import annotations

from click.testing import CliRunner

from mctutil.cli import main


def test_mctutil_root_help_lists_category_groups():
	result = CliRunner().invoke(main, ["--help"])
	assert result.exit_code == 0, result.output
	for name in ["transform", "sino", "als832", "flats", "ng", "mesh", "transport", "mem", "parse", "hpc"]:
		assert name in result.output


def test_transform_group_help_lists_unified_commands():
	result = CliRunner().invoke(main, ["transform", "--help"])
	assert result.exit_code == 0, result.output
	for name in ["normalize", "trim", "transpose", "flip", "reslice", "gunzip", "strip-gz-suffix"]:
		assert name in result.output


def test_mem_group_help_lists_collapsed_commands():
	result = CliRunner().invoke(main, ["mem", "--help"])
	assert result.exit_code == 0, result.output
	for name in ["clean", "mark", "from-file", "from-range"]:
		assert name in result.output


def test_mem_leaf_help_lists_config_and_node_selection_options():
	result = CliRunner().invoke(main, ["mem", "clean", "--help"])
	assert result.exit_code == 0, result.output
	assert "--config" in result.output
	assert "--shared-base" in result.output
	assert "--execute" in result.output
	assert "--dry-run" in result.output
	assert "--apply" not in result.output

	result = CliRunner().invoke(main, ["mem", "mark", "--help"])
	assert result.exit_code == 0, result.output
	assert "--config" in result.output
	assert "--node-list" in result.output
	assert "--node-file" in result.output
	assert "--node-call" in result.output
	assert "--job-preamble" in result.output
	assert "--sbatch-output" in result.output
	assert "--sbatch-error" in result.output
	assert "--execute" in result.output
	assert "--dry-run" in result.output
	assert "--apply" not in result.output


def test_mesh_group_exposes_only_unified_build_command():
	result = CliRunner().invoke(main, ["mesh", "--help"])
	assert result.exit_code == 0, result.output
	assert "build" in result.output
	assert "build-igneous" not in result.output


def test_issue72_groups_help_lists_promoted_commands():
	result = CliRunner().invoke(main, ["als832", "--help"])
	assert result.exit_code == 0, result.output
	for name in ["extract-projections", "extract-refs", "h5-tree"]:
		assert name in result.output

	result = CliRunner().invoke(main, ["flats", "--help"])
	assert result.exit_code == 0, result.output
	for name in ["beam-tracking", "series-digest", "medianize"]:
		assert name in result.output


def test_unified_leaf_help_works(stubbed_modules):
	result = CliRunner().invoke(main, ["sino", "convert", "--help"])
	assert result.exit_code == 0, result.output
	assert "--mode" in result.output

	result = CliRunner().invoke(main, ["ng", "layer-tag", "--help"])
	assert result.exit_code == 0, result.output
	assert "--segment_radius" in result.output

	result = CliRunner().invoke(main, ["als832", "extract-projections", "--help"])
	assert result.exit_code == 0, result.output
	assert "--dry-run" in result.output

	result = CliRunner().invoke(main, ["flats", "series-digest", "--help"])
	assert result.exit_code == 0, result.output
	assert "--dry-run" in result.output
