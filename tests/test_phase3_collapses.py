from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from click.testing import CliRunner


def test_collect_idle_nodes_merges_the_old_nodeinfo_scripts(load_module, tmp_path):
	module = load_module("mctutil/mem/from_file.py")
	node_file = tmp_path / "nodes.txt"
	node_file.write_text("NODE PART STATE\nnode-a sas idle\nnode-b sas alloc\nnode-c gpu mix\n")

	assert module.collect_idle_nodes(node_file) == ["node-a", "node-c"]


def test_expand_node_range_replaces_hardcoded_from_list(load_module):
	module = load_module("mctutil/mem/from_range.py")
	assert module.expand_node_range("node", 3, 5) == ["node3", "node4", "node5"]


def test_parse_sample_list_replaces_embedded_meta_paths(load_module, tmp_path):
	module = load_module("mctutil/parse/meta_shift.py")
	list_file = tmp_path / "samples.txt"
	list_file.write_text("/tmp/one.yaml\n\n/tmp/two.yaml\n")

	assert module.parse_sample_list(list_file) == [Path("/tmp/one.yaml"), Path("/tmp/two.yaml")]


def test_sinogram_full_mode_requires_flat_dir(load_module, tmp_path):
	module = load_module("mctutil/transform/sinogram.py")
	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	output_dir.mkdir()

	result = CliRunner().invoke(module.sino_convert, ["--mode", "full", "-i", str(input_dir), "-o", str(output_dir)])

	assert result.exit_code != 0
	assert "--flat-dir is required when --mode=full" in result.output


def test_transpose_naive_mode_replaces_f_transpose(load_module, tmp_path):
	module = load_module("mctutil/transform/transpose.py")
	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	tifffile.imwrite(input_dir / "a.tif", np.array([[1, 2], [3, 4]], dtype=np.uint8))
	tifffile.imwrite(input_dir / "b.tif", np.array([[5, 6], [7, 8]], dtype=np.uint8))

	result = CliRunner().invoke(
		module.transpose_stack,
		["--mode", "naive", "-p", str(input_dir), "-n", "tp", str(output_dir)],
	)

	assert result.exit_code == 0, result.output
	assert tifffile.imread(output_dir / "tp_0000.tif").tolist() == [[1, 5], [3, 7]]
	assert tifffile.imread(output_dir / "tp_0001.tif").tolist() == [[2, 6], [4, 8]]
