from __future__ import annotations

import json
from pathlib import Path

import pytest

from mctutil.shared.cloudpaths import (
	default_queue_root,
	local_layer_path,
	normalize_cloudpath,
	normalize_layer_path,
	select_layer_encoding,
)
from mctutil.shared.sharded_tree import (
	read_sharded_scales,
	sharded_scale_complete,
	sharded_tree_complete,
)


def make_tree(root: Path) -> Path:
	root.mkdir()
	(root / "info").write_text(
		json.dumps(
			{
				"scales": [
					{
						"key": "mip0",
						"sharding": {"@type": "sharded"},
					},
					{
						"key": "mip1",
						"sharding": {"@type": "sharded"},
					},
				]
			}
		),
		encoding="utf-8",
	)
	for key in ("mip0", "mip1"):
		scale = root / key
		scale.mkdir()
		(scale / "0.shard").write_bytes(key.encode())
	return root


def test_cloudpath_helpers_preserve_layer_and_cloudfiles_contracts(tmp_path):
	local = tmp_path / "layer"

	assert normalize_layer_path(local) == local.resolve().as_uri()
	assert normalize_layer_path("precomputed://s3://bucket/layer") == (
		"precomputed://s3://bucket/layer"
	)
	assert normalize_cloudpath("precomputed://s3://bucket/layer") == (
		"s3://bucket/layer"
	)
	assert normalize_cloudpath(f"precomputed://{local}") == (
		local.resolve().as_uri()
	)
	assert local_layer_path(f"precomputed://{local.resolve().as_uri()}") == (
		local.resolve()
	)
	assert local_layer_path("s3://bucket/layer") is None
	assert default_queue_root(local) == local.resolve() / ".mctutil-queues"
	with pytest.raises(ValueError, match="--queue is required"):
		default_queue_root("s3://bucket/layer")


@pytest.mark.parametrize(
	("requested", "layer_type", "source", "expected"),
	(
		("auto", "image", "jpeg", "raw"),
		("auto", "segmentation", "raw", "compressed_segmentation"),
		("auto", "segmentation", "compresso", "compresso"),
		("auto", "unknown", "raw", "raw"),
		("jpeg", "image", "raw", "jpeg"),
	),
)
def test_encoding_auto_select_is_shared(requested, layer_type, source, expected):
	assert select_layer_encoding(requested, layer_type, source) == expected


def test_sharded_tree_helpers_share_one_completion_walk(tmp_path):
	root = make_tree(tmp_path / "tree")

	assert read_sharded_scales(root) == [
		(0, "mip0", root / "mip0"),
		(1, "mip1", root / "mip1"),
	]
	assert sharded_scale_complete(root, 0) is True
	assert sharded_scale_complete(root.resolve().as_uri(), 1) is True
	assert sharded_tree_complete(root) is True

	(root / "mip1" / "0.shard").unlink()
	assert sharded_scale_complete(root, 1) is False
	assert sharded_tree_complete(root) is False


def test_sharded_scale_completion_ignores_malformed_sibling_scale(tmp_path):
	root = tmp_path / "tree"
	scale = root / "mip0"
	scale.mkdir(parents=True)
	(scale / "0.shard").write_bytes(b"mip0")
	info = {
		"scales": [
			{"key": "mip0", "sharding": {"@type": "sharded"}},
			None,
		]
	}

	assert sharded_scale_complete(root, 0, info) is True


def test_ng_commands_do_not_import_sibling_command_helpers():
	shard_source = Path("mctutil/ng/shard.py").read_text(encoding="utf-8")
	assert "from mctutil.ng.downsample_pyramid import" not in shard_source
	for path in Path("mctutil/ng").glob("*.py"):
		source = path.read_text(encoding="utf-8")
		assert "def local_layer_path(" not in source
		assert "def normalize_layer_path(" not in source
