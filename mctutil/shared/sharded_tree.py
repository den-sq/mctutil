"""Shared structural inspection for local sharded Neuroglancer trees."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from mctutil.shared.cloudpaths import local_layer_path


@dataclass(frozen=True)
class ShardedScale:
	"""One declared scale and its local sharding state."""

	mip: int
	metadata: dict
	key: str | None
	path: Path | None
	sharded: bool
	has_shards: bool

	@property
	def complete(self) -> bool:
		return bool(self.sharded and self.key and self.has_shards)


def load_info(root: Path) -> dict:
	"""Load local precomputed metadata with stable validation errors."""
	info_path = root / "info"
	if not info_path.is_file():
		raise ValueError(f"sharded tree is missing info: {info_path}")
	try:
		return json.loads(info_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		raise ValueError(f"invalid precomputed info: {info_path}") from exc


def inspect_sharded_tree(
	root: Path,
	info: dict | None = None,
) -> tuple[ShardedScale, ...]:
	"""Inspect every declared scale through one sharding-completion walk."""
	root = Path(root)
	info = load_info(root) if info is None else info
	return tuple(
		_describe_scale(root, mip, scale)
		for mip, scale in enumerate(info.get("scales", []))
	)


def _describe_scale(root: Path, mip: int, scale: dict) -> ShardedScale:
	if not isinstance(scale, dict):
		return ShardedScale(mip, {}, None, None, False, False)
	key_value = scale.get("key")
	key = str(key_value) if key_value else None
	path = root / key if key is not None else None
	sharded = bool(scale.get("sharding"))
	has_shards = bool(
		path is not None
		and path.is_dir()
		and any(path.glob("*.shard"))
	)
	return ShardedScale(mip, scale, key, path, sharded, has_shards)


def sharded_scale_complete(
	root: str | Path,
	mip: int,
	info: dict | None = None,
) -> bool:
	"""Conservatively recognize one declared scale with local shard data."""
	try:
		local_root = local_layer_path(root)
		if local_root is None:
			return False
		scales = inspect_sharded_tree(local_root, info)
		return scales[mip].complete
	except (IndexError, KeyError, TypeError, ValueError):
		return False


def sharded_tree_complete(root: str | Path, info: dict | None = None) -> bool:
	"""Match publish's existing rule: every sharded scale must have data."""
	try:
		local_root = local_layer_path(root)
		if local_root is None:
			return False
		sharded = tuple(
			scale
			for scale in inspect_sharded_tree(local_root, info)
			if scale.sharded
		)
	except (KeyError, TypeError, ValueError):
		return False
	return bool(sharded and all(scale.complete for scale in sharded))


def read_sharded_scales(
	root: Path,
	include_mip0: bool = True,
) -> list[tuple[int, str, Path]]:
	"""Return uploadable declared scales with legacy validation behavior."""
	root = Path(root)
	info = load_info(root)
	declared = info.get("scales", [])
	if not declared:
		raise ValueError("sharded tree has no scales")

	selected = []
	for scale in inspect_sharded_tree(root, info):
		if scale.mip == 0 and not include_mip0:
			continue
		if scale.key is None:
			raise ValueError(f"scale {scale.mip} has no key")
		if not scale.sharded:
			raise ValueError(f"scale {scale.mip} is not sharded")
		if scale.path is None or not scale.path.is_dir():
			raise ValueError(
				f"scale {scale.mip} directory is missing: {scale.path}"
			)
		selected.append((scale.mip, scale.key, scale.path))
	return selected
