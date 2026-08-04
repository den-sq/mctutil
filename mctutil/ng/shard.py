"""Stage an unsharded precomputed pyramid as sharded Neuroglancer data."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import click

from mctutil.ng.downsample_pyramid import (
	default_queue_root,
	local_layer_path,
	normalize_layer_path,
)
from mctutil.ng.resource_planning import (
	log_resource_plan,
	parse_size,
	plan_resources,
	ResourcePlan,
)
from mctutil.shared.cli import XYZ
from mctutil.shared.igneous_output import (
	capture_igneous_call,
	igneous_output_command,
)
from mctutil.shared.log import log, LOG
from mctutil.shared.persistent_queue import (
	read_state,
	run_persistent_tasks,
	stable_fingerprint,
	write_state,
)

SHARD_COMPRESSION = "gzip"


def _require_dependencies():
	try:
		import igneous.task_creation as task_creation
		from cloudvolume import CloudVolume
	except ImportError as exc:
		raise RuntimeError(
			"ng shard requires igneous-pipeline and task-queue; "
			"install with pip install -e '.[ng,mesh]'"
		) from exc
	return CloudVolume, task_creation


def parse_mips(_context, _parameter, value: str | None) -> tuple[int, ...] | None:
	if value is None:
		return None
	try:
		mips = tuple(sorted({int(part.strip()) for part in value.split(",")}))
	except ValueError as exc:
		raise click.BadParameter("must be a comma-separated list of integers") from exc
	if not mips or any(mip < 0 for mip in mips):
		raise click.BadParameter("must contain non-negative mip levels")
	return mips


def inspect_source(source: str) -> tuple[str, str, dict]:
	CloudVolume, _task_creation = _require_dependencies()
	volume = CloudVolume(normalize_layer_path(source), parallel=False)
	info = volume.info
	scales = info.get("scales", [])
	if not scales:
		raise ValueError("source precomputed volume has no scales")
	return (
		info.get("type", "image"),
		scales[0].get("encoding", "raw"),
		info,
	)


def detect_mips(source: str, info: dict) -> tuple[int, ...]:
	"""Detect declared scales plus local scale-key/numeric directories."""
	scales = info.get("scales", [])
	detected = {
		int(str(scale["key"]))
		if scale.get("key") is not None and str(scale["key"]).isdecimal()
		else mip
		for mip, scale in enumerate(scales)
	}
	source_path = local_layer_path(source)
	if source_path is None or not source_path.is_dir():
		return tuple(sorted(detected))

	for child in source_path.iterdir():
		if not child.is_dir():
			continue
		if child.name.isdecimal():
			detected.add(int(child.name))
	return tuple(sorted(detected))


def destination_scale_complete(destination: str, mip: int) -> bool:
	"""Conservatively recognize a locally staged scale with shard data."""
	destination_path = local_layer_path(destination)
	if destination_path is None:
		return False
	info_path = destination_path / "info"
	if not info_path.is_file():
		return False
	try:
		info = json.loads(info_path.read_text(encoding="utf-8"))
		scale = info["scales"][mip]
	except (IndexError, KeyError, json.JSONDecodeError):
		return False
	if not scale.get("sharding") or not scale.get("key"):
		return False
	scale_path = destination_path / scale["key"]
	return scale_path.is_dir() and any(scale_path.glob("*.shard"))


def create_shard_tasks(
	source: str,
	destination: str,
	mip: int,
	chunk_size: tuple[int, int, int],
	encoding: str,
	memory: int,
):
	"""Create sharded transfer tasks across current and legacy Igneous APIs."""
	_CloudVolume, task_creation = _require_dependencies()
	source = normalize_layer_path(source)
	destination = normalize_layer_path(destination)
	sharded_factory = getattr(
		task_creation,
		"create_image_shard_transfer_tasks",
		None,
	)
	if sharded_factory is not None:
		return capture_igneous_call(
			sharded_factory,
			source,
			destination,
			mip=mip,
			chunk_size=chunk_size,
			fill_missing=True,
			encoding=encoding,
			memory_target=memory,
			compress=SHARD_COMPRESSION,
			# Preserve absolute mip positions when staging multiple or
			# non-contiguous scales through sequential factory calls.
			truncate_scales=False,
		)

	legacy_factory = task_creation.create_transfer_tasks
	if "sharded" not in inspect.signature(legacy_factory).parameters:
		raise RuntimeError(
			"installed igneous-pipeline has no sharded image transfer API"
		)
	return capture_igneous_call(
		legacy_factory,
		source,
		destination,
		mip=mip,
		chunk_size=chunk_size,
		fill_missing=True,
		encoding=encoding,
		memory_target=memory,
		compress=SHARD_COMPRESSION,
		sharded=True,
	)


def all_shard_tasks(
	source: str,
	destination: str,
	scale_plans,
	encoding: str,
):
	for mip, chunk, _count, capacity in scale_plans:
		yield from create_shard_tasks(
			source,
			destination,
			mip,
			chunk,
			encoding,
			capacity,
		)


def shard_configuration(
	source: str,
	destination: str,
	resources: ResourcePlan,
	encoding: str,
) -> dict:
	"""Return output-affecting settings; executor concurrency is excluded."""
	return {
		"stage": "shard",
		"source": normalize_layer_path(source),
		"destination": normalize_layer_path(destination),
		"mips": tuple(shard[0] for shard in resources.shards),
		"capacity_ceiling": resources.shard_ceiling,
		"scale_plans": [
			{
				"mip": mip,
				"chunk_size": chunk,
				"chunks_per_shard": count,
				"capacity": capacity,
			}
			for mip, chunk, count, capacity in resources.shards
		],
		"encoding": encoding,
		"compression": SHARD_COMPRESSION,
	}


def shard_volume(
	source: str,
	destination: str,
	queue_dir: Path,
	resources: ResourcePlan,
	encoding: str,
	parallel: int,
	lease_seconds: int,
	release_leases: bool = True,
) -> None:
	mips = tuple(shard[0] for shard in resources.shards)
	configuration = shard_configuration(source, destination, resources, encoding)
	fingerprint = stable_fingerprint(configuration)
	stage_root = queue_dir / "shard" / fingerprint
	state_path = stage_root / "pipeline.json"
	state = read_state(
		state_path,
		{
			"configuration": configuration,
			"completed_mips": [],
			"attempt": 0,
			"attempt_started": False,
			"complete": False,
		},
	)
	verified_completed = {
		mip
		for mip in state["completed_mips"]
		if destination_scale_complete(destination, mip)
	}
	if state["complete"] and all(
		destination_scale_complete(destination, mip)
		for mip in mips
	):
		log.write(
			"Shard",
			"Staging is already complete for this configuration.",
			log_level=LOG.STATUS,
		)
		return
	if state["complete"]:
		state["attempt"] = state.get("attempt", 0) + 1
		state["attempt_started"] = False
		state["completed_mips"] = sorted(verified_completed)
		state["complete"] = False
		write_state(state_path, state)

	pending = tuple(
		mip
		for mip in mips
		if mip not in verified_completed
	)
	if not pending:
		state["complete"] = True
		write_state(state_path, state)
		return

	task_fingerprint = stable_fingerprint(
		{
			"configuration": configuration,
			"attempt": state.get("attempt", 0),
			"pending": pending,
		}
	)
	expected_existing = state.get("attempt_started", False)
	if not expected_existing:
		state["attempt_started"] = True
		write_state(state_path, state)
	run_persistent_tasks(
		stage_root / f"tasks-{state.get('attempt', 0)}",
		task_fingerprint,
		lambda: all_shard_tasks(
			source,
			destination,
			tuple(
				shard
				for shard in resources.shards
				if shard[0] in pending
			),
			encoding,
		),
		parallel,
		lease_seconds,
		release_leases=release_leases,
		expected_existing=expected_existing,
		progress_label="Shard Tasks",
	)
	state["completed_mips"] = sorted(set(state["completed_mips"]) | set(pending))
	state["complete"] = True
	write_state(state_path, state)


@click.command("shard")
@click.argument("source")
@click.argument("destination")
@click.option(
	"--mips",
	callback=parse_mips,
	help="Comma-separated mip levels; default auto-detects source scales.",
)
@click.option(
	"--low-chunk",
	type=XYZ,
	default="96,96,96",
	show_default=True,
	help="Shard chunk size for mips 0-2.",
)
@click.option(
	"--mid-chunk",
	type=XYZ,
	default="64,64,64",
	show_default=True,
	help="Shard chunk size for mips 3-4.",
)
@click.option(
	"--high-chunk",
	type=XYZ,
	default="16,16,16",
	show_default=True,
	help="Shard chunk size for mips 5 and above.",
)
@click.option(
	"--shard-capacity",
	"capacity_override",
	metavar="SIZE",
	help=(
		"Override the automatic 2/4/8 GiB uncompressed capacity ceiling; "
		"accepts bytes or binary units such as 4GiB."
	),
)
@click.option(
	"--parallel",
	type=click.IntRange(min=1),
	default=8,
	show_default=True,
	help="Maximum workers; effective memory capacity and CPU count may lower it.",
)
@click.option("--include-mip0/--exclude-mip0", default=True, show_default=True)
@click.option(
	"--encoding",
	default="auto",
	show_default=True,
	help="Destination encoding; auto chooses from layer type/source metadata.",
)
@click.option("--queue", "queue_dir", type=click.Path(path_type=Path))
@click.option("--lease-seconds", type=click.IntRange(min=10), default=3600, show_default=True)
@click.option(
	"--release-leases/--preserve-leases",
	default=True,
	show_default=True,
	help="Release existing FileQueue leases when resuming; preserve for shared queues.",
)
@click.option("--execute/--dry-run", default=True, show_default=True)
@igneous_output_command
def shard(
	source: str,
	destination: str,
	mips: tuple[int, ...] | None,
	low_chunk: tuple[int, int, int],
	mid_chunk: tuple[int, int, int],
	high_chunk: tuple[int, int, int],
	capacity_override: str | int | None,
	parallel: int,
	include_mip0: bool,
	encoding: str,
	queue_dir: Path | None,
	lease_seconds: int,
	release_leases: bool,
	execute: bool,
) -> None:
	"""Stage a precomputed pyramid into sharded Neuroglancer scales."""
	try:
		layer_type, source_encoding, info = inspect_source(source)
		if encoding == "auto":
			encoding = "raw" if layer_type == "image" else source_encoding
			if layer_type == "segmentation" and encoding == "raw":
				encoding = "compressed_segmentation"
		mips = mips or detect_mips(source, info)
		if not include_mip0:
			mips = tuple(mip for mip in mips if mip != 0)
		if not mips:
			raise ValueError("no mip levels selected for sharded staging")
		queue_dir = queue_dir or default_queue_root(destination)
		if isinstance(capacity_override, str):
			capacity_override = parse_size(capacity_override)
		resources = plan_resources(
			info,
			mips,
			parallel,
			capacity_override=capacity_override,
			low_chunk=low_chunk,
			mid_chunk=mid_chunk,
			high_chunk=high_chunk,
		)
		log.write(
			"Shard",
			(
				f"Plan: {normalize_layer_path(source)} -> "
				f"{normalize_layer_path(destination)}; {layer_type}/{encoding}; "
				f"queue={queue_dir.resolve()}."
			),
			log_level=LOG.INFO,
		)
		log_resource_plan("Shard", resources, include_shards=True)
		if not execute:
			return
		shard_volume(
			source,
			destination,
			queue_dir,
			resources,
			encoding,
			resources.workers,
			lease_seconds,
			release_leases,
		)
		log.write("Shard", "Staging complete.", log_level=LOG.STATUS)
	except click.ClickException:
		raise
	except Exception as exc:
		raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
	shard()
