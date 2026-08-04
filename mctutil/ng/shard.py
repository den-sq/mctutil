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
	format_binary_size,
	parse_binary_size,
	plan_shard_capacities,
	plan_worker_limit,
	ShardCapacityPlan,
	WorkerPlan,
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


def parse_capacity(_context, _parameter, value: str | None) -> int | None:
	if value is None:
		return None
	try:
		return parse_binary_size(value)
	except ValueError as exc:
		raise click.BadParameter(str(exc)) from exc


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
	for scale in scale_plans:
		yield from create_shard_tasks(
			source,
			destination,
			scale.mip,
			scale.chunk_size,
			encoding,
			scale.capacity,
		)


def shard_configuration(
	source: str,
	destination: str,
	capacity_plan: ShardCapacityPlan,
	encoding: str,
) -> dict:
	"""Return output-affecting settings; executor concurrency is excluded."""
	return {
		"stage": "shard",
		"source": normalize_layer_path(source),
		"destination": normalize_layer_path(destination),
		"mips": tuple(scale.mip for scale in capacity_plan.scales),
		"capacity_ceiling": capacity_plan.capacity_ceiling,
		"scale_plans": [
			{
				"mip": scale.mip,
				"chunk_size": scale.chunk_size,
				"chunks_per_shard": scale.chunks_per_shard,
				"capacity": scale.capacity,
			}
			for scale in capacity_plan.scales
		],
		"encoding": encoding,
		"compression": SHARD_COMPRESSION,
	}


def shard_volume(
	source: str,
	destination: str,
	queue_dir: Path,
	capacity_plan: ShardCapacityPlan,
	encoding: str,
	parallel: int,
	lease_seconds: int,
	release_leases: bool = True,
) -> None:
	mips = tuple(scale.mip for scale in capacity_plan.scales)
	configuration = shard_configuration(source, destination, capacity_plan, encoding)
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
				scale
				for scale in capacity_plan.scales
				if scale.mip in pending
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


def describe_shard_plan(
	source: str,
	destination: str,
	layer_type: str,
	encoding: str,
	queue_dir: Path,
	capacity_plan: ShardCapacityPlan,
	worker_plan: WorkerPlan,
) -> None:
	for statement in (
		f"Source: {normalize_layer_path(source)}",
		f"Destination: {normalize_layer_path(destination)}",
		f"Layer type: {layer_type}; encoding: {encoding}",
		f"Queue root: {queue_dir.resolve()}",
		(
			f"Logical MIP 0: "
			f"{format_binary_size(capacity_plan.logical_mip0_bytes)}; "
			f"shard-capacity ceiling: "
			f"{format_binary_size(capacity_plan.capacity_ceiling)}"
		),
		(
			f"Available RAM: {format_binary_size(worker_plan.available_ram)}; "
			f"reserve: {format_binary_size(worker_plan.reserve)}; "
			f"headroom: 3x "
			f"{format_binary_size(worker_plan.capacity_budget)} capacity budget"
		),
		(
			f"Worker ceilings: requested={worker_plan.requested_limit}, "
			f"cpu={worker_plan.cpu_limit}, memory={worker_plan.memory_limit}; "
			f"selected={worker_plan.workers}"
		),
	):
		log.write("Shard", statement, log_level=LOG.INFO)
	if worker_plan.warning is not None:
		log.write("Shard", f"Warning: {worker_plan.warning}", log_level=LOG.WARN)
	for scale in capacity_plan.scales:
		status = (
			"complete"
			if destination_scale_complete(destination, scale.mip)
			else "pending"
		)
		log.write(
			"Shard",
			f"Mip {scale.mip}: chunk={scale.chunk_size}, "
			f"chunks/shard={scale.chunks_per_shard}, "
			f"capacity={format_binary_size(scale.capacity)}, "
			f"fill_missing=True, compression={SHARD_COMPRESSION}, "
			f"status={status}",
			log_level=LOG.INFO,
		)


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
	callback=parse_capacity,
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
	help="Maximum workers; available RAM and CPU count may lower it.",
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
	capacity_override: int | None,
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
		capacity_plan = plan_shard_capacities(
			info,
			mips,
			capacity_override,
			low_chunk,
			mid_chunk,
			high_chunk,
		)
		worker_plan = plan_worker_limit(
			parallel,
			capacity_plan.maximum_actual_capacity,
		)

		describe_shard_plan(
			source,
			destination,
			layer_type,
			encoding,
			queue_dir,
			capacity_plan,
			worker_plan,
		)
		if not execute:
			return
		shard_volume(
			source,
			destination,
			queue_dir,
			capacity_plan,
			encoding,
			worker_plan.workers,
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
