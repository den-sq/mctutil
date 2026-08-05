"""Build the two-pass volumetric MIP pyramid used by the sharded pipeline."""

from __future__ import annotations

from pathlib import Path

import click

from mctutil.ng.completeness import Mip0Completeness, check_mip0_completeness
from mctutil.ng.resource_planning import (
	log_resource_plan,
	parse_size,
	plan_resources,
)
from mctutil.shared.cli import XYZ
from mctutil.shared.cloudpaths import (
	default_queue_root,
	normalize_layer_path,
	select_layer_encoding,
)
from mctutil.shared.deps import require
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


def _require_dependencies():
	cloudvolume, task_creation = require(
		("cloudvolume", "igneous.task_creation"),
		("ng", "mesh"),
		purpose="ng downsample-pyramid requires igneous-pipeline and task-queue",
	)
	return cloudvolume.CloudVolume, task_creation


def inspect_volume(layer_path: str) -> dict:
	CloudVolume, _task_creation = _require_dependencies()
	volume = CloudVolume(normalize_layer_path(layer_path), parallel=False)
	info = volume.info
	if not info.get("scales", []):
		raise ValueError("precomputed volume has no scales")
	return info


def mip0_failure_message(completeness: Mip0Completeness) -> str:
	if completeness.verifiable:
		return f"source MIP 0 is incomplete: {completeness.summary()}"
	return (
		"source MIP 0 completeness could not be verified: "
		f"{completeness.summary()}"
	)


def create_downsample_tasks(
	layer_path: str,
	source_mip: int,
	chunk_size: tuple[int, int, int],
	encoding: str,
	memory: int,
):
	_CloudVolume, task_creation = _require_dependencies()
	return capture_igneous_call(
		task_creation.create_downsampling_tasks,
		normalize_layer_path(layer_path),
		mip=source_mip,
		fill_missing=True,
		chunk_size=chunk_size,
		encoding=encoding,
		compress="br",
		factor=(2, 2, 2),
		memory_target=memory,
	)


def run_pass(
	layer_path: str,
	queue_dir: Path,
	pass_name: str,
	source_mip: int,
	chunk_size: tuple[int, int, int],
	encoding: str,
	memory: int,
	parallel: int,
	lease_seconds: int,
	release_leases: bool = True,
	expected_existing: bool = False,
) -> None:
	specification = {
		"stage": "downsample",
		"pass": pass_name,
		"layer_path": normalize_layer_path(layer_path),
		"source_mip": source_mip,
		"chunk_size": chunk_size,
		"encoding": encoding,
		"compression": "br",
		"factor": (2, 2, 2),
		"memory": memory,
	}
	fingerprint = stable_fingerprint(specification)
	run_persistent_tasks(
		queue_dir / pass_name / fingerprint,
		fingerprint,
		lambda: create_downsample_tasks(
			layer_path,
			source_mip,
			chunk_size,
			encoding,
			memory,
		),
		parallel,
		lease_seconds,
		release_leases=release_leases,
		expected_existing=expected_existing,
		progress_label=f"Downsample {pass_name.replace('-', ' ')}",
	)


def _pipeline_state_path(queue_dir: Path, configuration: dict) -> Path:
	return queue_dir / "downsample" / stable_fingerprint(configuration) / "pipeline.json"


def downsample_volume(
	layer_path: str,
	queue_dir: Path,
	initial_chunk: tuple[int, int, int],
	extend_chunk: tuple[int, int, int],
	max_extend_passes: int,
	initial_parallel: int,
	extend_parallel: int,
	memory: int,
	encoding: str,
	lease_seconds: int,
	release_leases: bool = True,
) -> None:
	configuration = {
		"layer_path": normalize_layer_path(layer_path),
		"initial_chunk": initial_chunk,
		"extend_chunk": extend_chunk,
		"max_extend_passes": max_extend_passes,
		"memory": memory,
		"encoding": encoding,
	}
	state_path = _pipeline_state_path(queue_dir, configuration)
	state = read_state(
		state_path,
		{
			"configuration": configuration,
			"initial_started": False,
			"initial_complete": False,
			"extensions": [],
			"complete": False,
		},
	)
	if state["complete"]:
		log.write(
			"Downsample",
			"Pyramid is already complete for this configuration.",
			log_level=LOG.STATUS,
		)
		return

	pass_root = state_path.parent
	if not state["initial_complete"]:
		expected_existing = state.get("initial_started", False)
		if not expected_existing:
			state["initial_started"] = True
			write_state(state_path, state)
		log.write(
			"Downsample",
			f"Initial pass from mip 0 with chunks {initial_chunk}.",
			log_level=LOG.STATUS,
		)
		run_pass(
			layer_path,
			pass_root,
			"initial",
			0,
			initial_chunk,
			encoding,
			memory,
			initial_parallel,
			lease_seconds,
			release_leases=release_leases,
			expected_existing=expected_existing,
		)
		state["initial_complete"] = True
		write_state(state_path, state)

	for pass_index in range(max_extend_passes):
		if pass_index < len(state["extensions"]):
			extension = state["extensions"][pass_index]
		else:
			source_mip = len(inspect_volume(layer_path)["scales"]) - 1
			extension = {
				"source_mip": source_mip,
				"started": False,
				"complete": False,
			}
			state["extensions"].append(extension)
			write_state(state_path, state)

		if not extension["complete"]:
			expected_existing = extension.get("started", False)
			if not expected_existing:
				extension["started"] = True
				write_state(state_path, state)
			log.write(
				"Downsample",
				f"Extension pass {pass_index + 1} from mip {extension['source_mip']} "
				f"with chunks {extend_chunk}.",
				log_level=LOG.STATUS,
			)
			run_pass(
				layer_path,
				pass_root,
				f"extend-{pass_index + 1}",
				extension["source_mip"],
				extend_chunk,
				encoding,
				memory,
				extend_parallel,
				lease_seconds,
				release_leases=release_leases,
				expected_existing=expected_existing,
			)
			extension["complete"] = True
			write_state(state_path, state)

		current_max_mip = len(inspect_volume(layer_path)["scales"]) - 1
		if current_max_mip <= extension["source_mip"]:
			log.write(
				"Downsample",
				f"No new mip appeared after extension pass {pass_index + 1}.",
				log_level=LOG.STATUS,
			)
			state["complete"] = True
			write_state(state_path, state)
			return

	state["complete"] = True
	write_state(state_path, state)


@click.command("downsample-pyramid")
@click.argument("layer_path")
@click.option("--queue", "queue_dir", type=click.Path(path_type=Path))
@click.option(
	"--initial-chunk",
	type=XYZ,
	default="64,64,64",
	show_default=True,
	help="Destination chunk size for the first pass.",
)
@click.option(
	"--extend-chunk",
	type=XYZ,
	default="16,16,16",
	show_default=True,
	help="Destination chunk size for extension passes.",
)
@click.option("--max-extend-passes", type=click.IntRange(min=0), default=3, show_default=True)
@click.option("--initial-parallel", type=click.IntRange(min=1), default=16, show_default=True)
@click.option("--extend-parallel", type=click.IntRange(min=1), default=16, show_default=True)
@click.option("--memory", type=click.IntRange(min=1), default=10_000_000_000, show_default=True)
@click.option(
	"--shard-capacity",
	"capacity_override",
	metavar="SIZE",
	help=(
		"Override the automatic 2/4/8 GiB capacity ceiling used to size "
		"post-MIP-0 worker concurrency."
	),
)
@click.option(
	"--encoding",
	default="auto",
	show_default=True,
	help="Destination encoding; auto chooses from the layer type/source metadata.",
)
@click.option("--lease-seconds", type=click.IntRange(min=10), default=3600, show_default=True)
@click.option(
	"--release-leases/--preserve-leases",
	default=True,
	show_default=True,
	help="Release existing FileQueue leases when resuming; preserve for shared queues.",
)
@click.option(
	"--force",
	is_flag=True,
	help="Allow downsampling when local MIP-0 completeness cannot be confirmed.",
)
@click.option("--execute/--dry-run", default=True, show_default=True)
@igneous_output_command
def downsample_pyramid(
	layer_path: str,
	queue_dir: Path | None,
	initial_chunk: tuple[int, int, int],
	extend_chunk: tuple[int, int, int],
	max_extend_passes: int,
	initial_parallel: int,
	extend_parallel: int,
	memory: int,
	capacity_override: str | int | None,
	encoding: str,
	lease_seconds: int,
	release_leases: bool,
	force: bool,
	execute: bool,
) -> None:
	"""Build a volumetric MIP pyramid with durable task-level resume."""
	try:
		info = inspect_volume(layer_path)
		layer_type = info.get("type", "image")
		scales = info["scales"]
		source_encoding = scales[0].get("encoding", "raw")
		encoding = select_layer_encoding(
			encoding,
			layer_type,
			source_encoding,
		)
		queue_dir = queue_dir or default_queue_root(layer_path)
		if isinstance(capacity_override, str):
			capacity_override = parse_size(capacity_override)
		resources = plan_resources(
			info,
			(0, 3, 5),
			max(initial_parallel, extend_parallel),
			capacity_override=capacity_override,
		)
		initial_workers = min(
			initial_parallel,
			resources.cpu_limit,
			resources.downsample_memory_limit,
		)
		extend_workers = min(
			extend_parallel,
			resources.cpu_limit,
			resources.downsample_memory_limit,
		)
		log.write(
			"Downsample",
			(
				f"Plan: mip 0 at {initial_chunk}; up to {max_extend_passes} "
				f"extension(s) at {extend_chunk}; queue={queue_dir.resolve()}."
			),
			log_level=LOG.INFO,
		)
		log_resource_plan("Downsample", resources)
		if not execute:
			return
		completeness = check_mip0_completeness(layer_path)
		if completeness.complete:
			log.write(
				"Downsample",
				f"MIP 0 completeness check passed: {completeness.summary()}",
				log_level=LOG.INFO,
			)
		else:
			failure = mip0_failure_message(completeness)
			if force:
				log.write(
					"Downsample",
					f"Warning: forcing downsample despite {failure}",
					log_level=LOG.WARN,
				)
			else:
				raise ValueError(f"{failure}; use --force to override")
		downsample_volume(
			layer_path,
			queue_dir,
			initial_chunk,
			extend_chunk,
			max_extend_passes,
			initial_workers,
			extend_workers,
			memory,
			encoding,
			lease_seconds,
			release_leases,
		)
		log.write("Downsample", "Pyramid complete.", log_level=LOG.STATUS)
	except click.ClickException:
		raise
	except Exception as exc:
		raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
	downsample_pyramid()
