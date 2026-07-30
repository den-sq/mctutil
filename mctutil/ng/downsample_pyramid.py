"""Build the two-pass volumetric MIP pyramid used by the sharded pipeline."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import click

from mctutil.ng.completeness import check_mip0_completeness
from mctutil.shared.cli import XYZ
from mctutil.shared.persistent_queue import (
	read_state,
	run_persistent_tasks,
	stable_fingerprint,
	write_state,
)


def _require_dependencies():
	try:
		import igneous.task_creation as task_creation
		from cloudvolume import CloudVolume
	except ImportError as exc:
		raise RuntimeError(
			"ng downsample-pyramid requires igneous-pipeline and task-queue; "
			"install with pip install -e '.[ng,mesh]'"
		) from exc
	return CloudVolume, task_creation


def normalize_layer_path(layer_path: str) -> str:
	if "://" in layer_path:
		return layer_path
	return Path(layer_path).resolve().as_uri()


def local_layer_path(layer_path: str) -> Path | None:
	value = layer_path.removeprefix("precomputed://")
	if "://" not in value:
		return Path(value).resolve()
	parsed = urlparse(value)
	if parsed.scheme != "file":
		return None
	return Path(unquote(parsed.path)).resolve()


def default_queue_root(layer_path: str) -> Path:
	local_path = local_layer_path(layer_path)
	if local_path is None:
		raise ValueError("--queue is required for non-local layer paths")
	return local_path / ".mctutil-queues"


def inspect_volume(layer_path: str) -> tuple[str, str, int]:
	CloudVolume, _task_creation = _require_dependencies()
	volume = CloudVolume(normalize_layer_path(layer_path), parallel=False)
	layer_type = volume.info.get("type", "image")
	scales = volume.info.get("scales", [])
	if not scales:
		raise ValueError("precomputed volume has no scales")
	encoding = scales[0].get("encoding", "raw")
	return layer_type, encoding, len(scales) - 1


def create_downsample_tasks(
	layer_path: str,
	source_mip: int,
	chunk_size: tuple[int, int, int],
	encoding: str,
	memory: int,
):
	_CloudVolume, task_creation = _require_dependencies()
	return task_creation.create_downsampling_tasks(
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
		click.echo("Downsample pyramid is already complete for this configuration.")
		return

	pass_root = state_path.parent
	if not state["initial_complete"]:
		expected_existing = state.get("initial_started", False)
		if not expected_existing:
			state["initial_started"] = True
			write_state(state_path, state)
		click.echo(f"Initial downsample from mip 0 with chunks {initial_chunk}.")
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
			_layer_type, _source_encoding, source_mip = inspect_volume(layer_path)
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
			click.echo(
				f"Extension pass {pass_index + 1} from mip {extension['source_mip']} "
				f"with chunks {extend_chunk}."
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

		_layer_type, _source_encoding, current_max_mip = inspect_volume(layer_path)
		if current_max_mip <= extension["source_mip"]:
			click.echo(f"No new mip appeared after extension pass {pass_index + 1}.")
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
def downsample_pyramid(
	layer_path: str,
	queue_dir: Path | None,
	initial_chunk: tuple[int, int, int],
	extend_chunk: tuple[int, int, int],
	max_extend_passes: int,
	initial_parallel: int,
	extend_parallel: int,
	memory: int,
	encoding: str,
	lease_seconds: int,
	release_leases: bool,
	force: bool,
	execute: bool,
) -> None:
	"""Build a volumetric MIP pyramid with durable task-level resume."""
	try:
		layer_type, source_encoding, max_mip = inspect_volume(layer_path)
		if encoding == "auto":
			encoding = "raw" if layer_type == "image" else source_encoding
			if layer_type == "segmentation" and encoding == "raw":
				encoding = "compressed_segmentation"
		queue_dir = queue_dir or default_queue_root(layer_path)
		click.echo(f"Layer: {normalize_layer_path(layer_path)}")
		click.echo(f"Layer type: {layer_type}; encoding: {encoding}; current max mip: {max_mip}")
		click.echo(f"Queue root: {queue_dir.resolve()}")
		click.echo(
			f"Plan: mip 0 at {initial_chunk}, then up to {max_extend_passes} "
			f"extension pass(es) at {extend_chunk}; factor=(2, 2, 2), compression=br"
		)
		if not execute:
			return
		completeness = check_mip0_completeness(layer_path)
		if completeness.complete:
			click.echo(f"MIP 0 completeness check passed: {completeness.summary()}")
		elif force:
			click.echo(
				f"Warning: forcing downsample despite MIP 0 completeness failure: "
				f"{completeness.summary()}",
				err=True,
			)
		else:
			raise ValueError(
				f"source MIP 0 is incomplete: {completeness.summary()}; "
				"use --force to override"
			)
		downsample_volume(
			layer_path,
			queue_dir,
			initial_chunk,
			extend_chunk,
			max_extend_passes,
			initial_parallel,
			extend_parallel,
			memory,
			encoding,
			lease_seconds,
			release_leases,
		)
	except click.ClickException:
		raise
	except Exception as exc:
		raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
	downsample_pyramid()
