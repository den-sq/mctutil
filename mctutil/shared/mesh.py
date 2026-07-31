"""Shared Igneous mesh-building workflow."""

from multiprocessing import cpu_count
from pathlib import Path

import click

from mctutil.shared.aws import (
	configure_aws_profile,
	preflight_s3_info,
	s3_location,
)
from mctutil.shared.igneous_output import (
	capture_igneous_call,
	igneous_output_command,
)
from mctutil.shared.log import log, LOG
from mctutil.shared.persistent_queue import run_persistent_tasks, stable_fingerprint


def _require_mesh_dependencies():
	try:
		import igneous.task_creation as task_creation
		from taskqueue import LocalTaskQueue
	except ImportError as exc:
		raise click.ClickException(
			"Mesh support requires igneous-pipeline and task-queue; "
			"install with pip install -e '.[mesh]'"
		) from exc

	return LocalTaskQueue, task_creation


@igneous_output_command
def build_mesh(
	layer_path,
	mip=0,
	num_lod=4,
	parallel=None,
	shape=(448, 448, 448),
	simplification=True,
	max_simplification_error=40,
	mesh_dir=None,
	cdn_cache=False,
	dust_threshold=None,
	object_ids=None,
	fill_missing=False,
	encoding="precomputed",
	spatial_index=True,
	magnitude=3,
	vertex_quantization_bits=16,
	min_chunk_size=(256, 256, 256),
	execute=True,
	queue_dir=None,
	lease_seconds=3600,
	aws_profile=None,
):
	"""Build and merge an unsharded multiresolution mesh.

	:param layer_path: Precomputed segmentation layer URL.
	:param mip: Resolution level to mesh.
	:param num_lod: Number of additional mesh levels of detail.
	:param parallel: Number of local TaskQueue workers.
	:param shape: First-pass task shape in voxels.
	:param simplification: Whether to simplify first-pass mesh fragments.
	:param max_simplification_error: Maximum simplification error in physical units.
	:param mesh_dir: Optional mesh directory override.
	:param cdn_cache: Whether generated fragments may be CDN cached.
	:param dust_threshold: Optional per-cutout label-size threshold.
	:param object_ids: Optional iterable of labels to mesh.
	:param fill_missing: Whether missing image chunks are treated as background.
	:param encoding: First-pass fragment encoding.
	:param spatial_index: Whether to generate a mesh spatial index.
	:param magnitude: Prefix partition magnitude for the merge pass.
	:param vertex_quantization_bits: Vertex precision for multiresolution meshes.
	:param min_chunk_size: Minimum highest-resolution mesh chunk size.
	:param execute: Whether to run tasks or only describe the workflow.
	:param queue_dir: Optional durable file-queue root for resumable execution.
	:param lease_seconds: FileQueue task lease duration.
	:param aws_profile: Named AWS profile used when layer_path is on S3.
	:return: None.
	"""
	parallel = cpu_count() if parallel is None else parallel
	if parallel < 1:
		raise click.ClickException("parallel must be at least 1.")

	resolved_aws_profile = None
	location = s3_location(layer_path)
	if location is not None:
		resolved_aws_profile = configure_aws_profile(
			aws_profile,
			location[0],
		)
		log.write(
			"Mesh",
			f"AWS profile: {resolved_aws_profile}",
			log_level=LOG.INFO,
		)
		if execute:
			preflight_s3_info(layer_path, resolved_aws_profile)

	if not execute:
		log.write(
			"Mesh",
			(
				f"Would mesh {layer_path} at mip {mip} with shape {tuple(shape)} "
				f"on {parallel} workers, then merge {num_lod} additional LODs "
				f"with magnitude {magnitude}"
			),
			log_level=LOG.INFO,
		)
		return

	LocalTaskQueue, task_creation = _require_mesh_dependencies()
	if queue_dir is not None:
		queue_dir = Path(queue_dir)
		forge_specification = {
			"stage": "mesh-forge",
			"layer_path": layer_path,
			"mip": mip,
			"shape": tuple(shape),
			"simplification": simplification,
			"max_simplification_error": max_simplification_error,
			"mesh_dir": mesh_dir,
			"cdn_cache": cdn_cache,
			"dust_threshold": dust_threshold,
			"object_ids": None if not object_ids else list(object_ids),
			"fill_missing": fill_missing,
			"encoding": encoding,
			"spatial_index": spatial_index,
		}
		forge_fingerprint = stable_fingerprint(forge_specification)
		run_persistent_tasks(
			queue_dir / "forge" / forge_fingerprint,
			forge_fingerprint,
			lambda: capture_igneous_call(
				task_creation.create_meshing_tasks,
				layer_path,
				mip,
				shape=tuple(shape),
				simplification=simplification,
				max_simplification_error=max_simplification_error,
				mesh_dir=mesh_dir,
				cdn_cache=cdn_cache,
				dust_threshold=dust_threshold,
				object_ids=None if not object_ids else list(object_ids),
				progress=False,
				fill_missing=fill_missing,
				encoding=encoding,
				spatial_index=spatial_index,
				sharded=False,
			),
			parallel,
			lease_seconds,
			progress_label="Mesh Forge",
		)
		log.write("Mesh", "Meshing pass complete", log_level=LOG.STATUS)

		merge_specification = {
			"stage": "mesh-merge",
			"layer_path": layer_path,
			"num_lod": num_lod,
			"magnitude": magnitude,
			"mesh_dir": mesh_dir,
			"vertex_quantization_bits": vertex_quantization_bits,
			"min_chunk_size": tuple(min_chunk_size),
		}
		merge_fingerprint = stable_fingerprint(merge_specification)
		run_persistent_tasks(
			queue_dir / "merge" / merge_fingerprint,
			merge_fingerprint,
			lambda: capture_igneous_call(
				task_creation.create_unsharded_multires_mesh_tasks,
				layer_path,
				num_lod=num_lod,
				magnitude=magnitude,
				mesh_dir=mesh_dir,
				vertex_quantization_bits=vertex_quantization_bits,
				min_chunk_size=tuple(min_chunk_size),
			),
			parallel,
			lease_seconds,
			progress_label="Mesh Merge",
		)
		log.write("Mesh", "Multiresolution merge pass complete", log_level=LOG.STATUS)
		return

	task_queue = LocalTaskQueue(parallel=parallel)

	mesh_tasks = capture_igneous_call(
		task_creation.create_meshing_tasks,
		layer_path,
		mip,
		shape=tuple(shape),
		simplification=simplification,
		max_simplification_error=max_simplification_error,
		mesh_dir=mesh_dir,
		cdn_cache=cdn_cache,
		dust_threshold=dust_threshold,
		object_ids=None if not object_ids else list(object_ids),
		progress=False,
		fill_missing=fill_missing,
		encoding=encoding,
		spatial_index=spatial_index,
		sharded=False,
	)
	task_queue.insert(mesh_tasks)
	task_queue.execute()
	log.write("Mesh", "Meshing pass complete", log_level=LOG.STATUS)

	merge_tasks = capture_igneous_call(
		task_creation.create_unsharded_multires_mesh_tasks,
		layer_path,
		num_lod=num_lod,
		magnitude=magnitude,
		mesh_dir=mesh_dir,
		vertex_quantization_bits=vertex_quantization_bits,
		min_chunk_size=tuple(min_chunk_size),
	)
	task_queue.insert(merge_tasks)
	task_queue.execute()
	log.write("Mesh", "Multiresolution merge pass complete", log_level=LOG.STATUS)
