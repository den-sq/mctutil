"""Shared Igneous mesh-building workflow."""

from multiprocessing import cpu_count

import click

from mctutil.shared.log import log, LOG


def _require_mesh_dependencies():
	try:
		import igneous.task_creation as task_creation
		from taskqueue import LocalTaskQueue
	except ImportError as exc:
		raise click.ClickException(
			"Mesh support requires igneous-pipeline and task-queue; "
			"install the project environment from environment.yml."
		) from exc

	return LocalTaskQueue, task_creation


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
	:return: None.
	"""
	parallel = cpu_count() if parallel is None else parallel
	if parallel < 1:
		raise click.ClickException("parallel must be at least 1.")

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
	task_queue = LocalTaskQueue(parallel=parallel)

	mesh_tasks = task_creation.create_meshing_tasks(
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

	merge_tasks = task_creation.create_unsharded_multires_mesh_tasks(
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
