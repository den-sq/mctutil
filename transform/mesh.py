'''
Supported Formats: None (precomputed), graphene, precomputed, boss, n5
Supported Protocols: gs, file, s3, http, https, mem, matrix, tigerdata
'''
from pathlib import Path
import sys

from taskqueue import LocalTaskQueue
import click

import igneous.task_creation as tc

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


@click.command()
@click.option("-p", "--proj-dir", type=click.Path(file_okay=False), required=True, help="Path of input data.")
@click.option("-l", "--layer-path", type=click.STRING, required=True, help="Path of Layer data, including remote URLs.")
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually enqueue meshing tasks or just describe the planned passes.")
def mesh(proj_dir, layer_path, execute):
	mip = 0

	if not execute:
		log.log("Mesh",
				f"Would create meshing tasks for {layer_path} (mip={mip}, shape=512^3) and follow with manifest tasks",
				log_level=log.DEBUG.INFO)
		return

	with LocalTaskQueue(parallel=8) as tq:
		tasks = tc.create_meshing_tasks( 	# First Pass
			layer_path, 					# Which data layer
			mip,							# Which resolution level to mesh at (we often choose near isotropic resolutions)
			shape=(512, 512, 512),			# Size of a task to mesh, chunk alignment not needed
			simplification=True,			# Whether to enable quadratic edge collapse mesh simplification
			max_simplification_error=40, 	# Maximum physical deviation of mesh vertices during simplification
			mesh_dir=None,					# Optionally choose a non-default location for saving meshes
			cdn_cache=False,				# Disable caching in the cloud so updates aren't painful to view
			dust_threshold=None,			# Don't bother meshing below this number of voxels
			object_ids=None,				# Optionally, only mesh these labels.
			progress=False,					# Display a progress bar (more useful locally than in the cloud)
			fill_missing=False,				# If part of the data is missing, fill with zeros instead of raising an error
			encoding='precomputed',			# 'precomputed' or 'draco' (don't change this unless you know what you're doing)
			spatial_index=True,				# generate a spatial index for querying meshes by bounding box
			sharded=False,					# generate intermediate shard fragments for later processing into sharded format
		)
		tq.insert_all(tasks)
	log.log("Mesh", "create_meshing_tasks complete", log_level=log.DEBUG.STATUS)

	with LocalTaskQueue(parallel=8) as tq:
		tasks = tc.create_mesh_manifest_tasks(layer_path, magnitude=3) 	# Second Pass
		tq.insert_all(tasks)

	log.log("Mesh", "create_mesh_manifest_tasks complete", log_level=log.DEBUG.STATUS)


if __name__ == "__main__":
	mesh()
