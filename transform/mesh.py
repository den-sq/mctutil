'''
Supported Formats: None (precomputed), graphene, precomputed, boss, n5
Supported Protocols: gs, file, s3, http, https, mem, matrix, tigerdata
'''
from taskqueue import LocalTaskQueue
import click

import igneous.task_creation as tc

# layer_path = os.path.join(proj_dir, '33dpf_seg_test')
# layer_path = 'precomputed://s3://3d.fish/assets/precomputed_repository/B2_daphnia/AAA399/AAA399_seg_out/'


@click.commmand
@click.option("-p", "--proj-dir", type=click.Path(file_okay=False), required=True, help="Path of input data.")
@click.option("-l", "--layer-path", type=click.STRING, required=True, help="Path of Layer data, including remote URLs.")
def mesh(proj_dir, layer_path):
	mip = 0

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
	print("Done create_meshing_tasks !!")

	with LocalTaskQueue(parallel=8) as tq:
		tasks = tc.create_mesh_manifest_tasks(layer_path, magnitude=3) 	# Second Pass
		tq.insert_all(tasks)

	print("Done create_mesh_manifest_tasks !!")


if __name__ == "__main__":
	mesh()
