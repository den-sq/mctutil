"""Unified mesh build command."""

from collections import namedtuple
from multiprocessing import cpu_count

import click

from mctutil.shared.cli import DelimitedRecord
from mctutil.shared.mesh import build_mesh


MeshVector = namedtuple("MeshVector", ["x", "y", "z"])
MESH_VECTOR = DelimitedRecord(
	MeshVector,
	[int, int, int],
	delimiter=",",
	name="X,Y,Z voxel dimensions",
)


@click.command()
@click.option(
	"-p",
	"--parallel",
	type=click.IntRange(min=1),
	default=cpu_count,
	show_default="CPU count",
	help="Number of local TaskQueue workers.",
)
@click.option("--mip", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--num-lod", type=click.IntRange(min=0), default=4, show_default=True,
				help="Number of additional mesh levels of detail.")
@click.option("--shape", type=MESH_VECTOR, default="448,448,448", show_default=True,
				help="First-pass task shape as X,Y,Z voxels.")
@click.option("--simplify/--skip-simplify", default=True, show_default=True)
@click.option("--max-error", type=click.IntRange(min=0), default=40, show_default=True,
				help="Maximum simplification error in physical units.")
@click.option("--mesh-dir", type=click.STRING, default=None,
				help="Override the mesh directory named in the layer info.")
@click.option("--cdn-cache/--no-cdn-cache", default=False, show_default=True)
@click.option("--dust-threshold", type=click.IntRange(min=0), default=None,
				help="Skip labels smaller than this many voxels within a cutout.")
@click.option("--object-id", type=click.INT, multiple=True,
				help="Mesh only this label; may be repeated.")
@click.option("--fill-missing", is_flag=True,
				help="Treat missing image chunks as background.")
@click.option(
	"--encoding",
	type=click.Choice(["precomputed", "draco"], case_sensitive=False),
	default="precomputed",
	show_default=True,
)
@click.option("--spatial-index/--skip-spatial-index", default=True, show_default=True)
@click.option("--magnitude", type=click.IntRange(min=1), default=3, show_default=True,
				help="Prefix partition magnitude for the merge pass.")
@click.option(
	"--vertex-quantization-bits",
	type=click.Choice([10, 16]),
	default=16,
	show_default=True,
)
@click.option("--min-chunk-size", type=MESH_VECTOR, default="256,256,256", show_default=True,
				help="Minimum multiresolution chunk size as X,Y,Z voxels.")
@click.option("--execute/--dry-run", default=True,
				help="Whether to run mesh tasks or only describe the workflow.")
@click.argument("layer_path", type=click.STRING)
def mesh(
	parallel,
	mip,
	num_lod,
	shape,
	simplify,
	max_error,
	mesh_dir,
	cdn_cache,
	dust_threshold,
	object_id,
	fill_missing,
	encoding,
	spatial_index,
	magnitude,
	vertex_quantization_bits,
	min_chunk_size,
	execute,
	layer_path,
):
	"""Build an unsharded multiresolution mesh for LAYER_PATH."""
	build_mesh(
		layer_path,
		mip=mip,
		num_lod=num_lod,
		parallel=parallel,
		shape=shape,
		simplification=simplify,
		max_simplification_error=max_error,
		mesh_dir=mesh_dir,
		cdn_cache=cdn_cache,
		dust_threshold=dust_threshold,
		object_ids=object_id,
		fill_missing=fill_missing,
		encoding=encoding,
		spatial_index=spatial_index,
		magnitude=magnitude,
		vertex_quantization_bits=vertex_quantization_bits,
		min_chunk_size=min_chunk_size,
		execute=execute,
	)


if __name__ == "__main__":
	mesh()
