"""Unified mesh command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="mesh",
	help="Meshing helpers for igneous/neuroglancer workflows.",
	lazy_subcommands={
		"build": "mctutil.transform.mesh:mesh",
		"build-igneous": "mctutil.transform.mesh_ig:mesh_ig",
	},
)
