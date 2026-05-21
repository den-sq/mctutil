"""Unified sinogram command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="sino",
	help="Sinogram conversion and preprocessing workflows.",
	lazy_subcommands={
		"convert": "transform.sinogram:sino_convert",
	},
)
