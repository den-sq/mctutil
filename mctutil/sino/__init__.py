"""Unified sinogram command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="sino",
	help="Sinogram conversion and preprocessing workflows.",
	lazy_subcommands={
		"convert": "mctutil.transform.sinogram:sino_convert",
	},
)
