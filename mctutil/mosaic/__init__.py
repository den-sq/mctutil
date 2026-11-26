"""Multi-scan registration, layout, and stitching command group."""

from mctutil.lazy import LazyGroup


main = LazyGroup(
	name="mosaic",
	help="Register overlapping reconstructions and build a multi-scan stitch layout.",
	lazy_subcommands={
		"coarse": "mctutil.mosaic.coarse:coarse",
		"register": "mctutil.mosaic.register:register",
		"link-depth": "mctutil.mosaic.depth:link_depth",
		"refine": "mctutil.mosaic.refine:refine",
		"tilt-points": "mctutil.mosaic.tilt:tilt_points",
		"fit": "mctutil.mosaic.fit:fit",
		"layout": "mctutil.mosaic.layout:layout",
	},
)
