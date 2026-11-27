"""Multi-scan registration, layout, and stitching command group."""

from mctutil.lazy import LazyGroup


main = LazyGroup(
	name="mosaic",
	help="Register overlapping reconstructions, build a stitch layout, and stitch and denoise the canvas.",
	lazy_subcommands={
		"coarse": "mctutil.mosaic.coarse:coarse",
		"register": "mctutil.mosaic.register:register",
		"link-depth": "mctutil.mosaic.depth:link_depth",
		"refine": "mctutil.mosaic.refine:refine",
		"tilt-points": "mctutil.mosaic.tilt:tilt_points",
		"fit": "mctutil.mosaic.fit:fit",
		"layout": "mctutil.mosaic.layout:layout",
		"norm-fit": "mctutil.mosaic.norm:norm_fit",
		"profile": "mctutil.mosaic.profile:profile",
		"stitch": "mctutil.mosaic.stitch:stitch",
		"spectra": "mctutil.mosaic.spectra:spectra",
		"wiener": "mctutil.mosaic.wiener:wiener",
		"window": "mctutil.mosaic.window:window",
		"to-u16": "mctutil.mosaic.window:to_u16_command",
	},
)
