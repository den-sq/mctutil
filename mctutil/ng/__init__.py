"""Unified Neuroglancer command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="ng",
	help="Neuroglancer JSON, layer, and annotation helpers.",
	lazy_subcommands={
		"build": "mctutil.transform.ng:neuroglance",
		"downsample-pyramid": "mctutil.ng.downsample_pyramid:downsample_pyramid",
		"layer-copy": "mctutil.ng.layer_copy:layer_copy",
		"layer-extract": "mctutil.ng.layer_extract:layer_extract",
		"layer-recolor": "mctutil.ng.change_color:change_color",
		"layer-tag": "mctutil.ng.layer_tag:layer_tag",
		"layer-urlshift": "mctutil.ng.layer_urlshift:layer_urlshift",
		"point-add": "mctutil.ng.point_add:point_add",
		"point-merge": "mctutil.ng.point_merge:point_merge",
		"point-shift": "mctutil.ng.point_shift:point_shift",
		"point-sort": "mctutil.ng.point_sort:point_sort",
		"position-copy": "mctutil.ng.position_copy:position_copy",
		"precompute": "mctutil.ng.precompute:precompute",
		"shift-angle": "mctutil.ng.shift_angle:shift_angle",
	},
)
