"""Unified Neuroglancer command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="ng",
	help="Neuroglancer JSON, layer, and annotation helpers.",
	lazy_subcommands={
		"build": "transform.ng:neuroglance",
		"layer-copy": "ng.layer_copy:layer_copy",
		"layer-extract": "ng.layer_extract:layer_extract",
		"layer-recolor": "ng.change_color:change_color",
		"layer-tag": "ng.layer_tag:layer_tag",
		"layer-urlshift": "ng.layer_urlshift:layer_urlshift",
		"point-add": "ng.point_add:point_add",
		"point-merge": "ng.point_merge:point_merge",
		"point-shift": "ng.point_shift:point_shift",
		"point-sort": "ng.point_sort:point_sort",
		"position-copy": "ng.position_copy:position_copy",
		"shift-angle": "ng.shift_angle:shift_angle",
	},
)
