"""Phase estimation and material-table command group."""

from mctutil.lazy import LazyGroup


main = LazyGroup(
	name="phase",
	help="Estimate phase parameters and construct material tables.",
	lazy_subcommands={
		"harvest": "mctutil.phase.commands:harvest",
		"estimate-line-slope": "mctutil.phase.commands:estimate_line_slope",
		"refine": "mctutil.phase.commands:refine",
		"make-material-table": "mctutil.phase.commands:make_material_table",
		"export": "mctutil.phase.export:export",
	},
)
