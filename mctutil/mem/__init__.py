"""Unified shared-memory cleanup command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="mem",
	help="Shared-memory cleanup and node submission helpers.",
	lazy_subcommands={
		"clean": "mctutil.mem.clean_command:command",
		"mark": "mctutil.mem.mark_command:command",
		"from-file": "mctutil.mem.from_file:from_file",
		"from-range": "mctutil.mem.from_range:from_range",
	},
)
