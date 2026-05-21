"""Unified HPC command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="hpc",
	help="HPC runtime and scheduler-side helpers.",
	lazy_subcommands={
		"time-check": "hpc_work.timecheck:timecheck",
	},
)
