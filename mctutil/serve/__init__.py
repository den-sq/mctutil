"""Unified local serving command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="serve",
	help="Local data and visualization servers.",
	lazy_subcommands={
		"ng": "mctutil.serve.ng:ng",
	},
)
