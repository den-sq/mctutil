"""Unified parsing command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="parse",
	help="Metadata, config, and scanlog parsing helpers.",
	lazy_subcommands={
		"meta-shift": "parsing.meta_shift:meta_shift",
		"prune-empty": "parsing.empty_dir_removal:prune_empty",
		"pull-config": "parsing.pull_config:get_conf",
		"scanlog-fetch": "parsing.scanlog_fetch:scanlog_fetch",
	},
)
