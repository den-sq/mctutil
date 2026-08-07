"""Unified parsing command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="parse",
	help="Metadata, config, and scanlog parsing helpers.",
	lazy_subcommands={
		"find-errs": "mctutil.parse.find_err_general:find_errs",
		"meta-shift": "mctutil.parse.meta_shift:meta_shift",
		"prune-empty": "mctutil.parse.empty_dir_removal:prune_empty",
		"pull-config": "mctutil.parse.pull_config:get_conf",
		"scanlog-fetch": "mctutil.parse.scanlog_fetch:scanlog_fetch",
		"xaid-log": "mctutil.parse.xaid_reconstruction_log:xaid_log",
	},
)
