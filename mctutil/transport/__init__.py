"""Unified transport command group."""

from mctutil.lazy import LazyGroup

main = LazyGroup(
	name="transport",
	help="Remote storage and data movement helpers.",
	lazy_subcommands={
		"cv-fetch": "transport.cv_import:cloudvolume_fetch",
		"s3-upload": "transport.s3upload:s3upload",
	},
)
