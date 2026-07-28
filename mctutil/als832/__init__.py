"""ALS Beamline 8.3.2 data extraction command group."""

from mctutil.lazy import LazyGroup


main = LazyGroup(
	name="als832",
	help="ALS Beamline 8.3.2 Data Exchange HDF5 extraction helpers.",
	lazy_subcommands={
		"extract-projections": "mctutil.als832.extract_projections:extract_projections",
		"extract-refs": "mctutil.als832.extract_refs:extract_refs",
		"h5-tree": "mctutil.als832.h5_tree:h5_tree",
	},
)
