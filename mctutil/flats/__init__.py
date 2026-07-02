"""Flat/gain-field processing command group."""

from mctutil.lazy import LazyGroup


main = LazyGroup(
	name="flats",
	help="Flat-field drift tracking, digest, and medianization helpers.",
	lazy_subcommands={
		"beam-tracking": "mctutil.flats.beam_tracking:beam_tracking",
		"medianize": "mctutil.flats.medianize:medianize",
		"series-digest": "mctutil.flats.series_digest:series_digest",
	},
)
