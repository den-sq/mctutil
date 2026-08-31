# Importer mapping safety boundary

Importer YAML is declarative and non-executable. A user-supplied `--mapping`
file cannot execute code and cannot reach the filesystem beyond artifacts that
the selected source adapter has already opened. It may redirect a reader to a
different locator *inside* those artifacts, which remains a correctness and
mislabeling risk. Pre-discovery mapping validation checks schema membership,
reader and transform vocabulary, types, locators, and registered unit
conversions; it cannot prove that a scientifically incorrect locator is right.

Mappings are complete replacements, not partial overlays. Complex joins,
aliases, defaults, precedence, and cross-field formulas belong to source
adapters or the shared engine.
