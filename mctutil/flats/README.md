# mctutil flats

Flat-field drift tracking, digest, and medianization helpers.

Run `mctutil flats --help` to list commands and `mctutil flats <task> --help`
for a command's options.

## Commands

- **`beam-tracking`** — Diagnose beam drift and optionally split flat fields into static/dynamic components.
- **`series-digest`** — Build `digest_stack.tif` and `drift_trajectory.csv` from flat-field frames.
- **`medianize`** — Median TIFF flats by filename prefix, writing one median image per group.

These commands accept `--dry-run` to plan the writes instead of performing them.

Install the extra with `pip install -e .[flats]`.
