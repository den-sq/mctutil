# mctutil als832

ALS Beamline 8.3.2 Data Exchange HDF5 extraction helpers.

Run `mctutil als832 --help` to list commands and
`mctutil als832 <task> --help` for a command's options.

## Commands

- **`extract-projections`** — Extract projection frames from ALS 8.3.2 HDF5 files or directories.
- **`extract-refs`** — Extract ALS 8.3.2 flat/bright and dark reference frames.
- **`h5-tree`** — Read HDF5 structure or values without modifying the source files.

`extract-projections` and `extract-refs` accept `--dry-run` to plan the writes.
`h5-tree` opens sources read-only; datasets with more than 10,000 values are not
loaded unless `--max-values` is raised or set to `0`.

Install the extra with `pip install -e .[als832]`.
