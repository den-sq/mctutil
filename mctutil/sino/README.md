# mctutil sino

Sinogram conversion and preprocessing workflows.

Run `mctutil sino --help` to list commands and `mctutil sino <task> --help`
for a command's options.

## Commands

- **`convert`** — Build sinograms from projections + flats, or preprocess existing sinograms. Select the path with `--mode full|preproc` (`full` requires a flats directory).

`convert` accepts `--dry-run` to log the planned outputs instead of writing them.
