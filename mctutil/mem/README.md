# mctutil mem

Shared-memory cleanup and node-submission helpers.

Run `mctutil mem --help` to list commands and `mctutil mem <task> --help` for a
command's options.

## Commands

- **`clean`** — List or clean configured shared-memory entries on this node. Prefix sets are selected through repeatable `--config` options.
- **`mark`** — Submit one shared-memory cleanup job per selected Slurm partition.
- **`from-file`** — Submit an sbatch script for each node listed in a file.
- **`from-range`** — Submit an sbatch script for each node in a `--prefix` / `--start` / `--stop` range.

Safety: `clean` and `mark` **default to `--dry-run`** (unlinking shared-memory
segments and submitting cleanup jobs are both consequential); pass `--execute`
to act. The node-submission commands `from-file` and `from-range` instead
**default to `--execute`**; pass `--dry-run` to list the planned submissions
without submitting.
