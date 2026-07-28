# mctutil parse

Metadata, config, and scanlog parsing helpers.

Run `mctutil parse --help` to list commands and `mctutil parse <task> --help`
for a command's options.

## Commands

- **`meta-shift`** — Run the per-sample meta-shift engine, delegating lab-specific schema (folder conventions, status enum, sbatch parsing, sheet layout) to a `--schema` adapter (e.g. `chenglab`).
- **`pull-config`** — Copy config files found under a root into a target directory.
- **`scanlog-fetch`** — Copy scanlogs into a target location.
- **`prune-empty`** — Remove empty subdirectories under each match of a pattern beneath a root.

Defaults: `prune-empty` **defaults to `--dry-run`** because `rmdir` is
destructive. `pull-config` and `scanlog-fetch` **default to `--execute`** (they
copy); pass `--dry-run` to list the planned copies instead.
