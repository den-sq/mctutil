# mctutil parse

Metadata, config, and scanlog parsing helpers.

Run `mctutil parse --help` to list commands and `mctutil parse <task> --help`
for a command's options.

## Commands

- **`find-errs`** — Classify directories containing non-empty versus empty
  scheduler error files. Matching defaults to `err*`; `--errors-out` and
  `--clean-out` optionally write the two sorted directory lists.
- **`meta-shift`** — Run the per-sample meta-shift engine, delegating lab-specific schema (folder conventions, status enum, sbatch parsing, sheet layout) to a `--schema` adapter (e.g. `chenglab`).
- **`pull-config`** — Copy config files found under a root into a target directory.
- **`scanlog-fetch`** — Copy scanlogs into a target location.
- **`xaid-log`** — Convert a MITOS X-AID reconstruction `config.txt` into a
  CSV row matching columns A-R of the lab's `Reconstructions` sheet. Metadata
  absent from X-AID can be supplied with options such as `--scan-number`,
  `--stain`, and `--energy`; center conversion is explicit through
  `--center-convention`.
- **`prune-empty`** — Remove empty subdirectories under each match of a pattern beneath a root.

Example:

```bash
mctutil parse xaid-log config.txt \
  --output reconstruction_log.csv \
  --scan-number 147 \
  --stain unstained \
  --energy "40 kV" \
  --center-convention width-half
```

The default `offset` center convention preserves the labeled X-AID
`rotation_axis_offset`. Use `width-half` only when the target sheet uses
`detector_width / 2 + offset`, or `pixel-center` when it uses
`(detector_width - 1) / 2 + offset`. An explicit `--center` takes precedence.

To append directly to Google Sheets instead of creating a CSV:

```bash
mctutil parse xaid-log config.txt \
  --upload \
  --spreadsheet SPREADSHEET_ID \
  --sheet Reconstructions
```

The command uses `conf/gsheets_credentials.json` for the initial OAuth desktop
flow and caches the resulting token in `conf/gsheets_token.json`. Change the
directory with `--google-conf` or `MCTUTIL_GOOGLE_CONF`. The Google client
libraries are included in `environment.yml`; pip-only installations can use
`mctutil[google-sheets]`.

Uploads verify that the destination header in A1:R1 exactly matches the
expected reconstruction schema, then append one row using raw values. Use
`--no-verify-header` only when intentionally targeting a differently labeled
but positionally compatible tab.

Defaults: `prune-empty` **defaults to `--dry-run`** because `rmdir` is
destructive. `pull-config` and `scanlog-fetch` **default to `--execute`** (they
copy); pass `--dry-run` to list the planned copies instead.
