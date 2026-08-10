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
- **`xaid-log`** — Convert a MITOS X-AID reconstruction `config.txt` into one
  88-column CSV row. Each configured value is written to its clear-name field
  from the X-AID field mapping, in mapping order (columns A-CJ).
- **`prune-empty`** — Remove empty subdirectories under each match of a pattern beneath a root.

Example:

```bash
mctutil parse xaid-log config.txt \
  --output reconstruction_log.csv
```

The CSV header uses the mapping's clear names, from `Software Release Version`
through `Internal Export Descriptor ⚠`. Values are preserved as recorded in
the config; a mapped field absent from a particular config is left blank.

To append directly to Google Sheets instead of creating a CSV:

```bash
mctutil parse xaid-log config.txt \
  --upload \
  --spreadsheet SPREADSHEET_ID \
  --sheet Reconstructions \
  --create-tab
```

The command uses `conf/gsheets_credentials.json` for the initial OAuth desktop
flow and caches the resulting token in `conf/gsheets_token.json`. Change the
directory with `--google-conf` or `MCTUTIL_GOOGLE_CONF`. The Google client
libraries are included in `environment.yml`; pip-only installations can use
`mctutil[google-sheets]`.

Uploads verify that the destination header in A1:CJ1 exactly matches the
expected reconstruction schema, then append one row using raw values. Use
`--no-verify-header` only when intentionally targeting a differently labeled
but positionally compatible tab. `--create-tab` creates the requested tab and
writes the expected header only when that tab is absent. If the tab already
exists, it is left unchanged and the normal header verification still applies.

Defaults: `prune-empty` **defaults to `--dry-run`** because `rmdir` is
destructive. `pull-config` and `scanlog-fetch` **default to `--execute`** (they
copy); pass `--dry-run` to list the planned copies instead.
