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
- **`scan-log`** — Import canonical scan records from ALS832, Sigray, APS 7-BM,
  or ChengLab Camera into CSV or Google Sheets.
- **`reconstruction-log`** — Import canonical reconstruction records from
  ALS832, X-AID, or 7-BM/TomoCuPy into CSV or Google Sheets.
- **`scanlog-fetch`** — Copy scanlogs into a target location.
- **`sigray-scan-log`** — Extract one concise scan row per numbered Sigray
  projection HDF5 acquisition. FLAT, DARK, and POST files enrich those rows but
  do not become scan rows.
- **`xaid-log`** — Convert a MITOS X-AID reconstruction `config.txt` into one
  88-column CSV row. Each configured value is written to its clear-name field
  from the X-AID field mapping, in mapping order (columns A-CJ).
- **`prune-empty`** — Remove empty subdirectories under each match of a pattern beneath a root.

Example:

```bash
mctutil parse sigray-scan-log /mnt/e/20260514_sample \
  --output scans.csv

mctutil parse xaid-log config.txt \
  --output reconstruction_log.csv
```

The canonical commands can append directly to Google Sheets through the same
destination implementation used by the legacy commands:

```bash
mctutil parse scan-log /data/aps-7bm \
  --source aps-7bm \
  --upload \
  --spreadsheet SPREADSHEET_ID \
  --sheet Scans \
  --create-tab

mctutil parse reconstruction-log /data/reconstructions \
  --source tomocupy-7bm \
  --upload \
  --spreadsheet SPREADSHEET_ID \
  --sheet Reconstructions \
  --create-tab
```

Use `--output FILE.csv` instead of `--upload` for local output. Uploads match
canonical columns by exact header name, so destination columns may be reordered
and unrelated columns are preserved. Credentials default to
`~/.creds/gsheets`; `MCTUTIL_GSHEET_ID`, `MCTUTIL_GSHEET_SHEET`, and
`MCTUTIL_GOOGLE_CONF` provide the established environment defaults. For a
ChengLab import, `--input-spreadsheet` and `--input-sheet` select the source;
`--spreadsheet` and `--sheet` select the independent destination.

`sigray-scan-log` accepts one or more HDF5 files or acquisition directories.
Directory discovery inspects direct children only so it does not traverse large
TIFF reconstruction trees; pass multiple directories when needed. It recognizes
numbered projection names such as `sample_000.h5` and writes a separate row for
every FOV acquisition, even when several acquisitions share a containing
folder. `Projection Data File` stores the portable HDF5 basename and can be
matched to the basename of X-AID's `Input Projection Data File`;
`Acquisition Group` records the parent folder.

The initial reader intentionally targets the observed Sigray/Data Exchange
layout and requires exact HDF5 paths. It reads projection shape and metadata,
theta, completion flags, timestamps, and detector IDs without loading the
projection image array. Stage positions are per-file medians. The status and
warning fields expose incomplete frames, frame-averaging-aware detector-ID
gaps, invalid theta, missing metadata, stage motion, and uncertain references.
`Dropped Frames` counts raw detector IDs missing beyond the normal
`Exposures per Projection` stride.
External POST references are associated by detector shape and sample stage;
FLAT and DARK files may be shared across the acquisition group. Ambiguous
matches are left blank and reported rather than inferred from filename suffixes.

The canonical scan header contains 42 machine-derived fields. User-managed
Sheet columns such as Project, Sample ID, Stain, Operator, and Notes can remain
alongside them: header-name upload mode leaves extra columns unwritten.

To append scan rows directly to Google Sheets:

```bash
mctutil parse sigray-scan-log /mnt/e/20260514_sample \
  --upload \
  --spreadsheet SPREADSHEET_ID \
  --sheet Scans \
  --create-tab
```

The Sigray and X-AID commands share the same OAuth cache, create-tab behavior,
exact header-name matching, strict-order option, and explicit positional mode.
Pip-only installs can use `mctutil[sigray]` for CSV extraction and add
`mctutil[google-sheets]` for uploads; the full conda environment includes both.

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

Uploads match the 88 mapped fields to exact names in the destination's live
header row, then append one row using raw values in that Sheet's column order.
Columns may be reordered, and extra leading, middle, or trailing columns are
left unwritten. Missing or duplicate mapped field names stop the upload before
anything is appended; names containing `⚠` must match exactly.

Pass `--strict-header-order` to require the canonical A1:CJ1 header order. Use
`--no-verify-header` only for an intentionally positional upload; it skips the
header read and writes the canonical 88-value order to A:CJ. These two options
cannot be combined. `--create-tab` creates the requested tab in canonical
mapping order only when that tab is absent. If the tab already exists, it is
left unchanged and the selected header behavior still applies.

Defaults: `prune-empty` **defaults to `--dry-run`** because `rmdir` is
destructive. `pull-config` and `scanlog-fetch` **default to `--execute`** (they
copy); pass `--dry-run` to list the planned copies instead.
