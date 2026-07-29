# mctutil transport

Remote storage and data-movement helpers.

Run `mctutil transport --help` to list commands and
`mctutil transport <task> --help` for a command's options.

## Commands

- **`s3-upload`** — Upload a stack to an S3 bucket, optionally meshing after
  upload (`--mesh`). `--from-sharded-tree` incrementally syncs root metadata and
  declared scale directories, with optional MIP-0 exclusion. Uses the `chenglab`
  boto profile.
- **`cv-fetch`** — Fetch a region of a CloudVolume URL as a stack, with MIP binning, resolution, and output-dtype control.

Both commands accept `--dry-run` to plan the transfer (and any meshing) without
mutating remote or local state.
