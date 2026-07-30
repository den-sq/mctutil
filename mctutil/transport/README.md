# mctutil transport

Remote storage and data-movement helpers.

Run `mctutil transport --help` to list commands and
`mctutil transport <task> --help` for a command's options.

## Commands

- **`s3-upload`** — Upload a stack to an S3 bucket, optionally meshing after
  upload (`--mesh`). `--from-sharded-tree` incrementally syncs root metadata and
  declared scale directories, with optional MIP-0 exclusion. Legacy uploads
  execute by default for backward compatibility; sharded-tree uploads plan by
  default until `--execute` is supplied. `--aws-profile` overrides
  `AWS_PROFILE`; the fallback profile is `chenglab`. The same profile is passed
  to upload worker processes and optional S3 meshing.
- **`cv-fetch`** — Fetch a region of a CloudVolume URL as a stack, with MIP binning, resolution, and output-dtype control.

Both commands accept an explicit `--dry-run` to plan the transfer (and any
meshing) without mutating remote or local state.

`s3-upload` refuses legacy CloudVolume/CloudFiles AWS secret JSON files and raw
AWS access-key environment variables because those sources would override the
selected named profile inside CloudFiles.
