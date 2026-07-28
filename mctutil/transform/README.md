# mctutil transform

TIFF-stack transforms and related local data-reshaping helpers.

Run `mctutil transform --help` to list commands and
`mctutil transform <task> --help` for a command's options.

## Commands

- **`trim`** — Crop an image stack (per-axis absolute or percentage trims).
- **`normalize`** — Normalize an image stack over a percentile value range.
- **`convert`** — Convert an image stack's dtype, optionally splitting into horizontal sections.
- **`downsample`** — Downsample an image stack, with output-dtype control.
- **`transpose`** — Transpose a reconstruction stack (`--mode shared|naive`), tracking angular vertical shift.
- **`flip`** — Flip a TIFF stack along the depth, row, or column axis.
- **`reslice`** — Write XY, XZ, and YZ TIFF slices through a stack coordinate.
- **`stack-split`** — Split a multi-page TIFF stack into one TIFF per Z slice.
- **`stitch`** — Stitch samples vertically, scanning for overlap.
- **`channelize`** — Write channelized (multi-channel) TIFF output.
- **`denoise`** — Block-based denoising by intensity-difference threshold.
- **`find-bounds`** — Scan a TIFF stack for global min/max intensity bounds.
- **`fix-name`** — Zero-pad the numeric suffix in `prefix_N` filenames to five digits.
- **`decompress-tiff`** — Rewrite every TIFF under a path with compression removed.
- **`gunzip`** — Decompress gzipped input files.
- **`strip-gz-suffix`** — Strip the `.gz` suffix from filenames without decompressing.
- **`hdf-convert`** — Convert HDF (usually HDF4) files to TIFF.
- **`h5-convert`** — Export image-like datasets from an HDF5 file as TIFF stacks.
- **`raw-convert`** — Convert a raw 3D image volume to a TIFF stack or per-Z folder.
- **`dicom-conv`** — Convert DICOM files to TIFF.
- **`df-write-tiff`** — Export TIFFs from ORS/Dragonfly objects by class+title or id (Dragonfly-only; paths via `DRAGONFLY_*` env vars).

Commands that write output accept `--dry-run` to log the planned writes instead of performing them.
