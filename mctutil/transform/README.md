# mctutil transform

TIFF-stack transforms and related local data-reshaping helpers.

Run `mctutil transform --help` to list commands and
`mctutil transform <task> --help` for a command's options.

## Commands

- **`trim`** — Crop an image stack (per-axis absolute or percentage trims).
- **`normalize`** — Normalize an image stack over a percentile value range.
- **`pipeline`** — Read a TIFF stack once into shared memory and apply the
  ordered `normalize → trim → MIP → circular mask → dtype conversion → spatial
  binning → compression/write` chain. Normalization, MIP, masking, binning, and
  nonzero trims are optional; output conversion defaults to `uint8`.
  `--mips-axis z|y|x` selects the rolling-projection dimension and
  `--bin-power N` averages `2**N`-wide XY blocks.
- **`convert`** — Convert an image stack's dtype, optionally splitting into
  horizontal sections. Use `--preserve-names --uncompressed` for the former
  dtype-only `downsample` behavior.
- **`downsample`** — Deprecated alias for filename-preserving dtype conversion;
  despite its name it performs no spatial downsampling. Existing scripts remain
  supported during the migration window. Use `pipeline --bin-power` for real
  spatial downsampling (implemented by #132).
- **`transpose`** — Transpose a reconstruction stack (`--mode shared|naive`), tracking angular vertical shift.
- **`flip`** — Flip a TIFF stack along the depth, row, or column axis.
- **`reslice`** — Write XY, XZ, and YZ TIFF slices through a stack coordinate.
- **`stack-split`** — Split a multi-page TIFF stack into one TIFF per Z slice.
- **`stitch`** — Stitch samples vertically, scanning for overlap.
- **`stitch-reconstructions`** — Join two reconstructed TIFF directories at
  explicit half-open Z cuts (`A[:a_stop] + B[b_start:]`) without registration
  or blending. Inputs are naturally ordered, output is transactionally written
  as `slice_00000.tif`, and `--dtype` uses clip-then-cast conversion without
  rescaling.
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
- **`memmap-prep`** — Stream a 3D TIFF into an uncompressed contiguous TIFF for fast memory-mapped reads.
- **`dicom-conv`** — Convert DICOM files to TIFF.
- **`df-write-tiff`** — Export TIFFs from ORS/Dragonfly objects by class+title or id (Dragonfly-only; paths via `DRAGONFLY_*` env vars).

Commands that write output accept `--dry-run` to log the planned writes instead of performing them.

The pipeline preserves the filename of each output plane. For a Z-axis MIP,
each rolling window uses the filename of its trailing input, so a width of 3
starts at the third selected filename. Spatial binning affects Y and X only;
trim and MIP update dimensions before the later operations consume them.
