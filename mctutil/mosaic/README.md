# mosaic

Register overlapping tomography scans, write the stitch layout that maps a
common canvas onto every scan, then stitch and denoise the canvas.

Inputs are per-scan reconstructions stored as row stacks: one TIFF square slice
per detector row, centred on the rotation axis. Input stacks use the shared
slice discovery: immediate `.tif` and `.tiff` files (case-insensitive) in natural
order, such as `slice_000.tif`, `slice_001.tif`, … or `slice_2.tif` before
`slice_10.tif`. Row numbers are zero-based positions in that list, not numbers
parsed from filenames. Names that have the same natural key, such as `slice_1.tif`
and `slice_01.tiff`, are rejected as ambiguous. This also applies to stitched
stacks read by `window`, `to-u16`, and `wiener`; provide the complete canvas stack
when filtering a span with `--g0`/`--g1`. Stitch and Wiener outputs retain
`NNNNN.tif` names, and `to-u16` retains its input filenames.

Scans fall into groups that share a lateral frame (same mount, vertical stage
moves only). One
group is the *reference*: its frame becomes the canvas. A second, *fixed* group
may be related to it by a rigid 3-D transform: a rotation `phi` about the axis,
a lateral shift that drifts linearly with depth, and a depth tilt.

`cv2` (OpenCV) and `tomopy` come from the conda runtime (`environment.yml`); the
`mosaic` extra provides `h5py`, `scipy`, and `tifffile`.

## Workflow

| Step | Command | Output |
|---|---|---|
| Coarse volumes | `mosaic coarse SCAN.h5 OUT.npy --center C` | binned gridrec of each DXchange scan |
| Coarse links | `mosaic register MANIFEST OUT.json --scale S` | phi and (dz, dy, dx) per scan pair, with significance |
| Same-group depth | `mosaic link-depth FIXED MOVING OUT.jsonl --fixed-row R --guess T --radius R` | row offset T per sampled block (appended) |
| Cross-group in-plane | `mosaic refine FIXED MOVING OUT.json ROWS... --dz --phi --dy --dx` | per-row phi, scale, dz, dy, dx |
| Cross-group depth | `mosaic tilt-points CONFIG OUT.json --radius R` | lateral d(Z) fit and sub-block depth points |
| Joint fit | `mosaic fit TILT OUT.json --links ... --reference ... --fixed ...` | group offsets and the depth relation |
| Layout | `mosaic layout FIT MANIFEST OUT.json --fov-radius R` | canvas size and per-scan 3x4 maps |
| Radial profile | `mosaic profile RECON OUT.npy --r-lo R --r-ref R` | per-scan mean radial profile and FOV-edge excess (layout `radcorr`) |
| Intensity maps | `mosaic norm-fit LAYOUT --reference S --pair M:R ... --fov-feather R_IN R_OUT` | per-scan `a`, `b` in the layout |
| Stitch | `mosaic stitch LAYOUT OUT --fov-feather R_IN R_OUT [--noise-out N.npy]` | float32 canvas slices and HP noise maps |
| Spectra | `mosaic spectra FULL.npy NOISY.npy OUT.json` | per-scan signal/noise spectra (layout `snr`) |
| Wiener | `mosaic wiener LAYOUT STITCHED NOISE.npy OUT --window LO HI` | uint16 canvas slices |
| Window | `mosaic window STITCHED`, `mosaic to-u16 STITCHED OUT --window LO HI` | display percentiles; uint16 stack |

Stitching splits each scan into high and low bands at `--sigma-lp`. A fixed-group
scan is one with an `off` field (written by `mosaic layout`): `--hp-inner` moves
its high-band handover inside the reference FOV edge and `--lp-inner` does the
same for the reference low band. Noise maps carry the reference and fixed-group
parts separately, each scan contributing a^2 * sum(nps) of its `snr` spectra, and
the Wiener filter mixes the two groups' spectra by them. A map whose lateral terms
`A[1:, 0]` are nonzero moves the scan's FOV with depth; the stitcher follows it.
In every uint16 output, 0 means no scan covers the pixel.

Conventions: a depth link's T means *moving row = fixed row + T*; in-plane
transforms follow OpenCV's `getRotationMatrix2D` sign about the slice centre
`((n - 1) / 2, (n - 1) / 2)`.

## Layout file

```json
{
  "canvas": [H, W], "nz": NZ, "offset_yx": [oy, ox], "slice_size": n, "fov_radius": R,
  "fit": {...},
  "scans": [{"name", "dir", "nrow", "A": [[...4], [...4], [...4]], "lowpass_weight", "group"?, "off"?,
             "a"?, "b"?, "radcorr"?, "snr"?}]
}
```

`A` maps canvas `(g, y, x, 1)` to scan `(row, y, x)`. Reference scans have an
integer depth translation; fixed-group scans carry the full affine map of the
fitted relation.
