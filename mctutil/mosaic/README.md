# mosaic

Register overlapping tomography scans and write the stitch layout that maps a
common canvas onto every scan.

Inputs are per-scan reconstructions stored as row stacks: one `NNNNN.tif`
square slice per detector row, centred on the rotation axis. Scans fall into
groups that share a lateral frame (same mount, vertical stage moves only). One
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

Conventions: a depth link's T means *moving row = fixed row + T*; in-plane
transforms follow OpenCV's `getRotationMatrix2D` sign about the slice centre
`((n - 1) / 2, (n - 1) / 2)`.

## Layout file

```json
{
  "canvas": [H, W], "nz": NZ, "offset_yx": [oy, ox], "slice_size": n, "fov_radius": R,
  "fit": {...},
  "scans": [{"name", "dir", "nrow", "A": [[...4], [...4], [...4]], "lowpass_weight", "group"?, "off"?}]
}
```

`A` maps canvas `(g, y, x, 1)` to scan `(row, y, x)`. Reference scans have an
integer depth translation; fixed-group scans carry the full affine map of the
fitted relation.
