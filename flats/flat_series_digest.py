"""
flat_series_digest.py
=====================
Compact a long series of background flat-field frames (e.g. 178 GB acquired over
~10 min while the beam drifts) into a small, shareable digest that PRESERVES the
beam movement, so the drift can be tracked algorithmically afterwards.

Why sampling + a *small* median (not block-averaging)
-----------------------------------------------------
The beam is moving throughout the series. Averaging a large contiguous block of
frames into one digest frame would blend many different beam positions together,
smearing the instantaneous beam shape and biasing the result toward the block
mean -- which is especially damaging when the drift is NON-linear. That destroys
exactly the information we want to measure.

Instead we SAMPLE the series at regular intervals and, at each sample point, take
a MEDIAN of only a few frames (3-5) centred on that point. A short median removes
true noise (readout noise, Poisson, cosmic-ray "zingers") without averaging over
enough time to move the beam appreciably, so each digest frame is a clean,
undistorted snapshot of the beam at that instant.

Because we only read the sampled frames, this touches a few hundred frames out of
the whole series rather than reading all 178 GB.

Outputs
-------
  digest_stack.tif     : `keep` clean beam snapshots (one per sampled interval).
  drift_trajectory.csv : per-snapshot frame index, normalised intensity, beam
                         centroid (row,col) and centroid drift vs the first
                         snapshot -- a compact record of the motion itself.

Usage
-----
  python flat_series_digest.py FRAMES  --keep 120 --median-window 5 --bin 4
  python flat_series_digest.py FRAMES  --crop 0,3500,0,6352
FRAMES may be a directory of single-frame TIFFs (sorted by name) or one
multi-page TIFF stack.
"""
from __future__ import annotations
import argparse
import glob
import os
import numpy as np
import tifffile


# --------------------------------------------------------------------------- #
#  Random-access frame reader
# --------------------------------------------------------------------------- #
class FrameSource:
    """Reads individual frames on demand from either a directory of single-frame
    TIFFs (sorted by filename) or a single multi-page TIFF stack.

    Random access (read one frame by index) is what lets us sample a few hundred
    frames out of a huge series without ever loading the whole thing."""

    def __init__(self, path: str):
        if os.path.isdir(path):
            self.frame_paths = sorted(glob.glob(os.path.join(path, "*.tif")) +
                                      glob.glob(os.path.join(path, "*.tiff")))
            if not self.frame_paths:
                raise SystemExit(f"no TIFF frames found in directory {path}")
            self.n_frames = len(self.frame_paths)
            self._stack = None
        else:                                   # one multi-page stack
            self._stack = tifffile.TiffFile(path)
            self.frame_paths = None
            self.n_frames = len(self._stack.pages)

    def read(self, index: int) -> np.ndarray:
        """Return frame `index` as a float32 image."""
        if self.frame_paths is not None:
            return tifffile.imread(self.frame_paths[index]).astype(np.float32)
        return self._stack.pages[index].asarray().astype(np.float32)


# --------------------------------------------------------------------------- #
#  Per-frame helpers
# --------------------------------------------------------------------------- #
def beam_centroid(frame: np.ndarray):
    """Intensity-weighted centroid (row, col) and total intensity of a frame.
    The centroid is a robust, single-number-per-axis proxy for beam position;
    total intensity tracks flux (e.g. an energy ramp)."""
    img = np.clip(frame, 0, None).astype(np.float64)
    total_intensity = img.sum()
    if total_intensity <= 0:
        return np.nan, np.nan, 0.0
    n_rows, n_cols = img.shape
    centroid_row = (img.sum(axis=1) @ np.arange(n_rows)) / total_intensity
    centroid_col = (img.sum(axis=0) @ np.arange(n_cols)) / total_intensity
    return centroid_row, centroid_col, total_intensity


def spatial_bin(frame: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by an integer factor using a factor x factor box mean."""
    if factor <= 1:
        return frame
    n_rows, n_cols = frame.shape
    n_rows -= n_rows % factor           # trim to a multiple of `factor`
    n_cols -= n_cols % factor
    return (frame[:n_rows, :n_cols]
            .reshape(n_rows // factor, factor, n_cols // factor, factor)
            .mean(axis=(1, 3)))


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="directory of TIFF frames OR one multi-page stack")
    ap.add_argument("--keep", type=int, default=120,
                    help="number of snapshots in the digest (= number of sample "
                         "intervals). Default 120.")
    ap.add_argument("--median-window", type=int, default=5, dest="median_window",
                    help="frames to median around each sample centre (3-5 "
                         "recommended; odd is best). Kills noise without smearing "
                         "beam motion. Default 5.")
    ap.add_argument("--bin", type=int, default=1,
                    help="spatial downsample factor (box mean). Default 1 (none).")
    ap.add_argument("--crop", type=str, default=None,
                    help="row0,row1,col0,col1 crop applied to every frame.")
    ap.add_argument("--out", type=str, default=".", help="output directory.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    source = FrameSource(args.path)
    n_total = source.n_frames
    window = max(1, args.median_window)
    half_window = window // 2
    if args.crop:
        row0, row1, col0, col1 = map(int, args.crop.split(","))

    # ----- choose sample centres: the midpoint of each of `keep` equal-length
    #       intervals spanning the whole series --------------------------------
    interval_edges = np.linspace(0, n_total, args.keep + 1)
    sample_centres = ((interval_edges[:-1] + interval_edges[1:]) / 2).astype(int)
    print(f"{n_total} frames -> {args.keep} snapshots "
          f"(interval {n_total / args.keep:.1f} frames), "
          f"median window {window} frames")

    # ----- build one clean snapshot per sample centre -------------------------
    snapshots = []            # the digest frames
    trajectory = []           # (centre_index, total_intensity, cen_row, cen_col)
    for centre in sample_centres:
        # frame indices in the median window, clipped to the valid range so the
        # windows at the very start/end simply use fewer frames
        lo = max(0, centre - half_window)
        hi = min(n_total, centre + half_window + 1)

        window_frames = []
        for idx in range(lo, hi):
            frame = source.read(idx)
            if args.crop:
                frame = frame[row0:row1, col0:col1]
            window_frames.append(frame)

        # small median -> removes zingers / noise, negligible beam motion across
        # these few adjacent frames, so the beam shape stays sharp
        snapshot = np.median(np.stack(window_frames), axis=0)
        snapshot = spatial_bin(snapshot, args.bin)
        snapshots.append(snapshot.astype(np.float32))

        cen_row, cen_col, total_intensity = beam_centroid(snapshot)
        trajectory.append((centre, total_intensity, cen_row, cen_col))

    # ----- write the drift trajectory CSV -------------------------------------
    trajectory = np.array(trajectory, float)
    intensity_norm = trajectory[:, 1] / trajectory[:, 1].max()
    drift_row = trajectory[:, 2] - trajectory[0, 2]      # vs first snapshot
    drift_col = trajectory[:, 3] - trajectory[0, 3]
    csv_table = np.column_stack([trajectory[:, 0], intensity_norm,
                                 trajectory[:, 2], trajectory[:, 3],
                                 drift_row, drift_col])
    csv_path = os.path.join(args.out, "drift_trajectory.csv")
    np.savetxt(csv_path, csv_table, delimiter=",",
               header="frame_index,intensity_norm,centroid_row,centroid_col,"
                      "drift_row,drift_col", comments="")

    # linearity check: residual of a straight-line fit to the vertical drift
    line = np.polyval(np.polyfit(csv_table[:, 0], drift_row, 1), csv_table[:, 0])
    lin_resid = float(np.std(drift_row - line))
    print(f"wrote {csv_path}")
    print(f"  total drift (row,col) : {drift_row[-1]:+.2f}, {drift_col[-1]:+.2f} px")
    print(f"  intensity change      : {(intensity_norm[-1]-intensity_norm[0])*100:+.1f} %")
    print(f"  vertical-drift linearity residual : {lin_resid:.3f} px "
          f"({'NON-linear -> needs a schedule, not just 2 endpoints' if lin_resid > 0.5 else 'roughly linear'})")

    # ----- write the digest stack --------------------------------------------
    digest_path = os.path.join(args.out, "digest_stack.tif")
    tifffile.imwrite(digest_path, np.stack(snapshots))
    size_mb = os.path.getsize(digest_path) / 1e6
    print(f"wrote {digest_path}  ({len(snapshots)} frames, {size_mb:.1f} MB)")
    print("Share digest_stack.tif and/or drift_trajectory.csv -- both are small.")


if __name__ == "__main__":
    main()
