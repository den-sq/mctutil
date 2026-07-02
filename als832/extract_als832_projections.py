#!/usr/bin/env python3
"""
extract_als832_projections.py

Pull the PROJECTION stack (exchange/data) out of ALS 8.3.2 Data Exchange .h5
files into a folder of numbered TIFFs -- one folder per source file:

    <output>/<h5stem>/<h5stem>_0000.tif
                       <h5stem>_0001.tif
                       ...

Frames are read and written one at a time (the stack is often tens of GB), and
the original dtype (uint16) is preserved. Flats/darks are ignored -- use
extract_als832_refs.py for those.

USAGE
    python extract_als832_projections.py scan.h5 -o out
    python extract_als832_projections.py /data/scans -o /data/proj      # whole dir
    python extract_als832_projections.py scan.h5 -o out --step 50       # every 50th (quick peek)
    python extract_als832_projections.py scan.h5 -o out --range 0 100   # first 100 only
    python extract_als832_projections.py scan.h5 -o out --multipage     # one big multipage TIFF
    python extract_als832_projections.py scan.h5 --dry-run              # size estimate, no write
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def find_projection_ds(f):
    """Return the projection dataset (exchange/data), or None."""
    if "exchange/data" in f and isinstance(f["exchange/data"], h5py.Dataset):
        return f["exchange/data"]
    hits = []
    f.visititems(lambda n, o: hits.append(o)
                 if isinstance(o, h5py.Dataset) and n.split("/")[-1] == "data"
                 and o.ndim == 3 else None)
    return hits[0] if hits else None


def human_gb(nbytes):
    return f"{nbytes / 1e9:.2f} GB"


def process_file(path, out_root, step=1, rng=None, multipage=False, dry_run=False):
    path = Path(path)
    try:
        f = h5py.File(path, "r")
    except Exception as e:
        print(f"  [skip] cannot open {path.name}: {e}")
        return 0

    with f:
        ds = find_projection_ds(f)
        if ds is None:
            print(f"  [skip] {path.name}: no projection dataset (exchange/data)")
            return 0

        n = ds.shape[0]
        start, stop = (rng if rng else (0, n))
        stop = min(stop, n)
        indices = list(range(start, stop, step))
        frame_bytes = int(np.prod(ds.shape[1:])) * ds.dtype.itemsize
        est = len(indices) * frame_bytes
        print(f"  {path.name}: {n} projections {ds.shape[1:]} {ds.dtype}; "
              f"writing {len(indices)} frame(s) (~{human_gb(est)})"
              + (" [multipage]" if multipage else ""))
        if dry_run:
            return 0

        width = max(4, len(str(n - 1)))
        tick = max(1, len(indices) // 20)

        if multipage:
            folder = out_root
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{path.stem}_projections.tif"
            with tifffile.TiffWriter(str(target), bigtiff=True) as tw:
                for j, i in enumerate(indices):
                    tw.write(np.asarray(ds[i]), contiguous=True)
                    if j % tick == 0 or j == len(indices) - 1:
                        print(f"\r    {j + 1}/{len(indices)}", end="", flush=True)
            print(f"\n    -> {target}")
        else:
            folder = out_root / path.stem
            folder.mkdir(parents=True, exist_ok=True)
            for j, i in enumerate(indices):
                tifffile.imwrite(str(folder / f"{path.stem}_{i:0{width}d}.tif"),
                                 np.asarray(ds[i]))
                if j % tick == 0 or j == len(indices) - 1:
                    print(f"\r    {j + 1}/{len(indices)}", end="", flush=True)
            print(f"\n    -> {folder}/")
        return len(indices)


def iter_h5_inputs(inputs):
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            seen = set()
            for pat in ("*.h5", "*.hdf5", "*.he5"):
                for q in p.rglob(pat):
                    if q not in seen:
                        seen.add(q); yield q
        elif p.is_file():
            yield p
        else:
            print(f"[warn] not found: {item}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Extract ALS 8.3.2 (Data Exchange) projections to TIFF stacks.")
    ap.add_argument("inputs", nargs="+", help=".h5 files or directories")
    ap.add_argument("-o", "--output", default="als832_projections",
                    help="output root folder (default: ./als832_projections)")
    ap.add_argument("--step", type=int, default=1,
                    help="take every Nth projection (default 1 = all)")
    ap.add_argument("--range", type=int, nargs=2, metavar=("START", "STOP"),
                    default=None, help="projection index range [START, STOP)")
    ap.add_argument("--multipage", action="store_true",
                    help="write one multipage BigTIFF per file instead of a folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="print counts and size estimate; write nothing")
    args = ap.parse_args()

    files = sorted(iter_h5_inputs(args.inputs), key=natural_key)
    if not files:
        print("No .h5/.hdf5 files found.", file=sys.stderr); sys.exit(1)

    out_root = Path(args.output)
    print(f"{'Planning' if args.dry_run else 'Processing'} {len(files)} file(s)"
          + ("" if args.dry_run else f" -> {out_root}"))
    total = 0
    for path in files:
        total += process_file(path, out_root, step=args.step,
                              rng=args.range, multipage=args.multipage,
                              dry_run=args.dry_run)
    verb = "would write" if args.dry_run else "wrote"
    print(f"\nTotal: {verb} {total} projection frame(s).")


if __name__ == "__main__":
    main()