#!/usr/bin/env python3
"""
extract_als832_refs.py

Scan ALS Beamline 8.3.2 microCT .h5 files (Scientific Data Exchange / DXfile
format, identified by a root `implements` dataset and an `exchange/` group) and
extract ONLY the reference frames into folders:

    <output>/gains/    each slice of  exchange/data_white   (flat / bright fields)
    <output>/darks/    each slice of  exchange/data_dark    (dark fields)

The projection stack (exchange/data) is never read, so this is fast.

A manifest.csv is written to <output>/ recording, for every extracted frame:
its source file, type, index, output path, acquisition timestamp (when
available), and mean / min / max pixel value. The mean column is the quick way
to spot flats where the sample wasn't moved out of the beam -- those frames read
noticeably darker / different from clean flats.

USAGE
    python extract_als832_refs.py /data/scans --inspect      # summarize, write nothing
    python extract_als832_refs.py /data/scans -o /data/refs  # extract frames + manifest
    python extract_als832_refs.py /data/scans -o /data/refs --manifest-only  # manifest only
    python extract_als832_refs.py a.h5 -o out --dry-run      # plan only

Frames are written as TIFFs preserving the original dtype (uint16), named
<h5stem>__white_NNN.tif / <h5stem>__dark_NNN.tif.
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import tifffile

# dataset name -> (output subfolder, short label)
REF_STACKS = {
    "data_white": ("gains", "white"),
    "data_dark":  ("darks", "dark"),
}
IMAGE_KEY = {"white": 1, "dark": 2}   # NeXus image_key convention (0=proj)


def natural_key(s):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def find_exchange(f):
    """Return the group holding data_white / data_dark, or None."""
    if "exchange" in f and isinstance(f["exchange"], h5py.Group):
        g = f["exchange"]
        if any(name in g for name in REF_STACKS):
            return g
    # fallback: search anywhere
    hits = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and any(n in obj for n in REF_STACKS):
            hits.append(obj)
    f.visititems(visit)
    return hits[0] if hits else None


def load_timestamps(f, ex, counts):
    """Map per-frame timestamps to white/dark frames via image_key + image_date.

    Returns {'white': [...], 'dark': [...]} with entries only when the count
    matches the stack size; missing/mismatched -> that label omitted.
    """
    out = {}
    # locate image_key
    image_key = None
    for path in ("exchange/image_key", "image_key"):
        if path in f:
            image_key = np.asarray(f[path][()]); break
    if image_key is None:
        return out
    # locate image_date (seconds) and optional nanoseconds
    date = ns = None
    for path in ("process/acquisition/image_date", "exchange/image_date"):
        if path in f and len(f[path]) == len(image_key):
            date = np.asarray(f[path][()]); break
    if date is None:
        return out
    for path in ("process/acquisition/image_date_ns",):
        if path in f and len(f[path]) == len(image_key):
            ns = np.asarray(f[path][()]); break

    for label, key in IMAGE_KEY.items():
        sel = image_key == key
        ts = date[sel]
        if len(ts) != counts.get(label, -1):
            continue
        nss = ns[sel] if ns is not None else None
        formatted = []
        for i, t in enumerate(ts):
            try:
                s = datetime.fromtimestamp(int(t), tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S")
                if nss is not None:
                    s += f".{int(nss[i]):09d}"
                s += "Z"
            except Exception:
                s = str(int(t))
            formatted.append(s)
        out[label] = formatted
    return out


def _read_scalar(f, *paths):
    for p in paths:
        if p in f:
            try:
                return np.asarray(f[p][()]).ravel()[0].item()
            except Exception:
                pass
    return ""


def read_flat_meta(f):
    """Read flat-field acquisition params (same for every frame in the file).

    i0_move_x / i0_move_y : stage move applied to take a flat; 0 means the
    sample was NOT moved out of the beam (a likely cause of contaminated flats).
    i0cycle : flat cadence (0 = flats only at start/end of scan).
    """
    base = "process/acquisition/flat_fields"
    if base not in f:
        hits = []
        f.visititems(lambda n, o: hits.append(n)
                      if isinstance(o, h5py.Group) and n.lower().endswith("flat_fields")
                      else None)
        base = hits[0] if hits else None
    keys = ("i0_move_x", "i0_move_y", "i0cycle")
    if base is None:
        return {k: "" for k in keys}
    return {k: _read_scalar(f, f"{base}/{k}") for k in keys}


def _read_scalar_by_leaf(f, leaf):
    """Find a dataset anywhere whose final path component == leaf; read scalar."""
    hits = []
    f.visititems(lambda n, o: hits.append(n)
                 if isinstance(o, h5py.Dataset) and n.split("/")[-1] == leaf
                 else None)
    return _read_scalar(f, hits[0]) if hits else ""


def read_camera_meta(f):
    """Read camera stage geometry (same for every frame in the file):
    camera_distance, camera_elevation, tilt_motor."""
    base = "measurement/instrument/camera_motor_stack/setup"
    keys = ("camera_distance", "camera_elevation", "tilt_motor")
    out = {}
    for k in keys:
        v = _read_scalar(f, f"{base}/{k}")
        out[k] = v if v != "" else _read_scalar_by_leaf(f, k)
    return out


def process_file(path, out_root, manifest_rows, mode="extract", dry_run=False,
                 write_frames=True):
    path = Path(path)
    try:
        f = h5py.File(path, "r")
    except Exception as e:
        print(f"  [skip] cannot open {path.name}: {e}")
        return

    with f:
        ex = find_exchange(f)
        if ex is None:
            print(f"  [skip] {path.name}: no exchange group with "
                  f"data_white/data_dark (keys: {list(f.keys())[:6]})")
            return

        counts = {label: (ex[name].shape[0] if name in ex else 0)
                  for name, (_, label) in REF_STACKS.items()}
        timestamps = load_timestamps(f, ex, counts)
        flat_meta = read_flat_meta(f)
        cam_meta = read_camera_meta(f)

        if mode == "inspect":
            proj = ex["data"].shape[0] if "data" in ex else "?"
            print(f"  exchange: projections={proj}  "
                  + "  ".join(f"{lbl}={counts[lbl]}" for lbl in ("white", "dark")))
            if "white" in counts and counts["white"]:
                w = ex["data_white"]
                means = [float(w[i].mean()) for i in range(w.shape[0])]
                lo, hi = min(means), max(means)
                print(f"    white-frame means: min={lo:.1f} max={hi:.1f} "
                      f"spread={hi - lo:.1f}")
                # flag outliers (possible sample-in-beam flats)
                med = float(np.median(means))
                odd = [i for i, m in enumerate(means) if abs(m - med) > 0.15 * med]
                if odd:
                    print(f"    !! {len(odd)} flat(s) deviate >15% from median "
                          f"(indices {odd}) -- candidate contaminated flats")
            mx, my = flat_meta.get("i0_move_x"), flat_meta.get("i0_move_y")
            if mx == 0 and my == 0:
                print("    !! i0_move_x = i0_move_y = 0 -- sample not moved out "
                      "of beam for flats (likely contaminated flatfields)")
            print(f"    flat meta: i0_move_x={mx} i0_move_y={my} "
                  f"i0cycle={flat_meta.get('i0cycle')}")
            print(f"    geometry: camera_distance={cam_meta.get('camera_distance')} "
                  f"camera_elevation={cam_meta.get('camera_elevation')} "
                  f"tilt_motor={cam_meta.get('tilt_motor')}")
            return

        for name, (subfolder, label) in REF_STACKS.items():
            if name not in ex:
                continue
            ds = ex[name]
            n = ds.shape[0]
            dest = out_root / subfolder
            need_write = write_frames and not dry_run
            if need_write:
                dest.mkdir(parents=True, exist_ok=True)
            ts = timestamps.get(label)
            width = max(3, len(str(n - 1)))
            for i in range(n):
                fname = f"{path.stem}__{label}_{i:0{width}d}.tif"
                rel = f"{subfolder}/{fname}"
                if need_write:
                    frame = np.asarray(ds[i])          # only read when writing a TIFF
                    tifffile.imwrite(str(dest / fname), frame)
                    mean, mn, mx = round(float(frame.mean()), 2), int(frame.min()), int(frame.max())
                else:
                    if dry_run:
                        print(f"    would write {dest / fname}")
                    mean = mn = mx = ""                # manifest-only: no pixel work
                manifest_rows.append({
                    "source_file": path.name,
                    "type": label,
                    "index": i,
                    "output": rel if write_frames else "",
                    "timestamp": ts[i] if ts else "",
                    "mean": mean,
                    "min": mn,
                    "max": mx,
                    "i0_move_x": flat_meta.get("i0_move_x", ""),
                    "i0_move_y": flat_meta.get("i0_move_y", ""),
                    "i0cycle": flat_meta.get("i0cycle", ""),
                    "camera_distance": cam_meta.get("camera_distance", ""),
                    "camera_elevation": cam_meta.get("camera_elevation", ""),
                    "tilt_motor": cam_meta.get("tilt_motor", ""),
                })
            dest_note = f"{subfolder}/" if write_frames else "manifest only (frames not written)"
            print(f"  {path.name}: {label} x{n} -> {dest_note}")


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
        description="Extract ALS 8.3.2 (Data Exchange) gains/darks to folders.")
    ap.add_argument("inputs", nargs="+", help=".h5 files or directories")
    ap.add_argument("-o", "--output", default="als832_refs",
                    help="output root folder (default: ./als832_refs)")
    ap.add_argument("--inspect", action="store_true",
                    help="summarize each file (incl. flat means); write nothing")
    ap.add_argument("--manifest-only", action="store_true",
                    help="write manifest.csv from metadata only; skip the gain/dark "
                         "TIFFs AND the per-frame pixel reads (mean/min/max left "
                         "blank). Fast catalog. Use --inspect or a full run for means.")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be written without writing")
    args = ap.parse_args()

    files = sorted(iter_h5_inputs(args.inputs), key=natural_key)
    if not files:
        print("No .h5/.hdf5 files found.", file=sys.stderr); sys.exit(1)

    out_root = Path(args.output)
    mode = "inspect" if args.inspect else "extract"
    write_frames = not args.manifest_only
    if mode == "inspect":
        head = "Inspecting"
    elif args.dry_run:
        head = "Planning"
    elif not write_frames:
        head = "Building manifest for"
    else:
        head = "Processing"
    print(f"{head} {len(files)} file(s)"
          + ("" if mode == "inspect" or args.dry_run else f" -> {out_root}"))

    manifest_rows = []
    for path in files:
        if mode == "inspect":
            print(Path(path).name)
        process_file(path, out_root, manifest_rows, mode=mode,
                     dry_run=args.dry_run, write_frames=write_frames)

    if mode == "extract" and not args.dry_run and manifest_rows:
        out_root.mkdir(parents=True, exist_ok=True)
        man = out_root / "manifest.csv"
        with open(man, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader(); w.writerows(manifest_rows)
        ngain = sum(r["type"] == "white" for r in manifest_rows)
        ndark = sum(r["type"] == "dark" for r in manifest_rows)
        verb = "Catalogued" if not write_frames else "Extracted"
        print(f"\n{verb} gains={ngain} darks={ndark} across {len(files)} file(s)."
              f"\nManifest: {man}")


if __name__ == "__main__":
    main()