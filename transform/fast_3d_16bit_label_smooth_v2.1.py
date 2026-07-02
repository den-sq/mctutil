#!/usr/bin/env python3
"""
fast_3d_label_smooth.py – v3.3  (2025-07-16)

Ultra-fast majority/modal smoothing for 16-bit 3-D label volumes.

New flag
---------
--series      Write each smoothed Z-slice as an individual TIFF inside
              <input>_smooth_r<R>_series/ instead of one stack file.

Example
-------
python fast_3d_label_smooth.py "Brachial Nerves 5.6um on 2.8um resample" \
       --backend gpu --radius 3 --series

Dependencies
------------
tifffile  numba  (optional CuPy + CUDA for GPU)
"""

from __future__ import annotations
import argparse
import math
import re
import time
from pathlib import Path
from typing import List, Iterable

import numpy as np
import tifffile as tiff

# ───────────────────────────── CPU (Numba) ──────────────────────────────
try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range        # type: ignore

if njit:
    @njit(parallel=True, fastmath=True, cache=True)
    def _majority_cpu(data: np.ndarray, radius: int) -> np.ndarray:
        nz, ny, nx = data.shape
        out = np.empty_like(data)
        hist = np.zeros(65536, np.uint16)
        for z in prange(nz):
            for y in range(ny):
                for x in range(nx):
                    touched, best_cnt, best_lab = [], 0, 0
                    for dz in range(-radius, radius + 1):
                        zz = min(max(z + dz, 0), nz - 1)
                        for dy in range(-radius, radius + 1):
                            yy = min(max(y + dy, 0), ny - 1)
                            for dx in range(-radius, radius + 1):
                                xx = min(max(x + dx, 0), nx - 1)
                                lab = data[zz, yy, xx]
                                c = hist[lab] + 1
                                hist[lab] = c
                                if c == 1:
                                    touched.append(lab)
                                if c > best_cnt:
                                    best_cnt, best_lab = c, lab
                    out[z, y, x] = best_lab
                    for lab in touched:
                        hist[lab] = 0
        return out
else:
    def _majority_cpu(*_a, **_kw):  # type: ignore
        raise RuntimeError("Numba not installed; CPU backend disabled.")

# ───────────────────────────── GPU (CuPy) ───────────────────────────────
try:
    import cupy as cp
except ImportError:
    cp = None                                               # type: ignore

_GPU_KERNEL_CACHE: dict[int, "cp.RawKernel"] = {}
_GPU_TEMPLATE = r"""
extern "C" __global__
void majority3d(const unsigned short* __restrict__ vol,
                unsigned short* __restrict__ out,
                const int nx, const int ny, const int nz)
{
    const int R = {RADIUS};
    const int win = 2*R + 1;
    const int size = win*win*win;               // ≤343 when R≤3

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = blockIdx.z*blockDim.z + threadIdx.z;
    if (x>=nx || y>=ny || z>=nz) return;

    unsigned short labels[size], uniq[size], counts[size];
    unsigned short ulen = 0, idx = 0;

    for (int dz=-R; dz<=R; ++dz) {
        int zz = min(max(z + dz,0), nz - 1);
        for (int dy=-R; dy<=R; ++dy) {
            int yy = min(max(y + dy,0), ny - 1);
            for (int dx=-R; dx<=R; ++dx) {
                int xx = min(max(x + dx,0), nx - 1);
                labels[idx++] = vol[(zz*ny + yy)*nx + xx];
            }
        }
    }
    for (int i=0;i<size;++i){
        unsigned short v=labels[i]; bool found=false;
        for(int j=0;j<ulen;++j){ if(uniq[j]==v){counts[j]++;found=true;break;} }
        if(!found){ uniq[ulen]=v; counts[ulen]=1; ++ulen; }
    }
    unsigned short best_l=uniq[0], best_c=counts[0];
    for(int j=1;j<ulen;++j){ if(counts[j]>best_c){best_c=counts[j]; best_l=uniq[j];}}
    out[(z*ny + y)*nx + x]=best_l;
}
"""


def _compile_gpu_kernel(radius: int):
    if radius not in _GPU_KERNEL_CACHE:
        src = _GPU_TEMPLATE.replace("{RADIUS}", str(radius))
        _GPU_KERNEL_CACHE[radius] = cp.RawKernel(src, "majority3d",
                                                 backend="nvcc", options=("-O3",))
    return _GPU_KERNEL_CACHE[radius]


def _majority_gpu(arr: "cp.ndarray", radius: int):
    ker = _compile_gpu_kernel(radius)
    nz, ny, nx = arr.shape
    out = cp.empty_like(arr)
    blk = (8, 8, 8)
    grd = (math.ceil(nx / blk[0]), math.ceil(ny / blk[1]), math.ceil(nz / blk[2]))
    ker(grd, blk, (arr, out, nx, ny, nz))
    return out


# ───────────────────────────── helpers ──────────────────────────────────
def _detect_backend(req: str) -> str:
    if req != "auto":
        return req
    return "gpu" if cp and cp.cuda.runtime.getDeviceCount() else "cpu"


def _process_block(block: np.ndarray, radius: int, backend: str) -> np.ndarray:
    if backend == "gpu":
        cup = cp.asarray(block, dtype=cp.uint16)
        return cp.asnumpy(_majority_gpu(cup, radius))
    return _majority_cpu(block, radius)


_TIF_RE = re.compile(r"\.tif{1,2}$", re.I)


def _sorted_tiffs(dirp: Path) -> List[Path]:
    files = sorted(p for p in dirp.iterdir() if _TIF_RE.search(p.name))
    if not files:
        raise FileNotFoundError(f"No TIFFs in {dirp}")
    return files


def _load_slice(f: Path) -> np.ndarray:
    return tiff.imread(f).astype(np.uint16, copy=False)


def _iter_dir(dirp: Path, chunk: int) -> Iterable[np.ndarray]:
    files = _sorted_tiffs(dirp)
    for i in range(0, len(files), chunk):
        grp = files[i:i + chunk]
        yield np.stack([_load_slice(f) for f in grp], axis=0), grp


# ───────────────────────────── I/O core ─────────────────────────────────
def _process_input(in_path: Path, out_path: Path, radius: int,
                   backend: str, chunk: int, series: bool):
    if in_path.is_dir():
        files = _sorted_tiffs(in_path)
        total, processed = len(files), 0
        print(f"[fast-smooth] detected {total} slice files → streaming.")
        if series:
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            writer = tiff.TiffWriter(out_path, bigtiff=True)
        try:
            for block, names in _iter_dir(in_path, chunk):
                smooth = _process_block(block, radius, backend)
                if series:
                    for slc, fname in zip(smooth, names):
                        dst = out_path / f"{Path(fname).stem}_smooth.tif"
                        tiff.imwrite(dst, slc, compression="zlib")
                        processed += 1
                else:
                    writer.write(smooth, compression="zlib")
                    processed += block.shape[0]
                print(f"[fast-smooth] processed {processed}/{total} slices…",
                      end="\r", flush=True)
        finally:
            if not series:
                writer.close()
        print()
    else:
        vol = tiff.imread(in_path).astype(np.uint16, copy=False)
        if vol.ndim == 2:
            raise ValueError("Need 3-D stack, got 2-D.")
        smooth = _process_block(vol, radius, backend)
        if series:
            out_path.mkdir(parents=True, exist_ok=True)
            for i, slc in enumerate(smooth):
                dst = out_path / f"slice_{i:05d}.tif"
                tiff.imwrite(dst, slc, compression="zlib")
        else:
            tiff.imwrite(out_path, smooth, compression="zlib")


# ───────────────────────────── CLI ──────────────────────────────────────
def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Majority smoothing for 16-bit 3-D label volumes.")
    p.add_argument("input",
                   help="3-D TIFF *or* directory of 2-D TIFF slices")
    p.add_argument("output", nargs="?", default=None,
                   help="Explicit output path (file or dir). "
                        "If omitted, a sensible default is chosen.")
    p.add_argument("-r", "--radius", type=int, default=3, choices=range(1, 14),
                   help="Cubic radius (1–3) [default 3]")
    p.add_argument("-c", "--chunk", type=int, default=128,
                   help="Z-slice chunk size when streaming [default 128]")
    p.add_argument("-b", "--backend", default="auto",
                   choices=("auto", "gpu", "cpu"),
                   help="Force backend; default auto-detect")
    # ── new behaviour ────────────────────────────────────────────────────
    p.add_argument("--stack", action="store_true",
                   help="Write ONE multi-page TIFF instead of the default "
                        "slice-series directory")
    return p.parse_args()


def main() -> None:
    a = _cli()
    backend = _detect_backend(a.backend)
    in_p = Path(a.input)

    # Decide where output goes
    if a.output:
        out_p = Path(a.output)
    else:
        stem = in_p.name.rstrip("/\\")
        if a.stack:
            # single-file name
            out_p = in_p.parent / f"{stem}_smooth_r{a.radius}.tif"
        else:
            # directory for the slice series
            out_p = in_p.parent / f"{stem}_smooth_r{a.radius}_series"

    print(f"[fast-smooth] backend  = {backend.upper()}")
    print(f"[fast-smooth] output   = "
          f"{'stack file' if a.stack else 'slice series'} → {out_p}")

    t0 = time.perf_counter()
    _process_input(in_p, out_p, a.radius, backend, a.chunk, series=not a.stack)
    print(f"[fast-smooth] total time {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
