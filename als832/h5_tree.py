#!/usr/bin/env python3
"""
h5_tree.py -- dump HDF5 structure: groups, datasets (shape/dtype), attributes,
and the VALUES of small datasets (size <= 64), which is where ALS/DataExchange
keeps counts and timing info (num_flat_fields, i0cycle, theta_white, etc.).

Groups named like exchange/flat/dark/white/theta, and any group containing a 3D
image stack, are fully expanded. Big image stacks just show their shape.

    python h5_tree.py file.h5 [file2.h5 ...]
"""
import sys
import h5py
import numpy as np

EXPAND_HINTS = ("exchange", "flat", "dark", "white", "bright", "theta")
MAX_LIST = 8
VALUE_MAX = 64   # print values for datasets with <= this many elements


def fmt_val(a):
    a = np.asarray(a)
    flat = a.ravel()
    out = []
    for x in flat.tolist():
        out.append(x.decode() if isinstance(x, (bytes, bytearray)) else x)
    return repr(out[0]) if a.size == 1 else repr(out)


def show_attrs(obj, indent):
    for k in obj.attrs:
        v = np.asarray(obj.attrs[k])
        s = fmt_val(v) if v.size <= VALUE_MAX else f"<{v.dtype} {v.shape}>"
        print(f"{indent}@{k} = {s}")


def has_stack(grp):
    for k in grp.keys():
        o = grp[k]
        if isinstance(o, h5py.Dataset) and o.ndim >= 3:
            return True
    return False


def is_expand(grp):
    low = grp.name.lower()
    return any(h in low for h in EXPAND_HINTS) or has_stack(grp)


def dump(grp, indent=""):
    show_attrs(grp, indent + "  ")
    ds = [k for k in grp.keys() if isinstance(grp[k], h5py.Dataset)]
    sub = [k for k in grp.keys() if isinstance(grp[k], h5py.Group)]
    expand = True  # is_expand(grp) or len(ds) <= MAX_LIST
    shown = ds if expand else ds[:2]
    for k in shown:
        d = grp[k]
        line = f"{indent}  - {k}   shape={d.shape} dtype={d.dtype}"
        if d.size <= VALUE_MAX:
            try:
                line += f"  = {fmt_val(d[()])}"
            except Exception:
                pass
        print(line)
        show_attrs(d, indent + "        ")
    if not expand and len(ds) > 2:
        print(f"{indent}  - ... ({len(ds)} datasets total; showing first 2)")
    for k in sub:
        print(f"{indent}  {k}/")
        dump(grp[k], indent + "  ")


def main():
    for path in sys.argv[1:]:
        print(f"\n===== {path} =====")
        with h5py.File(path, "r") as f:
            print("ROOT:")
            dump(f)


if __name__ == "__main__":
    main()