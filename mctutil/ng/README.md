# mctutil ng

Neuroglancer JSON, layer, and annotation helpers.

Run `mctutil ng --help` to list commands and `mctutil ng <task> --help` for a
command's options.

## Commands

- **`build`** — Build a Neuroglancer precomputed volume (image or segmentation) from a stack.
- **`layer-copy`** — Merge annotation layers from one Neuroglancer JSON into another, writing a new file.
- **`layer-extract`** — Extract layers from a Neuroglancer JSON into a new file.
- **`layer-recolor`** — Recolor the segment colors of a named annotation.
- **`layer-tag`** — Tag annotation layers, applying a default segmentation radius where none is set.
- **`layer-urlshift`** — Revert layer naming to a scheme CloudVolume can still handle.
- **`point-add`** — Add a series of points as a new annotation.
- **`point-merge`** — Merge annotations from two Neuroglancer JSONs into one.
- **`point-shift`** — Shift all annotations in a Neuroglancer JSON by a fixed amount.
- **`point-sort`** — Sort annotation(s) along an axis (0–2 for X/Y/Z).
- **`position-copy`** — Copy position and orientation from one Neuroglancer JSON to another.
- **`shift-angle`** — Set a Neuroglancer JSON to a consistent forward-facing angled orientation.

Note: `ng build` wraps the former `transform/ng.py` neuroglance command.
