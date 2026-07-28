# mctutil mesh

Meshing helpers for igneous / Neuroglancer workflows.

Run `mctutil mesh --help` to list commands and `mctutil mesh <task> --help`
for a command's options.

## Commands

- **`build`** — Build an unsharded multiresolution mesh for a layer path (the Igneous forge plus the multiresolution merge passes).

`build` is the shared mesh path — `mctutil.shared.mesh.build_mesh` — that also
backs `transport s3-upload --mesh`. It accepts `--dry-run` to plan the passes
without loading Igneous / TaskQueue.
