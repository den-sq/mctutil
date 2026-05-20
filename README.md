# mctutil

General tools for working with MicroCT data at PSU.

## What it is

`mctutil` is a grab-bag repository of Cheng Lab scripts for transforming
MicroCT image stacks, building Neuroglancer-friendly artifacts, moving data to
remote storage, and doing a small amount of HPC-side housekeeping.

The codebase is in the middle of a staged cleanup. Phase 0 adds packaging,
smoke tests, CI, and contributor hygiene without renaming the existing module
layout yet.

## Installation

Core Poetry install:

```bash
poetry install --with dev
```

Optional extras for heavier stacks:

```bash
poetry install --with dev -E aws -E cloud -E dicom -E gdal -E gsheets -E hpc -E mesh -E ng -E sino
```

Notes:
- Python indentation uses tabs in this repository.
- No autoformatter is configured at this time.
- Linting is enforced with `flake8`, the pre-commit hooks in
  `.pre-commit-config.yaml`, and `scripts/check_python_tabs.py`.
- `transform/transform.py` also needs TomoPy. Poetry is now the dependency
  manager, but `poetry add tomopy` still fails for the same reason: TomoPy is
  not published on a standard Python package index that Poetry can resolve in
  this environment. It remains documented instead of declared so the base
  install and CI stay usable.
- The `mctutil` console script is intentionally a stub for now. The current
  command surface still lives at `python -m <module>` from a repo checkout;
  Phase 4 will unify it under `mctutil <category> <task>`.

## Quickstart

Most tools are still run as module entrypoints:

```bash
python -m transform.trim --help
python -m transform.normalize --help
python -m ng.point_add --help
python -m transport.s3upload --help
python -m mem.clean --help
```

The placeholder future entrypoint is already reserved:

```bash
mctutil --help
```

## Project map

- `transform/` — TIFF stack transforms, normalization, trimming, reconstruction helpers
- `ng/` — Neuroglancer JSON and point/layer helpers
- `transport/` — data movement helpers for S3 / CloudVolume workflows
- `mem/` — HPC memory cleanup helpers
- `shared/` — shared click parameter types, logging, and memory helpers

## Refactor plan

See [REFACTOR_PLAN.md](REFACTOR_PLAN.md) for the staged cleanup plan that Phase
0 is implementing.
