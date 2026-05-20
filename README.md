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

Conda-first bootstrap:

```bash
conda env create -f environment.yml
conda activate mctutil
python -m pip install --no-deps -e .
```

Notes:
- `environment.yml` is the authoritative dependency set for this repository.
- Conda-forge is the supported source for `tomopy`; plain `pip install tomopy`
  does not work.
- A few packages are still pulled through the `pip:` section because they are
  not published on conda-forge today: `cloud-volume`, `dicom2jpg`,
  `igneous-pipeline`, `neuroglancer-scripts`, and `task-queue`.
- Python indentation uses tabs in this repository.
- No autoformatter is configured at this time.
- Linting is enforced with `flake8`, the pre-commit hooks in
  `.pre-commit-config.yaml`, and `scripts/check_python_tabs.py`.
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
