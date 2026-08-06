# mctutil

General tools for working with MicroCT data at PSU.

## What it is

`mctutil` is a grab-bag repository of Cheng Lab scripts for transforming
MicroCT image stacks, building Neuroglancer-friendly artifacts, moving data to
remote storage, and doing a small amount of HPC-side housekeeping.

The staged cleanup in [REFACTOR_PLAN.md](REFACTOR_PLAN.md) is substantially
complete: every tool now lives in a single installable `mctutil` package and is
exposed through a unified `mctutil <category> <task>` console script. A
follow-up remains open for a handful of surveyed-but-unregistered commands
(#87).

## Installation

Conda-first bootstrap:

```bash
conda env create -f environment.yml
conda activate mctutil
python -m pip install --no-deps -e .
```

Notes:
- `environment.yml` is the authoritative dependency set for this repository.
- Core pip metadata carries the NumPy 1.x runtime contract; supported installs
  use `numpy>=1.24,<2` with TomoPy 1.x.
- Conda-forge is the supported source for `tomopy`; plain `pip install tomopy`
  does not work.
- A few packages are still pulled through the `pip:` section because they are
  not published on conda-forge today: `cloud-volume`, `dicom2jpg`,
  `igneous-pipeline`, `neuroglancer-scripts`, and `task-queue`.
- Optional-dependency extras are declared for the heavy, orthogonal stacks:
  `[als832]`, `[flats]`, `[ng]`, `[serve]`, `[sino]`, `[mesh]`, `[aws]`, and
  `[dragonfly]`, so e.g. `python -m pip install -e .[mesh]` pulls only the
  mesh dependencies. The extras are complementary to `environment.yml`, not
  a replacement: conda-only packages stay in the conda env (`tomopy` for the
  sinogram/recon stack), and `[dragonfly]` is intentionally empty because
  ORS Dragonfly is Windows-only and not published on PyPI.
- The Tifffile/Zarr compatibility line is intentionally bounded to Tifffile
  before its Zarr-3/Python-3.11 cutover and to Zarr 2 (2.18 or newer). See
  [DEPENDENCIES.md](DEPENDENCIES.md) for the update policy and smoke checks.
- Python indentation uses tabs in this repository.
- No autoformatter is configured at this time.
- Linting is enforced with `flake8`, the pre-commit hooks in
  `.pre-commit-config.yaml`, and `scripts/check_python_tabs.py`.

## Quickstart

`pip install -e .` installs a real `mctutil` console script. The command surface
is `mctutil <category> <task>`:

```bash
mctutil --help
mctutil transform --help
mctutil transform trim --help
mctutil transform normalize --help
mctutil transform pipeline --help
mctutil sino convert --help
mctutil ng point-add --help
mctutil serve ng --help
mctutil mesh build --help
mctutil transport s3-upload --help
mctutil mem clean --help
mctutil parse meta-shift --help
```

Read one HDF5 dataset, or recursively read every dataset below a group:

```bash
mctutil als832 h5-tree scan.h5 \
  --path measurement/instrument/detector/actual_pixel_size
mctutil als832 h5-tree scan.h5 \
  --path measurement/instrument/detector \
  --path measurement/instrument/setup
```

`h5-tree` opens source files read-only. Selected datasets containing more than
10,000 values are not loaded unless `--max-values` is raised or set to `0`.

A worked trim example (equivalent to the old hardcoded `quick_crop` shape):

```bash
mctutil transform trim \
  --data-dir /path/to/projections \
  --output-dir /path/to/projections-tight \
  --vertical-trim 421,21 \
  --horizontal-trim 551,389 \
  --z-trim 803,0
```

Write-heavy commands accept `--dry-run`. Shared-memory cleanup
(`mctutil mem clean`) defaults to dry-run because unlinking is destructive; pass
`--execute` to actually unlink.

Verbosity is controlled at the top level: `--log-level [quiet|default|verbose|debug]`,
with `-q` / `-v` shorthands.

## Categories

Each category has its own README with the full command breakdown; run
`mctutil <category> --help` for the live task list:

- [`transform`](mctutil/transform/README.md) — TIFF-stack transforms (trim,
  normalize, single-pass pipeline, transpose, convert, downsample, find-bounds, denoise, stitch,
  stitch-reconstructions, decompress-tiff / strip-gz-suffix / gunzip,
  hdf-convert / h5-convert / raw-convert, stack-split, …)
- [`sino`](mctutil/sino/README.md) — sinogram conversion
  (`sino convert --mode full|preproc`)
- [`ng`](mctutil/ng/README.md) — Neuroglancer JSON, layer, and annotation
  helpers, plus `ng build`
- [`serve`](mctutil/serve/README.md) — local data and visualization servers
  (`serve ng`)
- [`mesh`](mctutil/mesh/README.md) — Igneous mesh generation (`mesh build`)
- [`transport`](mctutil/transport/README.md) — S3 / CloudVolume data movement
  (`s3-upload`, `cv-fetch`)
- [`mem`](mctutil/mem/README.md) — shared-memory cleanup (`clean`, `mark`) and
  node-list submission (`from-file`, `from-range`)
- [`parse`](mctutil/parse/README.md) — metadata / config / scanlog parsing
  (`find-errs`, `meta-shift`, `pull-config`, `scanlog-fetch`, `prune-empty`)
- [`hpc`](mctutil/hpc/README.md) — HPC scheduler-side helpers (`time-check`)
- [`als832`](mctutil/als832/README.md) / [`flats`](mctutil/flats/README.md) —
  ALS Beamline 8.3.2 HDF5 extractors and flat-field drift helpers

## Repository layout

- `mctutil/` — the installable package; one module per category with leaves at
  `mctutil/<category>/<task>.py`, and shared parameter types / logging / memory
  helpers in `mctutil/shared/`
- `chenglab/` — Cheng-Lab schema adapter sitting behind the generic
  `parse meta-shift` engine
- `hpc_env/`, `hpc_work/` — non-Python data buckets (sbatch templates, yaml
  configs)
- `tests/` — per-command smoke tests and math-heavy fixture tests
- `scripts/` — repository hygiene (e.g. `check_python_tabs.py`)

## Refactor plan

See [REFACTOR_PLAN.md](REFACTOR_PLAN.md) for the full staged-cleanup record —
phase by phase, with per-file defect → resolution tables.
