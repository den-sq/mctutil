# mctutil — Sharded Neuroglancer Port Plan

## Status

**Design / breakout complete; implementation not started.** Tracking issue #90
(review-and-breakout) plus six shippable follow-ups (#92–#97) are filed, with the
perf-architecture and resume design captured on them. Nothing is coded yet.
Depends on the refactor landing (#85 — merged) and the optional-dependency extras
(#86 — open). Companion to [REFACTOR_PLAN.md](REFACTOR_PLAN.md).

## Why

`mctutil ng build` (`neuroglancer-scripts`) is sequential, single-process, and
**unsharded** — one chunk file per chunk, so a large CT volume becomes millions
of tiny files: slow to write, brutal to upload (one S3 PUT per chunk), slow to
serve. The lab has a much faster Neuroglancer conversion chain (CloudVolume +
igneous, **sharded** output) that we want as a first-class `mctutil` capability
rather than a set of hand-run scripts.

## Source of the fast pipeline

`ChengLabResearch/ReconChengLab`, `Neuroglancer/` subdirectory (@ `d8e7d8a`).
Orchestrator: `NG_Precompute_Shard_pipeline_v0.05_MIP0_v1.9_memmap_singles.sh`.
Six stages, resumable at `START_AT=all|stage|upload`:

1. **memmap-prep** — `convert_to_memmappable.py`: stream any TIFF (via
   `tif.aszarr()`, no full-RAM load) into a contiguous uncompressed BigTIFF that
   `tifffile.memmap()` can open. Standalone; no CloudVolume.
2. **precompute (MIP 0)** — `STEP_1_...memmap_singles.py`: `ProcessPoolExecutor`,
   one worker per Z, each with its own `CloudVolume(parallel=False)` + `memmap`,
   chunk `[512,512,1]`, write `VOL[:,:,z:z+1,0:1]` (chunk-aligned, contention-free).
3. **downsample pyramid** — `igneous image downsample --volumetric` (64³ then 16³
   extend passes).
4. **sharded staging** — `igneous image xfer --sharded` per mip (96³ for mips ≤2,
   64³ for 3–4, 16³ for ≥5). **The new capability — mctutil has no sharded
   producer today.**
5. **upload** — `aws s3 cp --recursive` / `cloudfiles cp` to a configurable
   prefix (`s3://<deployment-bucket>/…`).
6. **mesh** (segmentation only) — `igneous mesh forge` + `mesh merge --nlod`.

Legacy variants `ct_to_ng.py` / `ct_to_ng_.py` are **superseded** by STEP_1 + the
`.sh`; do not port off them.

## Why it's faster (must-preserve properties)

Ranked. `ng build` is slow because it is the inverse of each:

1. **Sharded output** — packs many chunks into a handful of indexed shard files
   per resolution; collapses local file count and S3 object count by orders of
   magnitude. Biggest lever for anything cloud-hosted.
2. **Process-parallel, chunk-aligned MIP0 writes** — disjoint Z per process, no
   locks / read-modify-write / GIL contention; `parallel=False` per CloudVolume
   to avoid N×M thread oversubscription.
3. **Deferred compression** — MIP0 written raw (`compress=False`); brotli applied
   later, in parallel, by igneous.
4. **Memmap source** — page-cache-shared across processes, no per-slice decode.
5. **Igneous task queue** (`-p N`, `--memory`, `--volumetric`) vs
   neuroglancer-scripts' inline `compute_scales`.

Any port that "simplifies" one of these away loses the speed it exists for.

## What mctutil already has (reuse)

- **`mesh build`** (`mctutil/shared/mesh.py::build_mesh`) = igneous
  `create_meshing_tasks` + `create_unsharded_multires_mesh_tasks(num_lod=…)` —
  exactly the pipeline's `mesh forge` + `mesh merge --nlod`. Takes any cloudpath.
  **No mesh port needed.** (One empirical check: meshing against a *sharded*
  source volume — igneous is nominally source-agnostic.)
- **`sino convert`** uses `mctutil/shared/io_helpers.py::distribute_read` — a
  raw-offset shared-memory read (`page.dataoffsets[0]` + `Pool`). This is the
  fast ingest a ported `ng precompute` needs; `memmap-prep` is the front-door
  that manufactures the uncompressed-contiguous format it requires. See #91.
- **`transport s3-upload` / `cv-fetch`** — the S3 / CloudVolume surface to extend.

## Proposed shape

Map each pipeline stage to a `mctutil` command. Dependency order = shippable order.

| Stage | Command (issue) | Notes |
|---|---|---|
| memmap-prep | `transform memmap-prep` (#92) | port `convert_to_memmappable.py`; self-contained; the normalize-to-memmappable front-door |
| precompute | `ng precompute` (#93) | CloudVolume-backed TIFF→precomputed (STEP_1); coexist with `ng build` initially, not an internal swap |
| downsample | `ng downsample-pyramid` (#94) | `create_downsampling_tasks`, two-pass |
| shard | `ng shard` (#95) | `create_transfer_tasks(sharded=True)`, per-mip chunk map — the headline new capability |
| upload | extend `transport s3-upload` (#96) | sharded-tree aware; parallel per-mip; `aws s3 sync` |
| mesh | existing `mesh build` | no port; orchestrator calls it |
| orchestrate | `ng publish` (#97) | chains the above + optional `ng serve` / `validate` / `http-check` |

## Resume architecture (verified)

Verified against installed `igneous-pipeline` / `task-queue` / `cloud-volume` in a
local venv:

- **igneous downsample/xfer do NOT skip already-written chunks** — no CLI flag,
  and the task `execute()` bodies unconditionally download→compute→write.
  Re-running recomputes and overwrites (idempotent, not skip).
- **`task-queue`'s `FileQueue` IS durable + crash-resumable** — purpose-built for
  HPC: durable task files `queue/{lease-ts}--{uuid}.json`, time-limited
  lease / renew / delete-ack, recirculating (a crashed task's lease expires and
  it is re-picked-up).

So the resume design is **two-tier**:

- **Orchestrator (`ng publish`)**: per-dataset state file
  `<dataset>/.mctutil_ng_publish.json` (last completed stage + input hash) plus
  explicit `--start-at {prep,precompute,downsample,shard,upload,mesh}`. Lets an
  interruption skip finished datasets and resume the in-flight one.
- **igneous stages (downsample / shard / mesh)**: **persistent FileQueue** —
  point `--queue` at a durable directory and **do not `rm -rf` it between runs**
  (the source `.sh` does, discarding this). `igneous execute` drains only the
  remaining tasks = automatic task-level crash resume.
- **precompute (#93)**: custom `ProcessPoolExecutor`, not a queue → **per-Z
  existence check** (skip a Z whose chunk already exists).
- **upload (#96)**: `aws s3 sync` (not `cp --recursive`; no `--delete`).
- **memmap-prep (#92)**: skip if a valid output already exists.

## Porting hazards

- STEP_1's committed `__main__` **hardcodes an absolute input path** and ignores
  the argparse it declares (the `.sh` passes real flags — the argparse is what to
  port; the `__main__` needs rewiring).
- Resolution `[700, 700, 700]` and voxel offset `[0, 0, 0]` are hardcoded in
  `CloudVolume.create_new_info` → must become CLI options.
- The default S3 prefix is baked into the `.sh` → make it CLI-configurable.
- Every CloudVolume-touching script carries a `cloudfiles.monitoring...end_io`
  null-interval monkey-patch → port once into a shared shim.

## Open design questions

- **Replace vs augment `ng build`?** Recommend *coexist* initially — a new
  `ng precompute` (CloudVolume) beside the existing neuroglancer-scripts
  `ng build`; revisit consolidation once the CloudVolume path has miles.
  Tradeoff: two backends to maintain vs a risky in-place swap.
- **Share the ingest core with `sino`? (#91)** The raw-offset shared-memory read
  is common to `sino convert` and a future `ng precompute`. Recommend factoring
  the *primitive* (offset computation + `readinto`-into-shared-memory), not the
  whole stage (sino's is coupled to sinogram ordering / flat handling). Probably
  after #93 lands, so it is factored against two real callers rather than one.
- **Extras packaging (#86).** `ng precompute` / shard / upload need the
  `[ng]` / `[aws]` / `[mesh]` extras (cloud-volume, igneous, boto3).

## What's not decided

- Exact CLI verb names (`ng precompute` vs `ng build --backend cloudvolume`, etc.).
- Whether to port the optional serve / validate helpers now or defer them.
- cloudfiles-path resume for upload (its sync equivalent vs an existence check).
- Whether the `ng publish` state file records content hashes for staleness
  detection or just stage completion.

## Suggested next steps

1. **#86** — declare the `[ng]` / `[aws]` / `[mesh]` extras (unblocks installs).
2. **#92** — `transform memmap-prep` (self-contained; no CloudVolume; foundational).
3. **#93** — `ng precompute` (CloudVolume backend; coexist with `ng build`; per-Z
   resume; reuse / refactor the `sino` ingest primitive per #91).
4. **#94 / #95** — downsample + shard (persistent FileQueue; per-mip skip).
5. **#96** — sharded-tree upload (`aws s3 sync`, parallel per-mip).
6. **#97** — `ng publish` orchestrator (state file + `--start-at`; calls existing
   `mesh build`); optional serve / validate helpers last.

## Pointers

- Tracking + breakout: #90 — has the pipeline overview, per-file classification,
  perf-architecture note, and resume-design comments.
- Stages: #92–#97. Consolidation scan: #91. Extras: #86. Refactor landing: #85.
- Source: `ChengLabResearch/ReconChengLab`, `Neuroglancer/` @ `d8e7d8a`.
- Reuse: `mctutil/shared/mesh.py::build_mesh`,
  `mctutil/shared/io_helpers.py::distribute_read`, `mctutil/transform/ng.py`.
