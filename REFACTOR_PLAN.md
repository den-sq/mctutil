# mctutil — Refactor & Unification Plan

Status: substantively done. Originally drafted as a proposal from a survey of
`master`. Annotated 2026-05-21 with phase-by-phase execution status (PR
references inline). Audience: `den-sq/mctutil` maintainers.

The refactor branch is proposed for landing onto `master` in #85. Remaining
follow-ups are tracked as #86 (optional-dependency extras) and #87
(surveyed-but-unregistered commands).

## Status at a glance (2026-05-21)

- **Phase 0** — shipped in PR #48. (CI was finished separately by the
  maintainer; closes issue #13.)
- **Phase 1** — shipped in PR #49 (all §1.1 and §1.2 items).
- **Phase 2** — shipped in PR #50.
- **Phase 3** — shipped in PR #51, plus follow-up PR #58 splitting
  `meta_shift` into a generic engine + `chenglab/` adapter.
- **Phase 4** — shipped in PR #52, with two follow-ups: PR #62 made
  `pip install -e .` actually ship the leaves (top-level leaf packages
  weren't being picked up by setuptools-find), and PR #63 finished the
  long-deferred restructure — every CLI-registered leaf moved from
  top-level `<category>/<task>.py` to `mctutil/<category>/<task>.py`,
  every `sys.path.append(parents[1])` shim removed, every intra-codebase
  import rewritten to the `mctutil.*` form, and the lazy CLI registries
  updated to match. Closes issues #38, #39, #40, and #41.
- **Phase 5** — shipped across PRs #53–#56 (real tests, full
  `print` → `shared.log.log` sweep, `--dry-run` on every write-heavy
  command, restructuring `parsing/empty_dir_removal.py` into
  `parse prune-empty`), with two follow-ups: PR #66 swept the remaining
  `os.path` / `os.walk` call-sites to pathlib, and PR #68 ported
  den-sq/lftomo's `Logger` class wholesale, renamed `log.log` ->
  `log.write` across all 115 call-sites, and wired `--log-level` /
  `--quiet` / `--verbose` on the top-level `mctutil` CLI group.
- **§1.4 security follow-up** — shipped in PR #57 (deleted
  `mem/clean_shared.py`, retired its `eval(argv[1])`, merged its extra
  shm prefixes into `mem/clean.py`).
- **Issue #72 / b11_flat_handling promotion** — promoted the ALS Beamline
  8.3.2 HDF5 extractors and flat-field drift helpers into the refactored
  package layout: `mctutil/als832` exposes `extract-projections`,
  `extract-refs`, and `h5-tree`; `mctutil/flats` exposes `beam-tracking`,
  `series-digest`, and `medianize`; both groups are registered on the
  unified CLI with lazy imports, smoke tests, `--dry-run` coverage on
  write-heavy commands, and `[als832]` / `[flats]` extras in `pyproject.toml`.

Still open after the chain:

- Optional-dependencies groups in `pyproject.toml` (§3.3 / Phase 0
  partial). `[als832]` and `[flats]` are declared for the issue #72
  promotions; `[ng]`, `[sino]`, `[mesh]`, `[aws]`, `[dragonfly]` extras
  remain planned but undeclared. Runtime deps still source from
  `environment.yml` (conda). **Tracked in #86.**
- Surveyed CLI commands not yet on the unified surface: `mem list`
  (unimplemented), `parse find-errs` (source
  `mctutil/parse/find_err_general.py` exists but is unregistered), and
  `hpc cuda-check` (source `hpc_env/cuda.py` still at top level,
  unregistered). **Tracked in #87.**

Later additions (June–July 2026), on top of the phase chain:

- Issue #72 / `b11_flat_handling` promotion — `als832` + `flats` groups (#74;
  see the status-at-a-glance entry above).
- Local transform command integration (#78, #79) — added `transform flip`
  and `transform reslice`.
- `transform h5-convert`, `transform raw-convert`, and `transform stack-split`
  (#81).
- Mesh-build unification into `shared/mesh.py::build_mesh` (#37, #82; §2.2).
- Shared-memory cleanup revised to config-driven prefixes with a
  safe-by-default `--execute` / `--dry-run` pair (#64, #84; §1.4).

Intentionally deferred (per maintainer instruction):

- `hpc_work/codeclist.txt` — kept in tree.
- `parsing/meta_shift.py` repo question — answered: kept in repo behind
  the chenglab adapter (see PR #58).

## 0. Scope of survey

51 Python files, 4 213 LoC, across `hpc_env/`, `hpc_work/`, `mem/`, `ng/`,
`parsing/`, `shared/`, `transform/`, `transport/`. README is one line;
no `pyproject.toml`, no `setup.py`, no tests, no CI. All six tracked
GitHub issues are closed (last activity 2024-03), so this plan reads
"outstanding issues" as in-code defects discovered during the survey.

The good news: `shared/cli.py` already contains the seeds of a unified
parameter-type vocabulary (`Range`, `Frange`, `EnumParameter`, `NumPyType`,
`SLICE`, `OptionList`) and `shared/log.py` is a respectable logging shim.
The bad news: most scripts don't use them, several reimplement them
inline, and a non-trivial fraction of scripts are silently broken.

---

## 1. Outstanding issues (in-code, by severity)

### 1.1 Scripts that cannot run as written

All resolved.

| File | Defect | Status |
|------|--------|--------|
| `transform/mesh.py:14` | `@click.commmand` (three m's). Decorator throws on import; script unusable. | Fixed in #49. |
| `transform/upload.py:22,25` | `-s` short flag declared for both `--source-folder` and `--secret-json`. Click raises at CLI build. | Moot: `transform/upload.py` deleted in #51 (Phase 3) in favor of `transport/s3upload.py`. |
| `parsing/scanlog_fetch.py:10` | `Path("logs").mkdir(exists_ok=True)` — kwarg should be `exist_ok`. `TypeError` at first call. | Fixed in #49. |
| `transform/normalize.py:77` | `--processes` option typed as `click.Path` but used as int (default `psutil.cpu_count()`). Will accept anything; downstream usage will fail. | Fixed in #49 (now `click.INT`). |
| `transform/df_write_tiff.py:52` | `roi.getTitle()` — `roi` is the imported class, not an instance. `AttributeError`. Also hard-codes Windows-only Dragonfly paths. | Fixed in #49 (uses `source.getTitle()`); Dragonfly paths parameterized via `DRAGONFLY_DIR` / `DRAGONFLY_ORS_DIR` / `DRAGONFLY_USER_DIR` env vars. |
| `transform/hdf_convert.py:19` | `folder.glob('raw\\*.hdf')` uses a literal backslash; never matches on Linux. | Fixed in #49. |
| `transport/s3upload.py:74` | `s3.close()` — boto3 clients have no `close()` method. `AttributeError`. | Fixed in #49. |

### 1.2 Logic bugs that silently produce wrong results

All resolved.

| File | Defect | Status |
|------|--------|--------|
| `ng/layer_urlshift.py:47` | `json_data["layers"] == shifted_layers` is a comparison, not an assignment. The output JSON is the original, unmodified. | Fixed in #49. |
| `ng/layer_tag.py:81-82` | `while …: replace(t_layer, intensity=next(intensity_gen))` discards the return value of `replace`, so the loop condition never changes — infinite loop when collision occurs. | Fixed in #49 (return value captured). |
| `transform/sino_preproc.py:188-190` | `image_bounds(sino_mem)` computes `np.array([min, max])` but never returns it; callers fill `bounds` with `None`. | Moot: `sino_preproc.py` collapsed into `transform/sinogram.py` in #51 (Phase 3), behind `sino convert --mode preproc`. |
| `transform/sino_preproc.py:261` | `pool.map(image_bounds, sino_mem)` passes the `SharedNP` object as the iterable; not how the function is shaped. | Moot as above. |
| `transform/multitrim.py:62-63` | `vertical_trim` is applied to `dim[1]` (the horizontal axis) and `horizontal_trim` to `dim[0]`. Inverted. | Moot: `multitrim.py` deleted in #50 (Phase 2) in favor of the canonical `transform/trim.py`. |
| `transform/find_bounds.py:11-18` | `global count` incremented from a ThreadPool without a lock; progress display is non-deterministic (not catastrophic, but the pattern bites later). | Replaced in #55 (Phase 5): per-image `\r` print swapped for a periodic structured log line every 50 images, removing the race entirely. |
| `ng/layer_extract.py:14` | Click command function named `layer_copy` (copy-pasted). Subcommand name will collide once unified. | Fixed in #49. |
| `ng/point_shift.py:34`, `ng/point_sort.py:44` | Same problem: functions named `point_merge`. | Fixed in #49. |

### 1.3 Hardcoded local state masquerading as scripts

All resolved.

| File | Notes | Status |
|------|-------|--------|
| `transform/quick_crop.py` | `xy_crop`, `z_crop`, and `output_dir = "Octo_7_Tight_Orthocrop"` baked into module top-level. Not a tool; an artifact of one run. | Deleted in #51 (Phase 3); see canonical `transform/trim.py`. |
| `mem/from_list.py` | Hardcoded `psh01com1hcom{16..25}` node list. | Collapsed in #51 (Phase 3) into `mem/from_range.py` (CLI: `mem from-range`) with `--prefix`, `--start`, `--stop` options. |
| `mem/check_nodeinfo.py`, `mem/from_nodeinfo.py` | Hardcoded filenames `node_status.txt`/`node_list.txt`; otherwise identical code paths. | Collapsed in #51 (Phase 3) into `mem/from_file.py` (CLI: `mem from-file`) taking the node file as an argument. |
| `parsing/meta_list.py`, `parsing/meta_parser.py` | ~95 % identical 360-line Jupyter `# %%` cell scripts; only the execution preamble differs. Contains an embedded list of 11 absolute paths under `/gpfs/Labs/Cheng/phenome/COVID_Influenza_Progression/…` — should not be in a general-purpose tool repo. | Collapsed in #51 into `parsing/meta_shift.py`; the embedded path list became `--sample-list-file`. Further refactored in #58 to split the chenglab schema (path conventions, sbatch parser, sheet layout, `STATUS` enum) into a `chenglab/` adapter behind a generic engine seam. |
| `parsing/empty_dir_removal.py` | `Path("/gpfs/Labs/Cheng/phenome/")` hardcoded. | Restructured in #56 (Phase 5) into a real click command (`parse prune-empty`) that takes `ROOT` as an argument and a `--pattern` option, defaulting to `--dry-run` because `rmdir` is destructive. |
| `parsing/meta_parser.py:289-296`, `meta_list.py:289-296` | Embedded Google Sheets spreadsheet ID and sheet name `"GPFS (DEN)"`. Probably want this in config. | Moved in #51 to env-var overrides (`MCTUTIL_GSHEET_ID` / `MCTUTIL_GSHEET_SHEET`); in #58 removed from engine defaults entirely and pushed into the chenglab adapter as `default_spreadsheet` / `default_sheet` attributes. |

### 1.4 Security / footguns

All resolved.

| File | Notes | Status |
|------|-------|--------|
| `mem/clean_shared.py:22` | `eval(argv[1])` for an apply/dry-run boolean. Comment already admits this is "Very Stupid". | Resolved in #57: `mem/clean_shared.py` was deleted and its four extra prefixes were merged into the canonical cleanup path. Issue #64 subsequently moved the standard and KMP prefixes into matching JSON dictionaries selected through repeatable `--config` options, renamed the destructive flag to the safe-by-default `--execute/--dry-run` pair, and updated the sbatch consumers to use `mctutil mem clean`. |
| `transform/upload.py`, `transport/s3upload.py` | Both ingest AWS credentials in different ways (JSON file vs `boto3.Session(profile_name='chenglab')`). Pick one. Don't pass credentials via JSON if a profile already works. | Resolved in #51 (Phase 3): `transform/upload.py` deleted; `transport/s3upload.py` is the one S3 path, using the chenglab boto profile. |

### 1.5 Lint / hygiene

- Mixed tabs (most files) and 4-space (e.g., `transform/gz_strip.py`); `.flake8` already ignores `W191` so flake8 doesn't flag it, but it makes diffs noisy. **Resolved in #48 (Phase 0):** `scripts/check_python_tabs.py` rejects leading-spaces indentation; CI enforces it.
- Mixed `os.path` and `pathlib` use within single files (e.g., `transform/quick_crop.py`, `transform/upload.py`). **Resolved in #66:** the worst offenders (`quick_crop.py`, `upload.py`) were already deleted in #51; #66 swept the remaining `os.path.join` / `os.walk` call-sites in `transport/cv_import.py`, `parse/find_err_general.py`, `transport/s3upload.py`, and `hpc/timecheck.py` to pathlib. `os.environ` and `from os import PathLike` (type hints) are left alone since they have no pathlib equivalent.
- Mixed `print` and `shared.log.log` for diagnostic output. **Resolved in #55–#56 (Phase 5)** for the substitution and **#68** for the threshold plumbing: `shared/log.py` was rewritten on the den-sq/lftomo Logger class shape, every call-site was renamed from `log.log("step", ...)` to `log.write("step", ...)`, the `DEBUG` enum was replaced by a `LOG` IntFlag at `0/1/2/4/8/16/32`, and bitwise per-destination filtering was wired through `set_threshold()` / `set_screen()`. The top-level `mctutil` CLI gained `--log-level [quiet|default|verbose|debug]` plus `-q` / `-v` shorthands; the group callback applies the threshold before subcommand dispatch.
- `sys.path.append(parents[1])` boilerplate at the top of nearly every script — symptom of the lack of a real package. **Resolved in #63:** every CLI-registered leaf moved from top-level to `mctutil/<category>/<task>.py`, the shim is gone from every leaf, and intra-codebase imports use the `mctutil.*` form throughout. `chenglab/` stays at top level by design (host for the meta_shift adapter); `hpc_env/` and `hpc_work/` stay at top level as non-Python data buckets (sbatch + yaml configs).
- README is empty for practical purposes. **Resolved in #48 (Phase 0):** README rewritten with overview + install + `mctutil --help` example.

---

## 2. Duplicated implementations / divergent twins

### 2.1 Twin pairs (copy-paste evolution)

All resolved in #51 (Phase 3); `meta_shift.py` further refactored in #58.

| Pair | Diff scope | Resolution | Status |
|------|------------|------------|--------|
| `parsing/meta_list.py` ↔ `parsing/meta_parser.py` | Only execution preamble + hardcoded path list. | Collapse to one module + CLI; load sample list from a file/argument. | Shipped in #51 as `parsing/meta_shift.py` (CLI: `parse meta-shift`); embedded sample list became `--sample-list-file`. In #58 the file was further split into a generic engine + `chenglab/` adapter so the chenglab-specific schema lives behind a swappable seam. |
| `mem/check_nodeinfo.py` ↔ `mem/from_nodeinfo.py` | Filename literal (`node_status.txt` vs `node_list.txt`). | Single `mem nodes-mark` command taking a path argument. | Shipped in #51 as `mem/from_file.py` (CLI: `mem from-file`) taking the node file as an argument. |
| `transform/sinogram.py` ↔ `transform/sino_preproc.py` | ~150 lines of `weighted_normalize`, `memmap_helper`, `byteread_helper`, `distribute_read`, `sino_write`, `minmaxscale`, `remove_outlier`, `preprocess`, `sh_imread`, `FLAT` enum verbatim. `sino_preproc.py` is a stripped no-flats variant of `sinogram.py` — and the variant that omits flats is also where `image_bounds` was broken (1.2). | Make `sinogram` the one module; `--no-flats` (or `--source proj/sino`) flag selects the preproc-only path. | Shipped in #51 as `transform/sinogram.py` with `--mode full|preproc`. |
| `transform/transpose.py` ↔ `transform/f_transpose.py` | Shared-memory vs RAM-only. | Keep both behaviors, but expose them as `transform transpose --mode shared|naive` (default `shared`). `f_transpose` was a workaround; document it as such or drop. | Shipped in #51 as `transform/transpose.py` with `--mode shared|naive` (default `shared`); `f_transpose.py` deleted. |

### 2.2 Three+ implementations of the same idea

| Concept | Implementations | Pick | Status |
|---------|-----------------|------|--------|
| Crop / trim | `transform/trim.py` (CropNumber pair-type, good), `transform/multitrim.py` (axis-confused, single-float), `transform/quick_crop.py` (hardcoded run), `transform/transform.py` (string→float comma-split inline) | `transform/trim.py` is closest to right. Promote its `CropNumberType` into `shared/cli.py`; everyone else delegates to it. | Shipped in #50 (Phase 2). `multitrim.py` and `quick_crop.py` deleted; `transform/trim.py` is the canonical (CLI: `transform trim`). |
| Normalize / convert dtype | `transform/normalize.py`, `transform/transform.py::norm`, `transform/convert.py::np_convert`, `shared/cli.py::NumpyCLI.convert_ar` | `convert.np_convert` is the cleanest pure helper; promote to `shared/np_convert.py`. Pin a single `normalize` command that uses it. | Shipped in #50: `shared/np_convert.py` is the helper; `transform normalize` is the one normalize command. |
| Decompress / unzip | `transform/uncompress.py` (rewrites TIFFs uncompressed), `transform/gz_strip.py` (just renames `.gz` → strips suffix), `transform/quickgunzip.py` (real gunzip + brotli) | Three different things mislabeled with similar names. Rename: `decompress-tiff`, `strip-gz-suffix`, `gunzip`. Keep all three under one verb group with distinct subcommands. | Shipped in #52 (Phase 4): all three registered under `transform` as `decompress-tiff`, `strip-gz-suffix`, `gunzip`. |
| S3 upload | `transform/upload.py`, `transport/s3upload.py` | `transport/s3upload.py` is the more complete one (uses profile, optionally meshes after upload). Retire `transform/upload.py`. | Shipped in #51: `transform/upload.py` deleted, `transport/s3upload.py` is the single S3 path (CLI: `transport s3-upload`). |
| Mesh generation | `transform/mesh.py` (broken, has hardcoded params), `transform/mesh_ig.py` (clean), `transport/s3upload.py --mesh` (inline) | Fix `transform/mesh.py` or drop it. `mesh_ig.py` and the `s3upload --mesh` path both call the same `tc.create_meshing_tasks` API; factor into one helper. | Resolved in #37: `shared/mesh.py::build_mesh` owns the two-pass unsharded multiresolution workflow; `mesh build` and `s3-upload --mesh` both call it, the old hardcoded implementation was removed, and `build-igneous` was retired. |
| `cleanup_mem` / `exit_cleanly` | Defined verbatim in both `shared/log.py` and `shared/mem.py`. | Pick one location (`shared/mem.py` — it's the natural owner of shared-memory cleanup). `shared/log.py` should import from `mem`, not redefine. | Shipped in #50: `shared/log.py` now imports the canonical implementations from `shared/mem.py`. |

### 2.3 Repeated Click `ParamType` patterns

**Resolved in #50 (Phase 2):** `shared/cli.py::DelimitedRecord` ships as the
canonical delimiter-coerced ParamType helper, and the `ng/` + `parsing/` call
sites listed below were ported. The original survey for posterity:

Every NG/parsing/transform script reinvents the same `"foo:bar"` or
`"a,b,c"` parse-and-validate pattern:

- `transform/trim.py::CropNumberType` (int-or-float pair)
- `transform/sinogram.py::FLAT` (enum)
- `transform/stitch.py::SampleParameter` (5-tuple)
- `ng/change_color.py::ColorPairParameter` (id:hexcolor)
- `ng/layer_tag.py::TaggedLayerParameter` (name:intensity:radius)
- `ng/point_merge.py::AnnotationPairParameter` (name:direction)
- `ng/point_sort.py::AnnotationPairParameter` (copy of above)
- `ng/point_shift.py::Coordinates` (x,y,z int triple)
- `parsing/meta_*::STATUS` (enum, duplicated)

Most are essentially the same shape: split on a delimiter, coerce per
field, raise via `self.fail`. Add a small helper in `shared/cli.py` that
takes a `dataclass`/`namedtuple` and a delimiter and generates the
ParamType, then port the call sites.

---

## 3. Existing CLI surface vs proposed unified surface

### 3.1 Current command-name / file-name drift

**Resolved in #52 (Phase 4):** the unified `mctutil <category> <task>`
surface gives every leaf a consistent verb name, regardless of the underlying
file name. The drift described below was the pre-Phase-4 state on `master`.

The Click command name almost never matches the filename, and almost
never matches the directory. Sample:

```
transform/transform.py    → norm         (same name as next entry)
transform/normalize.py    → norm
transform/transpose.py    → transpose_stack
transform/f_transpose.py  → f_transpose
transform/gz_strip.py     → stripgz
transform/quickgunzip.py  → gunzip
transform/ng.py           → neuroglance
transform/simple_noise.py → simple_denoise
transform/fix_name.py     → fix_names
mem/clean.py              → memclean (group: clean, mark)
```

Two scripts ship a Click command literally named `norm`. Three scripts
have copy-paste function-name leftovers from older siblings (`layer_copy`
in `layer_extract.py`; `point_merge` in `point_shift.py` and
`point_sort.py`).

### 3.2 Proposed top-level entry point

```
mctutil <category> <task> [options]
```

with categories taken from the existing directory layout, so the
mapping stays familiar:

| Category | Tasks (proposed) | Tasks (shipped) |
|----------|------------------|-----------------|
| `transform` | `normalize`, `trim`, `transpose`, `downsample`, `channelize`, `convert`, `find-bounds`, `denoise`, `stitch`, `decompress-tiff`, `gunzip`, `strip-gz-suffix` | All shipped (#52), plus `df-write-tiff`, `dicom-conv`, `fix-name`, `hdf-convert`, and `neuroglance` (was `transform/ng.py`). |
| `sino` | `convert` (full flats), `preprocess` (no flats), `find-bounds` | `sino convert` shipped in #52 with `--mode full|preproc`; `find-bounds` lives under `transform find-bounds` instead. |
| `ng` | `layer copy`, `layer extract`, `layer tag`, `layer urlshift`, `layer recolor`, `point add`, `point merge`, `point sort`, `point shift`, `position copy`, `shift-angle`, `build` (= current `transform/ng.py`) | All shipped (#52); `ng build` wraps the former `transform/ng.py`. |
| `mesh` | `build`, `manifest` (the two-pass igneous flow) | `mesh build` is the sole command after #37 and runs the forge plus unsharded multiresolution merge passes. |
| `transport` | `s3 upload`, `cv fetch` | Shipped as `transport s3-upload` and `transport cv-fetch` (#52). |
| `mem` | `clean`, `mark`, `list` | `mem clean`, `mem mark`, `mem from-file`, `mem from-range` shipped (#52). `list` not implemented — **open**. |
| `parse` | `meta-shift` (the consolidated `meta_*.py`), `scanlog-fetch`, `pull-config`, `find-errs`, `prune-empty` | `parse meta-shift`, `parse pull-config`, `parse scanlog-fetch` shipped (#52); `parse prune-empty` shipped in #56. `find-errs` not implemented — **open**. |
| `hpc` | `cuda-check`, `time-check` | `hpc time-check` shipped (#52); `cuda-check` not registered (`hpc_env/cuda.py` is a 2-line script) — **open**. |

Implementation: one Click `Group` per category, registered into the
top-level `Group` via `add_command`. Each leaf is the existing
`@click.command()` function moved into a category module. Nothing about
the per-task logic needs to change for the CLI rewrite — only the
decorator and import location.

### 3.3 Packaging

**Resolved: option (a) shipped in #48 (Phase 0) and #52 (Phase 4).**
`pyproject.toml` declares `mctutil = "mctutil.cli:main"`; `pip install -e .`
gives users a real `mctutil` console script. The original tradeoff for
posterity:

Two equally valid options. I lean toward (a) for this size:

(a) **Single `mctutil` package, `mctutil <category> <task>` console
    script.** Add `pyproject.toml` with `[project.scripts]
    mctutil = "mctutil.cli:main"`. After `pip install -e .`, users get
    a real binary instead of `python -m transform.normalize`. Removes
    every `sys.path.append(parents[1])` block in the repo.

(b) **Per-category `console_scripts` (`mctutil-transform`,
    `mctutil-ng`, …).** Lets users discover via tab-completion but
    spreads installation surface. Worth it only if categories evolve
    into independent dependency groups (igneous, neuroglancer-scripts,
    GDAL, Dragonfly — all heavy and orthogonal).

Either way: declare optional-dependencies groups (`ng`, `sino`, `mesh`,
`dragonfly`, `aws`) so installs don't drag in GDAL+ORS for someone who
just wants `transform trim`.

Optional-dependencies groups are **not yet declared** in `pyproject.toml`;
the runtime depends on `environment.yml` (conda) for the heavy
GDAL / igneous / neuroglancer-scripts deps. Splitting these into pip
extras remains **open**.

The "remove every `sys.path.append(parents[1])`" part of option (a) is
**resolved in #63** — every CLI-registered leaf moved into
`mctutil/<category>/<task>.py` and the shim was removed from every leaf.
`chenglab/` stays at top level by design (it hosts the meta_shift adapter
and is referenced from `mctutil.parse.meta_shift`'s adapter registry);
`hpc_env/` and `hpc_work/` stay at top level as non-Python data buckets
(sbatch templates + yaml configs + the 2-line `cuda.py` probe).

---

## 4. Phased plan

Each phase is independently shippable. Order is "make it correct → make
it un-duplicated → make it unified".

### Phase 0 — Project hygiene (½ day)

**Status: shipped in #48.** The maintainer subsequently wired up the GitHub
Actions workflow (`Phase 0 CI`: lint + smoke matrix on Python 3.10 / 3.11 /
3.12) on a separate change, closing issue #13. Optional-dependencies groups
remain open — runtime deps still source from `environment.yml` (conda).

- Add `pyproject.toml` (PEP 621), declare runtime deps (click, numpy,
  tifffile, psutil, natsort, ruamel.yaml, …) and extras
  (`[ng]`, `[sino]`, `[mesh]`, `[aws]`, `[dragonfly]`).
- Replace one-line `README.md` with a real overview + install section
  + quick `mctutil --help` example.
- Pre-commit hook with `flake8` (config already in `.flake8`) and
  `ruff` (or just ruff, it can replace flake8). Enforce tabs OR spaces
  but pick one; today both exist.
- A `tests/` directory with one smoke test per command
  (`mctutil <cat> <task> --help` exits 0). This alone catches the
  `@click.commmand` typo and `-s` collision class of bugs.
- GitHub Actions workflow: lint + smoke tests on PR.

Separate commit, no logic changes.

### Phase 1 — Fix the broken-on-arrival scripts (1 day)

**Status: shipped in #49.** All checkboxes below are checked. Phase 1.4 §1
(`mem/clean_shared.py` `eval(argv[1])`) shipped separately in #57.

Strict scope: only the bugs in §1.1 and §1.2. No refactors yet.

- [x] `transform/mesh.py:14` — `@click.commmand` → `@click.command`
- [x] `transform/upload.py:25` — rename `-s` to e.g. `-k` for
      `--secret-json` (later moot: `upload.py` deleted in #51)
- [x] `parsing/scanlog_fetch.py:10` — `exists_ok` → `exist_ok`
- [x] `transform/normalize.py:77` — `click.Path` → `click.INT`
- [x] `transform/df_write_tiff.py` — either fix `roi.getTitle()` (likely
      `source.getTitle()`) and parameterize the Dragonfly path block,
      or move the file into a `dragonfly/` subdir gated behind an
      extra and skipped on import failure (chose: `source.getTitle()`
      + `DRAGONFLY_DIR` / `DRAGONFLY_ORS_DIR` / `DRAGONFLY_USER_DIR`
      env-var parameterization)
- [x] `transform/hdf_convert.py:19` — `'raw\\*.hdf'` → `'raw/*.hdf'`
- [x] `transport/s3upload.py:74` — drop `s3.close()` (or
      `del s3` if a session-clean intent existed)
- [x] `ng/layer_urlshift.py:47` — `==` → `=`
- [x] `ng/layer_tag.py:81-82` — capture `replace(...)` return value
- [x] `transform/sino_preproc.py:188-190` — return the array;
      and fix the `pool.map(image_bounds, sino_mem)` call shape
      (later moot: `sino_preproc.py` collapsed into `sinogram.py` in #51)
- [x] `transform/multitrim.py:62-63` — swap `vertical_trim` /
      `horizontal_trim` application (later moot: `multitrim.py`
      deleted in #50)
- [x] `ng/layer_extract.py`, `ng/point_shift.py`, `ng/point_sort.py` —
      rename the click command function to match the file

These are mechanical and should land as one or several focused PRs,
each with a regression test under `tests/`.

### Phase 2 — De-duplicate the helper layer (1–2 days)

**Status: shipped in #50.**

- Promote `transform/convert.py::np_convert` to `shared/np_convert.py`;
  port `transform/normalize.py`, `transform/transform.py`,
  `transform/downsample.py` to call it.
- Promote `transform/trim.py::CropNumberType` into `shared/cli.py`;
  port `transform.py`, `multitrim.py` (which then becomes redundant —
  delete it).
- Pick one location for `cleanup_mem` / `exit_cleanly` (recommend
  `shared/mem.py`); have `shared/log.py` re-export only if needed for
  ergonomics.
- Move common shared-memory readers (`byteread_helper`,
  `memmap_helper`, `distribute_read`) and the `FLAT` enum out of
  `transform/sinogram.py` into a new `shared/io_helpers.py`. Both
  sino scripts and `transform/transpose.py` then import from there.
- Add a `shared/cli.py::DelimitedRecord` helper that wraps the
  "split on delimiter, coerce per field, build dataclass/namedtuple"
  pattern. Port the 7+ ParamTypes in `ng/` to it.

Each helper move is one commit; each call-site port is one commit.
Tests added per helper.

### Phase 3 — Collapse twin scripts (1 day)

**Status: shipped in #51**, with a follow-up in #58 splitting
`parsing/meta_shift.py` into a generic engine plus a `chenglab/` adapter so
the chenglab-specific schema (folder layout, sbatch parser, `STATUS` enum,
sheet row format) lives behind a swappable seam. The `[gsheets]` extra still
isn't declared in `pyproject.toml`.

- `parsing/meta_list.py` + `parsing/meta_parser.py` → `parsing/meta_shift.py`,
  with the sample list passed as a file/CLI argument instead of
  embedded. Extract the Google Sheets bits behind an optional `[gsheets]`
  extra; spreadsheet ID/sheet via env var or config file.
- `mem/check_nodeinfo.py` + `mem/from_nodeinfo.py` → single
  `mem from-file` command taking a path argument. `mem/from_list.py`
  becomes `mem from-range` (or just an arg form of the same command).
- `transform/sinogram.py` + `transform/sino_preproc.py` → single
  `sino convert` with `--mode full|preproc` (or `--no-flats`).
- `transform/upload.py` deleted; `transport/s3upload.py` becomes the
  one S3 path.
- `transform/quick_crop.py` deleted; its hardcoded run becomes a
  worked example in README under `transform trim`.
- `transform/f_transpose.py` either deleted (if `transpose` handles
  the same case via `--mode naive`) or kept and documented.

### Phase 4 — Unified entrypoint (1–2 days)

**Status: shipped in #52** with a lazy-loading variant that imports each leaf
only when its subcommand is invoked (so `mctutil --help` doesn't pull in
igneous / GDAL / Dragonfly). Two follow-ups completed the rest:

- **PR #62** patched packaging so `pip install -e .` actually ships the
  leaf modules. The top-level leaf directories were implicit namespace
  packages (no `__init__.py`) and `pyproject.toml`'s `packages.find`
  only listed `mctutil*`, so the install registered the console script
  but the leaves it imported were missing from any non-repo cwd.
- **PR #63** finished the long-deferred leaves-under-mctutil restructure
  — every CLI-registered leaf moved from top-level to
  `mctutil/<category>/<task>.py`, every `sys.path.append(parents[1])`
  shim removed, every intra-codebase import rewritten to the `mctutil.*`
  form, every lazy CLI registry updated to match. Closes issues #38,
  #39, #40, #41.

Subsequent additions: `parse prune-empty` (#56), `parse meta-shift --schema`
(#58).

- Create `mctutil/__init__.py` and `mctutil/cli.py` with the top-level
  Click group.
- For each leaf command, move it into `mctutil/<category>/<task>.py`
  and register via `add_command`. The scripts retain their bodies; the
  only edit is removing the `sys.path.append` shim and updating
  imports to `from mctutil.shared import …`.
- Keep the old top-level scripts as thin shims for one minor release
  (`python -m mctutil.transform.normalize` continues to work) — or
  delete outright if there are no external callers; ask the
  maintainers. (If they say "no external callers" then delete cleanly,
  which matches the user's "avoid backwards-compatibility hacks
  unless explicitly required" preference.)
- Document `mctutil --help` and `mctutil <category> --help` in README,
  with examples for the most common verbs.

### Phase 5 — Coverage & polish (open-ended)

**Status: shipped across #53–#56**, with two follow-ups: **#66** swept the
remaining `os.path` / `os.walk` call-sites to pathlib and **#68** rewrote
`shared/log.py` on the den-sq/lftomo `Logger` class shape (renaming
`log.log("step", ...)` to `log.write("step", ...)` across all 115 call-sites
and wiring `--log-level` / `--quiet` / `--verbose` through to the top-level
CLI group).

- [x] Real tests for the math-heavy verbs (`normalize`, `trim`, `sino
  convert`) against small fixture tiffs. **Shipped in #53.** Test count
  grew from 0 → 70 across Phase 0–5; 77 after #58.
- [x] Replace `print` calls with `shared.log.log` everywhere, so
  `--quiet` / `--log-level` work uniformly. **Shipped in #55 + #56** for
  the print substitution, **#68** for the threshold plumbing. The
  rewritten `shared/log.py` now honors a per-destination `LOG` IntFlag
  bitmask via `set_threshold()` / `set_screen()`; the top-level CLI's
  `--log-level [quiet|default|verbose|debug]` (with `-q` / `-v`
  shorthands) applies the threshold before subcommand dispatch.
- [x] Add a `--dry-run` flag to anything that writes files. **Shipped
  across #54, #55, #56, with the final `mem clean` consistency update
  in #64.** Coverage:
  `transform trim`, `transform normalize`, `sino convert`,
  `transform decompress-tiff`, `transform hdf-convert`,
  `transform stitch`, `transform channelize`, `transform df-write-tiff`,
  `mesh build`, `transport s3-upload`, `transport cv-fetch`,
  `parse pull-config`, `parse scanlog-fetch`, `parse prune-empty`,
  `mem from-file`, `mem from-range`, `mem clean`, and `mem mark`.
  Shared-memory cleanup defaults to dry-run because unlinking is
  destructive.
- [x] Consider whether `parsing/meta_shift.py` even belongs in this repo
  vs the chenglab automation; if it's Cheng-Lab specific, move it.
  **Answered in #58:** kept in repo, but the chenglab schema lives in a
  dedicated top-level `chenglab/` adapter behind a generic engine seam;
  a second lab adapter can land without touching the engine.
- [ ] Decide what to do with `hpc_work/codeclist.txt` (a 449-line ffmpeg
  codec dump that has no consumer in the repo) — almost certainly
  delete. **Deferred per maintainer instruction:** stays in tree.

---

## 5. Risk notes

These remain relevant for any future restructuring.

- The shared-memory machinery in `shared/mem.py` initializes a global
  `mem_tracker` at import time. Any refactor that changes import order
  (e.g., moving things into a package) must preserve "the first import
  of anything from `mctutil` allocates this segment". Worth a test that
  asserts the tracker exists post-import. (#63 moved this into
  `mctutil/shared/mem.py` and the existing tests passed unchanged, but
  the invariant still applies to any future restructure.)
- `transform/sinogram.py` is the most subtle code in the repo; tread
  carefully when factoring out `distribute_read`. The function's
  `generate_offset_pairs_*` closures depend on `target_mem[…].buffer_address.start`
  which is itself non-trivial — keep its tests close. (Phase 2 #50
  moved the helpers into `shared/io_helpers.py` and the closures still
  work; the warning carries forward.)
- The Google Sheets path in `parsing/meta_*.py` has a real credential
  shape (`creds`, `GSCOPES`, token files). Whoever owns that flow should
  confirm whether anything still calls it before removing it. (After #58
  this lives in `parsing/meta_shift.py` engine + `chenglab/meta_shift.py`
  adapter; still active.)

---

## 6. Open questions for maintainers

1. Are any external scripts / sbatch files calling `python
   transform/<x>.py` directly? If yes, Phase 4 needs a deprecation
   period; if no, do it as a hard cut. **Answered (hard cut taken in
   #63):** the only in-tree direct invocations found were
   `mem/memcheck.sbatch` / `mem/memclean.sbatch` calling
   `clean_shared.py`, which were updated in #57 to `clean.py [--apply]
   clean`. #63 then moved every CLI-registered leaf into
   `mctutil/<category>/<task>.py` and deleted the top-level
   `transform/`, `ng/`, `parsing/`, `mem/`, `transport/`, `shared/`
   directories outright. Any out-of-tree caller that imported a leaf by
   its old top-level path now needs to switch to either the installed
   console script (`mctutil <verb>`) or the `mctutil.<category>.<task>`
   import. #64 completed the in-tree migration to `mctutil mem clean
   --dry-run|--execute`.
2. Is `parsing/meta_*.py` still in active use, or a snapshot of a
   one-time migration? If snapshot, archive and remove. **Answered:**
   active. Kept in tree; #58 split the chenglab schema into a dedicated
   `chenglab/` adapter behind a generic engine seam.
3. `transform/df_write_tiff.py` is Windows-only and depends on a
   specific Dragonfly install. Is it actually a CLI tool or a script
   that gets pasted into the Dragonfly console? That changes whether
   it belongs under `mctutil` at all. **Answered in #49:** kept under
   `transform/` and registered as `transform df-write-tiff`; Dragonfly
   paths now read from `DRAGONFLY_DIR` / `DRAGONFLY_ORS_DIR` /
   `DRAGONFLY_USER_DIR` env vars instead of being hardcoded.
4. Where do `mem/clean.py`'s sbatch templates run today, and against
   which scheduler version? **Answered in #64:** `mem mark` no longer
   embeds a cluster environment or log directories. Callers can provide
   cluster shell setup with `--job-preamble` and opt into Slurm log
   locations with `--sbatch-output` / `--sbatch-error`. The shipped
   sbatch consumers likewise assume only that the installed `mctutil`
   entrypoint is on `PATH`.

---

## 7. TL;DR

Original plan (preserved for context):

- Ship Phase 0 + Phase 1 as a single bug-fix PR; the seven
  broken-on-arrival scripts and ten silent-bug scripts get fixed and
  smoke-tested.
- Then de-duplicate helpers (Phase 2) and collapse twin scripts
  (Phase 3) in small, focused PRs.
- Then introduce `mctutil <category> <task>` as a proper console
  script (Phase 4), at which point every file's name lines up with
  its CLI verb and the per-file `sys.path.append` block disappears.
- Phase 5 is the long tail: tests, logging consistency, and trimming
  files that have no business being in a shared toolbox.

What actually happened (2026-05-21):

- Phases 0–5 shipped as a linear PR chain (#48 → #56), with four
  follow-ups: #57 retired `clean_shared.py`'s `eval(argv[1])`, #58 split
  `meta_shift` into an engine + chenglab adapter, #62 patched packaging
  so `pip install -e .` actually shipped the leaves, #63 moved every
  CLI-registered leaf into `mctutil/<category>/<task>.py` while removing
  every `sys.path.append` shim, #66 swept the remaining `os.path` /
  `os.walk` call-sites to pathlib, and #68 ported den-sq/lftomo's
  `Logger` class wholesale (renaming `log.log` -> `log.write` across all
  115 call-sites and wiring `--log-level` / `--quiet` / `--verbose` on
  the top-level CLI).
- The maintainer separately wired up the Phase 0 CI workflow (lint +
  smoke matrix on Python 3.10 / 3.11 / 3.12), closing issue #13.
- Remaining concrete follow-up: optional-dependencies groups in
  `pyproject.toml`.
- `hpc_work/codeclist.txt` decision was: keep, per maintainer
  instruction.
