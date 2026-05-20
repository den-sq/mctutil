# mctutil — Refactor & Unification Plan

Status: proposal, drafted from a survey of `master` at the time of writing.
Audience: `den-sq/mctutil` maintainers.

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

| File | Defect |
|------|--------|
| `transform/mesh.py:14` | `@click.commmand` (three m's). Decorator throws on import; script unusable. |
| `transform/upload.py:22,25` | `-s` short flag declared for both `--source-folder` and `--secret-json`. Click raises at CLI build. |
| `parsing/scanlog_fetch.py:10` | `Path("logs").mkdir(exists_ok=True)` — kwarg should be `exist_ok`. `TypeError` at first call. |
| `transform/normalize.py:77` | `--processes` option typed as `click.Path` but used as int (default `psutil.cpu_count()`). Will accept anything; downstream usage will fail. |
| `transform/df_write_tiff.py:52` | `roi.getTitle()` — `roi` is the imported class, not an instance. `AttributeError`. Also hard-codes Windows-only Dragonfly paths. |
| `transform/hdf_convert.py:19` | `folder.glob('raw\\*.hdf')` uses a literal backslash; never matches on Linux. |
| `transport/s3upload.py:74` | `s3.close()` — boto3 clients have no `close()` method. `AttributeError`. |

### 1.2 Logic bugs that silently produce wrong results

| File | Defect |
|------|--------|
| `ng/layer_urlshift.py:47` | `json_data["layers"] == shifted_layers` is a comparison, not an assignment. The output JSON is the original, unmodified. |
| `ng/layer_tag.py:81-82` | `while …: replace(t_layer, intensity=next(intensity_gen))` discards the return value of `replace`, so the loop condition never changes — infinite loop when collision occurs. |
| `transform/sino_preproc.py:188-190` | `image_bounds(sino_mem)` computes `np.array([min, max])` but never returns it; callers fill `bounds` with `None`. |
| `transform/sino_preproc.py:261` | `pool.map(image_bounds, sino_mem)` passes the `SharedNP` object as the iterable; not how the function is shaped. |
| `transform/multitrim.py:62-63` | `vertical_trim` is applied to `dim[1]` (the horizontal axis) and `horizontal_trim` to `dim[0]`. Inverted. |
| `transform/find_bounds.py:11-18` | `global count` incremented from a ThreadPool without a lock; progress display is non-deterministic (not catastrophic, but the pattern bites later). |
| `ng/layer_extract.py:14` | Click command function named `layer_copy` (copy-pasted). Subcommand name will collide once unified. |
| `ng/point_shift.py:34`, `ng/point_sort.py:44` | Same problem: functions named `point_merge`. |

### 1.3 Hardcoded local state masquerading as scripts

| File | Notes |
|------|-------|
| `transform/quick_crop.py` | `xy_crop`, `z_crop`, and `output_dir = "Octo_7_Tight_Orthocrop"` baked into module top-level. Not a tool; an artifact of one run. |
| `mem/from_list.py` | Hardcoded `psh01com1hcom{16..25}` node list. |
| `mem/check_nodeinfo.py`, `mem/from_nodeinfo.py` | Hardcoded filenames `node_status.txt`/`node_list.txt`; otherwise identical code paths. |
| `parsing/meta_list.py`, `parsing/meta_parser.py` | ~95 % identical 360-line Jupyter `# %%` cell scripts; only the execution preamble differs. Contains an embedded list of 11 absolute paths under `/gpfs/Labs/Cheng/phenome/COVID_Influenza_Progression/…` — should not be in a general-purpose tool repo. |
| `parsing/empty_dir_removal.py` | `Path("/gpfs/Labs/Cheng/phenome/")` hardcoded. |
| `parsing/meta_parser.py:289-296`, `meta_list.py:289-296` | Embedded Google Sheets spreadsheet ID and sheet name `"GPFS (DEN)"`. Probably want this in config. |

### 1.4 Security / footguns

| File | Notes |
|------|-------|
| `mem/clean_shared.py:22` | `eval(argv[1])` for an apply/dry-run boolean. Comment already admits this is "Very Stupid". |
| `transform/upload.py`, `transport/s3upload.py` | Both ingest AWS credentials in different ways (JSON file vs `boto3.Session(profile_name='chenglab')`). Pick one. Don't pass credentials via JSON if a profile already works. |

### 1.5 Lint / hygiene

- Mixed tabs (most files) and 4-space (e.g., `transform/gz_strip.py`); `.flake8` already ignores `W191` so flake8 doesn't flag it, but it makes diffs noisy.
- Mixed `os.path` and `pathlib` use within single files (e.g., `transform/quick_crop.py`, `transform/upload.py`).
- Mixed `print` and `shared.log.log` for diagnostic output.
- `sys.path.append(parents[1])` boilerplate at the top of nearly every script — symptom of the lack of a real package.
- README is empty for practical purposes.

---

## 2. Duplicated implementations / divergent twins

### 2.1 Twin pairs (copy-paste evolution)

| Pair | Diff scope | Resolution |
|------|------------|------------|
| `parsing/meta_list.py` ↔ `parsing/meta_parser.py` | Only execution preamble + hardcoded path list. | Collapse to one module + CLI; load sample list from a file/argument. |
| `mem/check_nodeinfo.py` ↔ `mem/from_nodeinfo.py` | Filename literal (`node_status.txt` vs `node_list.txt`). | Single `mem nodes-mark` command taking a path argument. |
| `transform/sinogram.py` ↔ `transform/sino_preproc.py` | ~150 lines of `weighted_normalize`, `memmap_helper`, `byteread_helper`, `distribute_read`, `sino_write`, `minmaxscale`, `remove_outlier`, `preprocess`, `sh_imread`, `FLAT` enum verbatim. `sino_preproc.py` is a stripped no-flats variant of `sinogram.py` — and the variant that omits flats is also where `image_bounds` was broken (1.2). | Make `sinogram` the one module; `--no-flats` (or `--source proj/sino`) flag selects the preproc-only path. |
| `transform/transpose.py` ↔ `transform/f_transpose.py` | Shared-memory vs RAM-only. | Keep both behaviors, but expose them as `transform transpose --mode shared|naive` (default `shared`). `f_transpose` was a workaround; document it as such or drop. |

### 2.2 Three+ implementations of the same idea

| Concept | Implementations | Pick |
|---------|-----------------|------|
| Crop / trim | `transform/trim.py` (CropNumber pair-type, good), `transform/multitrim.py` (axis-confused, single-float), `transform/quick_crop.py` (hardcoded run), `transform/transform.py` (string→float comma-split inline) | `transform/trim.py` is closest to right. Promote its `CropNumberType` into `shared/cli.py`; everyone else delegates to it. |
| Normalize / convert dtype | `transform/normalize.py`, `transform/transform.py::norm`, `transform/convert.py::np_convert`, `shared/cli.py::NumpyCLI.convert_ar` | `convert.np_convert` is the cleanest pure helper; promote to `shared/np_convert.py`. Pin a single `normalize` command that uses it. |
| Decompress / unzip | `transform/uncompress.py` (rewrites TIFFs uncompressed), `transform/gz_strip.py` (just renames `.gz` → strips suffix), `transform/quickgunzip.py` (real gunzip + brotli) | Three different things mislabeled with similar names. Rename: `decompress-tiff`, `strip-gz-suffix`, `gunzip`. Keep all three under one verb group with distinct subcommands. |
| S3 upload | `transform/upload.py`, `transport/s3upload.py` | `transport/s3upload.py` is the more complete one (uses profile, optionally meshes after upload). Retire `transform/upload.py`. |
| Mesh generation | `transform/mesh.py` (broken, has hardcoded params), `transform/mesh_ig.py` (clean), `transport/s3upload.py --mesh` (inline) | Fix `transform/mesh.py` or drop it. `mesh_ig.py` and the `s3upload --mesh` path both call the same `tc.create_meshing_tasks` API; factor into one helper. |
| `cleanup_mem` / `exit_cleanly` | Defined verbatim in both `shared/log.py` and `shared/mem.py`. | Pick one location (`shared/mem.py` — it's the natural owner of shared-memory cleanup). `shared/log.py` should import from `mem`, not redefine. |

### 2.3 Repeated Click `ParamType` patterns

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

| Category | Tasks (proposed, post-cleanup) |
|----------|---------------------------------|
| `transform` | `normalize`, `trim`, `transpose`, `downsample`, `channelize`, `convert`, `find-bounds`, `denoise`, `stitch`, `decompress-tiff`, `gunzip`, `strip-gz-suffix` |
| `sino` | `convert` (full flats), `preprocess` (no flats), `find-bounds` |
| `ng` | `layer copy`, `layer extract`, `layer tag`, `layer urlshift`, `layer recolor`, `point add`, `point merge`, `point sort`, `point shift`, `position copy`, `shift-angle`, `build` (= current `transform/ng.py`) |
| `mesh` | `build`, `manifest` (the two-pass igneous flow) |
| `transport` | `s3 upload`, `cv fetch` |
| `mem` | `clean`, `mark`, `list` |
| `parse` | `meta-shift` (the consolidated `meta_*.py`), `scanlog-fetch`, `pull-config`, `find-errs`, `prune-empty` |
| `hpc` | `cuda-check`, `time-check` |

Implementation: one Click `Group` per category, registered into the
top-level `Group` via `add_command`. Each leaf is the existing
`@click.command()` function moved into a category module. Nothing about
the per-task logic needs to change for the CLI rewrite — only the
decorator and import location.

### 3.3 Packaging

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

---

## 4. Phased plan

Each phase is independently shippable. Order is "make it correct → make
it un-duplicated → make it unified".

### Phase 0 — Project hygiene (½ day)

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

Strict scope: only the bugs in §1.1 and §1.2. No refactors yet.

- [ ] `transform/mesh.py:14` — `@click.commmand` → `@click.command`
- [ ] `transform/upload.py:25` — rename `-s` to e.g. `-k` for
      `--secret-json`
- [ ] `parsing/scanlog_fetch.py:10` — `exists_ok` → `exist_ok`
- [ ] `transform/normalize.py:77` — `click.Path` → `click.INT`
- [ ] `transform/df_write_tiff.py` — either fix `roi.getTitle()` (likely
      `source.getTitle()`) and parameterize the Dragonfly path block,
      or move the file into a `dragonfly/` subdir gated behind an
      extra and skipped on import failure
- [ ] `transform/hdf_convert.py:19` — `'raw\\*.hdf'` → `'raw/*.hdf'`
- [ ] `transport/s3upload.py:74` — drop `s3.close()` (or
      `del s3` if a session-clean intent existed)
- [ ] `ng/layer_urlshift.py:47` — `==` → `=`
- [ ] `ng/layer_tag.py:81-82` — capture `replace(...)` return value
- [ ] `transform/sino_preproc.py:188-190` — return the array;
      and fix the `pool.map(image_bounds, sino_mem)` call shape
- [ ] `transform/multitrim.py:62-63` — swap `vertical_trim` /
      `horizontal_trim` application
- [ ] `ng/layer_extract.py`, `ng/point_shift.py`, `ng/point_sort.py` —
      rename the click command function to match the file

These are mechanical and should land as one or several focused PRs,
each with a regression test under `tests/`.

### Phase 2 — De-duplicate the helper layer (1–2 days)

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

- Real tests for the math-heavy verbs (`normalize`, `trim`, `sino
  convert`) against small fixture tiffs.
- Replace `print` calls with `shared.log.log` everywhere, so
  `--quiet` / `--log-level` work uniformly.
- Add a `--dry-run` flag to anything that writes files, mirroring the
  `mem clean --apply` pattern.
- Consider whether `parsing/meta_shift.py` even belongs in this repo
  vs the chenglab automation; if it's Cheng-Lab specific, move it.
- Decide what to do with `hpc_work/codeclist.txt` (a 449-line ffmpeg
  codec dump that has no consumer in the repo) — almost certainly
  delete.

---

## 5. Risk notes

- The shared-memory machinery in `shared/mem.py` initializes a global
  `mem_tracker` at import time. Any refactor that changes import order
  (e.g., moving things into a package) must preserve "the first import
  of anything from `mctutil` allocates this segment". Worth a test that
  asserts the tracker exists post-import.
- `transform/sinogram.py` is the most subtle code in the repo; tread
  carefully when factoring out `distribute_read`. The function's
  `generate_offset_pairs_*` closures depend on `target_mem[…].buffer_address.start`
  which is itself non-trivial — keep its tests close.
- The Google Sheets path in `parsing/meta_*.py` has a real credential
  shape (`creds`, `GSCOPES`, token files). Whoever owns that flow should
  confirm whether anything still calls it before removing it.

---

## 6. Open questions for maintainers

1. Are any external scripts / sbatch files calling `python
   transform/<x>.py` directly? If yes, Phase 4 needs a deprecation
   period; if no, do it as a hard cut.
2. Is `parsing/meta_*.py` still in active use, or a snapshot of a
   one-time migration? If snapshot, archive and remove.
3. `transform/df_write_tiff.py` is Windows-only and depends on a
   specific Dragonfly install. Is it actually a CLI tool or a script
   that gets pasted into the Dragonfly console? That changes whether
   it belongs under `mctutil` at all.
4. Where do `mem/clean.py`'s sbatch templates run today, and against
   which scheduler version? That code embeds an heredoc with module
   loads (`module load miniconda/3`, `source activate recon`) — worth
   parameterizing per-cluster.

---

## 7. TL;DR

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
