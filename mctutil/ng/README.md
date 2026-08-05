# mctutil ng

Neuroglancer JSON, layer, and annotation helpers.

Run `mctutil ng --help` to list commands and `mctutil ng <task> --help` for a
command's options.

## Commands

- **`build`** — Build a Neuroglancer precomputed volume (image or segmentation) from a stack.
- **`precompute`** — Write process-parallel CloudVolume MIP-0 output from TIFF input.
- **`downsample-pyramid`** — Build a volumetric MIP pyramid with durable Igneous task queues.
- **`http-check`** — Smoke-test `info` and explicit chunk URLs with GET or HEAD.
- **`publish`** — Run the stage-aware, resumable sharded publishing pipeline.
- **`shard`** — Stage a precomputed pyramid into sharded per-mip output.
- **`validate`** — Validate precomputed metadata and representative origin/center reads.
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
`ng precompute` is the CloudVolume backend used by the sharded publishing
pipeline and intentionally coexists with `ng build`.

## Sharded publishing

The complete pipeline requires:

```console
pip install -e '.[ng,mesh,aws]'
mctutil ng publish ROOT --s3-prefix s3://BUCKET/PREFIX
```

`publish` checks the selected range before writing anything. Short ranges only
require their stage groups: prep/precompute use `[ng]`, downsample/shard/mesh use
`[mesh]`, and upload or S3 meshing uses `[aws]`. For example,
`--stop-after precompute` needs only `[ng]`, `--start-at upload` is an
`[aws]`-only upload resume, and `--no-upload --mesh-at local` removes the
`[aws]` requirement. Use `--dry-run` to inspect every dataset's
run/skip/omitted decisions. Voxel resolution defaults to `700,700,700` nm and
voxel offset independently defaults to `0,0,0`.

Upload and in-place S3 meshing share one named AWS profile. `--aws-profile`
wins over `AWS_PROFILE`; if neither is supplied, the profile defaults to
`chenglab`. To prevent CloudFiles from silently using a different identity,
`publish` refuses legacy `.cloudvolume`/`.cloudfiles` AWS secret JSON files and
raw AWS access-key environment variables. Named profiles may use static,
temporary, SSO, or assume-role credentials through Boto3.

`ng precompute` deliberately rewrites all MIP-0 planes when invoked again;
individual chunk writes are fast enough that scanning every planned chunk before
writing is counterproductive. It verifies completion with one local scale-folder
enumeration. `ng downsample-pyramid` refuses an incomplete MIP 0 unless
`--force` is supplied.

On Linux, `ng publish --systemd-scope` optionally re-executes an actual publish
inside a uniquely named transient user-systemd scope. This gives resource
logging exact cgroup-v2 totals for the publish parent and every descendant.
Existing isolated Slurm/container scopes are used as-is. If cgroup v2, the user
manager, or delegated memory accounting is unavailable, publish emits one
warning and continues with recursive process-tree accounting; scope setup never
blocks the pipeline. Dry runs are not relaunched.

For an interactive Linux/WSL smoke test, run with verbose logging and inspect
the announced unit from another terminal while a worker stage is active:

```console
mctutil --verbose ng publish ROOT --systemd-scope --stop-after precompute
systemctl --user status mctutil-publish-....scope
systemd-cgls --user-unit=mctutil-publish-....scope
```

The publish process and its workers should appear under the same scope while
progress remains attached to the invoking terminal. The unit is collected when
the command succeeds, fails, or is interrupted.

Downsampling and sharding resume from their durable Igneous FileQueues. A
resumed queue releases existing leases before draining so tasks from a killed
run are immediately available. Use `--preserve-leases` on the leaf commands, or
`--preserve-queue-leases` on `ng publish`, when other workers intentionally
share the same queue.
