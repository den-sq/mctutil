# mctutil hpc

HPC runtime and scheduler-side helpers.

Run `mctutil hpc --help` to list commands and `mctutil hpc <task> --help` for a
command's options.

## Commands

- **`time-check`** — Derive projection start/stop times from image timestamps under a root.

## Templates

[`hpc_env/cuda.sbatch`](../../hpc_env/cuda.sbatch) is the retained
cluster-specific CUDA/PyTorch probe template. There is no `cuda-check` CLI
command: the probe requires the HPC reconstruction environment and is kept
inline in the batch template rather than adding PyTorch as an mctutil
dependency.
