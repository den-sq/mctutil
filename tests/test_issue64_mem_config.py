from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner


def test_prefix_configs_default_to_shm_and_merge_custom_overrides(load_module, tmp_path):
	module = load_module("mctutil/mem/clean.py")
	custom_config = tmp_path / "custom.json"
	custom_config.write_text(json.dumps({
		"rsm": "custom-rot",
		"custom": "custom-prefix",
	}))

	default_prefixes = module.load_prefix_configs(["shm"])
	assert default_prefixes["rsm"] == "rot"
	assert "__KMP_REGISTERED_LIB" not in default_prefixes.values()

	merged_prefixes = module.load_prefix_configs(["shm", "kmp", custom_config])
	assert merged_prefixes["rsm"] == "custom-rot"
	assert merged_prefixes["kmp"] == "__KMP_REGISTERED_LIB"
	assert merged_prefixes["custom"] == "custom-prefix"


def test_clean_command_uses_default_or_multiple_explicit_configs(load_module, monkeypatch, tmp_path):
	module = load_module("mctutil/mem/clean.py")
	calls = []
	monkeypatch.setattr(module, "mem_clean", lambda shared_base, execute, prefixes: calls.append(
		(shared_base, execute, prefixes)
	))

	default_result = CliRunner().invoke(module.clean, ["--shared-base", str(tmp_path)])
	assert default_result.exit_code == 0, default_result.output
	assert calls[-1][0] == tmp_path
	assert calls[-1][1] is False
	assert "__KMP_REGISTERED_LIB" not in calls[-1][2].values()

	multiple_result = CliRunner().invoke(
		module.clean,
		[
			"--shared-base", str(tmp_path),
			"--config", "shm",
			"--config", "kmp",
			"--execute",
		],
	)
	assert multiple_result.exit_code == 0, multiple_result.output
	assert calls[-1][1] is True
	assert calls[-1][2]["kmp"] == "__KMP_REGISTERED_LIB"

	kmp_only_result = CliRunner().invoke(
		module.clean,
		[
			"--shared-base", str(tmp_path),
			"--config", "kmp",
		],
	)
	assert kmp_only_result.exit_code == 0, kmp_only_result.output
	assert calls[-1][2] == {"kmp": "__KMP_REGISTERED_LIB"}


def test_mem_clean_applies_all_configured_prefixes_once(load_module, monkeypatch, tmp_path):
	module = load_module("mctutil/mem/clean.py")
	(tmp_path / "rot-volume").touch()
	(tmp_path / "__KMP_REGISTERED_LIB_123").touch()
	(tmp_path / "unrelated").touch()
	unlinked = []

	class FakeSharedMemory:
		def __init__(self, name):
			self.name = name

		def close(self):
			pass

		def unlink(self):
			unlinked.append(self.name)

	monkeypatch.setattr(module.shared_memory, "SharedMemory", FakeSharedMemory)
	monkeypatch.setattr(module.log, "write", lambda *_args, **_kwargs: None)

	module.mem_clean(
		tmp_path,
		True,
		module.load_prefix_configs(["shm", "kmp"]),
	)

	assert unlinked == ["rot-volume", "__KMP_REGISTERED_LIB_123"]


def test_mark_node_list_queries_partitions_and_submits_eligible_nodes(load_module, monkeypatch, tmp_path):
	module = load_module("mctutil/mem/clean.py")

	class FakeSlurm:
		def __init__(self):
			self.commands = []
			self.jobs = []

		def command(self, command):
			self.commands.append(command)
			return (
				"NODELIST NODES PARTITION STATE\n"
				"node-a 1 compute idle\n"
				"node-a 1 backup idle\n"
				"node-b 1 gpu mix\n"
				"node-c 1 compute alloc\n"
			)

		def sbatch(self, script, args):
			self.jobs.append((script, args))
			return f"job-{len(self.jobs)}"

	slurm = FakeSlurm()
	monkeypatch.setattr(module, "_get_slurm", lambda _remote: slurm)
	monkeypatch.setattr(module.log, "write", lambda *_args, **_kwargs: None)
	preamble = tmp_path / "cluster-setup.sh"
	preamble.write_text("module load site-python\nsource /cluster/mctutil-env")

	result = CliRunner().invoke(
		module.mark,
		[
			"--node-list", "node-a,node-b",
			"--node-list", "node-c",
			"--node-mixed",
			"--shared-base", str(tmp_path),
			"--config", "shm",
			"--config", "kmp",
			"--execute",
			"--job-preamble", str(preamble),
			"--sbatch-output", "logs/%j_{partition}.out",
			"--sbatch-error", "logs/%j_{partition}.err",
		],
	)

	assert result.exit_code == 0, result.output
	assert slurm.commands == ["sinfo -N --nodes=node-a,node-b,node-c"]
	assert len(slurm.jobs) == 2

	jobs_by_partition = {
		job_args[job_args.index("--partition") + 1]: (script, job_args)
		for script, job_args in slurm.jobs
	}
	assert set(jobs_by_partition) == {"compute", "gpu"}

	compute_script, compute_args = jobs_by_partition["compute"]
	assert compute_args[compute_args.index("--nodelist") + 1] == "node-a"
	assert compute_args[compute_args.index("--nodes") + 1] == "1"
	assert compute_args[compute_args.index("--output") + 1] == "logs/%j_compute.out"
	assert compute_args[compute_args.index("--error") + 1] == "logs/%j_compute.err"

	gpu_script, gpu_args = jobs_by_partition["gpu"]
	assert gpu_args[gpu_args.index("--nodelist") + 1] == "node-b"
	assert gpu_args[gpu_args.index("--nodes") + 1] == "1"
	assert gpu_args[gpu_args.index("--output") + 1] == "logs/%j_gpu.out"
	assert gpu_args[gpu_args.index("--error") + 1] == "logs/%j_gpu.err"

	for script in [compute_script, gpu_script]:
		assert script.startswith("#!/bin/bash -l")
		assert "module load site-python" in script
		assert "source /cluster/mctutil-env" in script
		assert "miniconda/3" not in script
		assert "source activate recon" not in script
		assert "mctutil mem clean" in script
		assert "--config shm --config kmp" in script
		assert "--execute" in script


def test_mark_merges_node_file_and_live_call_without_duplicates(load_module, monkeypatch, tmp_path):
	module = load_module("mctutil/mem/clean.py")

	class FakeSlurm:
		def __init__(self):
			self.commands = []
			self.jobs = []

		def command(self, command):
			self.commands.append(command)
			return (
				"NODELIST NODES PARTITION STATE\n"
				"node-a 1 compute idle\n"
				"node-c 1 compute idle\n"
				"node-d 1 gpu idle\n"
			)

		def sbatch(self, script, args):
			self.jobs.append((script, args))
			return f"job-{len(self.jobs)}"

	node_file = tmp_path / "nodes.txt"
	node_file.write_text(
		"NODELIST NODES PARTITION STATE\n"
		"node-a 1 compute idle\n"
		"node-b 1 gpu mix\n"
		"node-x 1 sas idle\n"
		"node-z 1 compute alloc\n"
	)

	slurm = FakeSlurm()
	monkeypatch.setattr(module, "_get_slurm", lambda _remote: slurm)
	monkeypatch.setattr(module.log, "write", lambda *_args, **_kwargs: None)

	result = CliRunner().invoke(
		module.mark,
		[
			"--node-file", str(node_file),
			"--node-call",
			"--node-mixed",
			"--shared-base", "/target/dev/shm",
		],
	)

	assert result.exit_code == 0, result.output
	assert slurm.commands == ["sinfo -N"]

	jobs_by_partition = {
		job_args[job_args.index("--partition") + 1]: (script, job_args)
		for script, job_args in slurm.jobs
	}
	assert set(jobs_by_partition) == {"compute", "gpu"}
	assert jobs_by_partition["compute"][1][jobs_by_partition["compute"][1].index("--nodelist") + 1] == "node-a,node-c"
	assert jobs_by_partition["gpu"][1][jobs_by_partition["gpu"][1].index("--nodelist") + 1] == "node-b,node-d"

	for script, job_args in jobs_by_partition.values():
		assert "mctutil mem clean" in script
		assert "--config shm" in script
		assert "--dry-run" in script
		assert "--output" not in job_args
		assert "--error" not in job_args


def test_mark_requires_a_node_source(load_module, monkeypatch, tmp_path):
	module = load_module("mctutil/mem/clean.py")
	monkeypatch.setattr(
		module,
		"_get_slurm",
		lambda _remote: (_ for _ in ()).throw(AssertionError("Slurm loaded without a node source")),
	)

	result = CliRunner().invoke(module.mark, ["--shared-base", str(tmp_path)])

	assert result.exit_code != 0
	assert "Provide --node-list, --node-file, or --node-call" in result.output


def test_shipped_sbatch_consumers_use_installed_cli_without_cluster_setup():
	mem_dir = Path(__file__).parents[1] / "mctutil" / "mem"
	memcheck = (mem_dir / "memcheck.sbatch").read_text()
	memclean = (mem_dir / "memclean.sbatch").read_text()

	for script in [memcheck, memclean]:
		assert "module load" not in script
		assert "source activate" not in script
		assert "python clean.py" not in script

	assert "mctutil mem clean --dry-run" in memcheck
	assert "mctutil mem clean --execute" in memclean
