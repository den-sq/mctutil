from enum import IntEnum
from multiprocessing import shared_memory
from io import StringIO
from pathlib import Path
from subprocess import run, PIPE

import click
import ipyslurm

# Names of shared memory to be script-specific (for now)
shm = {
	"rsm": "rot",
	"psm": "proj",
	"fsm": "flat",
	"ysm": "y_rot",
	"pfm": "phase",
	"log": "last_step"
}

other_shm = "__KMP_REGISTERED_LIB"


class NF(IntEnum):
	NAME = 0
	COUNT = 1
	PARTITION = 2
	STATUS = 3


def parse_sinfo(node_info, node_mixed):
	free_nodes = {}

	next(node_info)
	node_entries = [[field for field in line.strip('\n').strip().split(' ') if field != ''] for line in node_info]

	for node in node_entries:
		if node[NF.PARTITION] != "sas":
			if node[NF.STATUS] == "idle" or (node[NF.STATUS] == "mix" and node_mixed):
				# Add node to list of those to clear.
				if node[NF.PARTITION] not in free_nodes:
					free_nodes[node[NF.PARTITION]] = [node[NF.NAME]]
				else:
					free_nodes[node[NF.PARTITION]].append(node[NF.NAME])

	return free_nodes


def mem_clean(shared_base, apply, apply_kmp):
	host = run("hostname", stdout=PIPE).stdout.decode().strip('\n')

	for mem_path in shared_base.iterdir():
		mem_name = mem_path.name
		for prefix in shm.values():
			if mem_name[:len(prefix)] == prefix:
				if apply:
					clean_target = shared_memory.SharedMemory(name=mem_name)
					clean_target.close()
					clean_target.unlink()
				print(f"{host}:{mem_name}:{apply}")
			elif mem_name[:len(other_shm)] == other_shm:
				if apply_kmp:
					try:
						clean_target = shared_memory.SharedMemory(name=str(mem_name))
						clean_target.close()
						clean_target.unlink()
					except FileNotFoundError:
						pass
					print(f"{host}:{mem_name}:{apply}")


@click.group()
@click.option("--node_mixed", is_flag=True, help="Whether to scan/remove on mixed files.")
@click.option("--apply", is_flag=True, help="Whether to clean shared memory or just write what memory exists.")
@click.option("--apply_kmp", is_flag=True, help="Whether to clean KMP_REGISTERED_LIB entries.")
@click.option("--shared_base", type=click.Path(exists=True), default="/dev/shm")
@click.option("--remote", type=click.STRING, default=None)
@click.pass_context
def memclean(ctx, node_mixed, apply, apply_kmp, shared_base, remote):
	ctx.obj = ipyslurm.Slurm()
	if remote is not None:
		ctx.obj.login(remote.split("@")[1], remote.split("@")[0])


@memclean.command()
@click.pass_context
def clean(ctx):
	mem_clean(Path(ctx.parent.params["shared_base"]), ctx.parent.params["apply"], ctx.parent.params["apply_kmp"])


@memclean.command()
@click.option("--node_list", type=click.STRING, required=False, help="Comma separated list of nodes to clear.")
@click.option("--node_file", type=click.File(), required=False, help="Path to output of sinfo -N to identify nodes.")
@click.option("--node_call", is_flag=True,
							help="Pull a list of nodes to parse for idle and mixed nodes from server.")
@click.pass_context
def mark(ctx, node_list, node_file, node_call):
	base_params = ctx.parent.params.copy()

	node_mixed = base_params.pop("node_mixed") 	# Used Here
	base_params.pop("remote")					# Not Used in Batched Command

	clear_targets = {}

# 	if node_list is None:
# 	node_list = []
# 	else:
# 		node_list = node_list.split(",")

	if node_file is not None:
		for partition, nodes in parse_sinfo(node_file, node_mixed).items():
			if partition in clear_targets:
				clear_targets[partition].append(nodes)
			else:
				clear_targets[partition] = nodes

	if node_call:
		with StringIO(ctx.obj.command("sinfo -N")) as node_live:
			for partition, nodes in parse_sinfo(node_live, node_mixed).items():
				if partition in clear_targets:
					clear_targets[partition].append(nodes)
				else:
					clear_targets[partition] = nodes

	if len(clear_targets) > 0:
		param_str = ' '.join([f"--{key}={value}" if not isinstance(value, bool) else f"--{key}"
							for key, value in base_params.items()
							if ctx.parent.get_parameter_source(key) != click.core.ParameterSource.DEFAULT])
		for partition, nodes in clear_targets.items():
			res = ctx.obj.sbatch(f"""
#!/bin/bash -l

module load miniconda/3
source activate recon

python -X pycache_prefix=~/.pycache ~/mem/clean.py {param_str} clean
""", args=['--job-name', 'mem_clean',
			'--output', f'~/mem/log/%j_{partition}.log',
			'--error', f'~/mem/error/%j_{partition}.log',
			'--nodelist', ','.join(nodes),
			'--partition', partition,
			'--nodes', str(len(nodes))
])
			print(f"Job {partition}:{res}")


if __name__ == "__main__":
	memclean()
