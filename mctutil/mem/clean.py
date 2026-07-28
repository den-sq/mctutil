"""Shared-memory cleanup and Slurm submission commands."""

from enum import IntEnum
from io import StringIO
import json
from multiprocessing import shared_memory
from pathlib import Path
import shlex
from subprocess import run, PIPE

import click

from mctutil.shared.log import log, LOG


CONFIG_DIR = Path(__file__).with_name("config")
BUILTIN_CONFIGS = {
	"shm": CONFIG_DIR / "shm.json",
	"kmp": CONFIG_DIR / "kmp.json",
}


class NF(IntEnum):
	NAME = 0
	COUNT = 1
	PARTITION = 2
	STATUS = 3


class PrefixConfig(click.ParamType):
	name = "CONFIG"

	def convert(self, value, param, ctx):
		path = resolve_config_path(value)
		if not path.is_file():
			self.fail(f"{value} is not a built-in config name or an existing JSON file.", param, ctx)
		return path


PREFIX_CONFIG = PrefixConfig()


def resolve_config_path(value):
	"""Resolve a built-in config name or custom path.

	:param value: Built-in config name or filesystem path.
	:return: Resolved config path.
	"""
	if isinstance(value, Path):
		return value
	return BUILTIN_CONFIGS.get(str(value).lower(), Path(value))


def config_reference(config_path):
	"""Return a portable CLI reference for a resolved config path."""
	config_path = resolve_config_path(config_path)
	for name, built_in_path in BUILTIN_CONFIGS.items():
		if config_path == built_in_path:
			return name
	return str(config_path)


def load_prefix_configs(config_paths):
	"""Load and merge shared-memory prefix dictionaries.

	Later files override keys from earlier files.

	:param config_paths: Iterable of built-in config names or JSON paths.
	:return: Merged name-to-prefix dictionary.
	"""
	prefixes = {}

	for config_path in config_paths:
		config_path = resolve_config_path(config_path)
		try:
			with config_path.open() as config_file:
				config = json.load(config_file)
		except (OSError, json.JSONDecodeError) as exc:
			raise click.ClickException(f"Could not load prefix config {config_path}: {exc}") from exc

		if not isinstance(config, dict):
			raise click.ClickException(f"Prefix config {config_path} must contain a JSON object.")

		for name, prefix in config.items():
			if not isinstance(name, str) or not isinstance(prefix, str) or prefix == "":
				raise click.ClickException(
					f"Prefix config {config_path} must map non-empty string names to non-empty string prefixes."
				)
			prefixes[name] = prefix

	if not prefixes:
		raise click.ClickException("At least one shared-memory prefix must be configured.")

	return prefixes


def prefix_config_option(command):
	return click.option(
		"-c",
		"--config",
		"config_paths",
		type=PREFIX_CONFIG,
		multiple=True,
		default=("shm",),
		show_default=True,
		help=(
			"Prefix config to load. Use built-in 'shm' or 'kmp', or a JSON path. "
			"Repeat to merge multiple configs; explicit values replace the default selection."
		),
	)(command)


def parse_sinfo(node_info, node_mixed):
	"""Group eligible nodes from ``sinfo -N`` output by partition."""
	free_nodes = {}
	assigned_nodes = set()

	next(node_info, None)
	for line in node_info:
		node = line.split()
		if len(node) <= NF.STATUS:
			continue

		node_name = node[NF.NAME]
		partition = node[NF.PARTITION].rstrip("*")
		status = node[NF.STATUS]

		if node_name in assigned_nodes or partition == "sas":
			continue
		if status.startswith("idle") or (status.startswith("mix") and node_mixed):
			free_nodes.setdefault(partition, []).append(node_name)
			assigned_nodes.add(node_name)

	return free_nodes


def parse_node_list(node_lists):
	"""Flatten repeatable comma-separated node-list options."""
	nodes = []
	for node_list in node_lists:
		for node in node_list.split(","):
			node = node.strip()
			if node and node not in nodes:
				nodes.append(node)
	return nodes


def merge_node_targets(clear_targets, new_targets):
	"""Merge partition-to-node mappings without duplicate submissions."""
	assigned_nodes = {node for nodes in clear_targets.values() for node in nodes}
	for partition, nodes in new_targets.items():
		for node in nodes:
			if node not in assigned_nodes:
				clear_targets.setdefault(partition, []).append(node)
				assigned_nodes.add(node)


def collect_node_targets(slurm, node_lists, node_file, node_call, node_mixed):
	"""Collect and deduplicate eligible nodes from every selected source."""
	clear_targets = {}
	nodes = parse_node_list(node_lists)

	if nodes:
		node_query = shlex.quote(",".join(nodes))
		with StringIO(slurm.command(f"sinfo -N --nodes={node_query}")) as node_info:
			merge_node_targets(clear_targets, parse_sinfo(node_info, node_mixed))

	if node_file is not None:
		merge_node_targets(clear_targets, parse_sinfo(node_file, node_mixed))

	if node_call:
		with StringIO(slurm.command("sinfo -N")) as node_live:
			merge_node_targets(clear_targets, parse_sinfo(node_live, node_mixed))

	return clear_targets


def load_job_preamble(job_preamble):
	"""Read optional cluster-specific shell setup."""
	if job_preamble is None:
		return ""
	try:
		return job_preamble.read_text().rstrip()
	except OSError as exc:
		raise click.ClickException(f"Could not read job preamble {job_preamble}: {exc}") from exc


def build_clean_command(shared_base, config_paths, execute):
	"""Build the installed CLI invocation for a submitted cleanup job."""
	command = ["mctutil", "mem", "clean", "--shared-base", str(shared_base)]
	for config_path in config_paths:
		command.extend(["--config", config_reference(config_path)])
	command.append("--execute" if execute else "--dry-run")
	return shlex.join(command)


def build_sbatch_job(partition, nodes, clean_command, preamble, output_pattern, error_pattern):
	"""Build a portable batch script and its Slurm arguments."""
	script_parts = ["#!/bin/bash -l"]
	if preamble:
		script_parts.extend(["", preamble])
	script_parts.extend(["", clean_command, ""])

	args = [
		"--job-name", "mem_clean",
		"--nodelist", ",".join(nodes),
		"--partition", partition,
		"--nodes", str(len(nodes)),
	]
	if output_pattern is not None:
		args.extend(["--output", output_pattern.replace("{partition}", partition)])
	if error_pattern is not None:
		args.extend(["--error", error_pattern.replace("{partition}", partition)])

	return "\n".join(script_parts), args


def mem_clean(shared_base, execute, prefixes):
	"""List or unlink shared-memory entries matching configured prefixes."""
	host = run(["hostname"], stdout=PIPE, check=False).stdout.decode().strip("\n")
	prefix_values = tuple(dict.fromkeys(prefixes.values()))

	for mem_path in shared_base.iterdir():
		mem_name = mem_path.name
		if not mem_name.startswith(prefix_values):
			continue

		if execute:
			try:
				clean_target = shared_memory.SharedMemory(name=mem_name)
				clean_target.close()
				clean_target.unlink()
			except FileNotFoundError:
				continue

		log.write(
			"Mem Clean",
			f"{host}:{mem_name}:{execute}",
			log_level=LOG.STATUS,
		)


def _get_slurm(remote):
	try:
		import ipyslurm
	except ImportError as exc:
		raise click.ClickException(
			"The mem mark command requires ipyslurm; install the project HPC environment."
		) from exc

	slurm = ipyslurm.Slurm()
	if remote is not None:
		try:
			user, host = remote.split("@", 1)
		except ValueError as exc:
			raise click.ClickException("--remote must use USER@HOST format.") from exc
		slurm.login(host, user)
	return slurm


@click.command()
@prefix_config_option
@click.option(
	"--execute/--dry-run",
	default=False,
	show_default=True,
	help="Unlink matching entries; the safe default only lists them.",
)
@click.option(
	"--shared-base",
	"--shared_base",
	type=click.Path(exists=True, file_okay=False, path_type=Path),
	default="/dev/shm",
	show_default=True,
)
def clean(config_paths, execute, shared_base):
	"""List or clean configured shared-memory entries on this node."""
	mem_clean(shared_base, execute, load_prefix_configs(config_paths))


@click.command()
@prefix_config_option
@click.option("--node-list", "--node_list", multiple=True,
				help="Comma-separated Slurm nodes to inspect; may be repeated.")
@click.option("--node-file", "--node_file", type=click.File("r"),
				help="Saved output from sinfo -N.")
@click.option("--node-call", "--node_call", is_flag=True,
				help="Query sinfo -N for all eligible nodes.")
@click.option("--node-mixed", "--node_mixed", is_flag=True,
				help="Include nodes in the mixed state.")
@click.option(
	"--execute/--dry-run",
	default=False,
	show_default=True,
	help="Make submitted jobs unlink matching entries; the safe default only reports them.",
)
@click.option(
	"--shared-base",
	"--shared_base",
	type=click.Path(file_okay=False, path_type=Path),
	default="/dev/shm",
	show_default=True,
	help="Shared-memory directory on the target nodes.",
)
@click.option("--remote", type=click.STRING, default=None,
				help="Run Slurm commands through USER@HOST.")
@click.option(
	"--job-preamble",
	type=click.Path(exists=True, dir_okay=False, path_type=Path),
	help="File containing cluster-specific shell setup inserted before the mctutil command.",
)
@click.option(
	"--sbatch-output",
	help="Optional Slurm output pattern; supports {partition} and Slurm percent escapes.",
)
@click.option(
	"--sbatch-error",
	help="Optional Slurm error pattern; supports {partition} and Slurm percent escapes.",
)
def mark(
	config_paths,
	node_list,
	node_file,
	node_call,
	node_mixed,
	execute,
	shared_base,
	remote,
	job_preamble,
	sbatch_output,
	sbatch_error,
):
	"""Submit one shared-memory cleanup job per selected Slurm partition."""
	if not node_list and node_file is None and not node_call:
		raise click.UsageError("Provide --node-list, --node-file, or --node-call.")

	# Validate every config before submitting remote jobs.
	load_prefix_configs(config_paths)
	preamble = load_job_preamble(job_preamble)
	slurm = _get_slurm(remote)
	clear_targets = collect_node_targets(slurm, node_list, node_file, node_call, node_mixed)
	if not clear_targets:
		log.write("Mem Clean", "No eligible nodes found.", log_level=LOG.WARN)
		return

	clean_command = build_clean_command(shared_base, config_paths, execute)

	for partition, partition_nodes in clear_targets.items():
		script, args = build_sbatch_job(
			partition,
			partition_nodes,
			clean_command,
			preamble,
			sbatch_output,
			sbatch_error,
		)
		res = slurm.sbatch(script, args=args)
		log.write("Mem Clean", f"Job {partition}:{res}", log_level=LOG.STATUS)
