from pathlib import Path
from subprocess import run

import click


from mctutil.shared.log import log, LOG


def collect_idle_nodes(node_file: Path):
	node_list = []
	with open(node_file) as node_info:
		next(node_info)
		for line in node_info:
			fields = line.split()
			if len(fields) == 0:
				continue
			status = fields[-1]
			if status in {"idle", "mix"}:
				node_list.append(fields[0])
	return node_list


@click.command()
@click.option("--sbatch-script", type=click.Path(path_type=Path), required=True,
				help="Sbatch script to submit for each selected node.")
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually submit sbatch jobs or just list the planned submissions.")
@click.argument("node_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def from_file(sbatch_script: Path, execute: bool, node_file: Path):
	for node in collect_idle_nodes(node_file):
		if execute:
			run(["sbatch", "-w", node, str(sbatch_script)])
			log.write("Mem From File", f"sbatch -w {node} {sbatch_script}", log_level=LOG.STATUS)
		else:
			log.write("Mem From File", f"Would sbatch -w {node} {sbatch_script}", log_level=LOG.INFO)


if __name__ == "__main__":
	from_file()
