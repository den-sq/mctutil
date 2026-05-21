from subprocess import run

import click


from mctutil.shared import log


def expand_node_range(prefix: str, start: int, stop: int):
	return [f"{prefix}{index}" for index in range(start, stop + 1)]


@click.command()
@click.option("--prefix", default="psh01com1hcom", show_default=True)
@click.option("--start", type=click.INT, required=True)
@click.option("--stop", type=click.INT, required=True)
@click.option("--sbatch-script", type=click.STRING, default="memclean.sbatch", show_default=True)
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually submit sbatch jobs or just list the planned submissions.")
def from_range(prefix: str, start: int, stop: int, sbatch_script: str, execute: bool):
	for node in expand_node_range(prefix, start, stop):
		if execute:
			run(["sbatch", "-w", node, sbatch_script])
			log.log("Mem From Range", f"sbatch -w {node} {sbatch_script}", log_level=log.DEBUG.STATUS)
		else:
			log.log("Mem From Range", f"Would sbatch -w {node} {sbatch_script}", log_level=log.DEBUG.INFO)


if __name__ == "__main__":
	from_range()
