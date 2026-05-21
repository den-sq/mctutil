from subprocess import run

import click


def expand_node_range(prefix: str, start: int, stop: int):
	return [f"{prefix}{index}" for index in range(start, stop + 1)]


@click.command()
@click.option("--prefix", default="psh01com1hcom", show_default=True)
@click.option("--start", type=click.INT, required=True)
@click.option("--stop", type=click.INT, required=True)
@click.option("--sbatch-script", type=click.STRING, default="memclean.sbatch", show_default=True)
def from_range(prefix: str, start: int, stop: int, sbatch_script: str):
	for node in expand_node_range(prefix, start, stop):
		run(["sbatch", "-w", node, sbatch_script])


if __name__ == "__main__":
	from_range()
