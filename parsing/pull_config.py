from pathlib import Path
from shutil import copy

import click


@click.command()
@click.option("--source", type=click.Path(exists=True), help="Root Directory to Search for Configs.")
@click.option("--target", type=click.Path(exists=False), help="Target Directory to Place Copied Configs.")
def get_conf(source, target):
	source_conf_set = Path(source).glob("**/*.yaml")
	Path(target).mkdir(exist_ok=True, parents=True)
	for conf in source_conf_set:
		print(f"{conf.parent.name}_{conf.name}")
		copy(conf, Path(target, f"{conf.parent.name}_{conf.name}"))

if __name__ == "__main__":
	get_conf()
