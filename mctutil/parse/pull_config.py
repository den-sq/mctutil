from pathlib import Path
from shutil import copy

import click


from mctutil.shared import log


@click.command()
@click.option("--source", type=click.Path(exists=True), help="Root Directory to Search for Configs.")
@click.option("--target", type=click.Path(exists=False), help="Target Directory to Place Copied Configs.")
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually copy configs or just list the planned copies.")
def get_conf(source, target, execute):
	source_conf_set = Path(source).glob("**/*.yaml")
	if execute:
		Path(target).mkdir(exist_ok=True, parents=True)
	else:
		log.log("Pull Config", f"Would create {target}", log_level=log.DEBUG.INFO)
	for conf in source_conf_set:
		dest = Path(target, f"{conf.parent.name}_{conf.name}")
		if execute:
			log.log("Pull Config", f"{conf.parent.name}_{conf.name}", log_level=log.DEBUG.STATUS)
			copy(conf, dest)
		else:
			log.log("Pull Config", f"Would copy {conf} -> {dest}", log_level=log.DEBUG.INFO)


if __name__ == "__main__":
	get_conf()
