from pathlib import Path
from shutil import copy

import click


@click.command()
@click.argument("root_dir", nargs=-1, type=click.Path())
def scanlog_fetch(root_dir):
	Path("logs").mkdir(exists_ok=True)
	for dir_name in root_dir:
		for fullpath in Path(dir_name).rglob("scanlog.txt"):
			software_trigger_scans = ["_post", "_pre", "_uncrop", "_crop", "_focus", "_step"]
			if not any(stscan in fullpath.parent.name.lower() for stscan in software_trigger_scans):
				if fullpath.stat().st_size > 93:
					copy(fullpath, Path("logs", f"{fullpath.parent.name}_scanlog.txt"))
					click.echo(f"{fullpath.parent.name}:{fullpath.stat().st_size}")


if __name__ == "__main__":
	scanlog_fetch()
