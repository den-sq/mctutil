from pathlib import Path
import sys
from shutil import copy

import click

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


@click.command()
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually copy scanlogs or just list the planned copies.")
@click.argument("root_dir", nargs=-1, type=click.Path())
def scanlog_fetch(execute, root_dir):
	if execute:
		Path("logs").mkdir(exist_ok=True)
	else:
		log.log("Scanlog Fetch", "Would create logs/", log_level=log.DEBUG.INFO)
	for dir_name in root_dir:
		for fullpath in Path(dir_name).rglob("scanlog.txt"):
			software_trigger_scans = ["_post", "_pre", "_uncrop", "_crop", "_focus", "_step"]
			if not any(stscan in fullpath.parent.name.lower() for stscan in software_trigger_scans):
				if fullpath.stat().st_size > 93:
					target = Path("logs", f"{fullpath.parent.name}_scanlog.txt")
					if execute:
						copy(fullpath, target)
						log.log("Scanlog Fetch", f"{fullpath.parent.name}:{fullpath.stat().st_size}",
								log_level=log.DEBUG.STATUS)
					else:
						log.log("Scanlog Fetch",
								f"Would copy {fullpath} -> {target} (size {fullpath.stat().st_size})",
								log_level=log.DEBUG.INFO)


if __name__ == "__main__":
	scanlog_fetch()
