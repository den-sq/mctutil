from pathlib import Path
import sys

import click

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


@click.command()
@click.option("--pattern", type=click.STRING, default="*history", show_default=True,
				help="rglob pattern under ROOT for directories whose immediate children should be pruned.")
@click.option('--execute/--dry-run', default=False, show_default=True,
				help="Whether to actually rmdir empty directories or just list them. Defaults to dry-run.")
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
def prune_empty(pattern: str, execute: bool, root: Path):
	""" Remove empty subdirectories under each match of PATTERN beneath ROOT.

		Defaults to dry-run because the operation is destructive.
	"""
	for outer in root.rglob(pattern):
		for inner_dir in outer.iterdir():
			if inner_dir.is_dir() and not any(inner_dir.iterdir()):
				if execute:
					log.log("Prune Empty", f"Removing empty {inner_dir}", log_level=log.DEBUG.STATUS)
					inner_dir.rmdir()
				else:
					log.log("Prune Empty", f"Would remove empty {inner_dir}", log_level=log.DEBUG.INFO)


if __name__ == "__main__":
	prune_empty()
