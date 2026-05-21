from pathlib import Path

import click


from mctutil.shared.log import log, LOG


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
					log.write("Prune Empty", f"Removing empty {inner_dir}", log_level=LOG.STATUS)
					inner_dir.rmdir()
				else:
					log.write("Prune Empty", f"Would remove empty {inner_dir}", log_level=LOG.INFO)


if __name__ == "__main__":
	prune_empty()
