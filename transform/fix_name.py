from pathlib import Path

import click


@click.command()
@click.argument("TARGETDIR", type=click.Path(exists=True, file_okay=False, readable=True, writable=True, path_type=Path))
def fix_names(targetdir):
	for target in targetdir.iterdir():
		first, second = target.with_suffix('').name.split("_")
		#print(target.with_name(f"{first}_{second.zfill(5)}{target.suffix}"))
		target.rename(target.with_name(f"{first}_{second.zfill(5)}{target.suffix}"))

if __name__ == "__main__":
	fix_names()
