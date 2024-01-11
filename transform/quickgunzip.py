from pathlib import Path
import gzip
import shutil

import click


@click.command()
@click.argument("root_path", type=click.Path(exists=True))
@click.argument("target_path", type=click.Path(exists=False))
def gunzip(root_path, target_path):
	for fname in Path(root_path).glob("**/*"):
		if fname.is_file():
			Path(target_path, fname.parent.name).mkdir(parents=True, exist_ok=True)
			with gzip.open(fname, 'rb') as infile, Path(target_path, fname.parent.name, fname.name).open("wb") as outfile:
				shutil.copyfileobj(infile, outfile)


if __name__ == '__main__':
	gunzip()
