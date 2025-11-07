from pathlib import Path
import gzip
import shutil

import brotli
import click


@click.command()
@click.argument("root_path", type=click.Path(exists=True))
@click.argument("target_path", type=click.Path(exists=False))
def gunzip(root_path, target_path):
	for fname in Path(root_path).glob("**/*"):
		if fname.is_file():
			Path(target_path, fname.parent.name).mkdir(parents=True, exist_ok=True)
			target_file = Path(target_path, fname.parent.name, fname.with_suffix("").name)
			try:
				if fname.suffix == ".gz":
					with gzip.open(fname, 'rb') as infile, target_file.open("wb") as outfile:
						shutil.copyfileobj(infile, outfile)
				elif fname.suffix == ".br":
					with open(fname, 'rb') as infile, target_file.open("wb") as outfile:
						outfile.write(brotli.decompress(infile.read()))
				else:
					shutil.copy(fname, target_file)
			except BaseException:
				print(f"Failed File: {fname}|{target_file}")


if __name__ == '__main__':
	gunzip()
