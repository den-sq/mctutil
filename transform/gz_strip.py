from pathlib import Path

import click


@click.command()
@click.argument("root_path", type=click.Path(exists=True))
def stripgz(root_path):
    for fname in Path(root_path).glob("**/*.gz"):
        fname.rename(fname.with_suffix(''))


if __name__ == '__main__':
    stripgz()
