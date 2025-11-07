import sys
from pathlib import Path 	# if you haven't already done so

import click
import dicom2jpg

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import cli 	# noqa::E402
from shared import log 	# noqa::E402


@click.command
@click.argument('input_loc', type=click.Path(exists=True, path_type=Path))
@click.argument('output_loc', type=click.Path(path_type=Path))
def dicom_conv(input_loc: Path, output_loc: Path):
	log.start()
	output_loc.mkdir(parents=True, exist_ok=True)
	path_list = input_loc.iterdir() if input_loc.is_dir() else [input_loc]
	for path in path_list:
		dicom2jpg.dicom2tiff(path, output_loc.joinpath(*path.parts[-2:]))
		log.log("File Written", path.parts[-2:])


if __name__ == "__main__":
	dicom_conv()
