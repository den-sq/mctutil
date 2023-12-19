import sys
from pathlib import Path 	# if you haven't already done so

import click
import numpy as np
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


@click.command
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for original dataset.', required=True)
@click.option('-o', '--output-dir', type=click.Path(), required=True,
				help='Output path for transformed dataset.')
@click.option('-v', '--vertical-trim', type=click.FLOAT, default=0.0,
				help='Vertical trim (top and bottom) as a percent')
@click.option('-h', '--horizontal-trim', type=click.FLOAT, default=0.0,
				help='Horizontal trim (top and bottom) as a percent')
def trim(data_dir, output_dir, vertical_trim, horizontal_trim):
	log.start()
	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	for path in Path(data_dir).iterdir():
		in_img = tf.imread(path)
		dim = np.shape(in_img)
		tf.imwrite(Path(out_dir, path.name), in_img[
			int(dim[0] * horizontal_trim):int(-1 * dim[0] * horizontal_trim),
			int(dim[1] * vertical_trim):int(-1 * dim[1] * vertical_trim)])
		log.log("File Written", path.name)


if __name__ == "__main__":
	trim()
