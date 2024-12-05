from multiprocessing import Pool
import sys
from pathlib import Path 	# if you haven't already done so

import click
import numpy as np
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


def write_crop(input, output, crop, compress):
	img = tf.imread(input)
	if compress:
		tf.imwrite(output, img[crop], compression=8)
	else:
		tf.imwrite(output, img[crop])
	log.log("File Written", f"{output.name}: ({img.shape}>{crop})")


@click.command
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for original dataset.', required=True)
@click.option('-o', '--output-dir', type=click.Path(), required=True,
				help='Output path for transformed dataset.')
@click.option('-v', '--vertical-trim', type=click.FLOAT, default=0.0,
				help='Vertical trim (top and bottom) as a percent')
@click.option('-h', '--horizontal-trim', type=click.FLOAT, default=0.0,
				help='Horizontal trim (top and bottom) as a percent')
@click.option('--compressed/--uncompressed', default=False,
				help='Whether to compress output data.')
def trim(data_dir, output_dir, vertical_trim, horizontal_trim, compressed):
	log.start()
	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	path_list = list(Path(data_dir).glob("*.tif*"))

	with tf.TiffFile(path_list[0]) as tif:
		dim = tif.pages[0].shape

	new_dim = np.s_[int(dim[0] * vertical_trim):int(dim[0] * (1.0 - vertical_trim)),
					int(dim[1] * horizontal_trim):int(dim[1] * (1.0 - horizontal_trim))]

	with Pool(64) as pool:
		pool.starmap(write_crop, [(path, Path(out_dir, path.name), new_dim, compressed) for path in path_list])


if __name__ == "__main__":
	trim()
