from multiprocessing import Pool
import sys
from pathlib import Path 	# if you haven't already done so

import click
import numpy as np
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


class CropNumberType(click.ParamType):
	name = "CropNumber"

	def convert(self, value, param, ctx):
		if "," in str(value):
			pair = value.split(",")
			print(f"is pair {pair}")
			if len(pair) != 2:
				self.fail(f"{value} must be a single or pair of values.")
		else:
			pair = [value, value]

		for x in range(0, 2):
			if str(pair[x]).isnumeric():
				pair[x] = int(pair[x])
			else:
				try:
					pair[x] = float(pair[x])
				except ValueError:
					self.fail(f"{value} must contain ints or floats: {pair[x]} is not.")
		return pair


CROP_NUMBER = CropNumberType()


def write_crop(input, output, crop, compress):
	img = tf.imread(input)
	if compress:
		tf.imwrite(output, img[crop], compression=8)
	else:
		tf.imwrite(output, img[crop])
	log.log("File Written", f"{output.name}: ({img.shape}>{crop})")


def crop_val(x, dim):
	return np.s_[int(x[0] * dim) if isinstance(x[0], float) else x[0]:
			int((1.0 - x[1]) * dim) if isinstance(x[1], float) else dim - x[1]]


@click.command
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for original dataset.', required=True)
@click.option('-o', '--output-dir', type=click.Path(), required=True,
				help='Output path for transformed dataset.')
@click.option('-v', '--vertical-trim', type=CROP_NUMBER, default=0.0,
				help='Vertical trim (top and bottom) as an absolute value (integer) or percent (float)')
@click.option('-h', '--horizontal-trim', type=CROP_NUMBER, default=0.0,
				help='Horizontal trim (top and bottom) as an absolute value (integer) or percent (float)')
@click.option('-z', '--z-trim', type=CROP_NUMBER, default=0.0,
				help='Z-dimension trim (top and bottom) as an absolute value (integer) or percent (float)')
@click.option('--compressed/--uncompressed', default=False,
				help='Whether to compress output data.')
def trim(data_dir, output_dir, vertical_trim, horizontal_trim, z_trim, compressed):
	""" Crop an image stack.
		Crop values can be a comma separated pair like 5,4 or a single value like 3.
		Float values are handled as % of image size; integer values as voxel values.
		"""
	log.start()
	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	path_list = sorted(list(Path(data_dir).glob("*.tif*")))
	path_list = path_list[crop_val(z_trim, len(path_list))]

	with tf.TiffFile(path_list[0]) as tif:
		dim = tif.pages[0].shape

	new_dim = np.s_[crop_val(vertical_trim, dim[0]), crop_val(horizontal_trim, dim[1])]

	with Pool(64) as pool:
		pool.starmap(write_crop, [(path, Path(out_dir, path.name), new_dim, compressed) for path in path_list])


if __name__ == "__main__":
	trim()
