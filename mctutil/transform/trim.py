from multiprocessing import Pool
from pathlib import Path

import click
import tifffile as tf


from mctutil.shared import cli
from mctutil.shared.log import log, LOG  # noqa: F401


def write_crop(input, output, crop, compress, execute=True):
	img = tf.imread(input)
	if execute:
		if compress:
			tf.imwrite(output, img[crop], compression=8)
		else:
			tf.imwrite(output, img[crop])
		log.write("File Written", f"{output.name}: ({img.shape}>{crop})")
	else:
		log.write("Dry Run", f"Would write {output.name}: ({img.shape}>{crop})")


@click.command
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for original dataset.', required=True)
@click.option('-o', '--output-dir', type=click.Path(), required=True,
				help='Output path for transformed dataset.')
@click.option('-v', '--vertical-trim', type=cli.CROP_NUMBER, default="0.0",
				help='Vertical trim (top and bottom) as an absolute value (integer) or percent (float)')
@click.option('-h', '--horizontal-trim', type=cli.CROP_NUMBER, default="0.0",
				help='Horizontal trim (top and bottom) as an absolute value (integer) or percent (float)')
@click.option('-z', '--z-trim', type=cli.CROP_NUMBER, default="0.0",
				help='Z-dimension trim (top and bottom) as an absolute value (integer) or percent (float)')
@click.option('--compressed/--uncompressed', default=False,
				help='Whether to compress output data.')
@click.option('--execute/--dry-run', default=True,
				help='Whether to write cropped files or only log the planned outputs.')
def trim(data_dir, output_dir, vertical_trim, horizontal_trim, z_trim, compressed, execute):
	"""Crop an image stack.
	Crop values can be a comma separated pair like 5,4 or a single value like 3.
	Float values are handled as % of image size; integer values as voxel values.
	"""
	log.start()
	out_dir = Path(output_dir)
	if execute:
		out_dir.mkdir(parents=True, exist_ok=True)
	path_list = sorted(list(Path(data_dir).glob("*.tif*")))
	path_list = path_list[cli.crop_val(z_trim, len(path_list))]

	with tf.TiffFile(path_list[0]) as tif:
		dim = tif.pages[0].shape

	new_dim = (cli.crop_val(vertical_trim, dim[0]), cli.crop_val(horizontal_trim, dim[1]))

	with Pool(64) as pool:
		pool.starmap(write_crop, [(path, Path(out_dir, path.name), new_dim, compressed, execute) for path in path_list])


if __name__ == "__main__":
	trim()
