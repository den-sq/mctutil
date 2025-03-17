from pathlib import Path

import click
import numpy as np
import tifffile as tf


@click.command()
@click.argument("INPUTDIR", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.argument("OUTPUTDIR", type=click.Path(path_type=Path, file_okay=False), required=True)
def f_transpose(inputdir: Path, outputdir: Path):
	"""Just a quick transpose script to load everything and switch z/x axes."""
	im_list = sorted(list(Path(inputdir).iterdir()))
	with tf.TiffFile(im_list[0]) as im:
		old_shape = (len(im_list), im.pages[0].shape[0], im.pages[0].shape[1])
		old_dtype = im.pages[0].dtype

	full_data = np.empty(old_shape, dtype=old_dtype)
	for i, im in enumerate(im_list):
		full_data[i, :, :] = tf.imread(im)
	transposed_data = np.transpose(full_data, [2, 1, 0])

	outputdir.mkdir(parents=True, exist_ok=True)
	for i in range(old_shape[2]):
		tf.imwrite(outputdir.joinpath(f"tp_{str(i).zfill(4)}.tif"), transposed_data[i])


if __name__ == "__main__":
	f_transpose()
