from pathlib import Path

import click
from natsort import natsorted
import tifffile as tf

@click.command()
@click.argument("image_path", type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path))
def uncompress(image_path: Path):
	images = natsorted(image_path.glob("*.tif*"))
	for im_path in images:
		image = tf.imread(im_path)
		tf.imwrite(im_path, image, compression=None)
		print(f"Rewrote {im_path}")

if __name__ == "__main__":
	uncompress()
