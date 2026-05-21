from pathlib import Path

import click
from natsort import natsorted
import tifffile as tf

from shared import log


@click.command()
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually rewrite the files or just plan the rewrites.")
@click.argument("image_path", type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path))
def uncompress(execute: bool, image_path: Path):
	""" Rewrite every TIFF under IMAGE_PATH with compression removed.

		With --dry-run, lists which files would be rewritten without touching them.
	"""
	images = natsorted(image_path.glob("*.tif*"))
	for im_path in images:
		if execute:
			image = tf.imread(im_path)
			tf.imwrite(im_path, image, compression=None)
			log.log("Uncompress", f"Rewrote {im_path}", log_level=log.DEBUG.STATUS)
		else:
			log.log("Uncompress", f"Would rewrite {im_path}", log_level=log.DEBUG.INFO)
	log.log("Uncompress", f"{len(images)} files {'rewritten' if execute else 'planned'}", log_level=log.DEBUG.STATUS)


if __name__ == "__main__":
	uncompress()
