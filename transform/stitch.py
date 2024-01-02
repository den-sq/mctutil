from pathlib import Path

import click
import numpy as np
import tifffile as tf


@click.command()
@click.option("-f", "--first-sample", type=click.Path(), required=True, help="First stitching sample.")
@click.option("---first-flat", type=click.Path(), help="Flatfield(s) for first sample.")
@click.option("--first-center", type=click.FLOAT, help="Rotational center of first image.")
@click.option("-f", "--second-sample", type=click.Path(), required=True, help="Second stitching sample.")
@click.option("--second-flat", type=click.Path(), help="Flatfield(s) for second sample.")
@click.option("--second-center", type=click.FLOAT, help="Rotational center of second image.")
@click.option("--top-stitch/--bottom-stitch", type=click.BOOL, default=False,
				help="Vertical side of first sample to stitch at.")
@click.option("--stitch-range", type=click.FLOAT, default=0.2, help="Maximum")
@click.argument("output_samples")
def stitch(first_sample, first_flat, first_center, second_sample, second_flat, second_center, top_stitch, stitch_range,
			output_samples):
	first_proj = tf.imread(sorted(list(Path(first_sample).iterdir()))[0])
	if first_flat is not None:
		first_flatset = np.array()
		np.divide(first_proj, np.median(first_flatset), out=first_flat)

	second_proj = tf.imread(sorted(list(Path(second_sample).iterdir()))[0])
	if second_flat is not None:
		second_flatset = np.array()
		np.divide(second_proj, np.median(second_flatset), out=second_flat)

	print(first_proj.shape)
	print(second_proj.shape)


if __name__ == "__main__":
	stitch()
