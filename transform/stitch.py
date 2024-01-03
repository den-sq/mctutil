from pathlib import Path

import click
import numpy as np
import tifffile as tf


@click.command()
@click.option("-f", "--first-sample", type=click.Path(), required=True, help="First stitching sample.")
@click.option("--first-flat", type=click.Path(), help="Flatfield(s) for first sample.")
@click.option("--first-center", type=click.FLOAT, help="Rotational center of first image.")
@click.option("-s", "--second-sample", type=click.Path(), required=True, help="Second stitching sample.")
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
		first_flatset = np.stack([tf.memmap(flat) for flat in Path(first_flat).iterdir() if flat.suffix[:4] == '.tif'], axis=0)
		first_proj = np.divide(first_proj, np.median(first_flatset, axis=0).astype(first_proj.dtype))

	second_proj = tf.imread(sorted(list(Path(second_sample).iterdir()))[0])
	if second_flat is not None:
		second_flatset = np.stack([tf.memmap(flat) for flat in Path(second_flat).iterdir() if flat.suffix[:4] == '.tif'], axis=0)
		second_proj = np.divide(second_proj, np.median(second_flatset, axis=0).astype(second_proj.dtype))

	target_width = first_proj.shape[1] // 2
	if first_center is not None:
		target_width = min(target_width, first_center, first_proj.shape[1] - first_center)
	else:
		first_center = first_proj.shape[1] // 2
	if second_center is not None:
		target_width = min(target_width, second_center, second_proj.shape[1] - second_center)
	else:
		second_center = second_proj.shape[1] // 2

	first_proj = first_proj[:, first_center - target_width: first_center + target_width]
	second_proj = second_proj[:, second_center - target_width: second_center + target_width]

	print(first_proj.shape)
	print(second_proj.shape)

	std_set = [(x, np.std(np.divide(first_proj[0:x, :] if top_stitch else first_proj[-x:, :],
								second_proj[-x:, :] if top_stitch else second_proj[0:x, :])))
								for x in range(20, int(first_proj.shape[0] * stitch_range))]

	std_set.sort(key=lambda x: x[1])

	print(std_set[:10])



if __name__ == "__main__":
	stitch()

