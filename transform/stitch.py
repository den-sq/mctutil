from pathlib import Path

import click
import numpy as np
import tifffile as tf


class SampleSet():
	def __init__(self, proj_folder, flat_folder=None, center=None, tilt=None):
		self.projs = sorted(list(Path(proj_folder).iterdir()))
		temp_proj = tf.imread(self.projs[0])
		if flat_folder is not None:
			flatset = np.stack([tf.memmap(flat) for flat in Path(flat_folder).iterdir() if flat.suffix[:4] == '.tif'], axis=0)
			self._proj = np.divide(temp_proj, np.median(flatset, axis=0).astype(temp_proj.dtype))
		self._center = center if center is not None else self._proj.shape[1] // 2
		self._half_width = min(self._center, self._proj.shape[1] - self._center)

	@property
	def half_width(self):
		return self._half_width

	@half_width.setter
	def half_width(self, value):
		self._half_width = value

	@property
	def proj(self):
		return self._proj[:, self._center - self._half_width: self._center + self._half_width]

	def proj_top(self, x):
		return self._proj[0:x, self._center - self._half_width: self._center + self._half_width]

	def proj_bot(self, x):
		return self._proj[-x:, self._center - self._half_width: self._center + self._half_width]


def stitch_proj(samples, overlaps):
	print(overlaps)


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
	samples = [SampleSet(first_sample, first_flat, first_center), SampleSet(second_sample, second_flat, second_center)]
	target_width = min([x.half_width for x in samples])

	for sample in samples:
		sample.half_width = target_width
		print(sample.proj.shape)

	# Stich the samples in reverse order if we're stitching starting at the top; simpler this way.
	if top_stitch:
		samples.reverse()

	overlaps = []
	stitch_scope = range(20, int(samples[0].proj.shape[0] * stitch_range))
	for y in range(0, len(samples) - 1):
		std_set = [(x, np.std(np.divide(samples[y].proj_bot(x), samples[y + 1].proj_top(x)))) for x in stitch_scope]
		std_set.sort(key=lambda x: x[1])
		print(std_set[:10])

		overlaps.append(std_set[0])

	stitch_proj(samples, overlaps)


if __name__ == "__main__":
	stitch()

