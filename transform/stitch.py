from pathlib import Path

import click
import numpy as np
import tifffile as tf


class SampleParameter(click.ParamType):
	name = "Sample Information"

	def convert(self, value, param, ctx):
		values = value.split(",")
		proj_folder = Path(values[0])
		flat_folder = None if len(values) < 2 else values[1]
		center = None if len(values) < 3 else values[2]
		tilt = None if len(values) < 4 else values[3]

		return SampleSet(proj_folder, flat_folder, center, tilt)


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
@click.option("-s", "--sample", type=SampleParameter(), multiple=True,
				help="Sample information; comma separated list of projection folder, flat folder, rotational center, and tilt.")
@click.option("--top-stitch/--bottom-stitch", type=click.BOOL, default=False,
				help="Vertical side of first sample to stitch at.")
@click.option("--stitch-range", type=click.FLOAT, default=0.2, help="Maximum")
@click.argument("output_samples")
def stitch(sample, top_stitch, stitch_range, output_samples):
	# samples = [SampleSet(first_sample, first_flat, first_center), SampleSet(second_sample, second_flat, second_center)]
	target_width = min([x.half_width for x in sample])

	for s in sample:
		s.half_width = target_width
		print(s.proj.shape)

	# Stich the samples in reverse order if we're stitching starting at the top; simpler this way.
	if top_stitch:
		sample.reverse()

	overlaps = []
	stitch_scope = range(20, int(sample[0].proj.shape[0] * stitch_range))
	for y in range(0, len(sample) - 1):
		std_set = [(x, np.std(np.divide(sample[y].proj_bot(x), sample[y + 1].proj_top(x)))) for x in stitch_scope]
		std_set.sort(key=lambda x: x[1])
		print(std_set[:10])

		overlaps.append(std_set[0])

	stitch_proj(sample, overlaps)


if __name__ == "__main__":
	stitch()

