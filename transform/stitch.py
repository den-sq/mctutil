from multiprocessing import Pool, shared_memory
from pathlib import Path
import sys

import click
import numpy as np
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402
from shared.mem import SharedNP, ProjOrder 	# noqa::E402


class SampleParameter(click.ParamType):
	name = "Sample Information"

	def convert(self, value, param, ctx):
		values = value.split(",")
		proj_folder = Path(values[0])
		pre_flat = None if len(values) < 2 else values[1]
		post_flat = None if len(values) < 3 else values[2]
		center = None if len(values) < 4 else values[3]
		tilt = None if len(values) < 5 else values[4]

		return SampleSet(proj_folder, pre_flat, post_flat, center, tilt)


class SampleSet():
	def __init__(self, proj_folder, pre_flat=None, post_flat=None, center=None, tilt=None):
		self.projs = sorted(list(Path(proj_folder).iterdir()))
		temp_proj = tf.imread(self.projs[0])
		if pre_flat is not None:
			flatset = np.stack([tf.memmap(flat) for flat in Path(pre_flat).iterdir() if flat.suffix[:4] == '.tif'], axis=0)
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

	def del_proj(self):
		self._proj = None

	def load_proj(self, x):
		self._proj = tf.imread(self.projs[x])


def stitch_single(samples, overlaps, new_dim, stitch_output, x):
	stitched = np.zeros(new_dim, dtype=np.uint16)

	for s in samples:
		s.load_proj(x)

	stitched[0:samples[0].proj.shape[0], :] = samples[0].proj
	offset = samples[0].proj.shape[0]

	for i, overlap in enumerate(overlaps):
		non_overlap = samples[i + 1].proj.shape[0] - overlap
		# print(f"{offset}|{overlap}|{non_overlap}")

		# Proj Merge
		np.median([stitched[offset - overlap:offset, :], samples[i + 1].proj_top(overlap)], axis=0,
					out=stitched[offset - overlap:offset, :])

		# Next section.
		stitched[offset: offset + non_overlap, :] = samples[i + 1].proj_bot(non_overlap)

		offset += non_overlap

	tf.imwrite(Path(stitch_output, f"AAA590_Stitched_{x}.tif"), stitched)


def stitch_samples(samples, overlaps, stitch_output):
	new_dim = (np.sum([s.proj.shape[0] for s in samples]) - np.sum(overlaps), samples[0].half_width * 2)

	for s in samples:
		s.del_proj()

	log.log("Dimension", new_dim)

	Path(stitch_output).mkdir(parents=True, exist_ok=True)

	log.log("Output Dir", stitch_output)

	with Pool(50) as pool:
		pool.starmap(stitch_single, [(samples, overlaps, new_dim, stitch_output, x) for x in range(len(samples[0].projs))])

	log.log("Stitching Complete", f"{len(samples[0].projs)} Projections Stitched")


@click.command()
@click.option("-s", "--sample", type=SampleParameter(), multiple=True,
				help="Sample information; comma separated list of projection folder, flat folder, rotational center, and tilt.")
@click.option("--top-stitch/--bottom-stitch", type=click.BOOL, default=False,
				help="Vertical side of first sample to stitch at.")
@click.option("--stitch-range", type=click.FLOAT, default=0.2,
				help="Maximum range (as percent of height) to scan for overlaps.")
@click.option("--stitch-output", type=click.Path(), required=True, help="Output path of stitching.")
def stitch(sample, top_stitch, stitch_range, stitch_output):
	# samples = [SampleSet(first_sample, first_flat, first_center), SampleSet(second_sample, second_flat, second_center)]
	target_width = min([x.half_width for x in sample])

	for s in sample:
		s.half_width = target_width

	log.log("Proj Shape", f"HW: {target_width}; P: {[s.proj.shape for s in sample]}")

	# Stich the samples in reverse order if we're stitching starting at the top; simpler this way.
	if top_stitch:
		sample.reverse()
		log.log("Reverse", "Reversed to handle top/bottom stitching.")

	overlaps = []
	stitch_scope = range(20, int(sample[0].proj.shape[0] * stitch_range))
	for y in range(len(sample) - 1):
		std_set = [(x, np.std(np.divide(sample[y].proj_bot(x), sample[y + 1].proj_top(x)))) for x in stitch_scope]
		std_set.sort(key=lambda x: x[1])
		log.log("Overlap", f"{std_set[0]}")

		overlaps.append(std_set[0][0])

	stitch_samples(sample, overlaps, stitch_output)


if __name__ == "__main__":
	stitch()

