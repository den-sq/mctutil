from multiprocessing import Pool
from pathlib import Path

import click
from natsort import natsorted
import numpy as np
import psutil
import tifffile as tf

from mctutil.shared.log import log, LOG


def _validate_triplet(images):
	triplet = np.asarray(images)
	if triplet.ndim != 3 or triplet.shape[0] != 3:
		raise ValueError(
			f"denoise core requires exactly three ZYX planes; got {triplet.shape}"
		)
	return triplet


def threshold_denoised_center(images, threshold):
	"""Replace a center-plane outlier with the mean of its two neighbors."""
	if threshold is None or not 0 <= threshold <= 1:
		raise ValueError("threshold denoise requires a fraction between 0 and 1")
	triplet = _validate_triplet(images)
	working = triplet.astype(np.float64, copy=False)
	gap = (np.max(working) - np.min(working)) * threshold
	mask = np.logical_and(
		np.abs(working[1] - working[0]) > gap,
		np.abs(working[1] - working[2]) > gap,
	)
	result = np.array(triplet[1], copy=True)
	if np.any(mask):
		replacement = (working[0] + working[2]) / 2
		result[mask] = replacement[mask]
	return result


def flat_denoised_center(images, threshold):
	"""Zero center pixels whose preceding and following pixels are both low."""
	if threshold is None:
		raise ValueError("flat denoise requires an absolute threshold")
	triplet = _validate_triplet(images)
	mask = np.logical_and(triplet[0] < threshold, triplet[2] < threshold)
	result = np.array(triplet[1], copy=True)
	result[mask] = 0
	return result


def denoised_volume(volume, mode, threshold, *, boundary="preserve"):
	"""Denoise every interior Z plane without performing any image I/O.

	``boundary='preserve'`` copies the first and last planes unchanged, while
	``boundary='drop'`` returns only the interior planes for legacy leaf output.
	"""
	array = np.asarray(volume)
	if array.ndim != 3:
		raise ValueError(f"denoise requires a three-dimensional ZYX volume; got {array.shape}")
	if mode not in {"threshold", "flat"}:
		raise ValueError(f"unsupported denoise mode: {mode}")
	if boundary not in {"preserve", "drop"}:
		raise ValueError(f"unsupported denoise boundary policy: {boundary}")
	operation = (
		threshold_denoised_center
		if mode == "threshold"
		else flat_denoised_center
	)
	if len(array) < 3:
		if boundary == "preserve":
			return np.array(array, copy=True)
		return np.empty((0,) + array.shape[1:], dtype=array.dtype)
	centers = np.stack(
		tuple(
			operation(array[index - 1:index + 2], threshold)
			for index in range(1, len(array) - 1)
		),
		axis=0,
	)
	if boundary == "drop":
		return centers
	result = np.array(array, copy=True)
	result[1:-1] = centers
	return result


def denoise_threshold(input_paths, output_path, threshold):
	base_data = np.stack(tuple(tf.imread(infile) for infile in input_paths))
	output = threshold_denoised_center(base_data, threshold)
	log.write(
		"Simple Denoise",
		f"threshold={threshold}; inputs={[path.name for path in input_paths]}",
		log_level=LOG.INFO,
	)
	tf.imwrite(output_path, output.astype(np.uint16))


def denoise_flat(input_paths, output_path, threshold):
	base_data = np.stack(tuple(tf.imread(infile) for infile in input_paths))
	tf.imwrite(output_path, flat_denoised_center(base_data, threshold))


@click.command()
@click.option("-a", "--area", type=click.INT, help="Area for block denoising.")
@click.option("-t", "--threshold", type=click.FLOAT,
				help="Difference threshold to mark as noise above.")
@click.option("-n", "--num-processes", type=click.INT, default=psutil.cpu_count(),
				help="Number of simultaneous processes.")
@click.option("--flat-denoise/--threshold-denoise", type=click.BOOL, default=False,
				help="Whether to use a ")
@click.argument("INPUTDIR", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.argument("OUTPUTDIR", type=click.Path(path_type=Path, file_okay=False), required=True)
def simple_denoise(threshold, area, num_processes, flat_denoise, inputdir, outputdir):
	input_paths = natsorted(list(inputdir.glob("**/*.tif*")))

	outputdir.mkdir(parents=True, exist_ok=True)

	with Pool(num_processes) as pool:
		if flat_denoise:
			log.write("Simple Denoise", "Mode: flat", log_level=LOG.STATUS)
			pool.starmap(denoise_flat, [(input_paths[i - 1: i + 2], outputdir.joinpath(input_paths[i].name), threshold)
										for i in range(1, len(input_paths) - 1)])
		else:
			log.write("Simple Denoise", f"Mode: threshold ({len(input_paths)} inputs)", log_level=LOG.STATUS)
			pool.starmap(denoise_threshold, [(input_paths[i - 1: i + 2], outputdir.joinpath(input_paths[i].name), threshold)
											for i in range(1, len(input_paths) - 1)])


if __name__ == "__main__":
	simple_denoise()
