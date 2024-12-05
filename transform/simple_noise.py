from multiprocessing import Pool
from pathlib import Path

import click
from natsort import natsorted
import numpy as np
import psutil
import tifffile as tf


def denoise_threshold(input_paths, output_path, threshold):
	print(input_paths)
	base_data = np.array([tf.imread(infile) for infile in input_paths]).astype(np.int32)

	floor = np.min(base_data)
	ceiling = np.max(base_data)
	gap = (ceiling - floor) * threshold

	mask = np.logical_and(np.abs(np.subtract(base_data[1], base_data[0])) > gap,
							np.abs(np.subtract(base_data[1], base_data[2]) > gap))

	if len(base_data[0][mask] > 0):
		base_data[1][mask] = np.average([base_data[0][mask], base_data[2][mask]], axis=0)

	tf.imwrite(output_path, base_data[1].astype(np.uint16))


def denoise_flat(input_paths, output_path, threshold):
	base_data = np.array([tf.imread(infile) for infile in input_paths])

	mask = np.logical_and((base_data[0] < threshold), (base_data[2] < threshold))
	base_data[1][mask] = 0
	tf.imwrite(output_path, base_data[1])


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
			print("flat")
			pool.starmap(denoise_flat, [(input_paths[i - 1: i + 2], outputdir.joinpath(input_paths[i].name), threshold)
										for i in range(1, len(input_paths) - 1)])
		else:
			print(f"threshold {len(input_paths)}")
			pool.starmap(denoise_threshold, [(input_paths[i - 1: i + 2], outputdir.joinpath(input_paths[i].name), threshold)
											for i in range(1, len(input_paths) - 1)])


if __name__ == "__main__":
	simple_denoise()
