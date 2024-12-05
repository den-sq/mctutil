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
# 	print(f"{floor}|{ceiling}|{type(norm_data)}|{type(ceiling - floor)}")
# 	np.divide(norm_data, np.float_(ceiling - floor), out=norm_data)

	mask = np.logical_and(np.abs(np.subtract(base_data[1], base_data[0])) > gap,
							np.abs(np.subtract(base_data[1], base_data[2]) > gap))

# 	print(f"c{np.abs(np.subtract(base_data[1], base_data[0]))[mask]}:{np.abs(np.subtract(base_data[1], base_data[2]))[mask]}|{gap}")
	if len(base_data[0][mask] > 0):
		print(f"c{np.subtract(base_data[1], base_data[0])[mask]}:{np.subtract(base_data[1], base_data[2])[mask]}|{gap}")
		print(f"b{base_data[0][mask]}:{base_data[1][mask]}:{base_data[2][mask]}")
		base_data[1][mask] = np.average([base_data[0][mask], base_data[2][mask]], axis=0)
	else:
		print(f"{np.max(np.abs(np.subtract(base_data[1], base_data[0])))} : {np.max(np.abs(np.subtract(base_data[1], base_data[2])))} : {gap}")
# 		print(f"b{base_data[0][mask]}:{base_data[1][mask]}:{base_data[2][mask]}")

	tf.imwrite(output_path, base_data[1].astype(np.uint16))


def denoise_flat(input_paths, output_path):
	print(input_paths)
	base_data = np.array([tf.imread(infile) for infile in input_paths])

	mask = np.logical_and((base_data[0] == 0), (base_data[2] == 0))
	print(f"b{base_data[0][mask]}:{np.sum(base_data[0][mask])}")
	print(f"m{base_data[1][mask]}:{np.sum(base_data[1][mask])}")
	print(f"t{base_data[2][mask]}:{np.sum(base_data[2][mask])}")
	base_data[1][mask] = np.average([base_data[0][mask], base_data[2][mask]], axis=0)
	print(f"f{base_data[1][mask]}:{np.sum(base_data[1][mask])}")
	tf.imwrite(output_path, base_data[1])


@click.command()
@click.option("-t", "--threshold", type=click.FLOAT,
				help="Difference threshold to mark as noise above.")
@click.option("-n", "--num-processes", type=click.INT, default=psutil.cpu_count(),
				help="Number of simultaneous processes.")
@click.option("--flat-denoise/--threshold-denoise", type=click.BOOL, default=False,
				help="Whether to use a ")
@click.argument("INPUTDIR", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.argument("OUTPUTDIR", type=click.Path(path_type=Path, file_okay=False), required=True)
def simple_denoise(threshold, num_processes, flat_denoise, inputdir, outputdir):
	input_paths = natsorted(list(inputdir.glob("**/*.tif*")))

	outputdir.mkdir(parents=True, exist_ok=True)

	with Pool(num_processes) as pool:
		if flat_denoise:
			print("flat")
			pool.starmap(denoise_flat, [(input_paths[i - 1: i + 2], outputdir.joinpath(input_paths[i].name))
										for i in range(1, len(input_paths) - 1)])
		else:
			print(f"threshold {len(input_paths)}")
			pool.starmap(denoise_threshold, [(input_paths[i - 1: i + 2], outputdir.joinpath(input_paths[i].name), threshold)
											for i in range(1, len(input_paths) - 1)])


if __name__ == "__main__":
	simple_denoise()
