from multiprocessing import Pool
from pathlib import Path
import sys

import click
from natsort import natsorted
import numpy as np
from numba import jit
import psutil
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))\

from shared import log 	# noqa::E402


@jit(nopython=True, cache=True)
def make_slice_3d(point, window):
	return (slice(point[0] - window // 2, point[0] + window // 2 + 1),
			slice(point[1] - window // 2, point[1] + window // 2 + 1),
			slice(point[2] - window // 2, point[2] + window // 2 + 1))


@jit(nopython=True, cache=True)
def make_slice_2d(point, window, is_3d=False):
	return (slice(point[0] - window // 2, point[0] + window // 2 + 1),
			slice(point[1] - window // 2, point[1] + window // 2 + 1))


@jit(nopython=True, cache=True)
def shift_point(point, flat_shift, d_shift):
	return np.add(np.subtract(point, flat_shift), d_shift)


@jit(nopython=True, cache=True)
def make_np_index(start, space):
	return [np.add(np.array([int(i // np.prod(space[:j])) % space[j] for j in range(len(space))]), start) for i in range(np.prod(space))]


# Function to calculate the weighted average value (Ip) for each pixel
@jit(nopython=True, cache=True)
def evaluateNorm2d(padImg, pixelWindow, point, small_window, big_window, Nw):

	# Get centers of neighboring regions.
	center_set = [shift_point(point, big_window // 2, index) for index in make_np_index(0, np.array([big_window] * len(padImg.shape)))]

	w_set = np.array([np.exp(-1 * ((np.sum((padImg[pixelWindow] - padImg[make_slice_2d(center, small_window)])**2)) / Nw))
				for center in center_set])
	Iq_set = np.array([padImg[center[0], center[1]] for center in center_set])

	Ip_Numerator = np.sum(w_set)
	Z = np.sum(w_set * Iq_set)

	if Z:
		return Ip_Numerator / Z
	else:
		return 0


# Function to calculate the weighted average value (Ip) for each pixel
@jit(nopython=True, cache=True)
def evaluateNorm3d(padImg, pixelWindow, point, small_window, big_window, Nw):

	# Get centers of neighboring regions.
	center_set = [shift_point(point, big_window // 2, index) for index in make_np_index(0, np.array([big_window] * len(padImg.shape)))]

	# Calculating norm if Ip - Iq
	w_set = [np.exp(-1 * ((np.sum((padImg[pixelWindow] - padImg[make_slice_3d(center, small_window)])**2)) / Nw))
				for center in center_set]
	Iq_set = [padImg[center] for center in center_set]

	Ip_Numerator, Z = np.sum([(w_set[i], w_set[i] * Iq_set[i]) for i in range(len(center_set))], axis=0)

	if Z:
		return Ip_Numerator / Z
	else:
		return 0


@jit(nopython=True, cache=True)
def nlm(padImg, shape, padding, sigma_h, small_window, big_window, is_3d):
	# Calculating neighborhood window
	Nw = (sigma_h**2) * (small_window**2)
	max_val, min_val = 0, 1.0
	
# 	np.iinfo(padImg.dtype).max, np.iinfo(padImg.dtype).min

# 	print(f"{shape}:{is_3d}:{padImg.shape}")

	# Calcualte NL Means
	if is_3d:
		pass
	else:
		Ip_set = [evaluateNorm2d(padImg, make_slice_2d(point, small_window), point, small_window, big_window, Nw)
					for point in make_np_index(padding, shape)]

# 	log.log("NLM", "IP Results calcualted.")

	result = np.array([max(min(max_val, Ip), min_val) for Ip in Ip_set])

# 	log.log("BOUNDS", "Bounds checked; min {np.min(result)}; max {np.max(result)}")
	return result


def nlm_solve(img_path, out_path, sigma_h, small_window, big_window, is_3d):
	"""
	Solve function to perform nlmeans filtering.

	:param img_path: path to noisy image.
	:param out_path: output path.
	:param sigma_h: sigma h (as mentioned in the paper)
	:param small_window: size of small window
	:param big_window: size of big window
	:param is_3d: whether to operate in 3d.
	:rtype: uint8 (w,h)
	:return: solved image
	"""
	# Padding the original image with reflect mode
	padding = big_window // 2 + small_window // 2 + 1
	padImg = np.pad(tf.imread(img_path)[1500:2500, 1500:2500], padding, mode='reflect')
	shape = np.subtract(padImg.shape, padding * 2)

	# dumb but needed for jit
# 	if not is_3d:
# 		padImg.reshape(padImg.shape[0], padImg.shape[1], np.int64(1))
# 		shape = np.array([shape[0], shape[1], np.int64(1)])

	log.log("FILE_READ", f"{img_path}:{padImg.shape}")
	tf.imwrite(out_path, nlm(padImg, shape, padding, sigma_h, small_window, big_window, is_3d).reshape(shape))
	log.log("FILE_WRITE", out_path)


@jit(nopython=True, cache=True)
def make_slice_npindex(point, window, is_3d=False):
	return [np.arange(point[x] - window // 2, point[x] + window // 2 + 1) for x in range(len(point))]


@click.command()
@click.option("-h", "--sigma-h", type=click.INT, default=30,
				help="---")
@click.option("-s", "--small-window", type=click.INT, default=7, help="Small window size for NLM denoising.")
@click.option("-l", "--big-window", type=click.INT, default=21, help="Big window size for NLM denoising.")
@click.option("-n", "--num-processes", type=click.INT, default=psutil.cpu_count(),
				help="Number of simultaneous processes.")
@click.option("--is-3d/--is-2d", type=click.BOOL, default=False, help="Operate in 2 or 3 dimensions.")
@click.argument("INPUTDIR", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.argument("OUTPUTDIR", type=click.Path(path_type=Path, file_okay=False), required=True)
def nlm_denoise(sigma_h, small_window, big_window, num_processes, is_3d, inputdir, outputdir):
	log.start()

	input_paths = natsorted(list(inputdir.glob("**/*.tif*")))

	outputdir.mkdir(parents=True, exist_ok=True)

	with Pool(num_processes) as pool:
		if is_3d:
			padding = big_window // 2 + small_window // 2 + 1
			pool.starmap(nlm_solve, [(input_paths[i - padding:i + padding], outputdir.joinpath(input_paths[i].name),
										sigma_h, small_window, big_window, is_3d)
										for i in range(padding, len(input_paths) - padding)])
		else:
			pool.starmap(nlm_solve, [(img_path, outputdir.joinpath(img_path.name), sigma_h, small_window, big_window, is_3d)
										for img_path in input_paths])

	log.log((f"{len(input_paths)} complete."))


if __name__ == "__main__":
	nlm_denoise()
