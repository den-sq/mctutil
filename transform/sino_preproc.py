from os import PathLike
from pathlib import Path
from multiprocessing import Pool
import sys
import uuid

import click
import natsort
import numpy as np
from psutil import cpu_count
from skimage.restoration import denoise_nl_means, estimate_sigma
import tifffile as tf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402
from shared import cli 	# noqa::E402
from shared.io_helpers import FLAT 	# noqa::E402
from shared.mem import SharedNP, ProjOrder, SinoOrder 	# noqa::E402


def weighted_normalize(sino_mem: SharedNP, input_mem: SharedNP, flats_mem: SharedNP, window, int_window,
					projection: int, projection_count: int, debug_folder: Path = None):
	"""Normalizes single projection using weighted pre/post flats."""
	with sino_mem[int_window] as sino, flats_mem as flats, input_mem as source:
		dark_map = np.average(flats[FLAT.PREDARK.index:FLAT.POSTDARK.index + 1, window, :], axis=0,
							weights=[projection_count - projection, projection])
		gain_map = np.subtract(np.average(flats[FLAT.PREGAIN.index:FLAT.POSTGAIN.index + 1, window, :], axis=0,
							weights=[projection_count - projection, projection]), dark_map)
		temp = np.subtract(source[projection, int_window, :].astype(sino_mem.dtype), dark_map)

		sino[:, projection, :] = np.divide(temp, gain_map)


def sino_write(sino_mem: SharedNP, path: PathLike, i, out_type: cli.NumpyCLI = None):
	with sino_mem as sino:
		if out_type is None:
			tf.imwrite(path, sino[i, :, :])
		else:
			tf.imwrite(path, out_type.convert_ar(sino[i, :, :]))


def image_bounds(sino_mem: SharedNP):
	with sino_mem as sino:
		return np.array([np.min(sino), np.max(sino)])


def minmaxscale(sino_mem, i, minval=None, maxval=None):
	with sino_mem[i] as sino:
		if minval is None:
			minval = np.min(sino)
		if maxval is None:
			maxval = np.max(sino)
		sino[:, :] = (sino - minval) / (maxval - minval)


def remove_outlier(sino_mem, i):
	with sino_mem[i] as sino:
		a_sigma_est = estimate_sigma(sino, channel_axis=None, average_sigmas=True)
		sino[:, :] = denoise_nl_means(sino, patch_size=9, patch_distance=5,
							fast_mode=True, sigma=0.001 * a_sigma_est,
							preserve_range=False, channel_axis=None)


def preprocess(sino_mem, i, minval=None, maxval=None):
	minmaxscale(sino_mem, i, minval, maxval)
	remove_outlier(sino_mem, i)


def sh_imread(sino_mem, i, path):
	with sino_mem[i] as sino:
		sino[:, :] = tf.imread(path)


@click.command()
@click.option("-i", "--input-dir", type=click.Path(path_type=Path, file_okay=False), required=True,
				help="Directory of Input Projections.")
@click.option("-o", "--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True,
				help="Directory of Output Sinograms.")
@click.option("-p", "--process-count", type=click.INT, default=cpu_count(),
				help="# of simulatenous processes during conversion.  Also used as window size.")
@click.option("--min-val", type=click.FLOAT, default=None, help="Minimum Value of Sinogram Set")
@click.option("--max-val", type=click.FLOAT, default=None, help="Maximum Value of Sinogram Set")
def sino_convert(input_dir: Path, output_dir: Path, process_count: int, min_val: int, max_val: int):
	image_paths = natsort.natsorted(list(input_dir.glob("**/*.tif*")))
	output_dir.mkdir(parents=True, exist_ok=True)
	output_paths = [output_dir.joinpath(x.name) for x in image_paths]

	segment_id = str(uuid.uuid4())
	internal_dtype = np.float32

	with tf.TiffFile(image_paths[0]) as tif:
		page = tif.pages[0]
		pj = {"dtype": page.dtype, "bytesize": page.dtype.itemsize, "offset": page.dataoffsets[0],
				"x": page.shape[1], "y": page.shape[0]}

	sino_shape = SinoOrder(process_count, pj["y"], pj["x"])
	bounds = []

	log.log("Setup", f"{pj}")

	if min_val is None or max_val is None:
		with (SharedNP(f"sino_{segment_id}", internal_dtype, sino_shape, create=True) as sino_mem):
			for x in range(0, len(image_paths), process_count):
				window = range(x, min(x + process_count, len(image_paths)))
				internal_window = range(0, len(window))

				log.log("Preprocess Cycle Start", f"Window {window}; Shape {sino_shape}")

				with Pool(process_count) as pool:
					pool.starmap(sh_imread, [(sino_mem, i, image_paths[window[i]]) for i in internal_window])

				log.log("Files Read", f"Window {window}; Shape {sino_shape}")

				bounds.append(image_bounds(sino_mem))

				log.log("Bounds Calculated", f"{window}", log.DEBUG.TIME)

		bounds = np.asarray(bounds)
		min_val = np.min(bounds[:, 0])
		max_val = np.max(bounds[:, 1])
		log.log("Final Bounds Calculated", f"{min_val} : {max_val}", log.DEBUG.TIME)

	with (SharedNP(f"sino_{segment_id}", internal_dtype, sino_shape, create=True) as sino_mem):
		for x in range(0, len(image_paths), process_count):
			window = range(x, min(x + process_count, len(image_paths)))
			internal_window = range(0, len(window))

			log.log("Preprocess Cycle Start", f"Window {window}; Shape {sino_shape}")

			with Pool(process_count) as pool:
				pool.starmap(sh_imread, [(sino_mem, i, image_paths[window[i]]) for i in internal_window])

			log.log("Files Read", f"Window {window}; Shape {sino_shape}")

			with Pool(process_count) as pool:
				pool.starmap(preprocess, [(sino_mem, i, min_val, max_val) for i in internal_window])

			log.log("Sinogram Preprocessing", f"{min_val} : {max_val}", log.DEBUG.TIME)

			with Pool(process_count) as pool:
				pool.starmap(sino_write, [(sino_mem, output_paths[i + window.start], i) for i in internal_window])

			log.log("Files Written", f"{output_dir} : {window}", log.DEBUG.TIME)


if __name__ == "__main__":
	sino_convert()
