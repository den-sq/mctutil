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
from shared.io_helpers import FLAT, distribute_read 	# noqa::E402
from shared.mem import SharedNP, ProjOrder, SinoOrder 	# noqa::E402


MODE = click.Choice(["full", "preproc"], case_sensitive=False)


def weighted_normalize(sino_mem: SharedNP, input_mem: SharedNP, flats_mem: SharedNP, window, int_window,
					projection: int, projection_count: int, debug_folder: Path = None):
	"""Normalizes single projection using weighted pre/post flats."""
	with sino_mem[int_window] as sino, flats_mem as flats, input_mem as source:
		dark_map = np.average(
			flats[FLAT.PREDARK.index:FLAT.POSTDARK.index + 1, window, :],
			axis=0,
			weights=[projection_count - projection, projection],
		)
		gain_map = np.subtract(
			np.average(
				flats[FLAT.PREGAIN.index:FLAT.POSTGAIN.index + 1, window, :],
				axis=0,
				weights=[projection_count - projection, projection],
			),
			dark_map,
		)
		gain_map[gain_map == 0] = np.min(gain_map[gain_map != 0])
		temp = np.subtract(source[projection, int_window, :].astype(sino_mem.dtype), dark_map)
		sino[int_window, projection, :] = np.divide(temp, gain_map)


def sino_write(sino_mem: SharedNP, path: PathLike, i, out_type: cli.NumpyCLI = None, execute=True):
	with sino_mem as sino:
		if execute:
			if out_type is None:
				tf.imwrite(path, sino[i, :, :])
			else:
				tf.imwrite(path, out_type.convert_ar(sino[i, :, :]))
			log.log("File Written", str(path))
		else:
			log.log("Dry Run", f"Would write {path}")


def image_bounds(sino_mem: SharedNP, i: int = None):
	if i is None:
		with sino_mem as sino:
			return np.array([np.min(sino), np.max(sino)])
	with sino_mem[i] as sino:
		if np.max(sino) > 2:
			log.log("Bounds Check", f"Peak value {np.max(sino):.4g}", log_level=log.DEBUG.INFO)
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
		sino[:, :] = denoise_nl_means(
			sino,
			patch_size=9,
			patch_distance=5,
			fast_mode=True,
			sigma=0.001 * a_sigma_est,
			preserve_range=False,
			channel_axis=None,
		)


def preprocess(sino_mem, i, minval=None, maxval=None):
	minmaxscale(sino_mem, i, minval, maxval)
	remove_outlier(sino_mem, i)


def sh_imread(sino_mem, i, path):
	with sino_mem[i] as sino:
		sino[:, :] = tf.imread(path)


def validate_mode(mode: str, flat_dir: Path | None):
	if mode == "full" and flat_dir is None:
		raise click.UsageError("--flat-dir is required when --mode=full.")


def run_full(input_dir: Path, output_dir: Path, flat_dir: Path, process_count: int, sectioning: int,
			sino_range: range, execute=True):
	image_paths = natsort.natsorted(list(input_dir.glob("**/*.tif*")))
	segment_id = str(uuid.uuid4())
	internal_dtype = np.float32

	with tf.TiffFile(image_paths[0]) as tif:
		page = tif.pages[0]
		pj = {
			"dtype": page.dtype,
			"bytesize": page.dtype.itemsize,
			"offset": page.dataoffsets[0],
			"x": page.shape[1],
			"y": page.shape[0],
		}

	if sino_range is None:
		sino_split = range(0, pj["y"], process_count)
	else:
		sino_split = range(sino_range.start, sino_range.stop, process_count)

	if sectioning:
		output_paths = [
			output_dir.joinpath(
				f"section_{x - x % sectioning:03}_{x - x % sectioning + sectioning:03}",
				f"sino_{x % sectioning:05}.tiff",
			)
			for x in range(pj["y"])
		]
	else:
		if execute:
			output_dir.mkdir(parents=True, exist_ok=True)
		output_paths = [output_dir.joinpath(f"sino_{x:05}.tiff") for x in range(pj["y"])]

	sino_shape = SinoOrder(process_count, len(image_paths), pj["x"])
	proj_shape = ProjOrder(len(image_paths), process_count, pj["x"])
	log.log("Setup", f"{pj}")

	with (
		SharedNP(
			f'flats_{segment_id}',
			pj["dtype"],
			ProjOrder(len(FLAT), pj["y"], pj["x"]),
			create=True,
		) as flats_mem,
		SharedNP(f"sino_{segment_id}", internal_dtype, sino_shape, create=True) as sino_mem,
		SharedNP(f"input_{segment_id}", pj["dtype"], proj_shape, create=True) as input_mem,
	):
		with flats_mem as flat_set:
			for flat in list(FLAT):
				flat_set[flat.index, :, :] = tf.imread(flat_dir.joinpath(f"{flat}_median.tiff")).astype(internal_dtype)

		for x in sino_split:
			window = range(x, min(x + process_count, sino_split.stop))
			internal_window = range(0, len(window))
			log.log("Cycle Start", f"Window {window}; Internal {internal_window}; Shape {sino_shape} from {proj_shape}")
			distribute_read(
				input_mem,
				pj,
				window,
				internal_window,
				enumerate(image_paths),
				thread_max=process_count,
				sino_order=False,
			)
			log.log("Files Read", f"Window {window}; Shape {proj_shape}")
			with Pool(process_count) as pool:
				pool.starmap(
					weighted_normalize,
					[(sino_mem, input_mem, flats_mem, window, internal_window, i, sino_mem.shape.Theta)
						for i in range(sino_mem.shape.Theta)],
				)
			if execute:
				for section_dir in set([fullpath.parent for fullpath in output_paths[window.start:window.stop]]):
					section_dir.mkdir(parents=True, exist_ok=True)
			with Pool(process_count) as pool:
				pool.starmap(
					sino_write,
					[(sino_mem, output_paths[i + window.start], i, None, execute) for i in internal_window],
				)
			log.log(
				"Files Written",
				f"{output_dir} : {window} ({'written' if execute else 'planned'})",
				log.DEBUG.TIME,
			)


def run_preproc(input_dir: Path, output_dir: Path, process_count: int, min_val: float | None, max_val: float | None,
			execute=True):
	image_paths = natsort.natsorted(list(input_dir.glob("**/*.tif*")))
	if execute:
		output_dir.mkdir(parents=True, exist_ok=True)
	output_paths = [output_dir.joinpath(x.name) for x in image_paths]

	segment_id = str(uuid.uuid4())
	internal_dtype = np.float32

	with tf.TiffFile(image_paths[0]) as tif:
		page = tif.pages[0]
		pj = {
			"dtype": page.dtype,
			"bytesize": page.dtype.itemsize,
			"offset": page.dataoffsets[0],
			"x": page.shape[1],
			"y": page.shape[0],
		}

	sino_shape = SinoOrder(process_count, pj["y"], pj["x"])
	bounds = []
	log.log("Setup", f"{pj}")

	if min_val is None or max_val is None:
		with SharedNP(f"sino_{segment_id}", internal_dtype, sino_shape, create=True) as sino_mem:
			for x in range(0, len(image_paths), process_count):
				window = range(x, min(x + process_count, len(image_paths)))
				internal_window = range(0, len(window))
				with Pool(process_count) as pool:
					pool.starmap(sh_imread, [(sino_mem, i, image_paths[window[i]]) for i in internal_window])
				bounds.append(image_bounds(sino_mem))
		bounds = np.asarray(bounds)
		min_val = np.min(bounds[:, 0])
		max_val = np.max(bounds[:, 1])
		log.log("Final Bounds Calculated", f"{min_val} : {max_val}", log.DEBUG.TIME)

	with SharedNP(f"sino_{segment_id}", internal_dtype, sino_shape, create=True) as sino_mem:
		for x in range(0, len(image_paths), process_count):
			window = range(x, min(x + process_count, len(image_paths)))
			internal_window = range(0, len(window))
			with Pool(process_count) as pool:
				pool.starmap(sh_imread, [(sino_mem, i, image_paths[window[i]]) for i in internal_window])
			with Pool(process_count) as pool:
				pool.starmap(preprocess, [(sino_mem, i, min_val, max_val) for i in internal_window])
			with Pool(process_count) as pool:
				pool.starmap(
					sino_write,
					[(sino_mem, output_paths[i + window.start], i, None, execute) for i in internal_window],
				)
			log.log(
				"Files Written",
				f"{output_dir} : {window} ({'written' if execute else 'planned'})",
				log.DEBUG.TIME,
			)


@click.command()
@click.option("--mode", type=MODE, default="full", show_default=True,
				help="Whether to build sinograms from projection+flat data or preprocess existing sinograms.")
@click.option("-i", "--input-dir", type=click.Path(path_type=Path, file_okay=False), required=True,
				help="Directory of input projections or sinograms.")
@click.option("-o", "--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True,
				help="Directory of output sinograms.")
@click.option("-f", "--flat-dir", type=click.Path(path_type=Path, file_okay=False), required=False,
				help="Directory of flats (required for full mode).")
@click.option("-p", "--process-count", type=click.INT, default=cpu_count(),
				help="# of simultaneous processes during conversion. Also used as window size.")
@click.option("-s", "--sectioning", type=click.INT, required=False,
				help="Divide full-mode results into serial sections of this size.")
@click.option("-r", "--sino_range", type=cli.RANGE, required=False,
				help="Sinogram (projection Y dimension) range to create in full mode.")
@click.option("--min-val", type=click.FLOAT, default=None, help="Minimum sinogram value for preproc mode.")
@click.option("--max-val", type=click.FLOAT, default=None, help="Maximum sinogram value for preproc mode.")
@click.option('--execute/--dry-run', default=True,
				help='Whether to write sinogram outputs or only log the planned outputs.')
def sino_convert(
		mode: str,
		input_dir: Path,
		output_dir: Path,
		flat_dir: Path | None,
		process_count: int,
		sectioning: int,
		sino_range: range,
		min_val: float | None,
		max_val: float | None,
		execute: bool,
):
	mode = mode.lower()
	validate_mode(mode, flat_dir)
	if mode == "full":
		run_full(input_dir, output_dir, flat_dir, process_count, sectioning, sino_range, execute=execute)
	else:
		run_preproc(input_dir, output_dir, process_count, min_val, max_val, execute=execute)


if __name__ == "__main__":
	sino_convert()
