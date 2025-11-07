from multiprocessing import Pool
from pathlib import Path
import sys

import click
import numpy as np
import numpy.typing as npt
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


def np_convert(target_dtype: np.dtype, source: npt.ArrayLike, normalize=True, safe_bool=False):
	""" TODO: Confirm Fix for Negative Values """
	if safe_bool and target_dtype == bool:
		return source.astype(target_dtype).astype(np.uint8)
	elif np.issubdtype(target_dtype, np.integer) and normalize:
		dtype_range = np.iinfo(target_dtype).max - np.iinfo(target_dtype).min
		source_floor = np.min(source) * -1
		source_range = np.max(source) + source_floor

		# Avoid divide by 0, esp. as numpy segfaults when you do.
		if source_range == 0.0:
			source_range = 1.0

		return ((source + source_floor) * max(dtype_range / source_range, 1)).astype(target_dtype)
	elif np.issubdtype(target_dtype, np.floating) and normalize:
		source_floor = np.min(source) * -1
		source_range = np.max(source) + source_floor

		# Avoid divide by 0, esp. as numpy segfaults when you do.
		if source_range == 0.0:
			source_range = 1.0

		return ((source + source_floor) / source_range).astype(target_dtype)
	else:
		return source.astype(target_dtype)


def write_split(source: Path, output_folder: Path, sections: int, dtype: type, compression: bool):
	log.log("Preload", f"Loading {source.name}")
	source_meta = tf.TiffFile(source)
	log.log("Preload Meta", f"{source_meta}", log_level=log.DEBUG.INFO)
	for k in range(0, len(source_meta.series)):
		for j in range(0, len(source_meta.series[k].levels)):
			source_data = tf.imread(source, series=k, level=j)
			log.log("Loaded", f"Loaded {source.name}: {source_data.shape} (series {j}, level {k})")
			split_size = source_data.shape[1] // sections
			for i in range(0, sections):
				target_path = output_folder.joinpath(source.with_suffix("").name + f"_s{k}_l{j}_p{i + 1}.tiff")
				tf.imwrite(target_path,
							np_convert(dtype, source_data[:, i * split_size: (i + 1) * split_size]),
							compression=tf.COMPRESSION.LZW if compression else None)
				log.log("Writing", f"Wrote {target_path}")


@click.command()
@click.option("-t", "--output_type", type=click.STRING, required=True, help="Type of output data.")
@click.option("-h", "--horizontal_sections", type=click.INT, default=1,
				help="Number of horizontal sections to split image into.")
@click.option("--uncompressed", is_flag=True)
@click.argument("input_folder", type=click.Path(exists=True, path_type=Path))
@click.argument("output_folder", type=click.Path(path_type=Path))
def convert(output_type, horizontal_sections, uncompressed, input_folder, output_folder):
	output_folder.mkdir(exist_ok=True, parents=True)
	log.log("Setup", f"Converting each file into {horizontal_sections} {output_type} files.")

	with Pool(24) as pool:
		pool.starmap(write_split, ([img, output_folder, horizontal_sections, np.dtype(output_type), not uncompressed]
									for img in input_folder.iterdir()))


if __name__ == "__main__":
	convert()
