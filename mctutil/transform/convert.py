from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
import tifffile as tf


from mctutil.shared.log import log, LOG
from mctutil.shared.np_convert import np_convert


def write_split(source: Path, output_folder: Path, sections: int, dtype: type, compression: bool):
	log.write("Preload", f"Loading {source.name}")
	source_meta = tf.TiffFile(source)
	log.write("Preload Meta", f"{source_meta}", log_level=LOG.INFO)
	for k in range(0, len(source_meta.series)):
		for j in range(0, len(source_meta.series[k].levels)):
			source_data = tf.imread(source, series=k, level=j)
			log.write("Loaded", f"Loaded {source.name}: {source_data.shape} (series {j}, level {k})")
			split_size = source_data.shape[1] // sections
			for i in range(0, sections):
				target_path = output_folder.joinpath(source.with_suffix("").name + f"_s{k}_l{j}_p{i + 1}.tiff")
				tf.imwrite(target_path,
							np_convert(dtype, source_data[:, i * split_size: (i + 1) * split_size]),
							compression=tf.COMPRESSION.LZW if compression else None)
				log.write("Writing", f"Wrote {target_path}")


@click.command()
@click.option("-t", "--output_type", type=click.STRING, required=True, help="Type of output data.")
@click.option("-h", "--horizontal_sections", type=click.INT, default=1,
				help="Number of horizontal sections to split image into.")
@click.option("--uncompressed", is_flag=True)
@click.argument("input_folder", type=click.Path(exists=True, path_type=Path))
@click.argument("output_folder", type=click.Path(path_type=Path))
def convert(output_type, horizontal_sections, uncompressed, input_folder, output_folder):
	output_folder.mkdir(exist_ok=True, parents=True)
	log.write("Setup", f"Converting each file into {horizontal_sections} {output_type} files.")

	with Pool(24) as pool:
		pool.starmap(write_split, ([img, output_folder, horizontal_sections, np.dtype(output_type), not uncompressed]
								for img in input_folder.iterdir()))


if __name__ == "__main__":
	convert()
