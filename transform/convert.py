from pathlib import Path
import sys

import click
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


@click.command()
@click.option("-t", "--output_type", type=click.STRING, required=True, help="Type of output data.")
@click.option("-h", "--horizontal_sections", type=click.INT, default=1,
				help="Number of horizontal sections to split image into.")
@click.argument("input_folder", type=click.Path(exists=True, path_type=Path))
@click.argument("output_folder", type=click.Path(path_type=Path))
def convert(output_type, horizontal_sections, input_folder, output_folder):
	output_folder.mkdir(exist_ok=True, parents=True)
	log.log("Setup", f"Converting each file into {horizontal_sections} {output_type} files.")

	for source in input_folder.iterdir():
		log.log("Preload", f"Loading {source.name}")
		source_meta = tf.TiffFile(source)
		log.log("Preload Meta", f"{source_meta}", log_level=log.DEBUG.INFO)
		for k in range(0, len(source_meta.series)):
			for j in range(0, len(source_meta.series[k].levels)):
				source_data = tf.imread(source, series=k, level=j)
				log.log("Loaded", f"Loaded {source.name}: {source_data.shape} (series {j}, level {k})")
				split_size = source_data.shape[1] // horizontal_sections
				for i in range(0, horizontal_sections):
					target_name = source.with_suffix("").name + f"_s{k}_l{j}_p{i + 1}_of_{horizontal_sections}.{output_type}"
					tf.imwrite(output_folder.joinpath(target_name), source_data[:, i * split_size: (i + 1) * split_size, :],
									compression=tf.COMPRESSION.LZW, bigtiff=True)
					log.log("Writing", f"Wrote {target_name}")


if __name__ == "__main__":
	convert()
