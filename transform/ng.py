from pathlib import Path

from neuroglancer_scripts.scripts.generate_scales_info import generate_scales_info
from neuroglancer_scripts.scripts.slices_to_precomputed import convert_slices_in_directory
from neuroglancer_scripts.scripts.compute_scales import compute_scales
import json
import tifffile
import click


@click.command()
@click.option('-c', '--chunk-size', type=click.INT, default=128,
				help="Chunk size.  Smaller values faster loads but more files; Larger values have slower but load less files.")
@click.option('-r', '--resolution', type=click.INT, default=1400, help="Dataset Resolution, in nanometers")
@click.option('--segmentation/--raw_data', type=click.BOOL, default=False,
				help="Whether we are writing segmentations or raw image data.")
@click.option('-i', '--input-path', type=click.Path(), required=True, help="Path to input files.")
@click.option('-m', '--metadata-info', type=click.Path(dir_okay=False, writable=True), required=True,
				help="Location for the Neuroglancer Metadata Info File.")
@click.option("--strip-gz/--keep-gz", type=click.BOOL, default=False,
				help="Whether to strip gz extennsions from filenames, as they can confuse neuroglancer.")
@click.option('-o', '--output-location', type=click.STRING, required=True)
def neuroglance(chunk_size, resolution, segmentation, strip_gz, input_path, metadata_info, output_location):
	print(f'input folder: {input_path}')
	image_paths = Path(input_path).glob("**/*.tif")

	memmap_ = tifffile.memmap(image_paths[0])
	size = [memmap_.shape[1], memmap_.shape[0], len(image_paths)]
	dtype_ = str(memmap_.dtype)

	print(f'volume size is :{size} datatype is {dtype_}')

	if segmentation:
		json_metadata = {
			"type": "segmentation",
			"mesh": "mesh",
			"encoding": 'compressed_segmentation',
			"data_type": "uint64",
			"num_channels": 1,
			"compressed_segmentation_block_size": [8, 8, 8],
			"scales":
			[
				{
					"size": size,
					"resolution": [resolution, resolution, resolution],
					"voxel_offset": [0, 0, 0]
				}
			]
		}
	else:
		json_metadata = {
			"type": "image",
			"data_type": dtype_,
			"num_channels": 1,
			"scales":
			[
				{
					"size": size,
					"encoding": 'raw',
					"resolution": [resolution, resolution, resolution],
					"voxel_offset": [0, 0, 0]
				}
			]
		}

	with open(metadata_info) as handle:
		json.dump(json_metadata, handle)

	generate_scales_info(metadata_info, output_location, chunk_size)
	convert_slices_in_directory(input_path, output_location, options={"flat": True})
	compute_scales(output_location, "stride" if segmentation else "average", options={"flat": True})

	if strip_gz:
		for out_file in Path(output_location).glob("**/*.gz"):
			out_file.rename(out_file.with_suffix(""))


if __name__ == '__main__':
	neuroglance()
