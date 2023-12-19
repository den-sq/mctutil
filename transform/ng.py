import glob
import os
import sys

import json
import tifffile


def write_json_file(data, filename):
	filepath = os.path.join(os.path.dirname(__file__), filename)

	with open(filepath, 'w') as file:
		json.dump(data, file)


def get_image_paths(folder):
	return sorted(glob.glob(os.path.join(folder, '*.tif*')))


proj_dir = os.path.dirname(sys.argv[0])
input_path = sys.argv[1]
resolution = int(sys.argv[2])

print(f'input folder: {input_path}')
image_paths = get_image_paths(input_path)

memmap_ = tifffile.memmap(image_paths[0])
z = len(image_paths)
y = memmap_.shape[0]
x = memmap_.shape[1]
dtype_ = str(memmap_.dtype)

print(f'volume size is :{x},{y},{z} datatype is {dtype_}')
info_file = {
	"type": "image",
	"data_type": dtype_,
	"num_channels": 1,
	"scales":
	[
		{
			"size": [x, y, z],
			"encoding": 'raw',
			"resolution": [resolution, resolution, resolution],
			"voxel_offset": [0, 0, 0]
		}
	]
}

filename = "info_MI"
write_json_file(info_file, filename)
print("JSON file created: ", filename)
