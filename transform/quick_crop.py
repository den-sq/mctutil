import tifffile
from pathlib import Path
import sys
import os
from datetime import datetime
import numpy as np
from multiprocessing import Pool


def get_image_paths(folder, z_crop):
	return (sorted(folder.glob('*.tif*')))[z_crop]


def quick_crop(image_path, xy_crop):
	img = tifffile.memmap(image_path) 	# [4200:6500,6000:8400]

	tifffile.imwrite(os.path.join(output_dir, os.path.basename(image_path)), img[xy_crop])
	print(f'processed {image_path}')


startTime = datetime.now()

proj_dir = os.path.dirname(sys.argv[0])
input_dir = Path(sys.argv[1])
output_dir = os.path.join(proj_dir, 'Octo_7_Tight_Orthocrop')

xy_crop = np.s_[421:4779, 551:1611]
z_crop = np.s_[803:]

Path(output_dir).mkdir(parents=True, exist_ok=True)

file_names = get_image_paths(input_dir, z_crop)
if __name__ == '__main__':
	startTime_full = datetime.now()

	with Pool(16) as pool:
		pool.starmap(quick_crop, [(file_name, xy_crop) for file_name in file_names])

	print('\nfull time ' + str(datetime.now() - startTime_full))
