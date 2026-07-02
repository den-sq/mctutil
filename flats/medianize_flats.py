from pathlib import Path
import sys
import tifffile
import numpy as np


def main():
	flat_sets = {}
	for full_path in Path(sys.argv[1]).iterdir():
		flats_key = "_".join(full_path.name.split("_")[:-1])
		if flats_key in flat_sets:
			flat_sets[flats_key].append(full_path)
		else:
			flat_sets[flats_key] = [full_path]
	
	Path(sys.argv[2]).mkdir(exist_ok=True, parents=True)
	for flats_key, flats_files in flat_sets.items():
		flats_data = [tifffile.imread(flat) for flat in flats_files]
		median_flat = np.median(flats_data, axis=0)
		Path(sys.argv[2], str(median_flat.shape)).mkdir(exist_ok=True, parents=True)
		tifffile.imwrite(Path(sys.argv[2], str(median_flat.shape), f"{flats_key}_median.tif"), median_flat)

	print(flat_sets)

if __name__ == "__main__":
	main()