from datetime import datetime
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import click
from cloudvolume import CloudVolume
import numpy as np
import psutil
import tifffile

if __name__ == '__main__':
	# Needed to run script from subfolder
	sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402
from shared import cli 	# noqa::E402


def fetch_slices(remote, use_https, region, bin_power, output_dir):
	log.log("Fetching Slices", f"{remote}: {region} with {bin_power}")
	vol = CloudVolume(remote, mip=bin_power, use_https=use_https, progress=True)
	for i, slice_data in enumerate(vol[region], start=region[0].start):
		tifffile.imwrite(os.path.join(output_dir, f"slice_{str(i).zfill(4)}.tif"), slice_data)


def bin_slices(base_slice, bin_power, base_dim):
	if bin_power:
		out_slice = ()

		for i, slice_dim in enumerate(base_slice):
			new_start = 0 if slice_dim.start is None else slice_dim.start // (2 ** bin_power)
			new_stop = base_dim[i] if slice_dim.stop is None else slice_dim.stop // (2 ** bin_power)
			out_slice += (np.s_[new_start: new_stop], )
		log.log("Slice Calculation", out_slice)
		return out_slice
	else:
		return base_slice


@click.command()
@click.option("-u", "--cloud-url", type=click.STRING, required=True, help="Cloudvolume URL to fetch.")
@click.option("-s", "--cloud-slice", type=cli.SLICE(), required=True, help="Slice of input image to use.")
@click.option("-r", "--resolution", type=click.FLOAT, required=True,
				help="Resolution of data to fetch in microns.")
@click.option("-b", "--bin-power", type=click.INT, required=True,
				help="Number of additional voxels in each dimension to bin together as a MIP.")
@click.option("--use-https", is_flag=True, help="Whether to use an https connection.")
@click.option("-n", "--num-processes", type=click.INT, default=psutil.cpu_count(),
				help="Number of simultaneous processes.")
@click.argument("output-dir")
def cloudvolume_fetch(cloud_url, cloud_slice, resolution, bin_power, use_https, num_processes, output_dir):
	log.log("Start")

	cloud_slice = bin_slices(cloud_slice, bin_power,
							CloudVolume(cloud_url, mip=bin_power, use_https=use_https, progress=True).shape)
	batch_size = (cloud_slice[0].stop - cloud_slice[0].start) // num_processes

	# directory management
	output_dir = Path(output_dir, f'CV_bin{bin_power}_{resolution*2**bin_power}um_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}')
	output_dir.mkdir(parents=True, exist_ok=True)

	with Pool(num_processes) as pool:
		pool.starmap(fetch_slices,
			[(cloud_url, use_https, (np.s_[i:min(i + batch_size, cloud_slice[0].stop)],) + cloud_slice[1:],
				bin_power, output_dir) for i in range(cloud_slice[0].start, cloud_slice[0].stop, batch_size)])

	log.log("Complete")


if __name__ == '__main__':
	cloudvolume_fetch()
