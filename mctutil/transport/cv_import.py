from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import click
from cloudvolume import CloudVolume
import numpy as np
import psutil
import tifffile

from mctutil.shared.log import log, LOG
from mctutil.shared import cli


def fetch_slices(remote, use_https, region, bin_power, output_dir, out_type=None, transpose_axes=False, execute=True):
	log.write("Fetching Slices", f"{remote}: {region} with {bin_power}")
	if not execute:
		log.write("Fetching Slices",
				f"Would fetch slices {region[0].start}..{region[0].stop} to {output_dir}",
				log_level=LOG.INFO)
		return
	vol = CloudVolume(remote, mip=bin_power, use_https=use_https, progress=True)
	log.write("Volume Set", f"{remote}: {region} with {bin_power}", log_level=LOG.INFO)
	for i, slice_data in enumerate(vol[region], start=region[0].start):
		if out_type is not None:
			slice_data = np.array(slice_data, dtype=out_type.nptype)
		if transpose_axes:
			slice_data = np.transpose(slice_data, (2, 0, 1))
		output_path = output_dir / f"slice_{str(i).zfill(4)}.tif"
		log.write("Writing Slice", output_path.name, log_level=LOG.INFO)
		tifffile.imwrite(output_path, slice_data)


def bin_slices(base_slice, bin_power, base_dim):
	if bin_power:
		out_slice = ()

		for i, slice_dim in enumerate(base_slice):
			new_start = 0 if slice_dim.start is None else slice_dim.start // (2 ** bin_power)
			new_stop = base_dim[i] if slice_dim.stop is None else slice_dim.stop // (2 ** bin_power)
			out_slice += (np.s_[new_start: new_stop], )
		log.write("Slice Calculation", out_slice)
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
@click.option("-t", "--out-dtype", type=cli.NUMPYTYPE, help="Target datatype.")
@click.option("--transpose-axes/--original-axes", type=click.BOOL, default=False,
				help="Whether to transpose downloaded slices from XYZ/channel order to ZYX/channel order.")
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually fetch and write slices or just plan the work.")
@click.argument("output-dir")
def cloudvolume_fetch(cloud_url, cloud_slice, resolution, bin_power, use_https, num_processes, out_dtype,
						transpose_axes, execute, output_dir):
	log.write("Start")

	cloud_slice = bin_slices(cloud_slice, bin_power,
							CloudVolume(cloud_url, mip=bin_power, use_https=use_https, progress=True).shape)
	batch_size = (cloud_slice[0].stop - cloud_slice[0].start) // num_processes

	# directory management
	effective_resolution = resolution * 2 ** bin_power
	timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
	output_dir = Path(output_dir, f'CV_bin{bin_power}_{effective_resolution}um_{timestamp}')
	if execute:
		output_dir.mkdir(parents=True, exist_ok=True)
	else:
		log.write("Cloudvolume Fetch", f"Would create {output_dir}", log_level=LOG.INFO)

	with Pool(num_processes) as pool:
		pool.starmap(fetch_slices,
			[(cloud_url, use_https, (np.s_[i:min(i + batch_size, cloud_slice[0].stop)],) + cloud_slice[1:],
				bin_power, output_dir, out_dtype, transpose_axes, execute)
				for i in range(cloud_slice[0].start, cloud_slice[0].stop, batch_size)])

	log.write("Complete")


if __name__ == '__main__':
	cloudvolume_fetch()
