from multiprocessing import Pool
from pathlib import Path
import sys

import click
import numpy as np
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


def channelize_file(randomize, source, target_dir, execute=True):
	target_path = target_dir.joinpath(source.name)
	if not execute:
		log.log("Channelize", f"Would write {target_path}", log_level=log.DEBUG.INFO)
		return

	source_data = tf.imread(source)

	if (len(source_data.shape) > 2 and source_data.shape[-1] == 1):
		new_shape = source_data.shape[:-1] + (3, )
	else:
		new_shape = source_data.shape + (3, )
		source_data = source_data[..., np.newaxis]

	if randomize:
		# Generates random color weights.
		starters = np.random.rand(2) / 2
		final = 1.0 - np.sum(starters)
		target_data = source_data * (list(starters) + [final])
	else:
		target_data = np.repeat(source_data, 3).reshape(new_shape)

	tf.imwrite(target_path, target_data.astype(source_data.dtype))
	log.log("Channelize", f"{target_path} written", log_level=log.DEBUG.STATUS)


@click.command()
@click.option("--randomize", is_flag=True)
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually write channelized TIFFs or just plan the writes.")
@click.argument("root_path", type=click.Path(exists=True, path_type=Path))
@click.argument("target_path", type=click.Path(exists=False, path_type=Path))
def channelize(randomize, execute, root_path, target_path):
	if execute:
		target_path.mkdir(parents=True)
	else:
		log.log("Channelize", f"Would create {target_path}", log_level=log.DEBUG.INFO)
	with Pool(12) as pool:
		pool.starmap(channelize_file,
					[(randomize, source, target_path, execute) for source in root_path.iterdir()])


if __name__ == '__main__':
	channelize()
