from multiprocessing import Pool
from pathlib import Path
import sys

import click
import numpy as np
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


def channelize_file(randomize, source, target_dir):
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

	# print(f"{source}|{source_data.shape}|{target_data.shape}|{target_path.joinpath(source.name)}")

	tf.imwrite(target_dir.joinpath(source.name), target_data.astype(source_data.dtype))
	log.log("Write Target", f"{target_dir.joinpath(source.name)} Written")


@click.command()
@click.option("--randomize", is_flag=True)
@click.argument("root_path", type=click.Path(exists=True, path_type=Path))
@click.argument("target_path", type=click.Path(exists=False, path_type=Path))
def channelize(randomize, root_path, target_path):
	target_path.mkdir(parents=True)
	with Pool(12) as pool:
		pool.starmap(channelize_file, [(randomize, source, target_path) for source in root_path.iterdir()])


if __name__ == '__main__':
	channelize()
