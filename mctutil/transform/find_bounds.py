from datetime import datetime
from multiprocessing.pool import ThreadPool
from pathlib import Path

import click
import numpy as np
import psutil
import tifffile as tf

from mctutil.shared.log import log, LOG

count = 0
start_time = datetime.now()


def image_bounds(path):
	x = tf.imread(path)
	global count
	count += 1
	if count % 50 == 0:
		log.write("Find Bounds", f"{count} calculated", log_level=LOG.INFO)
	return np.array([np.min(x), np.max(x)])


@click.command()
@click.option("--process-count", "-p", type=click.INT, default=psutil.cpu_count() * 3, help="")
@click.argument("input-path", type=click.Path(file_okay=False, exists=True, path_type=Path))
def find_bounds(process_count, input_path):
	log.write("Find Bounds", f"Scanning {input_path}", log_level=LOG.STATUS)
	with ThreadPool(process_count) as pool:
		bounds = np.array(pool.map(image_bounds, input_path.glob("**/*.tif*")))
	min_val = np.min(bounds[:, 0])
	max_val = np.max(bounds[:, 1])
	log.write("Find Bounds", f"{min_val}:{max_val}", log_level=LOG.STATUS)
	return min_val, max_val


if __name__ == "__main__":
	find_bounds()
