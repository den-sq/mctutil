import sys
from pathlib import Path 	# if you haven't already done so

import click
import numpy as np
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import cli 	# noqa::E402
from shared import log 	# noqa::E402


@click.command
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for original dataset.')
@click.option('-o', '--output-dir', type=click.Path(), help='Output path for transformed dataset.')
@click.option('-t', "--out-dtype", type=cli.NUMPYTYPE, default=np.uint8, help="Datatype of Output.")
def downsample(data_dir, output_dir, out_dtype):
	log.start()
	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	dtype = out_dtype.nptype
	for path in Path(data_dir).iterdir():
		in_img = tf.imread(path)
		# Assumes matching signed or unsigned should fix
		source_range = np.max(in_img) - np.min(in_img)
		target_range = np.iinfo(dtype).max - np.iinfo(dtype).min
		tf.imwrite(Path(out_dir, path.name), (in_img * target_range / source_range).astype(dtype), dtype=dtype)
		log.log("File Written", path.name)

if __name__ == "__main__":
	downsample()
