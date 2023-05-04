import sys
from pathlib import Path # if you haven't already done so

import click
import numpy as np
import tifffile as tf

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import cli
from shared import log

@click.command
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for original dataset.')
@click.option('-o', '--output-dir', type=click.Path(), help='Output path for transformed dataset.')
@click.option('-t', "--out-dtype", type=cli.NUMPYTYPE, default=np.uint8, help="Datatype of Output.")
def downsample(data_dir, output_dir, out_dtype):
	log.start()
	out = Path(output_dir)
	out.mkdir(parents=True, exist_ok=True)
	for path in Path(data_dir).iterdir():
		infile = tf.imread(path)
		outfile = infile[:].astype(out_dtype.nptype)
		tf.imwrite(Path(out, path.name), outfile)
		log.log("File Written", path.name)

if __name__ == "__main__":
	downsample()
