from collections import namedtuple
import json
from pathlib import Path

import click
import numpy as np

from mctutil.shared.log import log, LOG
from mctutil.shared.cli import DelimitedRecord

Coord = namedtuple("Coord", ["x", "y", "z"])


COORDINATES = DelimitedRecord(Coord, [int, int, int], delimiter=",", name="Integer Coordinates")


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to shift annotations in.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path),
				required=True, help="Neuroglancer JSON file to write shifted annotations.")
@click.option("--shift-dimensions", "-s", type=COORDINATES, required=True,
				help="Amount to shift all annotations, in 'x,y,z' format.")
def point_shift(json_file: Path, json_result: Path, shift_dimensions: Coord):
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	log.write("Point Shift", f"Loaded {json_file}", log_level=LOG.STATUS)

	for layer in json_data["layers"]:
		if layer["type"] == "annotation":
			for annotation in layer["annotations"]:
				if annotation["type"] == "point":
					annotation["point"] = list(np.add(annotation["point"], shift_dimensions))

	log.write("Point Shift", "Annotations updated", log_level=LOG.STATUS)

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	log.write("Point Shift", f"Wrote {json_result}", log_level=LOG.STATUS)


if __name__ == "__main__":
	point_shift()
