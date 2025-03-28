
from collections import namedtuple
import json
from pathlib import Path

import click
import numpy as np

Coord = namedtuple("Coord", ["x", "y", "z"])


# Click Parameter:
class Coordinates(click.ParamType):
	name = "Integer Coodrinates"

	def convert(self, value, param, ctx):
		try:
			coord = Coord(*[int(x) for x in value.split(",")])
			return coord
		except (ValueError, TypeError):
			self.fail(f'{value} is not a 3-value intteger coordinate.')


COORDINATES = Coordinates()


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to shift annotations in.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path),
				required=True, help="Neuroglancer JSON file to write shifted annotations.")
@click.option("--shift-dimensions", "-s", type=COORDINATES, required=True,
				help="Amount to shift all annotations, in 'x,y,z' format.")
def point_merge(json_file: Path, json_result: Path, shift_dimensions: Coord):
	""" Shift all annotations in a given file in 3 dimensions.

		Example run:
			python point_merge.py -j R2_L3.json -r R2_L3_Upd.json -s 10,5,3
		"""
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	print("json loaded")

	for layer in json_data["layers"]:
		if layer["type"] == "annotation":
			for annotation in layer["annotations"]:
				if annotation["type"] == "point":
					annotation["point"] = list(np.add(annotation["point"], shift_dimensions))

	print("annotation updated")

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	print(f"new annotation written to {json_result}")


if __name__ == "__main__":
	point_merge()
