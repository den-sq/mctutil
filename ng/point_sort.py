
from collections import namedtuple
from enum import IntEnum
import json
from pathlib import Path

import click


class Direction(IntEnum):
	BACKWARD = 0,
	FORWARD = 1


AnnotationPair = namedtuple("AnnotationPair", "name, direction")


# Click Parameter: Choice of an Enum from a list.
class AnnotationPairParameter(click.ParamType):
	name = "Annotation Information"

	def convert(self, value, param, ctx):
		source_pair = value.split(":")

		try:
			name = source_pair[0]
			direction = getattr(Direction, source_pair[1])
		except ValueError:
			self.fail(f'{value} is not an annotation pair, of format name:<FORWARD/BACKWARD>')

		return AnnotationPair(name, direction)


ANNOTATION_PAIR = AnnotationPairParameter()


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to sort annotation(s).")
@click.option("--json-result", "-r", type=click.Path(path_type=Path),
				required=True, help="Neuroglancer JSON file to write sorted annotation(s).")
@click.option("--axis", "-a", type=click.INT, required=True, help="Axis (0-2 for X/Y/Z) to sort on.")
@click.argument("source_annotations", type=ANNOTATION_PAIR, nargs=-1)
def point_merge(json_file: Path, json_result: Path, axis: int, source_annotations: tuple):
	""" Sort one or more annotation layers in a JSON file, according to one (X/Y/Z) dimension.
		Each SOURCE_ANNOTATION should be a name, followed by a colon and then whether to
		sort the points in FORWARD or BACKWARD order, e.g. R2-Root:FORWARD.

		Example run:
			python point_sort.py -j R2_L3.json -r R2_L3_Upd.json -a 0 R2A-Root:FORWARD L3B-Root:BACKWARD
		"""
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	print("json loaded")

	annotation_names = [layer["name"] for layer in json_data["layers"]]

	for pair in source_annotations:
		if pair.name not in annotation_names:
			print(f"Annotation {pair.name} missing.")
		else:
			base_annotations = json_data["layers"][annotation_names.index(pair.name)]["annotations"]
			sorted_annotations = sorted(base_annotations, key=lambda x: x['point'][axis])
			final_annotations = sorted_annotations if pair.direction else reversed(sorted_annotations)
			json_data["layers"][annotation_names.index(pair.name)]["annotations"] = list(final_annotations)

	print("annotations updated")

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	print(f"new annotation written to {json_result}")


if __name__ == "__main__":
	point_merge()
