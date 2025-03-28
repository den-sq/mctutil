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
				required=True, help="Neuroglancer JSON file to merge annothations from.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path),
				required=True, help="Neuroglancer JSON file to merge annothations from.")
@click.option("--target-name", "-t", type=click.STRING, required=True,
				help="Name of new, merged annotation.")
@click.argument("source_annotations", type=ANNOTATION_PAIR, nargs=-1)
def point_merge(json_file: Path, json_result: Path, target_name: str, source_annotations: tuple):
	""" Merge 2+ annotation layers contained in a neuroglancer json file into a new layer.
		Each SOURCE_ANNOTATION should be a name, followed by a colon and then whether to
		add the points in FORWARD or BACKWARD order, e.g. R2-Root:FORWARD.

		Example run:
			python point_merge.py -j R2_L3.json -r R2_L3_Upd.json -t R2_L3 R2A-Root:FORWARD L3B-Root:BACKWARD
		"""
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	print("json loaded")

	annotation_names = [layer["name"] for layer in json_data["layers"]]

	for pair in source_annotations:
		if pair.name not in annotation_names:
			print(f"Annotation {pair.name} missing.")
			return

	new_annotation_layer = {
		"type": "annotation",
		"tool": "annotatePoint",
		"tab": "annotations",
		"source": json_data["layers"][annotation_names.index(source_annotations[0].name)]["source"],
		"annotations": [],
		"name": target_name,
	}

	print("base new annotation created")

	for pair in source_annotations:
		base_annotations = json_data["layers"][annotation_names.index(pair.name)]["annotations"]
		in_order = base_annotations if pair.direction else list(reversed(base_annotations))
		new_annotation_layer["annotations"] += in_order

	print("new annotation created")

	json_data["layers"].append(new_annotation_layer)

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	print(f"new annotation written to {json_result}")


if __name__ == "__main__":
	point_merge()
