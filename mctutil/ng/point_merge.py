from collections import namedtuple
from enum import IntEnum
import json
from pathlib import Path

import click

from mctutil.shared import log
from mctutil.shared.cli import DelimitedRecord


class Direction(IntEnum):
	BACKWARD = 0,
	FORWARD = 1


AnnotationPair = namedtuple("AnnotationPair", "name, direction")


ANNOTATION_PAIR = DelimitedRecord(
	AnnotationPair,
	[str, lambda value: getattr(Direction, value)],
	name="Annotation Information",
)


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to merge annothations from.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path),
				required=True, help="Neuroglancer JSON file to merge annothations from.")
@click.option("--target-name", "-t", type=click.STRING, required=True,
				help="Name of new, merged annotation.")
@click.argument("source_annotations", type=ANNOTATION_PAIR, nargs=-1)
def point_merge(json_file: Path, json_result: Path, target_name: str, source_annotations: tuple):
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	log.log("Point Merge", f"Loaded {json_file}", log_level=log.DEBUG.STATUS)

	annotation_names = [layer["name"] for layer in json_data["layers"]]

	for pair in source_annotations:
		if pair.name not in annotation_names:
			log.log("Point Merge", f"Annotation {pair.name} missing.", log_level=log.DEBUG.ERROR)
			return

	new_annotation_layer = {
		"type": "annotation",
		"tool": "annotatePoint",
		"tab": "annotations",
		"source": json_data["layers"][annotation_names.index(source_annotations[0].name)]["source"],
		"annotations": [],
		"name": target_name,
	}

	log.log("Point Merge", "Base annotation created", log_level=log.DEBUG.STATUS)

	for pair in source_annotations:
		base_annotations = json_data["layers"][annotation_names.index(pair.name)]["annotations"]
		in_order = base_annotations if pair.direction else list(reversed(base_annotations))
		new_annotation_layer["annotations"] += in_order

	log.log("Point Merge", "Merged annotation created", log_level=log.DEBUG.STATUS)

	json_data["layers"].append(new_annotation_layer)

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	log.log("Point Merge", f"Wrote {json_result}", log_level=log.DEBUG.STATUS)


if __name__ == "__main__":
	point_merge()
