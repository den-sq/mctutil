from collections import namedtuple
from enum import IntEnum
import json
from pathlib import Path

import click

from mctutil.shared.log import log, LOG
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
				required=True, help="Neuroglancer JSON file to sort annotation(s).")
@click.option("--json-result", "-r", type=click.Path(path_type=Path),
				required=True, help="Neuroglancer JSON file to write sorted annotation(s).")
@click.option("--axis", "-a", type=click.INT, required=True, help="Axis (0-2 for X/Y/Z) to sort on.")
@click.argument("source_annotations", type=ANNOTATION_PAIR, nargs=-1)
def point_sort(json_file: Path, json_result: Path, axis: int, source_annotations: tuple):
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	log.write("Point Sort", f"Loaded {json_file}", log_level=LOG.STATUS)

	annotation_names = [layer["name"] for layer in json_data["layers"]]

	for pair in source_annotations:
		if pair.name not in annotation_names:
			log.write("Point Sort", f"Annotation {pair.name} missing.", log_level=LOG.WARN)
		else:
			base_annotations = json_data["layers"][annotation_names.index(pair.name)]["annotations"]
			sorted_annotations = sorted(base_annotations, key=lambda x: x['point'][axis])
			final_annotations = sorted_annotations if pair.direction else reversed(sorted_annotations)
			json_data["layers"][annotation_names.index(pair.name)]["annotations"] = list(final_annotations)

	log.write("Point Sort", "Annotations updated", log_level=LOG.STATUS)

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	log.write("Point Sort", f"Wrote {json_result}", log_level=LOG.STATUS)


if __name__ == "__main__":
	point_sort()
