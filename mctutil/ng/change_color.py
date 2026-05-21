from dataclasses import dataclass
import json
from pathlib import Path
import re

import click

from mctutil.shared import log
from mctutil.shared.cli import DelimitedRecord


@dataclass(frozen=True)
class ColorPair:
	segment: int
	hval: str

	def __post_init__(self):
		hex_color_pattern = re.compile(r"^#?([0-9a-fA-F]{3}){1,2}$|^#?([0-9a-fA-F]{4}){1,2}$")
		if not bool(hex_color_pattern.match(self.hval)):
			raise ValueError(f"{self.hval} is not a valid hexcolor.")


COLOR_PAIR = DelimitedRecord(ColorPair, [int, str], name="Annotation Information")


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to change to forward-facing angled projection.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path, exists=False, writable=True),
				required=True, help="Neuroglancer JSON file to write updated orientation to.")
@click.option("--annotation", "-a", type=click.STRING,
				required=True, help="Name of annotation to update the segment colors for.")
@click.argument("SEGMENT_COLORS", type=COLOR_PAIR, nargs=-1)
def change_color(json_file: Path, json_result: Path, annotation: str, segment_colors: ColorPair):
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	log.log("Change Color", f"Loaded {json_file}", log_level=log.DEBUG.STATUS)

	layer_names = [layer["name"] for layer in json_data["layers"]]
	if annotation not in layer_names:
		log.log("Change Color", f"Annotation {annotation} not found in JSON.", log_level=log.DEBUG.ERROR)
		exit(1)
	elif "segmentColors" not in json_data["layers"][layer_names.index(annotation)]:
		log.log("Change Color", f"No segmentation colors in layer {annotation}", log_level=log.DEBUG.ERROR)
		exit(1)
	else:
		color_layer = json_data["layers"][layer_names.index(annotation)]["segmentColors"]
		for color_pair in segment_colors:
			if str(color_pair.segment) in color_layer:
				color_layer[str(color_pair.segment)] = color_pair.hval
			else:
				log.log("Change Color",
						f"Segmentation ID {color_pair.segment} not found in layer {annotation}.",
						log_level=log.DEBUG.WARN)
		json_data["layers"][layer_names.index(annotation)]["segmentColors"] = color_layer

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	log.log("Change Color", f"Wrote {json_result}", log_level=log.DEBUG.STATUS)


if __name__ == "__main__":
	change_color()
