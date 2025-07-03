from dataclasses import dataclass
import json
from pathlib import Path
import re

import click


@dataclass(frozen=True)
class ColorPair:
	segment: int
	hval: str

	def __post_init__(self):
		hex_color_pattern = re.compile(r"^#?([0-9a-fA-F]{3}){1,2}$|^#?([0-9a-fA-F]{4}){1,2}$")
		if not bool(hex_color_pattern.match(self.hval)):
			raise ValueError(f"{self.hval} is not a valid hexcolor.")


# Click Parameter: Pair of segment ID and color
class ColorPairParameter(click.ParamType):
	name = "Annotation Information"

	def convert(self, value, param, ctx):
		source_pair = value.split(":")

		try:
			pair = ColorPair(segment=int(source_pair[0]), hval=source_pair[1])
		except ValueError as ve:
			self.fail(f'{value} is not a color pair, of format segment:hexcolor.\n{ve}')

		return pair


COLOR_PAIR = ColorPairParameter()


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to change to forward-facing angled projection.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path, exists=False, writable=True),
				required=True, help="Neuroglancer JSON file to write updated orientation to.")
@click.option("--annotation", "-a", type=click.STRING,
				required=True, help="Name of annotation to update the segment colors for.")
@click.argument("SEGMENT_COLORS", type=COLOR_PAIR, nargs=-1)
def change_color(json_file: Path, json_result: Path, annotation: str, segment_colors: ColorPair):
	""" Changes segmentation colors for an annotation in a neuroglancer file.

		Example:
			python change_color.py -j Octo_Subv.json -r Octo_Subv_Recolor.json 2166:#ff0000 2021:#00ff00
	"""
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	print("json loaded")

	layer_names = [layer["name"] for layer in json_data["layers"]]
	if annotation not in layer_names:
		print(f"Annotation {annotation} not found in JSON.")
		exit(1)
	elif "segmentColors" not in json_data["layers"][layer_names.index(annotation)]:
		print(f"No segmentation colors in layer {annotation}")
		exit(1)
	else:
		color_layer = json_data["layers"][layer_names.index(annotation)]["segmentColors"]
		for color_pair in segment_colors:
			if str(color_pair.segment) in color_layer:
				color_layer[str(color_pair.segment)] = color_pair.hval
			else:
				print(f"Segmentation ID {color_pair.segment} not found in layer {annotation}.")
		json_data["layers"][layer_names.index(annotation)]["segmentColors"] = color_layer

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	print(f"new arrangement written to {json_result}")


if __name__ == "__main__":
	change_color()
