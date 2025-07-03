from dataclasses import dataclass, replace
import json
from pathlib import Path

import click
import numpy as np


@dataclass(frozen=True)
class TaggedLayer:
	name: str
	intensity: int
	radius: int

	def __str__(self):
		return f"{self.name} ID{self.intensity}r{self.radius}"


# Click Parameter: Pair of segment ID and color
class TaggedLayerParameter(click.ParamType):
	name = "Annotation Information"

	def convert(self, value, param, ctx):
		tagged_layer = value.split(":")

		try:
			if len(tagged_layer) == 0:
				self.fail("Layer Name Empty")
			if len(tagged_layer) == 1:
				return TaggedLayer(name=tagged_layer[0], intensity=-1, radius=-1)
			elif len(tagged_layer) == 2:
				intensity = int(tagged_layer[1]) if len(tagged_layer[1]) > 0 else -1
				return TaggedLayer(name=tagged_layer[0], intensity=intensity, radius=-1)
			elif len(tagged_layer) == 3:
				intensity = int(tagged_layer[1]) if len(tagged_layer[1]) > 0 else -1
				radius = int(tagged_layer[2]) if len(tagged_layer[2]) > 0 else -1
				return TaggedLayer(name=tagged_layer[0], intensity=intensity, radius=radius)
			else:
				self.fail("Too many fields in tagged layer.")
		except ValueError as ve:
			self.fail(f'{value} is not a tagged layer.\n{ve}')


TAGGEDLAYER = TaggedLayerParameter()


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to merge annothations from.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path, exists=False),
				required=True, help="Neuroglancer JSON file to write to.")
@click.option("--segment_radius", "-s", type=click.INT, default=30,
				help="Segmentation radius used if not specified individually.")
@click.argument("tagged_layer", type=TAGGEDLAYER, nargs=-1)
def layer_tag(json_file: Path, json_result: Path, segment_radius, tagged_layer: TaggedLayer):
	""" Tag layers with ID and radius for Dan's scripts.  Layer format is 
			name:intensity:radius
		Intensity and radius can be left empty, in which case defaults will be assigned.
			- Generated intensities will be unique values spaced across 16 bit uint space.
			- Radius will default to segment_radius parameter.

		Example run:
			python layer_tag.py -j Untagged.json -r Tagged.json -s 30 LSVL_CC_A::20 Cross_Conn:570
		"""
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)
	print("json loaded")

	preset_intensities = [layer.intensity for layer in tagged_layer]
	intensity_step = np.iinfo(np.uint16).max // (len(tagged_layer) + 2)
	intensity_gen = np.nditer(np.arange(intensity_step, np.iinfo(np.uint16).max, intensity_step))

	source_layers = {layer["name"]: layer for layer in json_data["layers"]}
	for t_layer in tagged_layer:
		if t_layer.name not in source_layers:
			print(f"{t_layer} not found in source.")
		else:
			if t_layer.intensity == -1:
				# Generate an intensity that's not a repeat.
				t_layer = replace(t_layer, intensity=next(intensity_gen))
				while t_layer.intensity in preset_intensities:
					replace(t_layer, intensity=next(intensity_gen))

			if t_layer.radius == -1:
				# Update with default intensity.
				t_layer = replace(t_layer, radius=segment_radius)

			source_layers[t_layer.name]["name"] = str(t_layer)

			print(f"{t_layer.name} updated: {t_layer}.")

	json_data["layers"] = list(source_layers.values())

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	print(f"new annotation written to {json_result}")


if __name__ == "__main__":
	layer_tag()
