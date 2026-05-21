from dataclasses import dataclass, replace
import json
from pathlib import Path

import click
import numpy as np

from mctutil.shared import log
from mctutil.shared.cli import DelimitedRecord


@dataclass(frozen=True)
class TaggedLayer:
	name: str
	intensity: int
	radius: int

	def __str__(self):
		return f"{self.name} ID{self.intensity}r{self.radius}"


TAGGEDLAYER = DelimitedRecord(
	TaggedLayer,
	[
		lambda value: value if len(value) > 0 else (_ for _ in ()).throw(ValueError("Layer Name Empty")),
		lambda value: int(value) if len(value) > 0 else -1,
		lambda value: int(value) if len(value) > 0 else -1,
	],
	defaults=(None, "", ""),
	min_fields=1,
	name="Annotation Information",
)


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to merge annothations from.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path, exists=False),
				required=True, help="Neuroglancer JSON file to write to.")
@click.option("--segment_radius", "-s", type=click.INT, default=30,
				help="Segmentation radius used if not specified individually.")
@click.argument("tagged_layer", type=TAGGEDLAYER, nargs=-1)
def layer_tag(json_file: Path, json_result: Path, segment_radius, tagged_layer: TaggedLayer):
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)
	log.log("Layer Tag", f"Loaded {json_file}", log_level=log.DEBUG.STATUS)

	preset_intensities = [layer.intensity for layer in tagged_layer]
	intensity_step = np.iinfo(np.uint16).max // (len(tagged_layer) + 2)
	intensity_gen = np.nditer(np.arange(intensity_step, np.iinfo(np.uint16).max, intensity_step))

	source_layers = {layer["name"]: layer for layer in json_data["layers"]}
	for t_layer in tagged_layer:
		if t_layer.name not in source_layers:
			log.log("Layer Tag", f"{t_layer} not found in source.", log_level=log.DEBUG.WARN)
		else:
			if t_layer.intensity == -1:
				t_layer = replace(t_layer, intensity=next(intensity_gen))
				while t_layer.intensity in preset_intensities:
					t_layer = replace(t_layer, intensity=next(intensity_gen))

			if t_layer.radius == -1:
				t_layer = replace(t_layer, radius=segment_radius)

			source_layers[t_layer.name]["name"] = str(t_layer)

			log.log("Layer Tag", f"{t_layer.name} updated: {t_layer}.", log_level=log.DEBUG.STATUS)

	json_data["layers"] = list(source_layers.values())

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	log.log("Layer Tag", f"Wrote {json_result}", log_level=log.DEBUG.STATUS)


if __name__ == "__main__":
	layer_tag()
