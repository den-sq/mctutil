
import json
from pathlib import Path

import click

from mctutil.shared import log


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to merge annothations from.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path, exists=False),
				required=True, help="Neuroglancer JSON file to write to.")
@click.argument("layers", type=str, nargs=-1)
def layer_extract(json_file: Path, json_result: Path, layers: str):
	""" Extract layers contained in a neuroglancer json file into a new file.

		Example run:
			python layer_extracct.py -j Full_Annotation.json r Limited_Annotations.json
		"""
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)
	log.log("Layer Extract", f"Loaded {json_file}", log_level=log.DEBUG.STATUS)

	source_layers = {layer["name"]: layer for layer in json_data["layers"]}
	copy_layers = []
	for layer in layers:
		if layer not in source_layers:
			log.log("Layer Extract", f"{layer} not found in source.", log_level=log.DEBUG.WARN)
		else:
			log.log("Layer Extract", f"{layer} retained.", log_level=log.DEBUG.STATUS)
			copy_layers.append(source_layers[layer])

	json_data["layers"] = copy_layers

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	log.log("Layer Extract", f"Wrote {json_result}", log_level=log.DEBUG.STATUS)


if __name__ == "__main__":
	layer_extract()
