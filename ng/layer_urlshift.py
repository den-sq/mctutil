
import json
from pathlib import Path

import click


def upd_layer(name, source, shifted_layers):
	if isinstance(source, str):
		split_data = source.split("|")
		if len(split_data) > 1:
			shifted_layers[name]["source"] = f"precomputed://{split_data[0]}"
	elif isinstance(source, dict):
		split_data = source["url"].split("|")
		if len(split_data) > 1:
			shifted_layers[name]["source"]["url"] = f"precomputed://{split_data[0]}"
	else:
		split_data = source[0]["url"].split("|")
		split_data_two = source[1].split("|")
		if len(split_data) > 1:
			shifted_layers[name]["source"][0]["url"] = f"precomputed://{split_data[0]}"
		if len(split_data_two) > 1:
			shifted_layers[name]["source"][1] = f"precomputed://{split_data_two[0]}"


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to update annotations in.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path, exists=False),
				required=True, help="Neuroglancer JSON file to write to.")
def layer_urlshift(json_file: Path, json_result: Path):
	""" Reverts annotation layer naming scheme in neuroglancer to one that cloudvolume can still handle..

		Example run:
			python layer_urlshift.py -j Octo_7_stitch_sucker_6.json -r Octo_7_stitch_sucker_6_RN.json
		"""
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	print("json loaded")

	shifted_layers = {layer["name"]: layer for layer in json_data["layers"]}
	for name, layer in shifted_layers.items():
		if layer["type"] == "image" or layer["type"] == "segmentation":
			upd_layer(name, layer["source"], shifted_layers)

	json_data["layers"] == shifted_layers

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	print(f"new annotation written to {json_result}")


if __name__ == "__main__":
	layer_urlshift()
