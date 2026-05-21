import csv
import json
from pathlib import Path
import uuid

import click
import numpy as np

from shared import log


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to add an annotation layer to.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path),
				required=True, help="Neuroglancer JSON file to write with added annotations.")
@click.option("--base-layer", "-b", type=click.STRING,
				required=True, help="Base layer for annotation geometry.")
@click.option("--points", "-p", type=click.Path(path_type=Path),
				required=True, help="CSV or NPY File holding annotation coordinates.")
@click.argument("name", type=click.STRING)
def point_add(json_file: Path, json_result: Path, base_layer: str, points: Path, name: str):
	""" Add a series of points as a new annotation to the json file.

		Example run:
			python point_add.py -j R2_L3.json -r R2_L3_Upd.json -p points.csv
		"""
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	log.log("Point Add", f"Loaded {json_file}", log_level=log.DEBUG.STATUS)

	for layer in json_data["layers"]:
		if layer["name"] == base_layer:
			if "transform" in layer["source"]:
				tf = layer["source"]["transform"]
			else:
				tf = None

	new_layer = {
		"type": "annotation",
		"source": {
			"url": "local://annotations"
		},
		"tool": "annotatePoint",
		"tab": "annotations",
		"name": name
	}
	if tf is not None:
		new_layer["source"]["transform"] = tf

	if points.suffix == ".npy":
		point_array = np.load(points)
	elif points.suffix == ".csv":
		with open(points) as csvfile:
			point_reader = csv.reader(csvfile)
			point_array = np.array([row for row in point_reader])
	else:
		log.log("Point Add", "Points must be in .npy or .csv format.", log_level=log.DEBUG.ERROR)
		return

	if len(point_array.shape) != 2 or point_array.shape[-1] != 3:
		log.log("Point Add", "Points must be an array of 3D XYZ points.", log_level=log.DEBUG.ERROR)
		return

	uuids = [uuid.uuid1() for x in range(len(point_array))]
	new_layer["annotations"] = [{"type": "point", "id": str(idval), "point": list(point)}
								for point, idval in zip(point_array, uuids)]
	json_data["layers"].append(new_layer)
	log.log("Point Add", "Annotation added", log_level=log.DEBUG.STATUS)

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	log.log("Point Add", f"Wrote {json_result}", log_level=log.DEBUG.STATUS)


if __name__ == "__main__":
	point_add()
