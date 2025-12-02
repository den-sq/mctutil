import json
from pathlib import Path

import click


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to change to forward-facing angled projection.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path, exists=False, writable=True),
				required=True, help="Neuroglancer JSON file to write updated orientation to.")
@click.option("--json-target", "-t", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to use as base for new file.")
def position_copy(json_file, json_result, json_target):
	""" Copy position and orientation of a neuroglancer JSON from one file to another.

		Example:
			python shift_angle.py -j .json -t .json -r .json
"""
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)
	with open(json_target) as json_handle:
		json_upd = json.load(json_handle)

	print("json loaded")

	for field in ["position", "crossSectionOrientation", "crossSectionScale",
					"projectionOrientation", "projectionScale", "layout", "layerListPanel"]:
		json_upd[field] = json_data[field]

	print("orientation updated")

	with open(json_result, "w") as handle:
		json.dump(json_upd, handle)

	print(f"new arrangement written to {json_result}")


if __name__ == "__main__":
	position_copy()
