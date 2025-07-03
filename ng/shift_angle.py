import json
from pathlib import Path

import click


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to change to forward-facing angled projection.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path, exists=False, writable=True),
				required=True, help="Neuroglancer JSON file to write updated orientation to.")
@click.option("--make-ortho", is_flag=True, show_default=True, default=False,
				help="Make the 3D projection orthographic (hotkey 'o') in the target json.")
def shift_angle(json_file, json_result, make_ortho):
	""" Shifts orientation of an existing ng json to have a consistent (forward-facing at an angle) orientation.
		Can also make it orthographic.

		Example:
			python shift_angle.py -j Octo_Respiratory.json -r Octo_Respiratory.json --make-ortho
"""
	with open(json_file) as json_handle:
		json_data = json.load(json_handle)

	print("json loaded")

	json_data["projectionOrientation"] = [
		0.3462526500225067,
		-0.14639334380626678,
		-0.36085399985313416,
		0.8535001277923584
	]

	print("orientation updated")

	if make_ortho:
		if isinstance(json_data["layout"], str):
			json_data["layout"] = {
				"type": json_data["layout"],
				"orthographicProjection": True
			}
		else:
			json_data["layout"]["orthographicProjection"] = True
		print("Made orthographic, if not already.")

	with open(json_result, "w") as handle:
		json.dump(json_data, handle)

	print(f"new arrangement written to {json_result}")


if __name__ == "__main__":
	shift_angle()
