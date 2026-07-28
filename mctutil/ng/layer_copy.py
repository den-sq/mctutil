
import json
from pathlib import Path

import click

from mctutil.shared.log import log, LOG


@click.command()
@click.option("--json-file", "-j", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to merge annothations from.")
@click.option("--json-result", "-r", type=click.Path(path_type=Path, exists=False),
				required=True, help="Neuroglancer JSON file to write to.")
@click.option("--json-target", "-t", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
				required=True, help="Neuroglancer JSON file to use as base for new file.")
@click.argument("source_annotations", type=str, nargs=-1)
def layer_copy(json_file: Path, json_result: Path, json_target: Path, source_annotations: str):
	""" Merge annotation layers contained in a neuroglancer json file into another neuroglancer json, creating a new file.

		Example run:
			python layer_copy.py -j Octo_Digestive.json -t Octo_Respiratory.json -r Octo_Organs.json Esophagus Crop Lung
		"""
	with open(json_file) as json_handle:
		json_source = json.load(json_handle)
	with open(json_target) as json_handle:
		json_target = json.load(json_handle)
	log.write("Layer Copy", f"Loaded {json_file}", log_level=LOG.STATUS)

	existing_annotations = [layer["name"] for layer in json_source["layers"]]
	if len(source_annotations) > 0:
		duplicate_layers = set(source_annotations).intersection([layer["name"] for layer in json_target["layers"]])
	else:
		duplicate_layers = set(existing_annotations).intersection([layer["name"] for layer in json_target["layers"]])

	annotations = [layer for layer in json_source["layers"] if
					((layer["name"] in source_annotations) or (len(source_annotations) == 0))]

	log.write("Layer Copy", f"{len(annotations)} found to copy", log_level=LOG.STATUS)

	if len(source_annotations) > len(annotations):
		missing_annotations = [layer_name for layer_name in source_annotations if layer_name not in existing_annotations]
		log.write("Layer Copy", f"Annotations missing: {missing_annotations}", log_level=LOG.ERROR)
		return -1

	if len(duplicate_layers) > 0:
		log.write("Layer Copy", f"Some layers already exist in target: {duplicate_layers}", log_level=LOG.ERROR)
		return -1

	json_target["layers"].extend(annotations)

	with open(json_result, "w") as handle:
		json.dump(json_target, handle)

	log.write("Layer Copy", f"Wrote {json_result}", log_level=LOG.STATUS)


if __name__ == "__main__":
	layer_copy()
