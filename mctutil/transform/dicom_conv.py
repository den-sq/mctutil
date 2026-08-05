from pathlib import Path 	# if you haven't already done so

import click
import dicom2jpg


from mctutil.shared.log import log


@click.command
@click.argument('input_loc', type=click.Path(exists=True, path_type=Path))
@click.argument('output_loc', type=click.Path(path_type=Path))
@click.option("--dry-run", is_flag=True, help="Plan TIFF writes without decoding DICOM data.")
def dicom_conv(input_loc: Path, output_loc: Path, dry_run: bool):
	log.start()
	path_list = input_loc.iterdir() if input_loc.is_dir() else [input_loc]
	if not dry_run:
		output_loc.mkdir(parents=True, exist_ok=True)
	for path in path_list:
		target = output_loc.joinpath(*path.parts[-2:])
		if not dry_run:
			dicom2jpg.dicom2tiff(path, target)
		log.write(
			"Dry Run" if dry_run else "File Written",
			f"Would write {target}" if dry_run else str(target),
		)


if __name__ == "__main__":
	dicom_conv()
