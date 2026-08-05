from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
import tifffile as tf


from mctutil.shared.log import log, LOG
from mctutil.shared.np_convert import np_convert
from mctutil.shared import cli
from mctutil.shared.stack_apply import (
	apply_image_stack,
	plan_stack_map,
	run_parallel,
	tiff_paths,
)
from mctutil.shared.tiff_stack_writer import write_tiff_stack


def converted_image(image, dtype):
	"""Pure dtype conversion used by filename-preserving stack maps."""
	return np_convert(dtype, image)


def write_split(source: Path, output_folder: Path, sections: int, dtype: type, compression: bool):
	log.write("Preload", f"Loading {source.name}")
	source_meta = tf.TiffFile(source)
	log.write("Preload Meta", f"{source_meta}", log_level=LOG.INFO)
	for k in range(0, len(source_meta.series)):
		for j in range(0, len(source_meta.series[k].levels)):
			source_data = tf.imread(source, series=k, level=j)
			log.write("Loaded", f"Loaded {source.name}: {source_data.shape} (series {j}, level {k})")
			split_size = source_data.shape[1] // sections
			for i in range(0, sections):
				target_path = output_folder.joinpath(source.with_suffix("").name + f"_s{k}_l{j}_p{i + 1}.tiff")
				output = converted_image(
					source_data[:, i * split_size: (i + 1) * split_size],
					dtype,
				)
				write_tiff_stack(
					lambda _index, image=output: image,
					1,
					target_path,
					mode="image",
					compression=tf.COMPRESSION.LZW if compression else None,
				)
				log.write("Writing", f"Wrote {target_path}")


def convert_stack(
	input_folder: Path,
	output_folder: Path,
	dtype: type,
	*,
	horizontal_sections: int = 1,
	compression: bool = True,
	preserve_names: bool = False,
	workers: int = 24,
	dry_run: bool = False,
) -> None:
	"""Run the common parallel dtype-conversion implementation."""
	if preserve_names and horizontal_sections != 1:
		raise click.UsageError(
			"--preserve-names cannot be combined with --horizontal-sections"
		)
	input_folder = Path(input_folder)
	output_folder = Path(output_folder)
	sources = tiff_paths(input_folder)
	if preserve_names:
		items = plan_stack_map(sources, output_folder)
		apply_image_stack(
			items,
			converted_image,
			operation_args=(dtype,),
			compression=tf.COMPRESSION.LZW if compression else None,
			workers=workers,
			dry_run=dry_run,
			pool_factory=Pool,
		)
		return

	if dry_run:
		for source in sources:
			log.write(
				"Dry Run",
				f"Would convert {source} into {horizontal_sections} section(s) in {output_folder}",
				log_level=LOG.INFO,
			)
		return
	output_folder.mkdir(exist_ok=True, parents=True)
	run_parallel(
		write_split,
		(
			(
				source,
				output_folder,
				horizontal_sections,
				dtype,
				compression,
			)
			for source in sources
		),
		workers,
		pool_factory=Pool,
	)


@click.command()
@click.option("-t", "--output-type", "--output_type", type=click.STRING, required=True, help="Type of output data.")
@click.option("-h", "--horizontal-sections", "--horizontal_sections", type=click.INT, default=1,
				help="Number of horizontal sections to split image into.")
@click.option("--uncompressed", is_flag=True)
@click.option(
	"--preserve-names",
	is_flag=True,
	help="Write one dtype-converted output per input using the original filename.",
)
@click.option("-j", "--workers", type=click.IntRange(min=1), default=24, show_default=True)
@click.option("--dry-run", is_flag=True, help="Plan output files without writing them.")
@click.argument("input_folder", type=click.Path(exists=True, path_type=Path))
@click.argument("output_folder", type=click.Path(path_type=Path))
def convert(
	output_type,
	horizontal_sections,
	uncompressed,
	preserve_names,
	workers,
	dry_run,
	input_folder,
	output_folder,
):
	log.write("Setup", f"Converting each file into {horizontal_sections} {output_type} files.")
	convert_stack(
		input_folder,
		output_folder,
		np.dtype(output_type),
		horizontal_sections=horizontal_sections,
		compression=not uncompressed,
		preserve_names=preserve_names,
		workers=workers,
		dry_run=dry_run,
	)


@click.command("downsample")
@click.option('-d', '--data-dir', type=click.Path(exists=True), help='Input path for original dataset.')
@click.option('-o', '--output-dir', type=click.Path(), help='Output path for transformed dataset.')
@click.option('-t', "--out-dtype", type=cli.NUMPYTYPE, default=np.uint8, help="Datatype of Output.")
def downsample(data_dir, output_dir, out_dtype):
	"""Deprecated alias for filename-preserving dtype conversion."""
	click.echo(
		"Warning: transform downsample is deprecated; use transform convert "
		"--preserve-names --uncompressed --workers 1 instead.",
		err=True,
	)
	log.start()
	convert_stack(
		Path(data_dir),
		Path(output_dir),
		out_dtype.nptype,
		compression=False,
		preserve_names=True,
		workers=1,
	)


if __name__ == "__main__":
	convert()
