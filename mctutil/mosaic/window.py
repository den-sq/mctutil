"""Display windows for stitched float32 canvases, where 0 marks pixels no scan covers."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import tifffile, write_json
from mctutil.shared.log import log


def to_u16(values, low: float, high: float, covered) -> np.ndarray:
	"""clip((v - low) / (high - low)) * 65534 + 1, with 0 reserved for uncovered pixels."""
	scaled = (np.clip((values - low) / (high - low), 0, 1) * 65534 + 1).astype(np.uint16)
	scaled[~covered] = 0
	return scaled


def require_window(bounds) -> None:
	if not bounds[1] > bounds[0]:
		raise click.BadParameter("HI must be greater than LO", param_hint="--window")


def stack_paths(source: Path) -> list[Path]:
	paths = sorted(path for path in source.glob("*.tif") if path.stem.isdigit())
	if not paths:
		raise click.ClickException(f"no NNNNN.tif slices in {source}")
	return paths


@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--step", type=click.IntRange(min=1), default=97, show_default=True, help="Use every n-th slice.")
@click.option("--pixel-step", type=click.IntRange(min=1), default=97, show_default=True,
				help="Use every n-th covered pixel of each slice.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Also write the percentiles as JSON.")
def window(source, step, pixel_step, output):
	"""Print value percentiles of the covered (non-zero) pixels of a float32 slice stack."""
	tiff = tifffile()
	values = []
	for path in stack_paths(source)[::step]:
		image = tiff.imread(path)
		values.append(image[image != 0][::pixel_step])
	values = np.concatenate(values)
	result = {key: float(np.percentile(values, q)) for key, q in (("p0.1", 0.1), ("p99.9", 99.9), ("p1", 1), ("p99", 99))}
	if output:
		write_json(output, result)
	click.echo(json.dumps(result))


@click.command(name="to-u16")
@click.argument("source", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output", type=click.Path(file_okay=False, path_type=Path))
@click.option("--window", "bounds", type=(float, float), required=True, metavar="LO HI", help="Value window.")
def to_u16_command(source, output, bounds):
	"""Write a float32 slice stack as uint16 with a fixed window, keeping 0 for uncovered pixels (resumable)."""
	require_window(bounds)
	log.start()
	tiff = tifffile()
	output.mkdir(parents=True, exist_ok=True)
	written = 0
	for path in stack_paths(source):
		target = output / path.name
		if target.exists():
			continue
		image = tiff.imread(path)
		tiff.imwrite(target, to_u16(image, *bounds, image != 0))
		written += 1
	log.write("uint16 Stack", f"{written} slice(s) -> {output}")
