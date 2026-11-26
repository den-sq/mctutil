"""Depth offset between two scans sharing a lateral frame, from 3-D sub-block correlation."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from mctutil.mosaic.common import RowStack, band_pass, phase_correlation
from mctutil.shared.log import log


def block_centres(n: int, block: int, step: int, radius: float, grid=None) -> list[tuple[int, int]]:
	"""Sub-block centres inside ``radius`` of the slice centre; ``grid`` = (start, stop, step) overrides."""
	centre = (n - 1) / 2
	if grid is None:
		half = int(min(radius, n / 2 - block / 2) // step) * step
		axis = [int(round(centre + k)) for k in range(-half, half + 1, step)]
	else:
		axis = list(range(*grid))
	return [(yc, xc) for yc in axis for xc in axis if np.hypot(yc - centre, xc - centre) <= radius]


def windowed_block(volume, yc: int, xc: int, block: int) -> np.ndarray:
	return volume[:, yc - block // 2:yc + block // 2, xc - block // 2:xc + block // 2]


def block_offsets(fixed, moving, centres, block: int, search: int) -> list[tuple]:
	"""Per centre: (z, dy, dx, peak) of moving relative to fixed; z indexes the padded moving stack."""
	depth = fixed.shape[0]
	window = (np.hanning(depth)[:, None, None] * np.outer(np.hanning(block), np.hanning(block))[None]).astype(np.float32)
	shape = (depth + 2 * search + depth, block, block)
	offsets = []
	for yc, xc in centres:
		spectrum = np.fft.rfftn(windowed_block(fixed, yc, xc, block) * window, shape, axes=(0, 1, 2))
		peak, shift = phase_correlation(spectrum, windowed_block(moving, yc, xc, block), shape)
		offsets.append((*shift, peak))
	return offsets


def demeaned_stack(stack: RowStack, rows) -> np.ndarray:
	volume = np.stack([band_pass(stack.read(row), 1.5, 12) for row in rows])
	return volume - volume.mean(0)


def depth_link(fixed: RowStack, moving: RowStack, fixed_row: int, guess: int, *, depth: int, search: int,
				block: int, step: int, radius: float, min_peak: float, grid=None) -> dict:
	"""Median depth offset T (moving row = fixed row + T) over sub-blocks above ``min_peak``.

	Each band-passed stack has its mean over z removed, which drops z-invariant artefacts
	(rings, the axis line) that would otherwise lock the correlation onto zero shift.
	"""
	depth = min(depth, fixed.rows - fixed_row)
	a = demeaned_stack(fixed, range(fixed_row, fixed_row + depth))
	b = demeaned_stack(moving, range(fixed_row + guess - search, fixed_row + guess + depth + search))
	offsets = np.array(block_offsets(a, b, block_centres(fixed.size, block, step, radius, grid), block, search))
	good = offsets[:, 3] > min_peak
	if not good.any():
		raise click.ClickException(f"no sub-block correlation peak exceeds {min_peak}")
	t = guess - offsets[good, 0] - search
	return {
		"fy0": fixed_row, "n": int(good.sum()), "T_median": float(np.median(t)),
		"T_iqr": np.percentile(t, [25, 75]).tolist(),
		"dy_median": float(np.median(offsets[good, 1])), "dx_median": float(np.median(offsets[good, 2])),
	}


@click.command(name="link-depth")
@click.argument("fixed_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("moving_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--fixed-row", type=click.IntRange(min=0), required=True, help="First fixed row of the sampled block.")
@click.option("--guess", type=int, required=True, help="Expected offset: moving row = fixed row + guess.")
@click.option("--radius", type=float, required=True, help="Sub-block centres stay within this many px of the axis.")
@click.option("--fixed-name", help="Name recorded for the fixed scan (default: directory name).")
@click.option("--moving-name", help="Name recorded for the moving scan (default: directory name).")
@click.option("--depth", type=click.IntRange(min=2), default=48, show_default=True, help="Rows per block.")
@click.option("--search", type=click.IntRange(min=0), default=12, show_default=True,
				help="Extra moving rows on each side of the guess.")
@click.option("--block", type=click.IntRange(min=8), default=320, show_default=True, help="Sub-block width (px).")
@click.option("--step", type=click.IntRange(min=1), default=160, show_default=True, help="Sub-block spacing (px).")
@click.option("--grid", type=(int, int, int), help="Explicit centre range START STOP STEP for both axes.")
@click.option("--min-peak", type=float, default=0.003, show_default=True, help="Minimum correlation peak.")
def link_depth(fixed_dir, moving_dir, output, fixed_row, guess, radius, fixed_name, moving_name,
				depth, search, block, step, grid, min_peak):
	"""Append one depth link (moving row = fixed row + T) between two row stacks to OUTPUT (JSON lines)."""
	log.start()
	record = depth_link(
		RowStack(fixed_dir), RowStack(moving_dir), fixed_row, guess, depth=depth, search=search,
		block=block, step=step, radius=radius, min_peak=min_peak, grid=grid,
	)
	record = {"fixed": fixed_name or fixed_dir.name, "moving": moving_name or moving_dir.name, **record}
	with output.open("a", encoding="utf-8") as stream:
		stream.write(json.dumps(record) + "\n")
	log.write("Depth Link", json.dumps(record))
