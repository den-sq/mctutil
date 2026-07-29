"""Validate structural metadata and representative reads from a precomputed layer."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from mctutil.shared.cli import XYZ
from mctutil.shared.cloudfiles_monitoring import patch_cloudfiles_monitoring


def normalize_cloudpath(layer_path: str) -> str:
	value = layer_path.removeprefix("precomputed://")
	if "://" in value:
		return value
	return Path(value).resolve().as_uri()


def _require_cloudvolume():
	try:
		from cloudvolume import CloudVolume
	except ImportError as exc:
		raise RuntimeError(
			"ng validate requires CloudVolume; install with pip install -e '.[ng]'"
		) from exc
	return CloudVolume


def load_info(cloudpath: str):
	try:
		from cloudfiles import CloudFiles
	except ImportError as exc:
		raise RuntimeError(
			"ng validate requires CloudFiles; install with pip install -e '.[ng]'"
		) from exc
	payload = CloudFiles(cloudpath, progress=False).get("info")
	if payload is None:
		raise ValueError(f"layer has no info metadata: {cloudpath}")
	try:
		return json.loads(payload)
	except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
		raise ValueError(f"layer info is not valid JSON: {cloudpath}/info") from exc


def validate_triplet(
	mip: int,
	scale: dict,
	field: str,
	positive: bool,
	integer: bool = False,
) -> list[str]:
	value = scale.get(field)
	if not isinstance(value, list) or len(value) != 3:
		return [f"mip {mip}: {field} must contain three values"]
	if any(
		isinstance(item, bool)
		or not isinstance(item, int if integer else (int, float))
		for item in value
	):
		kind = "integer" if integer else "numeric"
		return [f"mip {mip}: {field} entries must be {kind}"]
	if positive and any(item <= 0 for item in value):
		return [f"mip {mip}: {field} entries must be positive"]
	return []


def validate_chunk_sizes(mip: int, scale: dict) -> list[str]:
	chunk_sizes = scale.get("chunk_sizes")
	if (
		not isinstance(chunk_sizes, list)
		or not chunk_sizes
		or any(
			not isinstance(chunk, list)
			or len(chunk) != 3
			or any(not isinstance(item, int) or item <= 0 for item in chunk)
			for chunk in chunk_sizes
		)
	):
		return [f"mip {mip}: invalid chunk_sizes"]
	return []


def validate_scale(mip: int, scale) -> list[str]:
	if not isinstance(scale, dict):
		return [f"mip {mip}: scale must be an object"]
	errors = []
	for field in (
		"key",
		"encoding",
		"resolution",
		"voxel_offset",
		"size",
		"chunk_sizes",
	):
		if field not in scale:
			errors.append(f"mip {mip}: missing {field}")
	for field in ("key", "encoding"):
		if field in scale and (
			not isinstance(scale[field], str) or not scale[field]
		):
			errors.append(f"mip {mip}: {field} must be a non-empty string")
	errors.extend(validate_triplet(mip, scale, "resolution", True))
	errors.extend(validate_triplet(mip, scale, "voxel_offset", False, integer=True))
	errors.extend(validate_triplet(mip, scale, "size", True, integer=True))
	errors.extend(validate_chunk_sizes(mip, scale))
	return errors


def validate_info(info) -> list[str]:
	if not isinstance(info, dict):
		return ["info must be a JSON object"]
	errors = []
	for field in ("type", "data_type", "num_channels", "scales"):
		if field not in info:
			errors.append(f"missing info field: {field}")
	if info.get("type") not in {"image", "segmentation"}:
		errors.append(f"unsupported layer type: {info.get('type')!r}")
	try:
		np.dtype(info.get("data_type"))
	except (TypeError, ValueError):
		errors.append(f"invalid data_type: {info.get('data_type')!r}")
	if (
		isinstance(info.get("num_channels"), bool)
		or not isinstance(info.get("num_channels"), int)
		or info.get("num_channels", 0) < 1
	):
		errors.append(f"invalid num_channels: {info.get('num_channels')!r}")

	scales = info.get("scales")
	if not isinstance(scales, list) or not scales:
		errors.append("scales must be a non-empty list")
		return errors
	for mip, scale in enumerate(scales):
		errors.extend(validate_scale(mip, scale))
	return errors


def clamp_block(
	start: tuple[int, int, int],
	size: tuple[int, int, int],
	minimum: tuple[int, int, int],
	maximum: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
	if any(length <= 0 for length in size):
		raise ValueError(f"block-size entries must be positive: {size}")
	clamped_start = tuple(
		max(low, min(position, high - 1))
		for position, low, high in zip(start, minimum, maximum)
	)
	end = tuple(
		min(position + length, high)
		for position, length, high in zip(clamped_start, size, maximum)
	)
	if any(stop <= begin for begin, stop in zip(clamped_start, end)):
		raise ValueError(
			f"empty block after bounds clamp: start={start}, size={size}"
		)
	return clamped_start, end


def read_block(
	volume,
	name: str,
	start: tuple[int, int, int],
	size: tuple[int, int, int],
	layer_type: str,
) -> dict:
	minimum = tuple(int(value) for value in volume.bounds.minpt)
	maximum = tuple(int(value) for value in volume.bounds.maxpt)
	start, end = clamp_block(start, size, minimum, maximum)
	data = np.asarray(
		volume[
			start[0]:end[0],
			start[1]:end[1],
			start[2]:end[2],
		]
	)
	if data.size == 0:
		raise ValueError(f"{name} block read returned no values")
	stats = {
		"name": name,
		"start": start,
		"end": end,
		"shape": tuple(int(value) for value in data.shape),
		"dtype": str(data.dtype),
		"minimum": float(np.min(data)),
		"maximum": float(np.max(data)),
		"mean": float(np.mean(data)),
		"nonzero": int(np.count_nonzero(data)),
	}
	if layer_type == "segmentation":
		stats["unique"] = int(np.unique(data).size)
	return stats


def echo_block(stats: dict) -> None:
	message = (
		f"{stats['name']}: {stats['start']}..{stats['end']} "
		f"shape={stats['shape']} dtype={stats['dtype']} "
		f"min={stats['minimum']:g} max={stats['maximum']:g} "
		f"mean={stats['mean']:g} nonzero={stats['nonzero']}"
	)
	if "unique" in stats:
		message += f" unique={stats['unique']}"
	click.echo(message)


def echo_metadata(cloudpath: str, info: dict) -> None:
	scales = info.get("scales")
	scale_count = len(scales) if isinstance(scales, list) else 0
	click.echo(f"Layer: {cloudpath}")
	click.echo(
		f"Type: {info.get('type')}; dtype: {info.get('data_type')}; "
		f"channels: {info.get('num_channels')}; mips: {scale_count}"
	)
	for scale_mip, scale in enumerate(scales if isinstance(scales, list) else []):
		if not isinstance(scale, dict):
			click.echo(f"Mip {scale_mip}: invalid scale metadata")
			continue
		click.echo(
			f"Mip {scale_mip}: key={scale.get('key')} "
			f"resolution={scale.get('resolution')} size={scale.get('size')} "
			f"encoding={scale.get('encoding')} "
			f"sharded={bool(scale.get('sharding'))}"
		)


@click.command("validate")
@click.argument("layer_path")
@click.option("--mip", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--block-size", type=XYZ, default="64,64,64", show_default=True)
@click.option("--origin-at", type=XYZ, help="Origin-block start coordinate.")
@click.option("--center-at", type=XYZ, help="Center-block center coordinate.")
@click.option("--origin/--skip-origin", "read_origin", default=True, show_default=True)
@click.option("--center/--skip-center", "read_center", default=True, show_default=True)
@click.option("--metadata-only", is_flag=True, help="Skip representative data reads.")
def validate(
	layer_path: str,
	mip: int,
	block_size: tuple[int, int, int],
	origin_at: tuple[int, int, int] | None,
	center_at: tuple[int, int, int] | None,
	read_origin: bool,
	read_center: bool,
	metadata_only: bool,
) -> None:
	"""Validate metadata and representative blocks in a precomputed layer."""
	try:
		patch_cloudfiles_monitoring()
		cloudpath = normalize_cloudpath(layer_path)
		info = load_info(cloudpath)
		errors = validate_info(info)
		if not isinstance(info, dict):
			raise ValueError("; ".join(errors))
		echo_metadata(cloudpath, info)
		if errors:
			raise ValueError("; ".join(errors))
		if metadata_only:
			click.echo("Structural validation passed; data reads skipped.")
			return

		CloudVolume = _require_cloudvolume()
		volume = CloudVolume(
			cloudpath,
			mip=mip,
			parallel=False,
			bounded=True,
			cache=False,
			fill_missing=False,
		)
		minimum = tuple(int(value) for value in volume.bounds.minpt)
		maximum = tuple(int(value) for value in volume.bounds.maxpt)
		if read_origin:
			echo_block(
				read_block(
					volume,
					"origin",
					origin_at or minimum,
					block_size,
					info["type"],
				)
			)
		if read_center:
			center = center_at or tuple(
				(low + high) // 2
				for low, high in zip(minimum, maximum)
			)
			start = tuple(
				position - length // 2
				for position, length in zip(center, block_size)
			)
			echo_block(
				read_block(
					volume,
					"center",
					start,
					block_size,
					info["type"],
				)
			)
		click.echo("Validation passed.")
	except click.ClickException:
		raise
	except Exception as exc:
		raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
	validate()
