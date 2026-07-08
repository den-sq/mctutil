"""Convert a raw 3D image volume to a TIFF stack or a folder of per-Z TIFF files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from mctutil.shared.log import LOG, log


DTYPES: dict[str, Any] = {
	"uint8": np.uint8,
	"uint16": np.uint16,
	"uint32": np.uint32,
	"int8": np.int8,
	"int16": np.int16,
	"int32": np.int32,
	"float32": np.float32,
	"float64": np.float64,
}
BYTE_ORDER_CHOICES = ("native", "little", "big")
OUTPUT_MODE_CHOICES = ("stack", "folder")


def _require_tifffile():
	try:
		import tifffile
	except ImportError as exc:
		raise click.ClickException(
			"tifffile is required for TIFF output; install mctutil[transform]."
		) from exc
	return tifffile


def make_dtype(dtype, byte_order: str) -> np.dtype:
	"""Return a numpy dtype with the requested byte order applied.

		:param dtype: Any argument accepted by ``np.dtype``.
		:param byte_order: One of ``native`` / ``little`` / ``big``.
		:return: A concrete ``np.dtype`` with the byte order set.
	"""
	dtype = np.dtype(dtype)
	if byte_order == "little":
		return dtype.newbyteorder("<")
	if byte_order == "big":
		return dtype.newbyteorder(">")
	if byte_order == "native":
		return dtype.newbyteorder("=")
	raise ValueError(f"Unsupported byte order: {byte_order}")


def expected_size_bytes(width: int, height: int, depth: int, dtype: np.dtype) -> int:
	"""Bytes a raw volume with the given shape and dtype should occupy on disk."""
	return width * height * depth * np.dtype(dtype).itemsize


def validate_raw_size(
	raw_path: Path,
	width: int,
	height: int,
	depth: int,
	dtype: np.dtype,
	header_bytes: int,
) -> None:
	"""Raise if the file size does not match the declared shape + dtype + header.

		:param raw_path: Input ``.raw`` file.
		:param width: X dimension in voxels.
		:param height: Y dimension in voxels.
		:param depth: Z dimension in voxels (slice count).
		:param dtype: Voxel dtype.
		:param header_bytes: Number of bytes to skip at the file start.
	"""
	file_size = raw_path.stat().st_size
	expected = expected_size_bytes(width, height, depth, dtype) + header_bytes
	if file_size != expected:
		raise click.ClickException(
			"File size does not match supplied dimensions.\n"
			f"File size: {file_size:,} bytes\n"
			f"Expected size: {expected:,} bytes\n"
			f"Shape: depth={depth}, height={height}, width={width}\n"
			f"Dtype: {dtype}\n"
			f"Header bytes: {header_bytes:,}"
		)


def open_raw_volume(
	raw_path: Path,
	width: int,
	height: int,
	depth: int,
	dtype: np.dtype,
	header_bytes: int,
):
	"""Return a numpy ``memmap`` view over the raw volume in ``(depth, height, width)`` order."""
	return np.memmap(
		raw_path,
		dtype=dtype,
		mode="r",
		offset=header_bytes,
		shape=(depth, height, width),
		order="C",
	)


def write_single_tiff_stack(volume, output_path: Path, compression: str | None = None) -> None:
	"""Write the volume as one BigTIFF stack, streamed slice-by-slice."""
	tifffile = _require_tifffile()
	output_path.parent.mkdir(parents=True, exist_ok=True)

	depth = volume.shape[0]
	log.write("Raw Stack", str(output_path), log_level=LOG.STATUS)

	with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
		for z in range(depth):
			tif.write(volume[z], compression=compression, contiguous=False)
			if z % 100 == 0 or z == depth - 1:
				log.write("Raw Progress", f"slice {z + 1}/{depth}", log_level=LOG.INFO)

	log.write("Raw Done", str(output_path), log_level=LOG.STATUS)


def write_tiff_slice_folder(
	volume,
	output_dir: Path,
	prefix: str,
	compression: str | None = None,
) -> None:
	"""Write the volume as one TIFF file per Z slice under ``output_dir``."""
	tifffile = _require_tifffile()
	output_dir.mkdir(parents=True, exist_ok=True)

	depth = volume.shape[0]
	digits = max(4, len(str(depth)))
	log.write("Raw Folder", str(output_dir), log_level=LOG.STATUS)

	for z in range(depth):
		out_path = output_dir / f"{prefix}_z{z:0{digits}d}.tif"
		tifffile.imwrite(out_path, volume[z], compression=compression)
		if z % 100 == 0 or z == depth - 1:
			log.write("Raw Progress", f"slice {z + 1}/{depth}: {out_path.name}", log_level=LOG.INFO)

	log.write("Raw Done", str(output_dir), log_level=LOG.STATUS)


@click.command()
@click.argument("input_raw", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
	"-o", "--output",
	type=click.Path(path_type=Path),
	required=True,
	help="Output path. In stack mode, the target .tif; in folder mode, the target directory.",
)
@click.option(
	"--output-mode",
	type=click.Choice(OUTPUT_MODE_CHOICES),
	default="stack",
	show_default=True,
	help="Emit a single BigTIFF stack or a folder of one TIFF per Z slice.",
)
@click.option("--width", type=int, required=True, help="X dimension.")
@click.option("--height", type=int, required=True, help="Y dimension.")
@click.option("--depth", type=int, required=True, help="Z dimension (slice count).")
@click.option(
	"--dtype",
	type=click.Choice(sorted(DTYPES.keys())),
	required=True,
	help="Voxel datatype.",
)
@click.option(
	"--byte-order",
	type=click.Choice(BYTE_ORDER_CHOICES),
	default="native",
	show_default=True,
	help="Byte order of the raw data.",
)
@click.option(
	"--header-bytes",
	type=int,
	default=0,
	show_default=True,
	help="Number of bytes to skip at the start of the file.",
)
@click.option("--compress", is_flag=True, default=False, help="Use zlib compression in the output TIFF files.")
@click.option(
	"--prefix",
	type=str,
	default=None,
	help="Filename prefix in folder mode. Defaults to the input file's stem.",
)
def raw_convert(
	input_raw: Path,
	output: Path,
	output_mode: str,
	width: int,
	height: int,
	depth: int,
	dtype: str,
	byte_order: str,
	header_bytes: int,
	compress: bool,
	prefix: str | None,
) -> None:
	"""Convert a raw 3D image volume to a TIFF stack or per-Z TIFF folder."""
	full_dtype = make_dtype(DTYPES[dtype], byte_order)
	validate_raw_size(
		raw_path=input_raw,
		width=width,
		height=height,
		depth=depth,
		dtype=full_dtype,
		header_bytes=header_bytes,
	)

	log.write("Raw Input", str(input_raw), log_level=LOG.STATUS)
	log.write(
		"Raw Shape",
		f"depth={depth} height={height} width={width} dtype={full_dtype} mode={output_mode}",
		log_level=LOG.INFO,
	)

	volume = open_raw_volume(
		raw_path=input_raw,
		width=width,
		height=height,
		depth=depth,
		dtype=full_dtype,
		header_bytes=header_bytes,
	)

	compression = "zlib" if compress else None
	resolved_prefix = prefix or input_raw.stem

	if output_mode == "stack":
		write_single_tiff_stack(volume, output_path=output, compression=compression)
	elif output_mode == "folder":
		write_tiff_slice_folder(volume, output_dir=output, prefix=resolved_prefix, compression=compression)
	else:
		raise click.ClickException(f"Unsupported output mode: {output_mode}")


if __name__ == "__main__":
	raw_convert()
