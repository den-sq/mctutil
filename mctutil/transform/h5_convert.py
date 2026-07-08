"""Extract image-like datasets from an HDF5 file into TIFF stacks or images."""

from __future__ import annotations

import re
from pathlib import Path

import click
import numpy as np

from mctutil.shared.log import LOG, log


DTYPE_CHOICES = ("uint8", "uint16", "uint32", "float32", "float64")


def _require_h5py():
	try:
		import h5py
	except ImportError as exc:
		raise click.ClickException(
			"h5py is required for HDF5 → TIFF extraction; install mctutil[transform]."
		) from exc
	return h5py


def _require_tifffile():
	try:
		import tifffile
	except ImportError as exc:
		raise click.ClickException(
			"tifffile is required for TIFF output; install mctutil[transform]."
		) from exc
	return tifffile


def safe_name(name: str) -> str:
	"""Turn an HDF5 dataset path into a filesystem-safe filename stem.

		:param name: HDF5 path like ``/exchange/data``.
		:return: A filename-safe string; ``/`` is replaced with ``__`` and any
			other character outside ``[A-Za-z0-9_.-]`` becomes ``_``. The empty
			root maps to ``"root"``.
	"""
	name = name.strip("/")
	if not name:
		return "root"
	name = name.replace("/", "__")
	name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
	return name


def is_image_like_dataset(dataset) -> bool:
	"""Heuristic filter for datasets worth writing out as image data.

		:param dataset: An ``h5py.Dataset`` object.
		:return: ``True`` if the dataset is at least 2-D and has a numeric dtype.
	"""
	if dataset.ndim < 2:
		return False
	if not np.issubdtype(dataset.dtype, np.number):
		return False
	return True


def write_dataset_as_tiffs(
	dataset,
	h5_path: str,
	output_dir: Path,
	dtype=None,
	compress: bool = True,
) -> None:
	"""Write one HDF5 dataset out as one or more TIFF files.

		Rules:
		- 2-D → one TIFF image.
		- 3-D → one BigTIFF stack, streamed page by page.
		- 4-D+ → one BigTIFF stack per first-axis index, streamed page by page.

		:param dataset: The ``h5py.Dataset`` to export.
		:param h5_path: Original HDF5 path (used to build the output filename).
		:param output_dir: Directory that will receive the TIFF files.
		:param dtype: Optional numpy dtype to cast the output to.
		:param compress: If true, write with zlib compression.
	"""
	tifffile = _require_tifffile()

	base = safe_name(h5_path)
	compression = "zlib" if compress else None

	log.write("H5 Dataset", f"{h5_path} shape={dataset.shape} dtype={dataset.dtype}", log_level=LOG.STATUS)

	if dataset.ndim == 2:
		arr = dataset[()]
		if dtype is not None:
			arr = arr.astype(dtype)

		out_path = output_dir / f"{base}.tif"
		tifffile.imwrite(out_path, arr, compression=compression, bigtiff=True)
		log.write("H5 Wrote", str(out_path), log_level=LOG.STATUS)

	elif dataset.ndim == 3:
		out_path = output_dir / f"{base}_stack.tif"
		with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
			for z in range(dataset.shape[0]):
				arr = dataset[z, :, :]
				if dtype is not None:
					arr = arr.astype(dtype)
				tif.write(arr, compression=compression, contiguous=False)
				if z % 100 == 0:
					log.write("H5 Progress", f"slice {z + 1}/{dataset.shape[0]}", log_level=LOG.INFO)

		log.write("H5 Wrote", str(out_path), log_level=LOG.STATUS)

	else:
		for i in range(dataset.shape[0]):
			out_path = output_dir / f"{base}_{i:04d}_stack.tif"
			with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
				for z in range(dataset.shape[1]):
					arr = dataset[i, z, :, :]
					if dtype is not None:
						arr = arr.astype(dtype)
					tif.write(arr, compression=compression, contiguous=False)
					if z % 100 == 0:
						log.write(
							"H5 Progress",
							f"stack {i + 1}/{dataset.shape[0]}, slice {z + 1}/{dataset.shape[1]}",
							log_level=LOG.INFO,
						)

			log.write("H5 Wrote", str(out_path), log_level=LOG.STATUS)


def visit_h5(handle, output_dir: Path, dtype=None, compress: bool = True) -> int:
	"""Walk an open HDF5 handle and export every image-like dataset.

		:param handle: Open ``h5py.File`` object.
		:param output_dir: Directory to receive the TIFF files.
		:param dtype: Optional numpy dtype to cast the output to.
		:param compress: If true, write with zlib compression.
		:return: Number of datasets exported.
	"""
	h5py = _require_h5py()
	found = 0

	def visitor(name, obj):
		nonlocal found
		if isinstance(obj, h5py.Dataset) and is_image_like_dataset(obj):
			h5_path = "/" + name
			write_dataset_as_tiffs(
				obj,
				h5_path=h5_path,
				output_dir=output_dir,
				dtype=dtype,
				compress=compress,
			)
			found += 1

	handle.visititems(visitor)
	return found


@click.command()
@click.argument("input_h5", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
	"-o", "--output-dir",
	type=click.Path(file_okay=False, path_type=Path),
	default=None,
	help="Output directory. Defaults to <input_stem>_tiff_stacks alongside the input.",
)
@click.option(
	"--dtype",
	type=click.Choice(DTYPE_CHOICES),
	default=None,
	help="Optionally cast output TIFF data to this dtype.",
)
@click.option(
	"--no-compress",
	is_flag=True,
	default=False,
	help="Disable TIFF (zlib) compression.",
)
def h5_convert(input_h5: Path, output_dir: Path | None, dtype: str | None, no_compress: bool) -> None:
	"""Export image-like datasets from an HDF5 file as TIFF stacks."""
	h5py = _require_h5py()

	if output_dir is None:
		output_dir = input_h5.with_suffix("").parent / f"{input_h5.stem}_tiff_stacks"
	output_dir.mkdir(parents=True, exist_ok=True)

	cast_dtype = np.dtype(dtype) if dtype is not None else None

	log.write("H5 Open", str(input_h5), log_level=LOG.STATUS)
	with h5py.File(input_h5, "r") as handle:
		found = visit_h5(handle, output_dir, dtype=cast_dtype, compress=not no_compress)

	if found == 0:
		log.write("H5 Empty", "No image-like datasets found.", log_level=LOG.WARN)
	else:
		log.write("H5 Done", f"exported {found} dataset(s) to {output_dir}", log_level=LOG.STATUS)


if __name__ == "__main__":
	h5_convert()
