"""Compact long flat-field frame series into a small beam-drift digest."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mctutil.shared.log import log


def _require_tifffile():
	try:
		import tifffile
	except ImportError as exc:
		raise click.ClickException(
			"tifffile is required for flat-field digest generation; install mctutil[flats]."
		) from exc
	return tifffile


class FrameSource:
	"""Random-access TIFF reader for a directory of frames or one multipage stack."""

	def __init__(self, path):
		tifffile = _require_tifffile()
		path = Path(path)
		if path.is_dir():
			self.frame_paths = sorted([*path.glob("*.tif"), *path.glob("*.tiff")])
			if not self.frame_paths:
				raise click.ClickException(f"no TIFF frames found in directory {path}")
			self.n_frames = len(self.frame_paths)
			self._stack = None
		else:
			self._stack = tifffile.TiffFile(path)
			self.frame_paths = None
			self.n_frames = len(self._stack.pages)

	def read(self, index):
		tifffile = _require_tifffile()
		if self.frame_paths is not None:
			return tifffile.imread(self.frame_paths[index]).astype(np.float32)
		return self._stack.pages[index].asarray().astype(np.float32)

	def close(self):
		if self._stack is not None:
			self._stack.close()


def beam_centroid(frame):
	"""Return intensity-weighted centroid row/col and total intensity."""
	image = np.clip(frame, 0, None).astype(np.float64)
	total_intensity = image.sum()
	if total_intensity <= 0:
		return np.nan, np.nan, 0.0
	n_rows, n_cols = image.shape
	centroid_row = (image.sum(axis=1) @ np.arange(n_rows)) / total_intensity
	centroid_col = (image.sum(axis=0) @ np.arange(n_cols)) / total_intensity
	return centroid_row, centroid_col, total_intensity


def spatial_bin(frame, factor):
	"""Downsample by an integer factor using a factor-by-factor box mean."""
	if factor <= 1:
		return frame
	n_rows, n_cols = frame.shape
	n_rows -= n_rows % factor
	n_cols -= n_cols % factor
	return (
		frame[:n_rows, :n_cols]
		.reshape(n_rows // factor, factor, n_cols // factor, factor)
		.mean(axis=(1, 3))
	)


def parse_crop(_ctx, _param, value):
	if value is None:
		return None
	try:
		row0, row1, col0, col1 = [int(part) for part in value.split(",")]
	except ValueError as exc:
		raise click.BadParameter("Use row0,row1,col0,col1.") from exc
	if row1 <= row0 or col1 <= col0:
		raise click.BadParameter("Crop stop values must be greater than start values.")
	return row0, row1, col0, col1


def sample_centres(n_total, keep):
	interval_edges = np.linspace(0, n_total, keep + 1)
	return ((interval_edges[:-1] + interval_edges[1:]) / 2).astype(int)


@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--keep", type=click.IntRange(1), default=120, show_default=True, help="Number of snapshots.")
@click.option(
	"--median-window",
	type=click.IntRange(1),
	default=5,
	show_default=True,
	help="Frames to median around each sample centre.",
)
@click.option("--bin", "bin_factor", type=click.IntRange(1), default=1, show_default=True, help="Spatial bin factor.")
@click.option("--crop", callback=parse_crop, help="row0,row1,col0,col1 crop applied to every frame.")
@click.option(
	"--out",
	"output_dir",
	type=click.Path(file_okay=False, path_type=Path),
	default=".",
	show_default=True,
	help="Output directory.",
)
@click.option("--dry-run", is_flag=True, help="Plan the digest without writing or reading sampled image data.")
def series_digest(path, keep, median_window, bin_factor, crop, output_dir, dry_run):
	"""Build digest_stack.tif and drift_trajectory.csv from flat-field frames."""
	log.start()
	source = FrameSource(path)
	try:
		n_total = source.n_frames
		centres = sample_centres(n_total, keep)
		window = max(1, median_window)
		half_window = window // 2
		log.write(
			"Digest Plan",
			f"{n_total} frames -> {keep} snapshots (interval {n_total / keep:.1f} frames), median window {window}",
		)

		if dry_run:
			log.write("Dry Run", f"Would write {output_dir / 'drift_trajectory.csv'}")
			log.write("Dry Run", f"Would write {output_dir / 'digest_stack.tif'}")
			return

		output_dir.mkdir(parents=True, exist_ok=True)
		snapshots = []
		trajectory = []
		for centre in centres:
			low = max(0, centre - half_window)
			high = min(n_total, centre + half_window + 1)

			window_frames = []
			for index in range(low, high):
				frame = source.read(index)
				if crop is not None:
					row0, row1, col0, col1 = crop
					frame = frame[row0:row1, col0:col1]
				window_frames.append(frame)

			snapshot = np.median(np.stack(window_frames), axis=0)
			snapshot = spatial_bin(snapshot, bin_factor)
			snapshots.append(snapshot.astype(np.float32))

			centroid_row, centroid_col, total_intensity = beam_centroid(snapshot)
			trajectory.append((centre, total_intensity, centroid_row, centroid_col))

		trajectory = np.array(trajectory, float)
		intensity_norm = trajectory[:, 1] / trajectory[:, 1].max()
		drift_row = trajectory[:, 2] - trajectory[0, 2]
		drift_col = trajectory[:, 3] - trajectory[0, 3]
		csv_table = np.column_stack(
			[trajectory[:, 0], intensity_norm, trajectory[:, 2], trajectory[:, 3], drift_row, drift_col]
		)

		csv_path = output_dir / "drift_trajectory.csv"
		np.savetxt(
			csv_path,
			csv_table,
			delimiter=",",
			header="frame_index,intensity_norm,centroid_row,centroid_col,drift_row,drift_col",
			comments="",
		)
		log.write("File Written", str(csv_path))

		line = np.polyval(np.polyfit(csv_table[:, 0], drift_row, 1), csv_table[:, 0])
		linear_residual = float(np.std(drift_row - line))
		log.write("Total Drift", f"row={drift_row[-1]:+.2f} col={drift_col[-1]:+.2f} px")
		log.write("Intensity Change", f"{(intensity_norm[-1] - intensity_norm[0]) * 100:+.1f}%")
		log.write(
			"Linearity",
			(
				f"vertical residual={linear_residual:.3f} px "
				f"({'nonlinear' if linear_residual > 0.5 else 'roughly linear'})"
			),
		)

		tifffile = _require_tifffile()
		digest_path = output_dir / "digest_stack.tif"
		tifffile.imwrite(digest_path, np.stack(snapshots))
		size_mb = digest_path.stat().st_size / 1e6
		log.write("File Written", f"{digest_path} ({len(snapshots)} frames, {size_mb:.1f} MB)")
	finally:
		source.close()


if __name__ == "__main__":
	series_digest()
