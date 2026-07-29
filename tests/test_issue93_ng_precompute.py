from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from click.testing import CliRunner
from cloudvolume import CloudVolume

from mctutil.ng.precompute import (
	build_plan,
	coerce_segmentation_dtype,
	create_volume_info,
	discover_input,
	natural_sort_key,
	precompute,
)
import mctutil.ng.precompute as precompute_module


def test_ng_precompute_writes_real_local_cloudvolume(tmp_path, monkeypatch):
	monkeypatch.setattr(precompute_module, "_require_cloudvolume", lambda: CloudVolume)
	input_path = tmp_path / "sample.tif"
	output_path = tmp_path / "sample_precomputed"
	source = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
	tifffile.imwrite(input_path, source, photometric="minisblack")

	result = CliRunner().invoke(
		precompute,
		[
			str(input_path),
			str(output_path),
			"--workers", "1",
			"--chunk-size", "2,2,1",
			"--voxel-offset", "10,20,30",
		],
	)

	assert result.exit_code == 0, result.output
	volume = CloudVolume(output_path.resolve().as_uri(), parallel=False)
	assert volume.info["scales"][0]["resolution"] == [700, 700, 700]
	assert volume.info["scales"][0]["voxel_offset"] == [10, 20, 30]
	written = np.asarray(volume[:, :, :, 0])[..., 0]
	assert written.shape == (4, 3, 2)
	assert np.array_equal(written, source.transpose(2, 1, 0))

	chunk_mtimes = {
		path.relative_to(output_path): path.stat().st_mtime_ns
		for path in output_path.rglob("*")
		if path.is_file() and path.name != "info"
	}
	resume_result = CliRunner().invoke(
		precompute,
		[
			str(input_path),
			str(output_path),
			"--workers", "1",
			"--chunk-size", "2,2,1",
			"--voxel-offset", "10,20,30",
		],
	)
	assert resume_result.exit_code == 0, resume_result.output
	assert "All Z planes are already present" in resume_result.output
	assert chunk_mtimes == {
		path.relative_to(output_path): path.stat().st_mtime_ns
		for path in output_path.rglob("*")
		if path.is_file() and path.name != "info"
	}


def test_ng_precompute_dry_run_uses_agreed_metadata_defaults(tmp_path):
	input_path = tmp_path / "sample.tif"
	output_path = tmp_path / "planned"
	tifffile.imwrite(
		input_path,
		np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3),
		photometric="minisblack",
	)

	result = CliRunner().invoke(
		precompute,
		[str(input_path), str(output_path), "--dry-run"],
	)

	assert result.exit_code == 0, result.output
	assert "Voxel resolution (nm): (700, 700, 700)" in result.output
	assert "Voxel offset: (0, 0, 0)" in result.output
	assert not output_path.exists()


def test_ng_precompute_directory_input_uses_natural_order(tmp_path):
	input_dir = tmp_path / "slices"
	input_dir.mkdir()
	for name, value in (("slice_10.tif", 10), ("slice_2.tif", 2), ("slice_1.tif", 1)):
		tifffile.imwrite(input_dir / name, np.full((2, 3), value, dtype=np.uint8))

	spec = discover_input(input_dir)

	assert spec.shape == (3, 2, 3)
	assert [Path(path).name for path in spec.source] == [
		"slice_1.tif",
		"slice_2.tif",
		"slice_10.tif",
	]
	assert natural_sort_key(Path("slice_2.tif")) < natural_sort_key(Path("slice_10.tif"))


def test_ng_precompute_segmentation_dtype_and_chunk_defaults(tmp_path, monkeypatch):
	monkeypatch.setattr(precompute_module, "_require_cloudvolume", lambda: CloudVolume)
	input_path = tmp_path / "labels_seg.tif"
	tifffile.imwrite(
		input_path,
		np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4),
		photometric="minisblack",
	)
	spec = discover_input(input_path)

	plan = build_plan(
		input_path,
		spec,
		"auto",
		"compressed_segmentation",
		None,
		None,
		(700, 700, 700),
		(0, 0, 0),
		(8, 8, 8),
	)

	assert plan.layer_type == "segmentation"
	assert plan.dtype == np.dtype("uint32")
	assert plan.chunk_size == (4, 3, 1)
	assert coerce_segmentation_dtype(np.dtype("uint64"), "compressed_segmentation", None) == np.dtype("uint64")
	assert create_volume_info(plan, spec)["scales"][0]["compressed_segmentation_block_size"] == [8, 8, 8]
