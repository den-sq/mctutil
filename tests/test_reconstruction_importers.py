from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import tifffile

from mctutil.parse import reconstruction_importers


def _put(handle, path: str, value):
	parent, name = path.rsplit("/", 1)
	return handle.require_group(parent).create_dataset(name, data=value)


def test_als832_imports_configured_setup_without_claiming_execution(tmp_path):
	path = tmp_path / "als-reconstruction.h5"
	base = "/process/tomo_rec/setup/algorithm"
	with h5py.File(path, "w") as handle:
		_put(handle, f"{base}/distance", 25.0)
		_put(handle, f"{base}/normalize_by_ROI", 1)
		_put(handle, f"{base}/normalization_ROI_left", 1)
		_put(handle, f"{base}/normalization_ROI_right", 2)
		_put(handle, f"{base}/normalization_ROI_top", 3)
		_put(handle, f"{base}/normalization_ROI_bottom", 4)
		_put(handle, f"{base}/remove_outliers", 1)
		_put(handle, f"{base}/kernel_size", 3)
		_put(handle, "/measurement/sample/file_name", np.bytes_("scan-1"))
		_put(handle, "/measurement/sample/experiment/proposal", np.bytes_("P-1"))

	record, = reconstruction_importers.read_als832_reconstructions((path,))
	values = record.values

	assert values["scan_id"] == "scan-1"
	assert values["propagation_distance_mm"] == 25.0
	assert values["normalization_method"] == "roi"
	assert values["normalization_roi"] == (1, 2, 3, 4)
	assert values["outlier_removal_enabled"] is True
	assert values["configuration_state"] == "configured"
	assert values["reconstruction_status"] == "configured"
	assert values["effective_command"] is None


def test_xaid_wraps_existing_parser_and_maps_composites(tmp_path, monkeypatch):
	path = tmp_path / "xaid.txt"
	path.write_text(
		"""a[General_Info]
software_version = 2026.5
path_to_data = Z:\\scans\\scan-7.h5

[ROI]
roi_posx = 1
roi_posy = 2
roi_posz = 3
roi_sizex = 4
roi_sizey = 5
roi_sizez = 6

[VolumeData]
nat_vx_size = 0.002
voxel_size = 0.004
volume_rotation_x = 1.0
volume_rotation_y = 2.0
volume_rotation_z = 3.0

[Proj_Filters]
outlier_size = 3
outlier_delta = 0.1

[Reconstruction_Settings]
reco_type = FDK
apply_roundmask = 1
img_binning = 2

[GC_Values]
drift_x = 0.1
drift_y = 0.2
drift_z = 0.3

[Final_Image_Settings]
save_path = Z:\\recon
export_type = tif16
""",
		encoding="utf-8",
	)
	real_parser = reconstruction_importers.xaid_reconstruction_log.parse_xaid_config
	calls = []

	def recording_parser(config_path):
		calls.append(config_path)
		return real_parser(config_path)

	monkeypatch.setattr(
		reconstruction_importers.xaid_reconstruction_log,
		"parse_xaid_config",
		recording_parser,
	)

	record, = reconstruction_importers.read_xaid_reconstructions((path,))
	values = record.values

	assert calls == [path]
	assert values["scan_id"] == "scan-7"
	assert values["volume_roi_start_xyz"] == (1, 2, 3)
	assert values["volume_roi_extent_xyz"] == (4, 5, 6)
	assert values["output_rotation_xyz_deg"] == (1.0, 2.0, 3.0)
	assert values["drift_correction_xyz"] == (0.1, 0.2, 0.3)
	assert values["outlier_removal_enabled"] is True
	assert values["cylindrical_mask_enabled"] is True
	assert record.warnings == ["repaired malformed first section header"]


def _tomocupy_fixture(tmp_path, *, matching_config=True):
	run = tmp_path / "run-1"
	output = run / "output slices"
	output.mkdir(parents=True)
	rec_line = run / "rec_line.txt"
	rec_line.write_text(
		" ".join(
			(
				"tomocupy recon",
				"--file-name '/data/scan one.h5'",
				"--out-path-name 'output slices'",
				"--bright-ratio 1.2",
				"--binning 1",
				"--pixel-size 2",
				"--rotation-axis 123.5",
				"--retrieve-phase-method paganin",
				"--dezinger 3",
				"--start-proj 1 --end-proj 10",
				"--dtype float32 --save-format tif --clear-folder true",
			)
		),
		encoding="utf-8",
	)
	config = tmp_path / "recon_params.json"
	config.write_text(
		json.dumps(
			{
				"records": [
					{
						"file_name": (
							"/data/scan one.h5" if matching_config else "/data/other.h5"
						),
						"bright_ratio": 0.9,
						"propagation_distance": 42,
						"flat_linear": True,
						"cor_method_full": "manual",
						"start_row": 2,
						"end_row": 5,
					}
				]
			}
		),
		encoding="utf-8",
	)
	center = tmp_path / "rot_cen.json"
	center.write_text(json.dumps({"rotation-axis": 124.0}), encoding="utf-8")
	for index in (0, 1):
		tifffile.imwrite(
			output / f"slice_{index:04d}.tif",
			np.zeros((4, 5), dtype=np.float32),
		)
	return rec_line, config, center


def test_tomocupy_uses_exact_join_command_precedence_and_output_metadata(tmp_path):
	rec_line, config, center = _tomocupy_fixture(tmp_path)

	record, = reconstruction_importers.read_tomocupy_7bm_reconstructions(
		(rec_line, config, center)
	)
	values = record.values

	assert values["scan_id"] == "scan one"
	assert values["bright_ratio"] == 1.2
	assert values["propagation_distance_mm"] == 42.0
	assert values["projection_binning"] == 2
	assert values["native_voxel_size_mm"] == pytest.approx(0.002)
	assert values["output_voxel_size_mm"] == pytest.approx(0.004)
	assert values["rotation_axis_coordinate_px"] == 123.5
	assert values["rotation_axis_auto_method"] == "manual"
	assert values["projection_range"] == (1, 10)
	assert values["sinogram_range"] == (2, 5)
	assert values["output_file_count"] == 2
	assert values["output_image_shape"] == (4, 5)
	assert values["output_dtype_observed"] == "float32"
	assert values["reconstruction_status"] == "completed"
	assert any("rot_cen.json disagrees" in warning for warning in record.warnings)


def test_tomocupy_does_not_join_a_different_scan_record(tmp_path):
	rec_line, config, center = _tomocupy_fixture(tmp_path, matching_config=False)

	record, = reconstruction_importers.read_tomocupy_7bm_reconstructions(
		(rec_line, config, center)
	)

	assert record.values["propagation_distance_mm"] is None
	assert record.values["sinogram_range"] is None
	assert any("no exact record" in warning for warning in record.warnings)
