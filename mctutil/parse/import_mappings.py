"""Direct source locators used by the lean import readers.

These dictionaries are lookup data only. Joins, selection, parsing, precedence,
and derived values belong in the source reader that consumes them.
"""


ALS832_SCAN_MAPPING = {
	"facility_name": {"locator": "/defaults/file_attrs/facility"},
	"project_id": {"locator": "/measurement/sample/experiment/proposal"},
	"scan_id": {"locator": "/measurement/sample/file_name"},
	"scan_method": {"locator": "/process/acquisition/name"},
	"projection_count_planned": {
		"locator": "/process/acquisition/rotation/num_angles",
		"source_unit": "count",
		"canonical_unit": "count",
	},
	"exposure_us": {
		"locator": "/measurement/instrument/detector/exposure_time",
		"source_unit": "s",
		"canonical_unit": "us",
		"multiplier": 1_000_000.0,
	},
	"flat_exposure_us": {
		"locator": "/process/acquisition/flat_fields/flat_field_exposure",
		"source_unit": "s",
		"canonical_unit": "us",
		"multiplier": 1_000_000.0,
	},
	"scan_start": {"locator": "/process/acquisition/start_date"},
}


SIGRAY_SCAN_MAPPING = {
	"sample_id": {"locator": "/exchange/sample_number"},
	"acquisition_system_id": {"locator": "/exchange/system_serial_number"},
	"scan_method": {"locator": "/exchange/scan_type"},
	"projection_count_planned": {
		"locator": "/measurement/process/acquisition/setup/number_of_projections",
		"source_unit": "count",
		"canonical_unit": "count",
	},
	"angular_step_deg": {
		"locator": "/measurement/process/acquisition/setup/angular_step",
		"source_unit": "deg",
		"canonical_unit": "deg",
	},
	"exposure_us": {
		"locator": "/measurement/process/acquisition/image_exposure_time",
		"source_unit": "s",
		"canonical_unit": "us",
		"multiplier": 1_000_000.0,
	},
	"exposures_per_projection": {
		"locator": "/measurement/instrument/detector/frame_averaging",
		"source_unit": "count",
		"canonical_unit": "count",
	},
	"binning_x": {"locator": "/measurement/instrument/detector/binning_x"},
	"binning_y": {"locator": "/measurement/instrument/detector/binning_y"},
	"image_width_px": {"locator": "/measurement/instrument/detector/dimension_x"},
	"image_height_px": {"locator": "/measurement/instrument/detector/dimension_y"},
	"detector_pixel_pitch_mm": {
		"locator": "/measurement/instrument/detector/physical_pixel_size",
		"source_unit": "um",
		"canonical_unit": "mm",
		"multiplier": 0.001,
	},
	"effective_pixel_size_mm": {
		"locator": "/measurement/instrument/detector/actual_pixel_size_x",
		"source_unit": "um",
		"canonical_unit": "mm",
		"multiplier": 0.001,
	},
	"geometric_magnification": {
		"locator": "/measurement/instrument/detector/geometric_magnification"
	},
	"sample_stage_x_mm": {
		"locator": "/measurement/instrument/sample/setup/sample_x",
		"source_unit": "um",
		"canonical_unit": "mm",
		"multiplier": 0.001,
	},
	"sample_stage_y_mm": {
		"locator": "/measurement/instrument/sample/setup/sample_y",
		"source_unit": "um",
		"canonical_unit": "mm",
		"multiplier": 0.001,
	},
	"sample_stage_z_mm": {
		"locator": "/measurement/instrument/sample/setup/sample_z",
		"source_unit": "um",
		"canonical_unit": "mm",
		"multiplier": 0.001,
	},
	"detector_stage_z_mm": {
		"locator": "/measurement/instrument/detector/setup/detector_z",
		"source_unit": "um",
		"canonical_unit": "mm",
		"multiplier": 0.001,
	},
	"source_stage_z_mm": {
		"locator": "/measurement/instrument/source/setup/source_z",
		"source_unit": "um",
		"canonical_unit": "mm",
		"multiplier": 0.001,
	},
	"source_target": {"locator": "/measurement/instrument/source/srps_target"},
}


SEVEN_BM_SCAN_MAPPING = {
	"scan_method": {"locator": "/process/acquisition/scan_type"},
	"projection_count_planned": {
		"locator": "/process/acquisition/rotation/num_angles",
		"source_unit": "count",
		"canonical_unit": "count",
	},
}


CHENGLAB_CAMERA_SCAN_MAPPING = {
	"project_id": {"locator": "Project"},
	"sample_id": {"locator": "Sample ID"},
	"scan_id": {"locator": "Scan ID"},
	"sample_stain": {"locator": "Stain"},
	"operator": {"locator": "Scan Operator"},
	"notes": {"locator": "Notes"},
	"projection_count_planned": {
		"locator": "Projection Total",
		"source_unit": "count",
		"canonical_unit": "count",
	},
	"projection_count_acquired": {
		"locator": "Projection Total Result",
		"source_unit": "count",
		"canonical_unit": "count",
	},
	"rotation_start_deg": {
		"locator": "Rotational Start",
		"source_unit": "deg",
		"canonical_unit": "deg",
	},
	"rotation_stop_planned_deg": {
		"locator": "Rotational Stop",
		"source_unit": "deg",
		"canonical_unit": "deg",
	},
	"rotation_stop_actual_deg": {
		"locator": "Rotational Stop Result",
		"source_unit": "deg",
		"canonical_unit": "deg",
	},
	"fov_count": {
		"locator": "FOV Count",
		"source_unit": "count",
		"canonical_unit": "count",
	},
	"helical": {"locator": "Helical?"},
	"exposure_us": {
		"locator": "Exposure (us)",
		"source_unit": "us",
		"canonical_unit": "us",
	},
	"flat_exposure_us": {
		"locator": "Flatfield Exposure (us)",
		"source_unit": "us",
		"canonical_unit": "us",
	},
	"exposures_per_projection": {
		"locator": "Exposures Per Projection",
		"source_unit": "count",
		"canonical_unit": "count",
	},
	"trigger_period_us": {
		"locator": "Trigger (us)",
		"source_unit": "us",
		"canonical_unit": "us",
	},
	"hardware_trigger_enabled": {"locator": "HW Trigger"},
	"scan_start": {"locator": "Scan Start"},
	"scan_stop": {"locator": "Scan Stop"},
	"image_width_px": {
		"locator": "Width",
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"image_height_px": {
		"locator": "Height",
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"stored_bit_depth": {"locator": "Bit Depth"},
	"crop_enabled": {"locator": "HW Crop"},
	"detector_gain": {"locator": "Gain"},
	"source_energy_kev": {"locator": "Energy"},
	"flat_frame_count_pre": {
		"locator": "Pre Gains",
		"source_unit": "count",
		"canonical_unit": "count",
	},
	"flat_frame_count_post": {
		"locator": "Post Gains",
		"source_unit": "count",
		"canonical_unit": "count",
	},
}


ALS832_RECONSTRUCTION_MAPPING = {
	"propagation_distance_mm": {
		"locator": "/process/tomo_rec/setup/algorithm/distance",
		"source_unit": "mm",
		"canonical_unit": "mm",
	},
	"normalization_method": {
		"locator": "/process/tomo_rec/setup/algorithm/normalize_by_ROI"
	},
	"outlier_removal_enabled": {
		"locator": "/process/tomo_rec/setup/algorithm/remove_outliers"
	},
	"outlier_kernel_px": {
		"locator": "/process/tomo_rec/setup/algorithm/kernel_size",
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"outlier_threshold": {"locator": "/process/tomo_rec/setup/algorithm/threshold"},
	"ring_stripe_method": {
		"locator": "/process/tomo_rec/setup/algorithm/ring_removal_method"
	},
	"phase_retrieval_enabled": {
		"locator": "/process/tomo_rec/setup/algorithm/phase_filt"
	},
	"phase_beta": {"locator": "/process/tomo_rec/setup/algorithm/beta"},
	"phase_delta": {"locator": "/process/tomo_rec/setup/algorithm/delta"},
	"reconstruction_type": {
		"locator": "/process/tomo_rec/setup/algorithm/reconstruction_type"
	},
	"output_format": {"locator": "/process/tomo_rec/setup/algorithm/output_type"},
	"output_min": {
		"locator": "/process/tomo_rec/setup/algorithm/output_scaling_min_value"
	},
	"output_max": {
		"locator": "/process/tomo_rec/setup/algorithm/output_scaling_max_value"
	},
}


XAID_RECONSTRUCTION_MAPPING = {
	"software_version": {"locator": ("General_Info", "software_version")},
	"input_projection_file": {"locator": ("General_Info", "path_to_data")},
	"native_voxel_size_mm": {
		"locator": ("VolumeData", "nat_vx_size"),
		"source_unit": "mm",
		"canonical_unit": "mm",
	},
	"output_voxel_size_mm": {
		"locator": ("VolumeData", "voxel_size"),
		"source_unit": "mm",
		"canonical_unit": "mm",
	},
	"cylindrical_mask_enabled": {
		"locator": ("Reconstruction_Settings", "apply_roundmask")
	},
	"normalization_method": {"locator": ("Reconstruction_Settings", "freeray")},
	"median_filter_size_px": {
		"locator": ("Proj_Filters", "median_filter_size"),
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"gaussian_sigma_px": {
		"locator": ("Proj_Filters", "gauss_filter_size"),
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"outlier_kernel_px": {
		"locator": ("Proj_Filters", "outlier_size"),
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"outlier_threshold": {"locator": ("Proj_Filters", "outlier_delta")},
	"source_object_distance_mm": {
		"locator": ("Reconstruction_Settings", "sod"),
		"source_unit": "mm",
		"canonical_unit": "mm",
	},
	"source_detector_distance_mm": {
		"locator": ("Reconstruction_Settings", "sdd"),
		"source_unit": "mm",
		"canonical_unit": "mm",
	},
	"projection_binning": {
		"locator": ("Reconstruction_Settings", "img_binning")
	},
	"reconstruction_algorithm": {
		"locator": ("Reconstruction_Settings", "reco_type")
	},
	"reconstruction_filter": {
		"locator": ("Reconstruction_Settings", "fdk_filter")
	},
	"iterative_regularization_method": {
		"locator": ("Reconstruction_Settings", "reg_type")
	},
	"iteration_count": {"locator": ("Reconstruction_Settings", "iterations")},
	"regularization_threshold": {
		"locator": ("Reconstruction_Settings", "reg_threshold")
	},
	"regularization_weight": {
		"locator": ("Reconstruction_Settings", "reg_lambda")
	},
	"rotation_axis_offset_px": {
		"locator": ("GC_Values", "rotation_axis_offset"),
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"rotation_axis_tilt_deg": {
		"locator": ("GC_Values", "rotation_axis_tilt"),
		"source_unit": "deg",
		"canonical_unit": "deg",
	},
	"rotation_axis_slant_deg": {
		"locator": ("Global_Reco_Settings", "rotation_axis_slant"),
		"source_unit": "deg",
		"canonical_unit": "deg",
	},
	"detector_rotation_eta_deg": {
		"locator": ("Global_Reco_Settings", "det_rot_eta"),
		"source_unit": "deg",
		"canonical_unit": "deg",
	},
	"detector_rotation_theta_deg": {
		"locator": ("Global_Reco_Settings", "det_rot_theta"),
		"source_unit": "deg",
		"canonical_unit": "deg",
	},
	"detector_rotation_phi_deg": {
		"locator": ("Global_Reco_Settings", "det_rot_phi"),
		"source_unit": "deg",
		"canonical_unit": "deg",
	},
	"detector_offset_u_px": {
		"locator": ("Global_Reco_Settings", "det_offset_u"),
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"detector_offset_v_px": {
		"locator": ("Global_Reco_Settings", "det_offset_v"),
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"output_path": {"locator": ("Final_Image_Settings", "save_path")},
	"output_format": {"locator": ("Final_Image_Settings", "export_type")},
	"output_axis_order": {"locator": ("Final_Image_Settings", "export_order")},
	"output_min": {"locator": ("Final_Image_Settings", "min")},
	"output_max": {"locator": ("Final_Image_Settings", "max")},
}


SEVEN_BM_RECONSTRUCTION_MAPPING = {
	"input_projection_file": {"locator": "--file-name"},
	"propagation_distance_mm": {
		"locator": "--propagation-distance",
		"source_unit": "mm",
		"canonical_unit": "mm",
	},
	"projection_binning": {"locator": "--binning"},
	"native_voxel_size_mm": {
		"locator": "--pixel-size",
		"source_unit": "um",
		"canonical_unit": "mm",
		"multiplier": 0.001,
	},
	"rotation_axis_coordinate_px": {
		"locator": "--rotation-axis",
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"rotation_axis_auto_method": {"locator": "--rotation-axis-auto"},
	"bright_ratio": {"locator": "--bright-ratio"},
	"flat_linear": {"locator": "--flat-linear"},
	"minus_log_enabled": {"locator": "--minus-log"},
	"outlier_kernel_px": {
		"locator": "--dezinger",
		"source_unit": "px",
		"canonical_unit": "px",
	},
	"outlier_threshold": {"locator": "--dezinger-threshold"},
	"ring_stripe_method": {"locator": "--remove-stripe-method"},
	"phase_retrieval_method": {"locator": "--retrieve-phase-method"},
	"reconstruction_type": {"locator": "--reconstruction-type"},
	"reconstruction_algorithm": {"locator": "--reconstruction-algorithm"},
	"reconstruction_filter": {"locator": "--fbp-filter"},
	"output_path": {"locator": "--out-path-name"},
	"output_format": {"locator": "--save-format"},
	"output_dtype": {"locator": "--dtype"},
	"overwrite_policy": {"locator": "--clear-folder"},
}
