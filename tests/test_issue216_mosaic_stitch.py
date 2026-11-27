from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from mctutil.mosaic.layout import GroupRelation, canvas_bounds, fixed_map
from mctutil.mosaic.norm import norm_fit
from mctutil.mosaic.profile import edge_excess, profile
from mctutil.mosaic.spectra import block_spectra, spectra
from mctutil.mosaic.stitch import fuse, samplers_for, stitch
from mctutil.mosaic.wiener import wiener
from mctutil.mosaic.window import to_u16, to_u16_command, window

pytest.importorskip("cv2")
ndimage = pytest.importorskip("scipy.ndimage")
tifffile = pytest.importorskip("tifffile")

OPTIONS = {"fov_feather": (20.0, 23.0), "row_feather": 4.0, "hp_inner": None, "lp_inner": None,
			"flat_cupping": False, "sigma_lp": 3.0}


def write_stack(path, volume):
	path.mkdir(parents=True)
	for row, image in enumerate(volume):
		tifffile.imwrite(path / f"{row:05d}.tif", np.asarray(image, np.float32))
	return path


def reference_scan(name, path, rows, depth_offset, **extra):
	return {"name": name, "dir": str(path), "nrow": rows, "lowpass_weight": 1.0,
			"A": [[1, 0, 0, -depth_offset], [0, 1, 0, 0], [0, 0, 1, 0]], **extra}


def test_edge_excess_keeps_the_trend_and_isolates_the_rim():
	radius = np.arange(200, dtype=np.float64)
	trend = 1.0 + 1e-4 * radius ** 2
	rim = np.where(radius >= 160, 0.5, 0.0)

	excess = edge_excess(trend + rim, 100, 150, 5.0, 0.5)

	assert np.all(excess[:146] == 0)
	assert excess[180] == pytest.approx(0.5, abs=1e-6)


def test_profile_writes_radius_excess_and_mean_rows(tmp_path):
	n = 64
	radius = np.hypot(*(np.mgrid[:n, :n] - (n - 1) / 2))
	stack = write_stack(tmp_path / "recon", np.repeat((1 + 0.001 * radius ** 2)[None], 4, axis=0))
	output = tmp_path / "radial.npy"

	result = CliRunner().invoke(profile, [str(stack), str(output), "--r-lo", "5", "--r-ref", "20", "--step", "2"])

	assert result.exit_code == 0, result.output
	saved = np.load(output)
	assert saved.shape[0] == 3
	np.testing.assert_allclose(saved[2][:20], 1 + 0.001 * np.arange(20) ** 2, atol=0.02)
	assert np.abs(saved[1][:31]).max() < 0.01


def test_single_scan_stitch_reproduces_the_scan_inside_its_disk(tmp_path):
	volume = ndimage.gaussian_filter(np.random.default_rng(0).normal(size=(20, 48, 48)), 1.5).astype(np.float32)
	document = {"canvas": [48, 48], "nz": 20, "slice_size": 48,
				"scans": [reference_scan("s", write_stack(tmp_path / "s", volume), 20, 0)]}

	values, covered = fuse(samplers_for(document, OPTIONS), 10, (48, 48), OPTIONS).result()

	assert covered[24, 24] and not covered[0, 0]
	np.testing.assert_allclose(values[covered], volume[10][covered], atol=1e-5)


def test_tilted_scan_interpolates_between_its_bracketing_rows(tmp_path):
	rows = np.repeat(np.arange(30, dtype=np.float32)[:, None, None], 48, axis=1).repeat(48, axis=2)
	tilt = 0.05
	scan = {"name": "t", "dir": str(write_stack(tmp_path / "t", rows)), "nrow": 30, "lowpass_weight": 1.0, "off": 0.0,
			"A": [[1, tilt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]}
	document = {"canvas": [48, 48], "nz": 30, "slice_size": 48, "scans": [scan]}

	values, covered = fuse(samplers_for(document, OPTIONS), 10, (48, 48), OPTIONS).result()

	y = np.arange(48)[:, None].repeat(48, axis=1)
	np.testing.assert_allclose(values[covered], (10 + tilt * y)[covered], atol=1e-4)


def nps_file(path, total):
	path.write_text(json.dumps({"nps": [total / 4] * 4}))
	return str(path)


def drifting_scan(path, values, a):
	return {"name": "d", "dir": str(write_stack(path, values)), "nrow": len(values), "lowpass_weight": 1.0,
			"off": 0.0, "A": a}


def test_planar_sampling_follows_lateral_drift_with_depth(tmp_path):
	columns = np.repeat(np.arange(64, dtype=np.float32)[None, None, :], 64, axis=1).repeat(30, axis=0)
	scan = drifting_scan(tmp_path / "d", columns, [[1, 0, 0, 0], [0, 1, 0, 0], [0.2, 0, 1, 0]])
	document = {"canvas": [64, 64], "nz": 30, "slice_size": 64, "scans": [scan]}
	options = {**OPTIONS, "fov_feather": (24.0, 28.0)}

	values, covered = fuse(samplers_for(document, options), 20, (64, 64), options).result()

	assert covered[32, 32]
	assert values[32, 32] == pytest.approx(36.0, abs=1e-3)


def test_fov_weight_moves_with_the_scan_at_each_depth(tmp_path):
	scan = drifting_scan(tmp_path / "d", np.ones((40, 64, 64), np.float32),
							[[1, -0.01, 0, 0], [0, 1, 0, 0], [0.2, -0.002, 1, 0]])
	document = {"canvas": [64, 64], "nz": 40, "slice_size": 64, "scans": [scan]}
	options = {**OPTIONS, "fov_feather": (24.0, 28.0)}
	sampler = samplers_for(document, options)[0]

	_, weight = sampler.sample(20, (64, 64))

	def expected(y, x):
		sy, sx = y, 0.2 * 20 - 0.002 * y + x
		return np.clip((28 - np.hypot(sy - 31.5, sx - 31.5)) / 4, 0, 1)

	assert weight[32, 2] == pytest.approx(expected(32, 2), abs=0.02) and expected(32, 2) > 0.5
	assert weight[32, 59] == pytest.approx(expected(32, 59), abs=0.02) and expected(32, 59) == 0


def test_short_fixed_scan_in_the_middle_of_a_deep_canvas_is_sampled(tmp_path):
	fit = {"phi": 0.0, "K": 500.0, "gy": 0.01, "gx": 0.0, "gz": 0.0, "dy": [0.0, 0.0], "dx": [0.2, 0.0]}
	relation = GroupRelation(fit, 31.5)
	oy, ox, height, width = canvas_bounds(relation, 28, (0.0, 100.0, 199.0))
	scan = drifting_scan(tmp_path / "d", np.ones((200, 64, 64), np.float32), fixed_map(relation, 0.0, oy, ox))
	document = {"canvas": [height, width], "nz": 1200, "slice_size": 64, "scans": [scan]}
	options = {**OPTIONS, "fov_feather": (24.0, 28.0)}
	sampler = samplers_for(document, options)[0]

	values, weight = sampler.sample(550, (height, width))

	inside = weight > 0.9
	assert inside.sum() > 1000
	np.testing.assert_allclose(values[inside], 1.0, atol=1e-6)


def test_noise_maps_require_spectra(tmp_path):
	scan = reference_scan("s", write_stack(tmp_path / "s", np.ones((4, 48, 48), np.float32)), 4, 0)
	layout_path = tmp_path / "layout.json"
	layout_path.write_text(json.dumps({"canvas": [48, 48], "nz": 4, "slice_size": 48, "scans": [scan]}))

	result = CliRunner().invoke(stitch, [str(layout_path), str(tmp_path / "out"), "--fov-feather", "20", "23",
											"--noise-out", str(tmp_path / "noise.npy")])

	assert result.exit_code != 0
	assert "no snr spectra" in result.output


def test_rerun_with_noise_out_fills_maps_for_existing_slices(tmp_path):
	volume = np.ones((6, 48, 48), np.float32)
	scan = reference_scan("s", write_stack(tmp_path / "s", volume), 6, 0, snr=nps_file(tmp_path / "s.json", 4.0))
	layout_path = tmp_path / "layout.json"
	layout_path.write_text(json.dumps({"canvas": [48, 48], "nz": 6, "slice_size": 48, "scans": [scan]}))
	arguments = [str(layout_path), str(tmp_path / "out"), "--fov-feather", "20", "23", "--row-feather", "1"]
	runner = CliRunner()

	first = runner.invoke(stitch, arguments)
	second = runner.invoke(stitch, arguments + ["--noise-out", str(tmp_path / "noise.npy")])

	assert first.exit_code == 0 and second.exit_code == 0, second.output
	assert np.load(tmp_path / "noise.npy")[3, 0, 3, 3] == pytest.approx(4.0, rel=1e-3)


def test_overlapping_scans_fuse_to_the_shared_content_and_write_noise_maps(tmp_path):
	volume = ndimage.gaussian_filter(np.random.default_rng(1).normal(size=(40, 48, 48)), 1.5).astype(np.float32)
	first = reference_scan("a", write_stack(tmp_path / "a", volume[:25]), 25, 0, snr=nps_file(tmp_path / "a.json", 4.0))
	second = reference_scan("b", write_stack(tmp_path / "b", volume[15:]), 25, 15, snr=nps_file(tmp_path / "b.json", 4.0))
	layout_path = tmp_path / "layout.json"
	layout_path.write_text(json.dumps({"canvas": [48, 48], "nz": 40, "slice_size": 48, "scans": [first, second]}))
	output, noise = tmp_path / "stitched", tmp_path / "noise.npy"

	result = CliRunner().invoke(stitch, [str(layout_path), str(output), "--fov-feather", "20", "23", "--row-feather",
											"4", "--sigma-lp", "3", "--noise-out", str(noise)])

	assert result.exit_code == 0, result.output
	stitched = tifffile.imread(output / "00020.tif")
	covered = stitched != 0
	np.testing.assert_allclose(stitched[covered], volume[20][covered], atol=1e-5)
	maps = np.load(noise)
	assert maps.shape == (40, 2, 6, 6)
	assert maps[2, 0, 3, 3] == pytest.approx(4.0, rel=1e-3)
	assert maps[20, 0, 3, 3] == pytest.approx(2.0, rel=1e-3)
	assert np.all(maps[:, 1] == 0)


def test_block_spectra_separate_signal_from_added_white_noise():
	rng = np.random.default_rng(2)
	signal = ndimage.gaussian_filter(rng.normal(size=(16, 64, 64)), 4.0) * 10
	noisy = signal + rng.normal(scale=0.1, size=signal.shape)

	result = block_spectra(signal, noisy, 32, (32, 32))

	snr = np.array(result["snr"])
	assert len(result["freq"]) == 16 and len(result["fz"]) == 9
	assert snr[1] > 100 > snr[-1]
	assert np.ptp(np.log(result["nps"][2:])) < 1.5


def test_to_u16_maps_the_window_and_reserves_zero():
	values = np.array([[-1.0, 0.0, 0.5, 2.0]], np.float32)

	scaled = to_u16(values, 0.0, 1.0, values != 0)

	assert scaled.tolist() == [[1, 0, 32768, 65535]]


def test_window_and_to_u16_commands_skip_uncovered_pixels(tmp_path):
	image = np.zeros((10, 10), np.float32)
	image[2:8, 2:8] = np.linspace(1, 2, 36).reshape(6, 6)
	source = write_stack(tmp_path / "f32", image[None])
	runner = CliRunner()

	percentiles = runner.invoke(window, [str(source), "--step", "1", "--pixel-step", "1"])
	written = runner.invoke(to_u16_command, [str(source), str(tmp_path / "u16"), "--window", "1", "2"])

	assert percentiles.exit_code == 0, percentiles.output
	assert json.loads(percentiles.output)["p0.1"] == pytest.approx(1.0, abs=0.01)
	assert written.exit_code == 0, written.output
	u16 = tifffile.imread(tmp_path / "u16" / "00000.tif")
	assert u16[0, 0] == 0 and u16[2, 2] == 1 and u16[7, 7] == 65535


def spectra_file(path, scale):
	freq = np.arange(16) / 32
	nps = np.full(16, 1.0)
	ps = nps + scale / (1 + 100 * freq ** 2)
	path.write_text(json.dumps({"freq": freq.tolist(), "nps": nps.tolist(), "ps": ps.tolist(),
								"fz": np.fft.rfftfreq(16).tolist(), "pz_noise": np.ones(9).tolist(),
								"pz_full": (1 + scale / (1 + 100 * np.fft.rfftfreq(16) ** 2)).tolist()}))
	return str(path)


def test_wiener_preserves_a_uniform_canvas(tmp_path):
	stitched = write_stack(tmp_path / "f32", np.full((30, 32, 32), 0.75, np.float32))
	scans = [
		{"name": "r", "a": 1.0, "snr": spectra_file(tmp_path / "r.json", 50)},
		{"name": "f", "a": 1.0, "off": 0.0, "snr": spectra_file(tmp_path / "f.json", 20)},
	]
	layout_path = tmp_path / "layout.json"
	layout_path.write_text(json.dumps({"canvas": [32, 32], "nz": 30, "scans": scans}))
	noise = np.zeros((30, 2, 4, 4), np.float16)
	noise[:, 0] = np.linspace(1.0, 4.0, 16).reshape(4, 4)
	noise[:, 1] = 1.0
	np.save(tmp_path / "noise.npy", noise)

	result = CliRunner().invoke(wiener, [str(layout_path), str(stitched), str(tmp_path / "noise.npy"),
											str(tmp_path / "out"), "--window", "0.5", "1.0", "--taps", "3"])

	assert result.exit_code == 0, result.output
	expected = to_u16(np.full((1, 1), 0.75, np.float32), 0.5, 1.0, np.ones((1, 1), bool))[0, 0]
	filtered = tifffile.imread(tmp_path / "out" / "00015.tif").astype(np.int64)
	assert np.abs(filtered - int(expected)).max() <= 1
	assert (tmp_path / "out_wiener_params.json").exists()


def wiener_inputs(tmp_path, noise_values):
	rng = np.random.default_rng(4)
	image = np.zeros((30, 32, 32), np.float32)
	image[:, 4:28, 4:28] = 0.75 + 0.05 * rng.normal(size=(30, 24, 24))
	stitched = write_stack(tmp_path / "f32", image)
	scans = [
		{"name": "r", "a": 1.0, "snr": spectra_file(tmp_path / "r.json", 50)},
		{"name": "f", "a": 1.0, "off": 0.0, "snr": spectra_file(tmp_path / "f.json", 20)},
	]
	layout_path = tmp_path / "layout.json"
	layout_path.write_text(json.dumps({"canvas": [32, 32], "nz": 30, "scans": scans}))
	noise = np.zeros((30, 2, 4, 4), np.float16)
	noise[:, 0, :, :] = noise_values
	np.save(tmp_path / "noise.npy", noise)
	return layout_path, stitched, image


def run_wiener(tmp_path, layout_path, stitched, *extra):
	return CliRunner().invoke(wiener, [str(layout_path), str(stitched), str(tmp_path / "noise.npy"),
										str(tmp_path / "out"), "--window", "0.5", "1.0", "--taps", "3", *extra])


def test_wiener_keeps_coverage_when_the_noise_is_uniform(tmp_path):
	layout_path, stitched, image = wiener_inputs(tmp_path, 2.0)

	result = run_wiener(tmp_path, layout_path, stitched)

	assert result.exit_code == 0, result.output
	filtered = tifffile.imread(tmp_path / "out" / "00015.tif")
	assert np.all(filtered[image[15] != 0] > 0)


@pytest.mark.parametrize("missing", [range(7, 10), range(20, 23)])
def test_wiener_refuses_covered_neighbours_without_noise_maps(tmp_path, missing):
	layout_path, stitched, _image = wiener_inputs(tmp_path, 2.0)
	noise = np.load(tmp_path / "noise.npy")
	noise[list(missing)] = 0
	np.save(tmp_path / "noise.npy", noise)

	result = run_wiener(tmp_path, layout_path, stitched, "--g0", "10", "--g1", "20")

	assert result.exit_code != 0
	assert "noise map is empty" in result.output


def test_wiener_slab_with_complete_neighbour_maps_matches_a_full_map(tmp_path):
	layout_path, stitched, _image = wiener_inputs(tmp_path, 0.0)
	noise = np.load(tmp_path / "noise.npy")
	noise[7:23, 0] = 2.0
	np.save(tmp_path / "noise.npy", noise)
	slab = run_wiener(tmp_path, layout_path, stitched, "--g0", "10", "--g1", "20")
	slab_output = [tifffile.imread(tmp_path / "out" / f"{g:05d}.tif") for g in range(10, 20)]
	noise[:, 0] = 2.0
	np.save(tmp_path / "noise.npy", noise)
	for g in range(10, 20):
		(tmp_path / "out" / f"{g:05d}.tif").unlink()
	full = run_wiener(tmp_path, layout_path, stitched, "--g0", "10", "--g1", "20")

	assert slab.exit_code == 0 and full.exit_code == 0, slab.output
	for g, image in zip(range(10, 20), slab_output):
		assert np.array_equal(image, tifffile.imread(tmp_path / "out" / f"{g:05d}.tif")), g


@pytest.mark.parametrize("noise_scale", [0.75, 0.0], ids=["filtered", "identity"])
@pytest.mark.parametrize("slab_span, span", [((7, 23), (10, 20)), ((3, 27), (6, 12))],
						ids=["index-outside-file-count", "index-inside-file-count"])
def test_wiener_stitched_slab_matches_full_canvas(tmp_path, noise_scale, slab_span, span):
	layout_path, source, _image = wiener_inputs(tmp_path, 2.0)
	document = json.loads(layout_path.read_text())
	document["slice_size"] = 32
	for scan in document["scans"]:
		scan.update(reference_scan(scan["name"], source, 30, 0))
	layout_path.write_text(json.dumps(document))
	runner = CliRunner()
	outputs = []
	for label, rows in (("full", (0, 30)), ("slab", slab_span)):
		canvas, noise = tmp_path / label, tmp_path / f"{label}_noise.npy"
		result = runner.invoke(stitch, [str(layout_path), str(canvas), "--noise-out", str(noise),
										"--g0", str(rows[0]), "--g1", str(rows[1]), "--fov-feather", "12", "14",
										"--row-feather", "1", "--sigma-lp", "3"])
		assert result.exit_code == 0, result.output
		assert {path.name for path in canvas.iterdir()} == {f"{g:05d}.tif" for g in range(*rows)}
		target = tmp_path / f"{label}_filtered"
		result = runner.invoke(wiener, [str(layout_path), str(canvas), str(noise), str(target),
										"--g0", str(span[0]), "--g1", str(span[1]), "--taps", "3",
										"--window", "0.5", "1.0", "--noise-scale", str(noise_scale), "--fft-workers", "1"])
		assert result.exit_code == 0, result.output
		assert {path.name for path in target.iterdir()} == {f"{g:05d}.tif" for g in range(*span)}
		outputs.append([tifffile.imread(target / f"{g:05d}.tif") for g in range(*span)])
		if noise_scale == 0:
			for g, actual in zip(range(*span), outputs[-1]):
				image = tifffile.imread(canvas / f"{g:05d}.tif")
				np.testing.assert_array_equal(actual, to_u16(image, 0.5, 1.0, image != 0))
	for full, slab in zip(*outputs):
		np.testing.assert_array_equal(slab, full)


def test_window_options_require_high_above_low(tmp_path):
	source = write_stack(tmp_path / "f32", np.ones((1, 4, 4), np.float32))

	result = CliRunner().invoke(to_u16_command, [str(source), str(tmp_path / "u16"), "--window", "1", "1"])

	assert result.exit_code != 0 and "HI must be greater than LO" in result.output


def test_spectra_extracts_odd_boxes_and_rejects_boxes_outside_the_slice(tmp_path):
	rng = np.random.default_rng(5)
	full = rng.normal(size=(8, 40, 40))
	np.save(tmp_path / "full.npy", full)
	np.save(tmp_path / "noisy.npy", full + rng.normal(scale=0.1, size=full.shape))
	runner = CliRunner()

	odd = runner.invoke(spectra, [str(tmp_path / "full.npy"), str(tmp_path / "noisy.npy"), str(tmp_path / "s.json"),
									"--box", "25"])
	outside = runner.invoke(spectra, [str(tmp_path / "full.npy"), str(tmp_path / "noisy.npy"),
										str(tmp_path / "t.json"), "--box", "48"])

	assert odd.exit_code == 0, odd.output
	assert len(json.loads((tmp_path / "s.json").read_text())["freq"]) == 12
	assert outside.exit_code != 0 and "does not fit" in outside.output


def test_wiener_is_the_identity_at_zero_noise(tmp_path):
	layout_path, stitched, image = wiener_inputs(tmp_path, 2.0)

	result = run_wiener(tmp_path, layout_path, stitched, "--noise-scale", "0")

	assert result.exit_code == 0, result.output
	expected = to_u16(image[15], 0.5, 1.0, image[15] != 0)
	assert np.array_equal(tifffile.imread(tmp_path / "out" / "00015.tif"), expected)


def test_norm_fit_rejects_scans_without_shared_slices(tmp_path):
	volume = np.ones((40, 48, 48), np.float32)
	reference = reference_scan("ref", write_stack(tmp_path / "ref", volume), 40, 0)
	moving = reference_scan("mov", write_stack(tmp_path / "mov", volume), 40, -100)
	layout_path = tmp_path / "layout.json"
	layout_path.write_text(json.dumps({"canvas": [48, 48], "nz": 140, "slice_size": 48, "scans": [reference, moving]}))

	result = CliRunner().invoke(norm_fit, [str(layout_path), "--reference", "ref", "--pair", "mov:ref",
											"--fov-feather", "20", "23"])

	assert result.exit_code != 0 and "share no canvas slices" in result.output


def test_norm_fit_matches_a_scaled_overlapping_scan(tmp_path):
	volume = ndimage.gaussian_filter(np.random.default_rng(3).normal(size=(300, 96, 96)), 2.0).astype(np.float32) * 5
	reference = reference_scan("ref", write_stack(tmp_path / "ref", volume[:260]), 260, 0)
	moving = reference_scan("mov", write_stack(tmp_path / "mov", 2 * volume[40:] + 0.5), 260, 40)
	layout_path = tmp_path / "layout.json"
	layout_path.write_text(json.dumps({"canvas": [96, 96], "nz": 300, "slice_size": 96, "scans": [reference, moving]}))

	result = CliRunner().invoke(norm_fit, [str(layout_path), "--reference", "ref", "--pair", "mov:ref",
											"--fov-feather", "40", "45", "--erode", "31"])

	assert result.exit_code == 0, result.output
	scans = {scan["name"]: scan for scan in json.loads(layout_path.read_text())["scans"]}
	assert (scans["ref"]["a"], scans["ref"]["b"]) == (1.0, 0.0)
	assert scans["mov"]["a"] == pytest.approx(0.5, rel=1e-3)
	assert scans["mov"]["b"] == pytest.approx(-0.25, abs=1e-3)
