from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from mctutil.mosaic import common
from mctutil.mosaic.coarse import binned_attenuation, chunk_starts
from mctutil.mosaic.depth import depth_link, link_depth, windowed_block
from mctutil.mosaic.fit import fit
from mctutil.mosaic.layout import GroupRelation, fixed_map, layout
from mctutil.mosaic.refine import StackCorrelator, refine_row
from mctutil.mosaic.register import prepare, register_pair, rotate_slices, scan_pairs
from mctutil.mosaic.tilt import tilt_points

pytest.importorskip("cv2")
ndimage = pytest.importorskip("scipy.ndimage")
tifffile = pytest.importorskip("tifffile")


def smooth_noise(shape, sigma, seed):
	return ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=shape), sigma).astype(np.float32)


def write_stack(path, volume):
	path.mkdir(parents=True)
	for row, image in enumerate(volume):
		tifffile.imwrite(path / f"{row:05d}.tif", np.asarray(image, np.float32))
	return path


def test_phase_correlation_reports_the_shift_that_moves_moving_onto_fixed():
	fixed = smooth_noise((16, 32, 32), 1.5, 0)
	moving = np.roll(fixed, (-3, 2, -5), axis=(0, 1, 2))
	shape = fixed.shape

	peak, shift = common.phase_correlation(np.fft.rfftn(fixed, shape, axes=(0, 1, 2)), moving, shape)

	assert shift == [3, -2, 5]
	assert peak > 0.9


def test_binned_attenuation_flat_corrects_then_mean_bins():
	counts = np.full((2, 4, 4), 50.0, np.float32)
	dark = np.full((4, 4), 10.0, np.float32)
	gain = np.full((4, 4), 80.0, np.float32)

	attenuation = binned_attenuation(counts, dark, gain, 2)

	assert attenuation.shape == (2, 2, 2)
	np.testing.assert_allclose(attenuation, -np.log(0.5), rtol=1e-6)


@pytest.mark.parametrize("rows", [253, 254, 256])
def test_coarse_chunks_cover_every_complete_bin_and_no_partial_tail(rows):
	starts = chunk_starts(rows, 128, 3)
	chunks = [(start, min(start + starts.step, starts.stop)) for start in starts]

	assert all(start % 3 == 0 and (stop - start) % 3 == 0 and stop > start for start, stop in chunks)
	assert sum((stop - start) // 3 for start, stop in chunks) == rows // 3


def test_wrapped_returns_the_shortest_displacement_for_odd_and_even_sizes():
	assert [common.wrapped(k, 63) for k in (0, 31, 32, 62)] == [0, 31, -31, -1]
	assert [common.wrapped(k, 64) for k in (0, 31, 32, 63)] == [0, 31, -32, -1]


def test_register_pair_reports_the_largest_positive_shift_on_an_odd_grid():
	base = smooth_noise((24, 63, 63), 2.0, 9)
	moving = np.roll(base, -31, axis=1)

	link = register_pair(prepare(base, 1), prepare(moving, 1), phi0=0.0, span=0.5, step=1.0, nulls=(90.0,), scale=1)

	assert link["dy"] == 31


def test_windowed_block_extracts_odd_widths_exactly():
	volume = np.zeros((2, 40, 40), np.float32)

	assert windowed_block(volume, 20, 20, 25).shape == (2, 25, 25)
	assert windowed_block(volume, 20, 20, 24).shape == (2, 24, 24)


def test_scan_pairs_cover_each_group_then_cross_pairs_with_the_earlier_group_fixed():
	scans = [{"name": "b1", "group": "B"}, {"name": "b2", "group": "B"}, {"name": "a1", "group": "A"}]

	pairs = [(fixed["name"], moving["name"]) for fixed, moving in scan_pairs(scans)]

	assert pairs == [("b1", "b2"), ("b1", "a1"), ("b2", "a1")]


def test_register_pair_recovers_rotation_and_shift_of_a_rotated_copy():
	base = smooth_noise((24, 96, 96), 2.0, 1)
	moving = np.roll(rotate_slices(base, -3.0), (2, 4, -3), axis=(0, 1, 2))
	fixed, moving = prepare(base, 1), prepare(moving, 1)

	link = register_pair(fixed, moving, phi0=0.0, span=5.0, step=1.0, nulls=(90.0,), scale=1)

	assert link["phi"] == pytest.approx(3.0)
	assert link["sig"] > 3


def test_depth_link_recovers_the_row_offset_between_overlapping_stacks(tmp_path):
	base = smooth_noise((140, 96, 96), 1.0, 2)
	fixed = write_stack(tmp_path / "fixed", base[:100])
	moving = write_stack(tmp_path / "moving", np.concatenate([smooth_noise((20, 96, 96), 1.0, 3), base[:100]]))

	record = depth_link(common.RowStack(fixed), common.RowStack(moving), 10, 17, depth=24, search=8,
						block=32, step=16, radius=30, min_peak=0.003)

	assert record["T_median"] == 20
	assert record["n"] > 0


def test_depth_link_accepts_odd_block_widths(tmp_path):
	base = smooth_noise((80, 64, 64), 1.0, 10)
	fixed = write_stack(tmp_path / "fixed", base[:60])
	moving = write_stack(tmp_path / "moving", np.concatenate([smooth_noise((5, 64, 64), 1.0, 11), base[:60]]))

	record = depth_link(common.RowStack(fixed), common.RowStack(moving), 5, 5, depth=16, search=4,
						block=25, step=8, radius=16, min_peak=0.003)

	assert record["T_median"] == 5


def test_link_depth_appends_json_lines(tmp_path):
	base = smooth_noise((80, 64, 64), 1.0, 4)
	fixed = write_stack(tmp_path / "f", base[:60])
	moving = write_stack(tmp_path / "m", np.concatenate([smooth_noise((5, 64, 64), 1.0, 5), base[:60]]))
	output = tmp_path / "links.jsonl"
	arguments = [str(fixed), str(moving), str(output), "--fixed-row", "5", "--guess", "5", "--radius", "16",
					"--depth", "16", "--search", "4", "--block", "24", "--step", "8"]

	for _ in range(2):
		result = CliRunner().invoke(link_depth, arguments)
		assert result.exit_code == 0, result.output

	records = [json.loads(line) for line in output.read_text().splitlines()]
	assert [(item["fixed"], item["moving"], item["T_median"]) for item in records] == [("f", "m", 5.0)] * 2


def test_refine_row_recovers_rotation_shift_and_depth(tmp_path):
	n = 128
	moving_volume = smooth_noise((40, n, n), 2.0, 6)
	truth = {"phi": 1.2, "dy": 3.4, "dx": -2.6}
	matrix = common.rotation(n, truth["phi"], 1.0, truth["dy"], truth["dx"])
	fixed_volume = np.stack([common.warp(image, matrix) for image in moving_volume[3:33]])
	correlator = StackCorrelator(common.RowStack(write_stack(tmp_path / "fixed", fixed_volume)),
									common.RowStack(write_stack(tmp_path / "moving", moving_volume)),
									slices=4, disk_fraction=0.86, disk_taper=10)

	record = refine_row(correlator, 4, {"dz": 2, "phi": 0.9, "s": 1.0, "dy": 2.5, "dx": -1.5})

	assert record["dz"] == 3
	assert record["phi"] == pytest.approx(truth["phi"], abs=0.1)
	assert record["dy"] == pytest.approx(truth["dy"], abs=0.3)
	assert record["dx"] == pytest.approx(truth["dx"], abs=0.3)


def test_fit_recovers_offsets_and_depth_relation_and_drops_outlier_points(tmp_path):
	reference = {"r0": 0.0, "r1": 900.0, "r2": 1795.0}
	fixed = {"f0": 0.0, "f1": 850.0}
	k, gy, gx, gz = 410.0, 0.01, -0.02, 0.001
	links = [
		{"fixed": "r1", "moving": "r0", "T_median": 900.0},
		{"fixed": "r2", "moving": "r1", "T_median": 895.0},
		{"fixed": "f1", "moving": "f0", "T_median": 850.0},
	]
	points = []
	rng = np.random.default_rng(7)
	for name, moving in (("f0", "r0"), ("f1", "r1"), ("f1", "r2")):
		for _ in range(30):
			y, x, row = rng.uniform(100, 900), rng.uniform(100, 900), rng.uniform(0, 1000)
			z = row + fixed[name]
			t = k + gy * y + gx * x + gz * z - reference[moving] + fixed[name]
			points.append({"link": name, "moving": moving, "row": row, "T": t, "Z": z, "y": y, "x": x, "peak": 0.1})
	points[0]["T"] += 40
	tilt_path, links_path, output = tmp_path / "tilt.json", tmp_path / "links.jsonl", tmp_path / "fit.json"
	tilt_path.write_text(json.dumps({"phi": -2.0, "dy": [0.01, 5.0], "dx": [-0.01, 3.0], "points": points}))
	links_path.write_text("".join(json.dumps(link) + "\n" for link in links))

	result = CliRunner().invoke(fit, [str(tilt_path), str(output), "--links", str(links_path),
										"--reference", "r0,r1,r2", "--fixed", "f0,f1"])

	assert result.exit_code == 0, result.output
	solved = json.loads(output.read_text())
	assert solved["reference"] == pytest.approx(reference, abs=1e-6)
	assert solved["fixed"] == pytest.approx(fixed, abs=1e-6)
	assert (solved["K"], solved["gy"], solved["gx"], solved["gz"]) == pytest.approx((k, gy, gx, gz), abs=1e-6)
	assert solved["points_used"] < solved["points_total"] == len(points)


def test_tilt_points_measure_the_cross_group_depth_offset(tmp_path):
	reference = smooth_noise((160, 96, 96), 1.0, 8)
	fixed_dir = write_stack(tmp_path / "fixed", reference[30:130])
	moving_dir = write_stack(tmp_path / "moving", reference)
	refine_path = tmp_path / "refine.json"
	refine_path.write_text(json.dumps([{"row": row, "dy": 0.0, "dx": 0.0, "peak": 0.5} for row in (0, 50)]))
	config = tmp_path / "tilt_config.json"
	config.write_text(json.dumps({
		"phi": 0.0, "depth_guess": 27, "fixed_offsets": {"f0": 0}, "moving_offsets": {"r0": 0},
		"links": [{"fixed": "f0", "fixed_dir": str(fixed_dir), "moving": "r0", "moving_dir": str(moving_dir),
					"rows": [20], "refine": str(refine_path)}],
	}))
	output = tmp_path / "tilt.json"

	result = CliRunner().invoke(tilt_points, [str(config), str(output), "--radius", "30", "--depth", "24",
												"--search", "8", "--block", "32", "--step", "16"])

	assert result.exit_code == 0, result.output
	points = json.loads(output.read_text())["points"]
	assert points
	assert np.median([point["G_minus_Z"] for point in points]) == 30
	assert {point["T"] for point in points} >= {30}


def test_fit_rejects_inputs_without_usable_cross_group_points(tmp_path):
	tilt_path, links_path, output = tmp_path / "tilt.json", tmp_path / "links.jsonl", tmp_path / "fit.json"
	tilt_path.write_text(json.dumps({"phi": 0.0, "dy": [0, 0], "dx": [0, 0], "points": [
		{"link": "f0", "moving": "r0", "row": 1, "T": 5, "Z": 1, "y": 1, "x": 1, "peak": 0.001}]}))
	links_path.write_text(json.dumps({"fixed": "r1", "moving": "r0", "T_median": 10.0}) + "\n")

	result = CliRunner().invoke(fit, [str(tilt_path), str(output), "--links", str(links_path),
										"--reference", "r0,r1", "--fixed", "f0"])

	assert result.exit_code != 0
	assert "no cross-group tilt points" in result.output
	assert not output.exists()


def fit_document(**changes):
	document = {"phi": -2.0, "K": 300.0, "gy": 0.004, "gx": -0.003, "gz": 0.0005,
				"dy": [0.01, 4.0], "dx": [-0.02, -6.0], "reference": {"r0": 0.0, "r1": 70.0}, "fixed": {"f0": 0.0}}
	return {**document, **changes}


def test_fixed_map_is_the_affine_inverse_of_the_group_relation():
	relation = GroupRelation(fit_document(), 31.5)
	matrix = np.array(fixed_map(relation, 0.0, 5, 7))

	for g, y, x in ((10.0, 20.0, 30.0), (250.0, 3.0, 60.0)):
		z, xs, ys = relation.fixed_point(g, x - 7, y - 5)
		np.testing.assert_allclose(matrix @ np.array([g, y, x, 1.0]), [z, ys, xs], atol=1e-9)


def test_layout_writes_integer_reference_maps_and_canvas_covering_every_disk(tmp_path):
	stacks = {name: write_stack(tmp_path / name, np.zeros((rows, 64, 64), np.float32))
				for name, rows in (("r0", 100), ("r1", 100), ("f0", 120))}
	manifest = tmp_path / "manifest.json"
	manifest.write_text(json.dumps({"scans": [{"name": name, "dir": str(path), "group": name[0]}
												for name, path in stacks.items()]}))
	fit_path, output = tmp_path / "fit.json", tmp_path / "layout.json"
	fit_path.write_text(json.dumps(fit_document()))

	result = CliRunner().invoke(layout, [str(fit_path), str(manifest), str(output), "--fov-radius", "30",
											"--fixed-lowpass", "0.02"])

	assert result.exit_code == 0, result.output
	document = json.loads(output.read_text())
	scans = {scan["name"]: scan for scan in document["scans"]}
	oy, ox = document["offset_yx"]
	assert scans["r1"]["A"] == [[1, 0, 0, -70], [0, 1, 0, -oy], [0, 0, 1, -ox]]
	assert document["nz"] == 170
	assert scans["f0"]["lowpass_weight"] == 0.02 and scans["f0"]["group"] == "f"
	height, width = document["canvas"]
	relation = GroupRelation(fit_document(), 31.5)
	for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
		x, y = 31.5 + 30 * np.cos(angle), 31.5 + 30 * np.sin(angle)
		assert 0 <= y + oy < height and 0 <= x + ox < width
		for z in (0.0, 119.0):
			xr, yr = relation.reference_point(z, x, y)
			assert 0 <= yr + oy < height and 0 <= xr + ox < width
