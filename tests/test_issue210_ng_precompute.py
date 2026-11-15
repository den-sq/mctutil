from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import psutil
import pytest
import tifffile


def _terminate_process_tree(process):
	children = psutil.Process(process.pid).children(recursive=True)
	for child in children:
		try:
			child.terminate()
		except psutil.NoSuchProcess:
			pass
	process.kill()
	process.wait(timeout=10)
	_, survivors = psutil.wait_procs(children, timeout=10)
	for child in survivors:
		try:
			child.kill()
		except psutil.NoSuchProcess:
			pass


@pytest.mark.parametrize("input_mode", ("memmap", "directory"))
def test_precompute_completes_beyond_worker_retirement_boundary(tmp_path, input_mode):
	CloudVolume = pytest.importorskip("cloudvolume").CloudVolume
	# Cross the former 4 workers * 500 tasks retirement boundary together.
	source = np.arange(2100 * 64 * 64, dtype=np.uint16).reshape(2100, 64, 64)
	if input_mode == "memmap":
		input_path = tmp_path / "volume.tif"
		tifffile.imwrite(input_path, source, photometric="minisblack")
	else:
		input_path = tmp_path / "slices"
		input_path.mkdir()
		for z_index, plane in enumerate(source):
			tifffile.imwrite(input_path / f"slice_{z_index}.tif", plane)
	output_path = tmp_path / "precomputed"
	log_path = tmp_path / "precompute.log"

	# A subprocess timeout bounds the actual stalled-pool failure, including
	# shutdown; a log file avoids pipe descriptors held open by pool children.
	with log_path.open("w", encoding="utf-8") as output:
		with subprocess.Popen(
			[
				sys.executable, "-m", "mctutil.ng.precompute",
				str(input_path), str(output_path),
				"--workers", "4", "--chunk-size", "64,64,1",
			],
			cwd=Path(__file__).resolve().parents[1],
			stdout=output,
			stderr=subprocess.STDOUT,
		) as process:
			try:
				process.wait(timeout=60)
			except subprocess.TimeoutExpired:
				_terminate_process_tree(process)
				pytest.fail(f"precompute did not finish within 60 seconds:\n{log_path.read_text()}")

	output = log_path.read_text(encoding="utf-8")
	assert process.returncode == 0, output
	assert "Precompute complete; wrote 2100 Z plane(s)." in output
	assert "resource_tracker:" not in output
	volume = CloudVolume(output_path.resolve().as_uri(), parallel=False)
	written = np.asarray(volume[:, :, :, 0])[..., 0]
	assert np.array_equal(written, source.transpose(2, 1, 0))
