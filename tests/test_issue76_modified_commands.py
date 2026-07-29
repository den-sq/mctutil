from __future__ import annotations

import importlib
import sys
import types

import numpy as np

from mctutil.shared.cli import NumpyCLI

try:
	import cloudvolume
except ImportError:
	cloudvolume = types.ModuleType("cloudvolume")
	cloudvolume.CloudVolume = type("CloudVolume", (), {})
	sys.modules["cloudvolume"] = cloudvolume

cv_import = importlib.import_module("mctutil.transport.cv_import")


def test_fetch_slices_converts_dtype_and_transposes(tmp_path, monkeypatch):
	class FakeVolume:
		def __init__(self, *_args, **_kwargs):
			self.data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

		def __getitem__(self, _region):
			return [self.data]

	writes = []
	monkeypatch.setattr(cv_import, "CloudVolume", FakeVolume)
	monkeypatch.setattr(
		cv_import.tifffile,
		"imwrite",
		lambda path, data: writes.append((path.name, np.array(data))),
	)

	cv_import.fetch_slices(
		"precomputed://example",
		False,
		(np.s_[5:6], np.s_[:], np.s_[:]),
		0,
		tmp_path,
		NumpyCLI(np.float32),
		True,
	)

	assert writes[0][0] == "slice_0005.tif"
	assert writes[0][1].dtype == np.float32
	np.testing.assert_array_equal(writes[0][1], np.transpose(FakeVolume().data.astype(np.float32), (2, 0, 1)))


def test_fetch_slices_dry_run_does_not_connect(tmp_path, monkeypatch):
	def fail_cloudvolume(*_args, **_kwargs):
		raise AssertionError("CloudVolume should not be created during dry runs")

	monkeypatch.setattr(cv_import, "CloudVolume", fail_cloudvolume)

	cv_import.fetch_slices(
		"precomputed://example",
		False,
		(np.s_[5:6], np.s_[:], np.s_[:]),
		0,
		tmp_path,
		execute=False,
	)
