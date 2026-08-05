from __future__ import annotations

from concurrent.futures import Future
from multiprocessing import shared_memory
import uuid

import numpy as np
import tifffile

from mctutil.ng import precompute as precompute_module
from mctutil.shared.io_helpers import distribute_read, offset_reads
from mctutil.shared.mem import ProjOrder, SharedNP
from mctutil.transform import sinogram


def test_raw_offset_reads_copy_exact_strided_spans_into_shared_memory(tmp_path):
	source = tmp_path / "source.bin"
	source.write_bytes(b"__abc--DEF++")
	memory = shared_memory.SharedMemory(create=True, size=10)
	try:
		memory.buf[:] = b"." * 10
		reads = offset_reads(
			source,
			source_offset=2,
			target_offset=1,
			size=3,
			count=2,
			source_stride=5,
			target_stride=5,
		)

		distribute_read(memory, reads, thread_max=1)

		assert bytes(memory.buf) == b".abc..DEF."
	finally:
		memory.close()
		memory.unlink()


def test_sinogram_projection_layout_matches_tifffile_pixels(tmp_path, monkeypatch):
	monkeypatch.setattr(sinogram.log, "write", lambda *_args, **_kwargs: None)
	images = []
	expected = []
	for index in range(2):
		data = np.arange(12, dtype=np.uint16).reshape(3, 4) + index * 100
		path = tmp_path / f"projection_{index}.tif"
		tifffile.imwrite(path, data, compression=None)
		images.append(path)
		expected.append(data[1:3, :])

	with tifffile.TiffFile(images[0]) as tif:
		page = tif.pages[0]
		projection = {
			"dtype": page.dtype,
			"bytesize": page.dtype.itemsize,
			"offset": page.dataoffsets[0],
			"x": page.shape[1],
			"y": page.shape[0],
		}

	name = f"issue138_{uuid.uuid4().hex}"
	with SharedNP(name, np.uint16, ProjOrder(2, 2, 4), create=True) as target_mem:
		sinogram.distribute_read(
			target_mem,
			projection,
			range(1, 3),
			range(0, 2),
			enumerate(images),
			thread_max=1,
			sino_order=False,
		)
		with target_mem as target:
			assert np.array_equal(target, np.asarray(expected))


def test_precompute_batches_memmap_offsets_through_shared_ingest(monkeypatch):
	spec = precompute_module.InputSpec(
		mode="memmap",
		source="volume.tif",
		shape=(3, 3, 4),
		dtype=np.dtype("uint16"),
		raw_offset=100,
		plane_stride=24,
	)
	plan = precompute_module.VolumePlan(
		layer_type="image",
		encoding="raw",
		dtype=np.dtype("uint16"),
		resolution=(700, 700, 700),
		voxel_offset=(0, 0, 5),
		chunk_size=(4, 3, 1),
		segmentation_block=(8, 8, 8),
	)
	read_batches = []
	executors = []

	def record_reads(_target, reads, thread_max):
		read_batches.append((tuple(reads), thread_max))

	class ImmediateExecutor:
		def __init__(self, **options):
			self.options = options
			executors.append(self)

		def submit(self, _function, work_item):
			future = Future()
			future.set_result(work_item[0])
			return future

		def shutdown(self, **_options):
			return None

	monkeypatch.setattr(precompute_module, "distribute_offset_reads", record_reads)
	monkeypatch.setattr(precompute_module, "ProcessPoolExecutor", ImmediateExecutor)

	result = precompute_module._execute_slices(
		"file:///output",
		spec,
		plan,
		[0, 1, 2],
		workers=2,
	)

	assert result == precompute_module.WorkerBatchResult(frozenset({0, 1, 2}), None)
	assert [[read.source_offset for read in reads] for reads, _ in read_batches] == [
		[100, 124],
		[148],
	]
	assert [[read.target_offset for read in reads] for reads, _ in read_batches] == [
		[0, 24],
		[0],
	]
	assert [thread_max for _, thread_max in read_batches] == [2, 1]
	initializer_args = executors[0].options["initargs"]
	assert initializer_args[3] is True
	assert initializer_args[4] == (2, 3, 4)
	assert initializer_args[6] == 5


def test_precompute_worker_writes_a_shared_plane_as_cloudvolume_xy(monkeypatch):
	class RecordingVolume:
		def __init__(self, *_args, **_kwargs):
			self.writes = []

		def __setitem__(self, key, value):
			self.writes.append((key, np.array(value)))

	memory = shared_memory.SharedMemory(create=True, size=2 * 3 * 4 * 2)
	try:
		source = np.ndarray((2, 3, 4), dtype=np.uint16, buffer=memory.buf)
		source[:] = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
		monkeypatch.setattr(precompute_module, "_require_cloudvolume", lambda: RecordingVolume)
		monkeypatch.setattr(precompute_module, "patch_cloudfiles_monitoring", lambda: None)

		precompute_module._init_worker(
			"file:///output",
			"uint16",
			memory.name,
			True,
			(2, 3, 4),
			"uint16",
			5,
		)
		assert precompute_module._write_slice((7, 1)) == 7
		key, written = precompute_module._WORKER_VOLUME.writes[0]
		assert key[2] == slice(12, 13)
		assert np.array_equal(written[:, :, 0, 0], source[1].T)
	finally:
		if precompute_module._WORKER_SHARED_MEMORY is not None:
			precompute_module._WORKER_SHARED_MEMORY.close()
			precompute_module._WORKER_SHARED_MEMORY = None
		memory.close()
		memory.unlink()


def test_discover_input_exposes_memmap_raw_plane_layout(tmp_path):
	path = tmp_path / "volume.tif"
	source = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
	tifffile.imwrite(path, source, photometric="minisblack")

	spec = precompute_module.discover_input(path)
	mapped = tifffile.memmap(path)
	try:
		assert spec.raw_offset == mapped.offset
		assert spec.plane_stride == mapped.strides[0]
	finally:
		del mapped
