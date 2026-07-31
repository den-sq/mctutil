from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from threading import get_ident, Thread
import types

import pytest

from mctutil.shared.log import log


class TtyBuffer(StringIO):
	def isatty(self):
		return True


class FakeClientError(Exception):
	def __init__(self, response, operation_name):
		super().__init__(operation_name)
		self.response = response


class RecordingProgress:
	def __init__(self, **configuration):
		self.configuration = configuration
		self.position = configuration.get("initial", 0)
		self.maximum = self.position
		self.update_threads = []
		self.exited_with = None

	def __enter__(self):
		return self

	def update(self, count):
		self.update_threads.append(get_ident())
		self.position += count
		self.maximum = max(self.maximum, self.position)

	def __exit__(self, exc_type, *_args):
		self.exited_with = exc_type
		return False


class CallbackClient:
	def __init__(self, *, fail_key=None):
		self.objects = {}
		self.uploads = []
		self.callback_calls = 0
		self.fail_key = fail_key

	def head_object(self, Bucket, Key):
		try:
			return self.objects[(Bucket, Key)]
		except KeyError:
			raise FakeClientError(
				{"Error": {"Code": "404"}},
				"HeadObject",
			)

	def upload_file(self, filename, bucket, key, ExtraArgs, Callback):
		path = Path(filename)
		size = path.stat().st_size
		self.uploads.append(key)
		if key == self.fail_key:
			self.callback_calls += 1
			# Even if callbacks report every byte, the transfer has not
			# succeeded until upload_file returns.
			Callback(size)
			raise OSError("simulated transfer failure")

		# Boto3 may invoke callbacks concurrently and in more than one chunk.
		# The extra byte also checks that the adapter cannot overrun an object.
		chunks = [max(1, size // 2), max(1, size - (size // 2)), 1]
		threads = [
			Thread(target=self._callback, args=(Callback, chunk))
			for chunk in chunks
		]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
		self.objects[(bucket, key)] = {
			"ContentLength": size,
			"Metadata": ExtraArgs["Metadata"],
		}

	def _callback(self, callback, amount):
		self.callback_calls += 1
		callback(amount)


def make_sharded_tree(root: Path, *, large_mip0=False) -> Path:
	root.mkdir()
	info = {
		"type": "image",
		"scales": [
			{
				"key": "mip0",
				"sharding": {"@type": "neuroglancer_uint64_sharded_v1"},
			},
			{
				"key": "mip1",
				"sharding": {"@type": "neuroglancer_uint64_sharded_v1"},
			},
		],
	}
	(root / "info").write_text(json.dumps(info), encoding="utf-8")
	(root / "provenance").write_bytes(b"provenance")
	for key in ("mip0", "mip1"):
		(root / key).mkdir()
	(root / "mip0" / "0.shard").write_bytes(
		b"x" * (16_384 if large_mip0 else 2_048)
	)
	(root / "mip1" / "1.shard").write_bytes(b"y" * 1_024)
	(root / ".mctutil-queues").mkdir()
	(root / ".mctutil-queues" / "state").write_bytes(b"private")
	return root


def configure_sync(module, monkeypatch, client):
	recorded = {}
	main_thread = get_ident()
	writes = []

	def progress_factory(_label, **configuration):
		progress = RecordingProgress(**configuration)
		recorded["label"] = _label
		recorded["progress"] = progress
		return progress

	def write(step, statement="", **kwargs):
		writes.append((get_ident(), step, statement, kwargs))

	monkeypatch.setattr(module, "ClientError", FakeClientError)
	monkeypatch.setattr(module, "configure_aws_profile", lambda *_args: "profile")
	monkeypatch.setattr(
		module,
		"_get_session",
		lambda _profile: types.SimpleNamespace(client=lambda _name: client),
	)
	monkeypatch.setattr(module.log, "progress", progress_factory)
	monkeypatch.setattr(module.log, "write", write)
	return main_thread, recorded, writes


def test_byte_position_formatter_is_used_by_interactive_progress(load_module):
	module = load_module("mctutil/transport/s3upload.py")
	terminal = TtyBuffer()

	with log.progress(
		"S3 Sync",
		length=2_048,
		out=terminal,
		start_message=None,
		final_message=None,
		position_formatter=module.format_byte_progress,
	) as progress:
		progress.update(2_047)

	assert "2.0 KiB (1 B remaining)/2.0 KiB" in terminal.getvalue()


def test_mixed_sync_reports_byte_progress_from_parent_thread(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transport/s3upload.py")
	source = make_sharded_tree(tmp_path / "sharded")
	client = CallbackClient()
	main_thread, recorded, writes = configure_sync(
		module,
		monkeypatch,
		client,
	)
	scales = module.read_sharded_scales(source)
	items = [
		item
		for group in module.inventory_sharded_tree(source, "dataset", scales)
		for item in group
	]
	unchanged = next(item for item in items if item.key.endswith("provenance"))
	client.objects[("bucket", unchanged.key)] = {
		"ContentLength": unchanged.size,
		"Metadata": unchanged.fingerprint,
	}

	counts = module.upload_sharded_tree(
		source,
		"dataset",
		"bucket",
		jobs=2,
		execute=True,
	)

	total = sum(item.size for item in items)
	progress = recorded["progress"]
	assert counts == {"planned": 0, "skipped": 1, "uploaded": len(items) - 1}
	assert recorded["label"] == "S3 Sync"
	assert progress.configuration["length"] == total
	assert progress.configuration["position_formatter"](1_536, 2_048) == (
		"1.5 KiB/2.0 KiB"
	)
	assert progress.configuration["position_formatter"](2_047, 2_048) == (
		"2.0 KiB (1 B remaining)/2.0 KiB"
	)
	assert progress.position == total
	assert progress.maximum == total
	assert set(progress.update_threads) == {main_thread}
	assert client.callback_calls > counts["uploaded"]
	assert all(thread_id == main_thread for thread_id, *_rest in writes)
	assert len(writes) == len(items) + 1
	assert all(
		write[3]["log_level"] == module.LOG.DEBUG
		for write in writes[:-1]
	)
	summary = writes[-1][2]
	assert f"uploaded={len(items) - 1}" in summary
	assert f"({module.format_bytes(total - unchanged.size)})" in summary
	assert "unchanged=1" in summary
	assert f"({module.format_bytes(unchanged.size)})" in summary
	assert ".mctutil-queues" not in str(writes)


def test_dry_run_accounts_for_every_planned_byte_without_s3(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transport/s3upload.py")
	source = make_sharded_tree(tmp_path / "sharded")
	main_thread, recorded, writes = configure_sync(
		module,
		monkeypatch,
		CallbackClient(),
	)
	monkeypatch.setattr(
		module,
		"_get_session",
		lambda _profile: (_ for _ in ()).throw(
			AssertionError("dry run constructed an S3 session")
		),
	)

	counts = module.upload_sharded_tree(
		source,
		"dataset",
		"bucket",
		jobs=2,
		execute=False,
	)

	progress = recorded["progress"]
	assert counts["planned"] == 4
	assert counts["skipped"] == counts["uploaded"] == 0
	assert progress.position == progress.configuration["length"]
	assert progress.maximum == progress.configuration["length"]
	assert set(progress.update_threads) == {main_thread}
	assert progress.configuration["start_message"].startswith("Dry run:")
	assert "planned=4" in writes[-1][2]
	assert f"({module.format_bytes(progress.position)})" in writes[-1][2]


def test_failed_upload_stops_short_and_names_the_object(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transport/s3upload.py")
	source = make_sharded_tree(tmp_path / "sharded", large_mip0=True)
	failed_key = "dataset/mip0/0.shard"
	client = CallbackClient(fail_key=failed_key)
	_main_thread, recorded, writes = configure_sync(
		module,
		monkeypatch,
		client,
	)

	with pytest.raises(
		RuntimeError,
		match=r"failed to upload s3://bucket/dataset/mip0/0\.shard",
	):
		module.upload_sharded_tree(
			source,
			"dataset",
			"bucket",
			jobs=2,
			execute=True,
		)

	progress = recorded["progress"]
	assert progress.exited_with is RuntimeError
	assert progress.position < progress.configuration["length"]
	assert progress.maximum < progress.configuration["length"]
	assert not any("sharded sync summary" in statement for _, _, statement, _ in writes)
