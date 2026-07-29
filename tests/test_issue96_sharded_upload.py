from __future__ import annotations

import json
from pathlib import Path
import types

from click.testing import CliRunner


def make_sharded_tree(root: Path) -> Path:
	root.mkdir()
	info = {
		"type": "image",
		"scales": [
			{
				"key": "700_700_700",
				"sharding": {"@type": "neuroglancer_uint64_sharded_v1"},
			},
			{
				"key": "1400_1400_1400",
				"sharding": {"@type": "neuroglancer_uint64_sharded_v1"},
			},
		],
	}
	(root / "info").write_text(json.dumps(info), encoding="utf-8")
	(root / "provenance").write_text("{}", encoding="utf-8")
	for key, content in (
		("700_700_700", b"mip0"),
		("1400_1400_1400", b"mip1"),
	):
		scale = root / key
		scale.mkdir()
		(scale / "0.shard").write_bytes(content)
	(root / ".mctutil-queues").mkdir()
	(root / ".mctutil-queues" / "state").write_text("private", encoding="utf-8")
	return root


class FakeClient:
	def __init__(self):
		self.objects = {}
		self.uploads = []

	def head_object(self, Bucket, Key):
		try:
			return self.objects[(Bucket, Key)]
		except KeyError:
			raise FakeClientError(
				{"Error": {"Code": "404"}},
				"HeadObject",
			)

	def upload_file(self, filename, bucket, key, ExtraArgs):
		path = Path(filename)
		self.uploads.append((path, bucket, key, ExtraArgs))
		self.objects[(bucket, key)] = {
			"ContentLength": path.stat().st_size,
			"Metadata": ExtraArgs["Metadata"],
		}


class FakeClientError(Exception):
	def __init__(self, response, operation_name):
		super().__init__(operation_name)
		self.response = response


def test_sharded_upload_excludes_mip0_and_private_dirs(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transport/s3upload.py")
	monkeypatch.setattr(module, "ClientError", FakeClientError)
	source = make_sharded_tree(tmp_path / "staged")
	client = FakeClient()
	monkeypatch.setattr(
		module,
		"_get_session",
		lambda: types.SimpleNamespace(client=lambda _name: client),
	)

	counts = module.upload_sharded_tree(
		source,
		"prefix/dataset",
		"bucket",
		jobs=2,
		include_mip0=False,
		execute=True,
	)

	keys = {upload[2] for upload in client.uploads}
	assert keys == {
		"prefix/dataset/info",
		"prefix/dataset/provenance",
		"prefix/dataset/1400_1400_1400/0.shard",
	}
	assert counts == {"planned": 0, "skipped": 0, "uploaded": 3}
	assert all(".mctutil-queues" not in key for key in keys)


def test_excluding_mip0_allows_it_to_be_unstaged(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transport/s3upload.py")
	monkeypatch.setattr(module, "ClientError", FakeClientError)
	source = make_sharded_tree(tmp_path / "staged")
	info = json.loads((source / "info").read_text(encoding="utf-8"))
	info["scales"][0].pop("sharding")
	(source / "info").write_text(json.dumps(info), encoding="utf-8")
	for path in (source / "700_700_700").iterdir():
		path.unlink()
	(source / "700_700_700").rmdir()
	client = FakeClient()
	monkeypatch.setattr(
		module,
		"_get_session",
		lambda: types.SimpleNamespace(client=lambda _name: client),
	)

	counts = module.upload_sharded_tree(
		source,
		"dataset",
		"bucket",
		include_mip0=False,
		execute=True,
	)

	assert counts["uploaded"] == 3
	assert all("700_700_700" not in upload[2] for upload in client.uploads)


def test_sharded_upload_is_incremental(load_module, tmp_path, monkeypatch):
	module = load_module("mctutil/transport/s3upload.py")
	monkeypatch.setattr(module, "ClientError", FakeClientError)
	source = make_sharded_tree(tmp_path / "staged")
	client = FakeClient()
	monkeypatch.setattr(
		module,
		"_get_session",
		lambda: types.SimpleNamespace(client=lambda _name: client),
	)

	first = module.upload_sharded_tree(
		source,
		"dataset",
		"bucket",
		jobs=2,
		execute=True,
	)
	second = module.upload_sharded_tree(
		source,
		"dataset",
		"bucket",
		jobs=2,
		execute=True,
	)

	assert first["uploaded"] == 4
	assert second == {"planned": 0, "skipped": 4, "uploaded": 0}
	assert len(client.uploads) == 4


def test_sharded_upload_dry_run_never_constructs_s3_client(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transport/s3upload.py")
	source = make_sharded_tree(tmp_path / "staged")
	monkeypatch.setattr(
		module,
		"_get_session",
		lambda: (_ for _ in ()).throw(AssertionError("S3 session created")),
	)

	result = CliRunner().invoke(
		module.s3upload,
		[
			"--bucket-prefix", "prefix",
			"--bucket-name", "bucket",
			"--from-sharded-tree",
			str(source),
			"dataset",
		],
	)

	assert result.exit_code == 0, result.output
	assert "planned=4" in result.output


def test_legacy_upload_without_flag_preserves_execute_default(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transport/s3upload.py")
	source = tmp_path / "legacy"
	source.mkdir()
	events = []

	class FakeS3:
		def put_object(self, **kwargs):
			events.append(("prefix", kwargs))

	fake_s3 = FakeS3()
	monkeypatch.setattr(
		module,
		"_get_session",
		lambda: types.SimpleNamespace(client=lambda _name: fake_s3),
	)
	monkeypatch.setattr(
		module,
		"upload_folder_to_s3_parallel",
		lambda *_args, **kwargs: events.append(
			("upload", kwargs["execute"])
		),
	)

	result = CliRunner().invoke(
		module.s3upload,
		[
			"--bucket-prefix", "prefix",
			"--bucket-name", "bucket",
			str(source),
			"dataset",
		],
	)

	assert result.exit_code == 0, result.output
	assert events[0][0] == "prefix"
	assert ("upload", True) in events


def test_explicit_dry_run_overrides_legacy_execute_default(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/transport/s3upload.py")
	source = tmp_path / "legacy"
	source.mkdir()
	events = []
	monkeypatch.setattr(
		module,
		"_get_session",
		lambda: (_ for _ in ()).throw(AssertionError("S3 session created")),
	)
	monkeypatch.setattr(
		module,
		"upload_folder_to_s3_parallel",
		lambda *_args, **kwargs: events.append(kwargs["execute"]),
	)

	result = CliRunner().invoke(
		module.s3upload,
		[
			"--bucket-prefix", "prefix",
			"--bucket-name", "bucket",
			"--dry-run",
			str(source),
			"dataset",
		],
	)

	assert result.exit_code == 0, result.output
	assert events == [False]


def test_sharded_tree_validation_rejects_unsharded_scale(load_module, tmp_path):
	module = load_module("mctutil/transport/s3upload.py")
	source = tmp_path / "not-sharded"
	source.mkdir()
	(source / "info").write_text(
		json.dumps({"scales": [{"key": "700_700_700"}]}),
		encoding="utf-8",
	)
	(source / "700_700_700").mkdir()

	result = CliRunner().invoke(
		module.s3upload,
		[
			"--bucket-prefix", "prefix",
			"--bucket-name", "bucket",
			"--from-sharded-tree",
			str(source),
			"dataset",
		],
	)

	assert result.exit_code != 0
	assert "scale 0 is not sharded" in result.output
