"""Deterministic manual demonstration of sharded S3 sync progress.

Run normally for redirected-output behavior, or under ``script`` for a
pseudo-terminal:

	PYTHONPATH=. python tests/s3_sync_progress_demo.py mixed
	script -q -c \
		'PYTHONPATH=. python tests/s3_sync_progress_demo.py mixed' \
		/dev/null
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import time
import types

from botocore.exceptions import ClientError

from mctutil.transport import s3upload


class FakeS3:
	def __init__(self, unchanged, fail_key=None):
		self.unchanged = unchanged
		self.fail_key = fail_key

	def head_object(self, Bucket, Key):
		if Key == self.unchanged["key"]:
			return {
				"ContentLength": self.unchanged["size"],
				"Metadata": self.unchanged["metadata"],
			}
		raise ClientError(
			{"Error": {"Code": "404"}},
			"HeadObject",
		)

	def upload_file(self, filename, bucket, key, ExtraArgs, Callback):
		size = Path(filename).stat().st_size
		if key == self.fail_key:
			Callback(size)
			raise OSError("deterministic simulated failure")
		chunks = [
			size // 4,
			size // 4,
			size // 4,
			size - 3 * (size // 4),
		]

		def report(amount):
			time.sleep(0.04)
			Callback(amount)

		threads = [
			Thread(target=report, args=(amount,))
			for amount in chunks
		]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()


def make_tree(root):
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
	root.mkdir()
	(root / "info").write_text(json.dumps(info), encoding="utf-8")
	(root / "provenance").write_bytes(b"p" * 1536)
	for key, size in (
		("mip0", 5 * 1024**2),
		("mip1", 2 * 1024**2),
	):
		(root / key).mkdir()
		(root / key / "0.shard").write_bytes(b"x" * size)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("mode", choices=("mixed", "dry-run", "failure"))
	args = parser.parse_args()
	with TemporaryDirectory() as temp:
		source = Path(temp) / "sharded"
		make_tree(source)
		provenance = source / "provenance"
		stat = provenance.stat()
		unchanged = {
			"key": "demo/provenance",
			"size": stat.st_size,
			"metadata": {
				"mctutil-size": str(stat.st_size),
				"mctutil-mtime-ns": str(stat.st_mtime_ns),
			},
		}
		fail_key = (
			"demo/mip0/0.shard"
			if args.mode == "failure"
			else None
		)
		client = FakeS3(unchanged, fail_key)
		s3upload.configure_aws_profile = lambda *_args: "demo"
		s3upload._get_session = lambda _profile: types.SimpleNamespace(
			client=lambda _name: client
		)
		try:
			s3upload.upload_sharded_tree(
				source,
				"demo",
				"example-bucket",
				jobs=2,
				execute=args.mode != "dry-run",
			)
		except RuntimeError as exc:
			print(f"EXPECTED ERROR: {exc}")


if __name__ == "__main__":
	main()
