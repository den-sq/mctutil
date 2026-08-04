from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from queue import Empty as ProgressQueueEmpty, Queue
from threading import Lock
from botocore.exceptions import ClientError

import click


from mctutil.shared.aws import (
	configure_aws_profile,
	create_boto3_session,
	resolve_aws_profile,
)
from mctutil.shared.log import log, LOG
from mctutil.shared.mesh import build_mesh
from mctutil.shared.resource_monitor import record_active_workers

_sessions = {}
SYNC_STATUSES = ("planned", "skipped", "uploaded")


@dataclass(frozen=True)
class SyncObject:
	"""One inventoried local file and its destination object key."""

	path: Path
	key: str
	size: int
	mtime_ns: int

	@property
	def fingerprint(self) -> dict[str, str]:
		return {
			"mctutil-size": str(self.size),
			"mctutil-mtime-ns": str(self.mtime_ns),
		}


@dataclass
class SyncSummary:
	"""Object and byte totals partitioned by sync result."""

	counts: dict[str, int] = field(
		default_factory=lambda: {status: 0 for status in SYNC_STATUSES}
	)
	bytes: dict[str, int] = field(
		default_factory=lambda: {status: 0 for status in SYNC_STATUSES}
	)
	details: list[tuple[str, str]] = field(default_factory=list)

	def add(self, status: str, size: int, key: str) -> None:
		self.counts[status] += 1
		self.bytes[status] += size
		self.details.append((status, key))

	def merge(self, other) -> None:
		for status in SYNC_STATUSES:
			self.counts[status] += other.counts[status]
			self.bytes[status] += other.bytes[status]
		self.details.extend(other.details)


class ObjectTransferProgress:
	"""Thread-safe, size-bounded adapter for Boto3 transfer callbacks."""

	def __init__(self, size: int, events: Queue):
		self.size = size
		self.events = events
		self.transferred = 0
		self.lock = Lock()

	def __call__(self, amount: int) -> None:
		# Keep one byte unreported until upload_file returns successfully. This
		# prevents a late transfer-manager failure from displaying completion.
		callback_limit = max(0, self.size - 1)
		with self.lock:
			delta = min(
				max(0, int(amount)),
				max(0, callback_limit - self.transferred),
			)
			self.transferred += delta
		if delta:
			self.events.put(delta)

	def complete(self) -> None:
		"""Account for any callback shortfall after a successful upload."""
		with self.lock:
			delta = self.size - self.transferred
			self.transferred += delta
		if delta:
			self.events.put(delta)


def _get_session(aws_profile):
	profile = resolve_aws_profile(aws_profile)
	if profile not in _sessions:
		_sessions[profile] = create_boto3_session(profile)
	return _sessions[profile]


def upload_file_to_s3(
	file_path,
	key,
	bucket_name,
	content_encoding,
	execute=True,
	aws_profile=None,
):
	if not execute:
		log.write("S3 Upload", f"Would upload {file_path} -> s3://{bucket_name}/{key}", log_level=LOG.INFO)
		return

	s3 = _get_session(aws_profile).client('s3')
	if file_path.is_dir():  # Handle directory
		try:
			s3.put_object(Bucket=bucket_name, Key=f"{key}/")
		except ClientError as e:
			log.write("S3 Upload", f"ClientError: {e.response}", log_level=LOG.ERROR)
		except Exception as e:
			log.write("S3 Upload", f"{e}", log_level=LOG.ERROR)
	else:  # Handle file
		extra_args = {}
		if content_encoding is not None:
			extra_args['ContentEncoding'] = content_encoding
		try:
			s3.upload_file(file_path, bucket_name, str(key), ExtraArgs=extra_args)
		except ClientError as e:
			log.write("S3 Upload", f"ClientError: {e.response}", log_level=LOG.ERROR)
		except Exception as e:
			log.write("S3 Upload", f"{e}", log_level=LOG.ERROR)
	log.write("S3 Upload", f"uploaded: {key}", log_level=LOG.STATUS)


def upload_folder_to_s3_parallel(
	folder_path,
	target_folder,
	bucket_name,
	num_processes,
	execute=True,
	aws_profile=None,
):
	folder_path = Path(folder_path)
	with ProcessPoolExecutor(max_workers=num_processes) as executor:
		for entry in folder_path.rglob("*"):
			key = target_folder.joinpath(entry.relative_to(folder_path))
			if entry.is_dir():
				executor.submit(
					upload_file_to_s3,
					entry,
					key,
					bucket_name,
					None,
					execute,
					aws_profile,
				)
			else:
				content_encoding = 'gzip' if entry.name != 'info' else None
				executor.submit(
					upload_file_to_s3,
					entry,
					key,
					bucket_name,
					content_encoding,
					execute,
					aws_profile,
				)


def _join_key(*parts) -> str:
	return "/".join(
		str(part).strip("/")
		for part in parts
		if str(part).strip("/")
	)


def read_sharded_scales(
	source_folder: Path,
	include_mip0: bool = True,
) -> list[tuple[int, str, Path]]:
	"""Return declared scale directories after validating sharding metadata."""
	info_path = source_folder / "info"
	if not info_path.is_file():
		raise ValueError(f"sharded tree is missing info: {info_path}")
	try:
		info = json.loads(info_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		raise ValueError(f"invalid precomputed info: {info_path}") from exc

	declared_scales = info.get("scales", [])
	if not declared_scales:
		raise ValueError("sharded tree has no scales")
	scales = []
	for mip, scale in enumerate(declared_scales):
		if mip == 0 and not include_mip0:
			continue
		key = scale.get("key")
		if not key:
			raise ValueError(f"scale {mip} has no key")
		if not scale.get("sharding"):
			raise ValueError(f"scale {mip} is not sharded")
		scale_path = source_folder / key
		if not scale_path.is_dir():
			raise ValueError(f"scale {mip} directory is missing: {scale_path}")
		scales.append((mip, str(key), scale_path))
	return scales


def format_bytes(value: int) -> str:
	"""Format a byte count with binary units."""
	amount = float(value)
	units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
	for unit in units[:-1]:
		if abs(amount) < 1024:
			return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
		amount /= 1024
	return f"{amount:.1f} {units[-1]}"


def format_byte_progress(position: int, total: int | None) -> str:
	"""Format byte progress without rounding an incomplete transfer to done."""
	if total is None:
		return format_bytes(position)
	position_text = format_bytes(position)
	total_text = format_bytes(total)
	if position < total and position_text == total_text:
		position_text = (
			f"{position_text} ({format_bytes(total - position)} remaining)"
		)
	return f"{position_text}/{total_text}"


def _sync_object(path: Path, key: str) -> SyncObject:
	stat = path.stat()
	return SyncObject(path, key, stat.st_size, stat.st_mtime_ns)


def inventory_sharded_tree(
	source_folder: Path,
	target_folder: str,
	scales: list[tuple[int, str, Path]],
) -> list[list[SyncObject]]:
	"""Enumerate root metadata and selected scale files exactly once."""
	root_objects = [
		_sync_object(path, _join_key(target_folder, path.name))
		for path in sorted(source_folder.iterdir())
		if path.is_file()
	]
	groups = [root_objects] if root_objects else []
	for _mip, key, scale_path in scales:
		scale_objects = [
			_sync_object(
				path,
				_join_key(
					target_folder,
					key,
					path.relative_to(scale_path).as_posix(),
				),
			)
			for path in sorted(scale_path.rglob("*"))
			if path.is_file()
		]
		if scale_objects:
			groups.append(scale_objects)
	return groups


def _object_matches(client, item: SyncObject, bucket_name: str) -> bool:
	try:
		remote = client.head_object(Bucket=bucket_name, Key=item.key)
	except ClientError as exc:
		code = str(exc.response.get("Error", {}).get("Code", ""))
		if code in {"404", "NoSuchKey", "NotFound"}:
			return False
		raise
	metadata = remote.get("Metadata", {})
	return (
		remote.get("ContentLength") == item.size
		and metadata.get("mctutil-size") == item.fingerprint["mctutil-size"]
		and metadata.get("mctutil-mtime-ns") == item.fingerprint["mctutil-mtime-ns"]
	)


def upload_incremental_file(
	client,
	item: SyncObject,
	bucket_name: str,
	execute: bool,
	progress_events: Queue,
) -> str:
	if not execute:
		progress_events.put(item.size)
		return "planned"
	if _object_matches(client, item, bucket_name):
		progress_events.put(item.size)
		return "skipped"
	transfer_progress = ObjectTransferProgress(item.size, progress_events)
	try:
		client.upload_file(
			str(item.path),
			bucket_name,
			item.key,
			ExtraArgs={"Metadata": item.fingerprint},
			Callback=transfer_progress,
		)
	except Exception as exc:
		raise RuntimeError(
			f"failed to upload s3://{bucket_name}/{item.key}: {exc}"
		) from exc
	transfer_progress.complete()
	return "uploaded"


def _sync_object_group(
	client,
	items: list[SyncObject],
	bucket_name: str,
	execute: bool,
	progress_events: Queue,
) -> SyncSummary:
	summary = SyncSummary()
	for item in items:
		status = upload_incremental_file(
			client,
			item,
			bucket_name,
			execute,
			progress_events,
		)
		summary.add(status, item.size, item.key)
	return summary


def _drain_progress_events(
	progress_events: Queue,
	progress,
	*,
	wait: bool,
	max_events: int | None = 1000,
) -> None:
	handled = 0
	while max_events is None or handled < max_events:
		try:
			if wait and handled == 0:
				delta = progress_events.get(timeout=0.1)
			else:
				delta = progress_events.get_nowait()
		except ProgressQueueEmpty:
			return
		progress.update(delta)
		handled += 1


def _execute_sync_groups(
	client,
	groups: list[list[SyncObject]],
	bucket_name: str,
	execute: bool,
	jobs: int,
	progress_events: Queue,
	progress,
) -> SyncSummary:
	summary = SyncSummary()
	if not groups:
		return summary
	max_workers = min(len(groups), jobs + 1)
	record_active_workers(max_workers)
	try:
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			pending = {
				executor.submit(
					_sync_object_group,
					client,
					group,
					bucket_name,
					execute,
					progress_events,
				)
				for group in groups
			}
			while pending:
				_drain_progress_events(
					progress_events,
					progress,
					wait=True,
				)
				completed = {
					future
					for future in pending
					if future.done()
				}
				for future in completed:
					summary.merge(future.result())
				pending.difference_update(completed)
	finally:
		_drain_progress_events(
			progress_events,
			progress,
			wait=False,
			max_events=None,
		)
	return summary


def _summary_statement(summary: SyncSummary) -> str:
	return (
		"sharded sync summary: "
		f"uploaded={summary.counts['uploaded']} "
		f"({format_bytes(summary.bytes['uploaded'])}), "
		f"unchanged={summary.counts['skipped']} "
		f"({format_bytes(summary.bytes['skipped'])}), "
		f"planned={summary.counts['planned']} "
		f"({format_bytes(summary.bytes['planned'])})"
	)


def _emit_sync_details(summary: SyncSummary) -> None:
	for status, key in sorted(summary.details, key=lambda detail: detail[1]):
		label = "unchanged" if status == "skipped" else status
		log.write("S3 Upload", f"{label}: {key}", log_level=LOG.DEBUG)


def upload_sharded_tree(
	source_folder: Path,
	target_folder,
	bucket_name: str,
	jobs: int = 6,
	include_mip0: bool = True,
	execute: bool = False,
	aws_profile: str | None = None,
) -> dict[str, int]:
	"""Incrementally upload root metadata and selected sharded scale dirs."""
	aws_profile = configure_aws_profile(aws_profile, bucket_name)
	source_folder = Path(source_folder)
	scales = read_sharded_scales(source_folder, include_mip0=include_mip0)
	target_folder = str(target_folder).strip("/")
	groups = inventory_sharded_tree(source_folder, target_folder, scales)
	items = [
		item
		for group in groups
		for item in group
	]
	total_bytes = sum(item.size for item in items)
	client = _get_session(aws_profile).client("s3") if execute else None
	progress_events = Queue()
	start_message = (
		f"Synchronizing {len(items)} object(s), "
		f"{format_bytes(total_bytes)}, with up to {jobs} scale worker(s)."
	)
	if not execute:
		start_message = (
			f"Dry run: planning {len(items)} object(s), "
			f"{format_bytes(total_bytes)}; no S3 writes."
		)
	with log.progress(
		"S3 Sync",
		length=total_bytes,
		start_message=start_message,
		final_message=None,
		position_formatter=format_byte_progress,
	) as progress:
		summary = _execute_sync_groups(
			client,
			groups,
			bucket_name,
			execute,
			jobs,
			progress_events,
			progress,
		)
	_emit_sync_details(summary)
	log.write(
		"S3 Upload",
		_summary_statement(summary),
		log_level=LOG.STATUS,
	)
	return summary.counts


@click.command()
@click.option("-p", "--bucket-prefix", type=click.Path(path_type=Path), required=True)
@click.option("-n", "--bucket-name", type=click.STRING, required=True, help="Name of target s3 bucket.")
@click.option("--aws-profile", help="Named AWS profile for upload and optional S3 mesh.")
@click.option("-t", "--process-count", type=click.IntRange(min=1), default=60,
				help="Number of simultaneous uploads.")
@click.option("--mesh", type=click.BOOL, is_flag=True, show_default=True, default=False,
				help="Whether to mesh the resulting upload.")
@click.option(
	"--from-sharded-tree",
	is_flag=True,
	help="Upload only root metadata and declared sharded scale directories.",
)
@click.option("--include-mip0/--exclude-mip0", default=True, show_default=True)
@click.option("--jobs", type=click.IntRange(min=1), default=6, show_default=True,
				help="Parallel scale-directory uploads in sharded-tree mode.")
@click.option(
	"--execute/--dry-run",
	default=None,
	help=(
		"Override execution. Legacy uploads execute by default; "
		"--from-sharded-tree plans by default."
	),
)
@click.argument("SOURCE_FOLDER", nargs=1, type=click.Path(exists=True, path_type=Path))
@click.argument("TARGET_FOLDER", nargs=1, type=click.Path(path_type=Path))
def s3upload(
	bucket_prefix,
	bucket_name,
	process_count,
	mesh,
	execute,
	source_folder,
	target_folder,
	from_sharded_tree=False,
	include_mip0=True,
	jobs=6,
	aws_profile=None,
):
	if execute is None:
		execute = not from_sharded_tree

	aws_profile = configure_aws_profile(aws_profile, bucket_name)
	target_full = bucket_prefix.joinpath(target_folder)

	log.write("S3 Upload", f"target bucket: {bucket_name}", log_level=LOG.STATUS)
	log.write("S3 Upload", f"target folder: {target_full}", log_level=LOG.STATUS)
	log.write("S3 Upload", f"AWS profile: {aws_profile}", log_level=LOG.STATUS)

	if execute and not from_sharded_tree:
		s3 = _get_session(aws_profile).client('s3')
		s3.put_object(Bucket=bucket_name, Key=f"{target_full}/")
	elif not execute:
		log.write("S3 Upload", f"Would create prefix s3://{bucket_name}/{target_full}/", log_level=LOG.INFO)

	if from_sharded_tree:
		try:
			upload_sharded_tree(
				source_folder,
				target_full,
				bucket_name,
				jobs=jobs,
				include_mip0=include_mip0,
				execute=execute,
				aws_profile=aws_profile,
			)
		except click.ClickException:
			raise
		except Exception as exc:
			raise click.ClickException(str(exc)) from exc
	else:
		upload_folder_to_s3_parallel(
			source_folder,
			target_full,
			bucket_name,
			num_processes=process_count,
			execute=execute,
			aws_profile=aws_profile,
		)

	if mesh:
		mesh_path = f"precomputed://s3://{bucket_name}/{target_full}"
		log.write("S3 Upload", f"full remote path: {mesh_path}", log_level=LOG.STATUS)
		build_mesh(
			mesh_path,
			mip=0,
			num_lod=4,
			parallel=max(1, process_count // 4),
			execute=execute,
			aws_profile=aws_profile,
		)


if __name__ == '__main__':
	s3upload()
