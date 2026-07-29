from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
from botocore.exceptions import ClientError

import boto3
import click


from mctutil.shared.log import log, LOG
from mctutil.shared.mesh import build_mesh

_session = None


def _get_session():
	global _session
	if _session is None:
		_session = boto3.Session(profile_name='chenglab')
	return _session


def upload_file_to_s3(file_path, key, bucket_name, content_encoding, execute=True):
	if not execute:
		log.write("S3 Upload", f"Would upload {file_path} -> s3://{bucket_name}/{key}", log_level=LOG.INFO)
		return

	s3 = _get_session().client('s3')
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


def upload_folder_to_s3_parallel(folder_path, target_folder, bucket_name, num_processes, execute=True):
	folder_path = Path(folder_path)
	with ProcessPoolExecutor(max_workers=num_processes) as executor:
		for entry in folder_path.rglob("*"):
			key = target_folder.joinpath(entry.relative_to(folder_path))
			if entry.is_dir():
				executor.submit(upload_file_to_s3, entry, key, bucket_name, None, execute)
			else:
				content_encoding = 'gzip' if entry.name != 'info' else None
				executor.submit(upload_file_to_s3, entry, key, bucket_name, content_encoding, execute)


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


def _local_fingerprint(path: Path) -> dict[str, str]:
	stat = path.stat()
	return {
		"mctutil-size": str(stat.st_size),
		"mctutil-mtime-ns": str(stat.st_mtime_ns),
	}


def _object_matches(client, path: Path, bucket_name: str, key: str) -> bool:
	try:
		remote = client.head_object(Bucket=bucket_name, Key=key)
	except ClientError as exc:
		code = str(exc.response.get("Error", {}).get("Code", ""))
		if code in {"404", "NoSuchKey", "NotFound"}:
			return False
		raise
	fingerprint = _local_fingerprint(path)
	metadata = remote.get("Metadata", {})
	return (
		remote.get("ContentLength") == path.stat().st_size
		and metadata.get("mctutil-size") == fingerprint["mctutil-size"]
		and metadata.get("mctutil-mtime-ns") == fingerprint["mctutil-mtime-ns"]
	)


def upload_incremental_file(
	client,
	path: Path,
	bucket_name: str,
	key: str,
	execute: bool,
) -> str:
	if not execute:
		log.write(
			"S3 Upload",
			f"Would sync {path} -> s3://{bucket_name}/{key}",
			log_level=LOG.INFO,
		)
		return "planned"
	if _object_matches(client, path, bucket_name, key):
		log.write("S3 Upload", f"unchanged: {key}", log_level=LOG.STATUS)
		return "skipped"
	client.upload_file(
		str(path),
		bucket_name,
		key,
		ExtraArgs={"Metadata": _local_fingerprint(path)},
	)
	log.write("S3 Upload", f"uploaded: {key}", log_level=LOG.STATUS)
	return "uploaded"


def _upload_scale_directory(
	client,
	scale_path: Path,
	target_folder: str,
	bucket_name: str,
	execute: bool,
) -> dict[str, int]:
	counts = {"planned": 0, "skipped": 0, "uploaded": 0}
	for path in sorted(scale_path.rglob("*")):
		if not path.is_file():
			continue
		key = _join_key(
			target_folder,
			scale_path.name,
			path.relative_to(scale_path).as_posix(),
		)
		status = upload_incremental_file(
			client,
			path,
			bucket_name,
			key,
			execute,
		)
		counts[status] += 1
	return counts


def upload_sharded_tree(
	source_folder: Path,
	target_folder,
	bucket_name: str,
	jobs: int = 6,
	include_mip0: bool = True,
	execute: bool = False,
) -> dict[str, int]:
	"""Incrementally upload root metadata and selected sharded scale dirs."""
	source_folder = Path(source_folder)
	scales = read_sharded_scales(source_folder, include_mip0=include_mip0)
	target_folder = str(target_folder).strip("/")
	client = _get_session().client("s3") if execute else None
	counts = {"planned": 0, "skipped": 0, "uploaded": 0}

	for path in sorted(source_folder.iterdir()):
		if not path.is_file():
			continue
		key = _join_key(target_folder, path.name)
		status = upload_incremental_file(
			client,
			path,
			bucket_name,
			key,
			execute,
		)
		counts[status] += 1

	with ThreadPoolExecutor(max_workers=jobs) as executor:
		futures = [
			executor.submit(
				_upload_scale_directory,
				client,
				scale_path,
				target_folder,
				bucket_name,
				execute,
			)
			for _mip, _key, scale_path in scales
		]
		for future in as_completed(futures):
			for status, count in future.result().items():
				counts[status] += count

	log.write(
		"S3 Upload",
		(
			f"sharded sync summary: uploaded={counts['uploaded']}, "
			f"unchanged={counts['skipped']}, planned={counts['planned']}"
		),
		log_level=LOG.STATUS,
	)
	return counts


@click.command()
@click.option("-p", "--bucket-prefix", type=click.Path(path_type=Path), required=True)
@click.option("-n", "--bucket-name", type=click.STRING, required=True, help="Name of target s3 bucket.")
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
@click.option('--execute/--dry-run', default=False, show_default=True,
				help="Whether to actually upload (and optionally mesh) or just plan the operations.")
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
):

	target_full = bucket_prefix.joinpath(target_folder)

	log.write("S3 Upload", f"target bucket: {bucket_name}", log_level=LOG.STATUS)
	log.write("S3 Upload", f"target folder: {target_full}", log_level=LOG.STATUS)

	if execute and not from_sharded_tree:
		s3 = _get_session().client('s3')
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
		)


if __name__ == '__main__':
	s3upload()
