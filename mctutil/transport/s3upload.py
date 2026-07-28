from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
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


@click.command()
@click.option("-p", "--bucket-prefix", type=click.Path(path_type=Path), required=True)
@click.option("-n", "--bucket-name", type=click.STRING, required=True, help="Name of target s3 bucket.")
@click.option("-t", "--process-count", type=click.IntRange(min=1), default=60,
				help="Number of simultaneous uploads.")
@click.option("--mesh", type=click.BOOL, is_flag=True, show_default=True, default=False,
				help="Whether to mesh the resulting upload.")
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually upload (and optionally mesh) or just plan the operations.")
@click.argument("SOURCE_FOLDER", nargs=1, type=click.Path(exists=True, path_type=Path))
@click.argument("TARGET_FOLDER", nargs=1, type=click.Path(path_type=Path))
def s3upload(bucket_prefix, bucket_name, process_count, mesh, execute, source_folder, target_folder):

	target_full = bucket_prefix.joinpath(target_folder)

	log.write("S3 Upload", f"target bucket: {bucket_name}", log_level=LOG.STATUS)
	log.write("S3 Upload", f"target folder: {target_full}", log_level=LOG.STATUS)

	if execute:
		s3 = _get_session().client('s3')
		s3.put_object(Bucket=bucket_name, Key=f"{target_full}/")
	else:
		log.write("S3 Upload", f"Would create prefix s3://{bucket_name}/{target_full}/", log_level=LOG.INFO)

	upload_folder_to_s3_parallel(source_folder, target_full, bucket_name, num_processes=process_count, execute=execute)

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
