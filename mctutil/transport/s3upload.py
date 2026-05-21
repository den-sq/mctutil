import os
from pathlib import Path
from taskqueue import LocalTaskQueue

from concurrent.futures import ProcessPoolExecutor
from botocore.exceptions import ClientError

import boto3
import click
import igneous.task_creation as tc


from mctutil.shared import log

_session = None


def _get_session():
	global _session
	if _session is None:
		_session = boto3.Session(profile_name='chenglab')
	return _session


def upload_file_to_s3(file_path, key, bucket_name, content_encoding, execute=True):
	if not execute:
		log.log("S3 Upload", f"Would upload {file_path} -> s3://{bucket_name}/{key}", log_level=log.DEBUG.INFO)
		return

	s3 = _get_session().client('s3')
	if file_path.is_dir():  # Handle directory
		try:
			s3.put_object(Bucket=bucket_name, Key=f"{key}/")
		except ClientError as e:
			log.log("S3 Upload", f"ClientError: {e.response}", log_level=log.DEBUG.ERROR)
		except Exception as e:
			log.log("S3 Upload", f"{e}", log_level=log.DEBUG.ERROR)
	else:  # Handle file
		extra_args = {}
		if content_encoding is not None:
			extra_args['ContentEncoding'] = content_encoding
		try:
			s3.upload_file(file_path, bucket_name, str(key), ExtraArgs=extra_args)
		except ClientError as e:
			log.log("S3 Upload", f"ClientError: {e.response}", log_level=log.DEBUG.ERROR)
		except Exception as e:
			log.log("S3 Upload", f"{e}", log_level=log.DEBUG.ERROR)
	log.log("S3 Upload", f"uploaded: {key}", log_level=log.DEBUG.STATUS)


def upload_folder_to_s3_parallel(folder_path, target_folder, bucket_name, num_processes, execute=True):
	with ProcessPoolExecutor(max_workers=num_processes) as executor:
		for root, dirs, files in os.walk(str(folder_path)):
			for dir_name in dirs:
				dir_path = Path(root).joinpath(dir_name)
				key = target_folder.joinpath(dir_path.relative_to(folder_path))
				executor.submit(upload_file_to_s3, dir_path, key, bucket_name, None, execute)
			for file_name in files:
				file_path = Path(root).joinpath(file_name)
				key = target_folder.joinpath(file_path.relative_to(folder_path))
				content_encoding = 'gzip' if file_name != 'info' else None
				executor.submit(upload_file_to_s3, file_path, key, bucket_name, content_encoding, execute)


@click.command()
@click.option("-p", "--bucket-prefix", type=click.Path(path_type=Path), required=True)
@click.option("-n", "--bucket-name", type=click.STRING, required=True, help="Name of target s3 bucket.")
@click.option("-t", "--process-count", type=click.INT, default=60, help="Number of simultaneous uploads.")
@click.option("--mesh", type=click.BOOL, is_flag=True, show_default=True, default=False,
				help="Whether to mesh the resulting upload.")
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually upload (and optionally mesh) or just plan the operations.")
@click.argument("SOURCE_FOLDER", nargs=1, type=click.Path(exists=True, path_type=Path))
@click.argument("TARGET_FOLDER", nargs=1, type=click.Path(path_type=Path))
def s3upload(bucket_prefix, bucket_name, process_count, mesh, execute, source_folder, target_folder):

	target_full = bucket_prefix.joinpath(target_folder)

	log.log("S3 Upload", f"target bucket: {bucket_name}", log_level=log.DEBUG.STATUS)
	log.log("S3 Upload", f"target folder: {target_full}", log_level=log.DEBUG.STATUS)

	if execute:
		s3 = _get_session().client('s3')
		s3.put_object(Bucket=bucket_name, Key=f"{target_full}/")
	else:
		log.log("S3 Upload", f"Would create prefix s3://{bucket_name}/{target_full}/", log_level=log.DEBUG.INFO)

	upload_folder_to_s3_parallel(source_folder, target_full, bucket_name, num_processes=process_count, execute=execute)

	if mesh:
		mesh_path = f"precomputed://s3://{bucket_name}/{target_full}"
		log.log("S3 Upload", f"full remote path: {mesh_path}", log_level=log.DEBUG.STATUS)
		if execute:
			tq = LocalTaskQueue(parallel=process_count // 4)
			tq.insert(tc.create_meshing_tasks(mesh_path, mip=0))
			tq.execute()
			tq.insert(tc.create_unsharded_multires_mesh_tasks(mesh_path, num_lod=4))
			tq.execute()
		else:
			log.log("S3 Upload", f"Would mesh {mesh_path}", log_level=log.DEBUG.INFO)


if __name__ == '__main__':
	s3upload()
