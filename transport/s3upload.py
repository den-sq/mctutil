import os
import sys
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

import boto3
import click

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402

session = boto3.Session(profile_name='chenglab')


def upload_file_to_s3(file_path, key, bucket_name, content_encoding):
	s3 = session.client('s3')
	if file_path.is_dir():  # Handle directory
		s3.put_object(Bucket=bucket_name, Key=f"{key}/")
	else:  # Handle file
		extra_args = {}
		if content_encoding is not None:
			extra_args['ContentEncoding'] = content_encoding
		s3.upload_file(file_path, bucket_name, str(key), ExtraArgs=extra_args)
	print(f'uploaded: {key}')


def upload_folder_to_s3_parallel(folder_path, target_folder, bucket_name, num_processes):
	with ProcessPoolExecutor(max_workers=num_processes) as executor:
		for root, dirs, files in os.walk(str(folder_path)):
			for dir_name in dirs:
				dir_path = Path(root).joinpath(dir_name)
				key = target_folder.joinpath(dir_path.relative_to(folder_path))
				executor.submit(upload_file_to_s3, dir_path, key, bucket_name, None)
			for file_name in files:
				file_path = Path(root).joinpath(file_name)
				key = target_folder.joinpath(file_path.relative_to(folder_path))
				content_encoding = 'gzip' if file_name != 'info' else None
				executor.submit(upload_file_to_s3, file_path, key, bucket_name, content_encoding)


@click.command()
@click.option("-p", "--bucket-prefix", type=click.Path(path_type=Path), required=True)
@click.option("-n", "--bucket-name", type=click.STRING, required=True, help="Name of target s3 bucket.")
@click.option("-t", "--process-count", type=click.INT, default=60, help="Number of simultaneous uploads.")
@click.argument("SOURCE_FOLDER", nargs=1, type=click.Path(exists=True, path_type=Path))
@click.argument("TARGET_FOLDER", nargs=1, type=click.Path(path_type=Path))
def s3upload(bucket_prefix, bucket_name, process_count, source_folder, target_folder):

	target_full = bucket_prefix.joinpath(target_folder)

	print(f'target bucket: {bucket_name}')
	print(f'target folder: {target_full}')

	upload_folder_to_s3_parallel(source_folder, target_full, bucket_name, num_processes=process_count)


if __name__ == '__main__':
	s3upload()
