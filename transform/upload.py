import json
import os
from concurrent.futures import ProcessPoolExecutor

import boto3
import click


def upload_file_to_s3(session, file_path, key, bucket_name, content_encoding):
	s3 = session.client('s3')
	if file_path.endswith(os.sep):  # Handle directory
		s3.put_object(Bucket=bucket_name, Key=key)
	else:  # Handle file
		extra_args = {}
		if content_encoding is not None:
			extra_args['ContentEncoding'] = content_encoding
		s3.upload_file(file_path, bucket_name, key, ExtraArgs=extra_args)
	print(f'uploaded: {key}')


@click.command()
@click.option("-s", "--source-folder", type=click.Path(), required=True, help="Data folder to upload.")
@click.option("-t", "--target-folder", type=click.Path(readable=False), required=True, help="Target folder name on S3")
@click.option("-b", "--bucket", type=click.STRING, required=True, help="Name of S3 bucket to upload to.")
@click.option("-s", "--secret-json", type=click.File(), required=True,
				help="Location of json file to load AWS credentials from.")
@click.option("-p", "--process-count", type=click.INT, default=60, show_default=True, help="Simultaneous Uploads.")
def upload(source_folder, target_folder, bucket, secret_json, process_count):

	# Loads credentials from JSON file (which will not be committed).
	credentials = json.load(secret_json)
	session = boto3.Session(
		aws_access_key_id=credentials['aws_access_key_id'],
		aws_secret_access_key=credentials['aws_secret_access_key']
	)

	with ProcessPoolExecutor(max_workers=process_count) as executor:
		for root, dirs, files in os.walk(source_folder):
			for dir_name in dirs:
				dir_path = os.path.join(root, dir_name)
				key = os.path.join(target_folder, os.path.relpath(dir_path, source_folder) + '/')
				executor.submit(upload_file_to_s3, session, dir_path, key, bucket, None)
			for file_name in files:
				file_path = os.path.join(root, file_name)
				key = os.path.join(target_folder, os.path.relpath(file_path, source_folder).replace(os.sep, '/'))
				content_encoding = 'gzip' if file_name != 'info' else None
				executor.submit(upload_file_to_s3, session, file_path, key, bucket, content_encoding)


if __name__ == '__main__':
	upload()
