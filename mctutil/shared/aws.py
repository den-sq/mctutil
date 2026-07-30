"""Deterministic AWS profile selection for mctutil cloud workflows."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import click


DEFAULT_AWS_PROFILE = "chenglab"
RAW_AWS_CREDENTIAL_VARIABLES = (
	"AWS_ACCESS_KEY_ID",
	"AWS_SECRET_ACCESS_KEY",
	"AWS_SESSION_TOKEN",
	"AWS_SECURITY_TOKEN",
)


def resolve_aws_profile(
	cli_profile: str | None,
	environ: Mapping[str, str] | None = None,
) -> str:
	"""Resolve CLI, environment, and default AWS profile precedence."""
	environ = os.environ if environ is None else environ
	if cli_profile is not None:
		profile = cli_profile.strip()
		if not profile:
			raise click.ClickException("--aws-profile must not be empty")
		return profile
	profile = environ.get("AWS_PROFILE", "").strip()
	return profile or DEFAULT_AWS_PROFILE


def s3_location(value: str) -> tuple[str, str] | None:
	"""Return bucket and key for s3:// and precomputed://s3:// paths."""
	raw = str(value).removeprefix("precomputed://")
	parsed = urlparse(raw)
	if parsed.scheme != "s3" or not parsed.netloc:
		return None
	return parsed.netloc, parsed.path.strip("/")


def cloudfiles_aws_secret_candidates(
	bucket: str,
	environ: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
	"""Return secret paths in the order CloudFiles resolves them."""
	environ = os.environ if environ is None else environ
	home = Path(environ.get("HOME", str(Path.home()))).expanduser()
	cloudvolume_dir = Path(
		environ.get("CLOUD_VOLUME_DIR", str(home / ".cloudvolume"))
	).expanduser()
	cloudfiles_dir = Path(
		environ.get("CLOUD_FILES_DIR", str(home / ".cloudfiles"))
	).expanduser()
	names = (f"{bucket}-aws-secret.json", "aws-secret.json")
	return tuple(
		path
		for name in names
		for path in (
			cloudvolume_dir / "secrets" / name,
			Path("/") / name,
			cloudfiles_dir / "secrets" / name,
		)
	)


def reject_legacy_aws_credentials(
	bucket: str,
	profile: str,
	environ: Mapping[str, str] | None = None,
) -> None:
	"""Refuse sources that CloudFiles would prioritize over AWS_PROFILE."""
	environ = os.environ if environ is None else environ
	for path in cloudfiles_aws_secret_candidates(bucket, environ):
		if path.is_file():
			raise click.ClickException(
				f"AWS profile '{profile}' cannot be enforced because CloudFiles "
				f"credential file exists: {path}"
			)
	for variable in RAW_AWS_CREDENTIAL_VARIABLES:
		if variable in environ:
			raise click.ClickException(
				f"AWS profile '{profile}' cannot be enforced because "
				f"{variable} is set"
			)


def configure_aws_profile(
	cli_profile: str | None,
	bucket: str,
	environ: MutableMapping[str, str] | None = None,
) -> str:
	"""Resolve, validate, and export a profile for child worker processes."""
	environ = os.environ if environ is None else environ
	profile = resolve_aws_profile(cli_profile, environ)
	reject_legacy_aws_credentials(bucket, profile, environ)
	environ["AWS_PROFILE"] = profile
	return profile


def create_boto3_session(profile: str):
	"""Construct a named Boto3 session with concise profile errors."""
	try:
		import boto3
		from botocore.exceptions import BotoCoreError
	except ImportError as exc:
		raise click.ClickException(
			"AWS support requires boto3; install with pip install -e '.[aws]'"
		) from exc

	try:
		session = boto3.Session(profile_name=profile)
		if session.get_credentials() is None:
			raise click.ClickException(
				f"AWS profile '{profile}' did not resolve any credentials"
			)
		return session
	except click.ClickException:
		raise
	except BotoCoreError as exc:
		raise click.ClickException(
			f"AWS profile '{profile}' could not be loaded: {exc}"
		) from exc


def preflight_s3_info(layer_path: str, profile: str) -> dict:
	"""Read and parse S3 precomputed metadata before creating a task queue."""
	location = s3_location(layer_path)
	if location is None:
		raise click.ClickException(f"not an S3 layer path: {layer_path}")
	bucket, prefix = location
	key = "/".join(part for part in (prefix, "info") if part)
	info_url = f"s3://{bucket}/{key}"
	session = create_boto3_session(profile)
	try:
		from botocore.exceptions import BotoCoreError, ClientError

		response = session.client("s3").get_object(Bucket=bucket, Key=key)
		body = response["Body"]
		try:
			payload = body.read()
		finally:
			close = getattr(body, "close", None)
			if close is not None:
				close()
	except (BotoCoreError, ClientError, KeyError) as exc:
		raise click.ClickException(
			f"AWS profile '{profile}' could not read {info_url}: {exc}"
		) from exc

	try:
		info = json.loads(payload)
	except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
		raise click.ClickException(
			f"S3 mesh input metadata is invalid JSON: {info_url}"
		) from exc
	if not isinstance(info, dict) or not info.get("scales"):
		raise click.ClickException(
			f"S3 mesh input metadata has no scales: {info_url}"
		)
	return info
