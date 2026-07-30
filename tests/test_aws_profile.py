from __future__ import annotations

import os
from pathlib import Path
import types

import click
from click.testing import CliRunner
import pytest

from mctutil.shared import aws


def clean_environment(tmp_path: Path, **values: str) -> dict[str, str]:
	return {
		"HOME": str(tmp_path),
		**values,
	}


def test_profile_resolution_precedence_and_export(tmp_path):
	environment = clean_environment(tmp_path, AWS_PROFILE="environment-profile")

	assert aws.resolve_aws_profile("cli-profile", environment) == "cli-profile"
	assert aws.resolve_aws_profile(None, environment) == "environment-profile"
	assert aws.resolve_aws_profile(None, clean_environment(tmp_path)) == "chenglab"
	assert aws.configure_aws_profile(
		"cli-profile",
		"bucket",
		environment,
	) == "cli-profile"
	assert environment["AWS_PROFILE"] == "cli-profile"


@pytest.mark.parametrize(
	"relative_path",
	(
		Path(".cloudvolume/secrets/bucket-aws-secret.json"),
		Path(".cloudvolume/secrets/aws-secret.json"),
		Path(".cloudfiles/secrets/bucket-aws-secret.json"),
		Path(".cloudfiles/secrets/aws-secret.json"),
	),
)
def test_profile_refuses_legacy_cloudfiles_secrets(tmp_path, relative_path):
	secret = tmp_path / relative_path
	secret.parent.mkdir(parents=True, exist_ok=True)
	secret.write_text("{}", encoding="utf-8")
	environment = clean_environment(tmp_path)

	with pytest.raises(click.ClickException, match=str(secret)):
		aws.configure_aws_profile("selected", "bucket", environment)

	assert "AWS_PROFILE" not in environment


@pytest.mark.parametrize("variable", aws.RAW_AWS_CREDENTIAL_VARIABLES)
def test_profile_refuses_raw_credential_environment_variables(tmp_path, variable):
	environment = clean_environment(tmp_path, **{variable: "legacy-value"})

	with pytest.raises(click.ClickException, match=variable):
		aws.configure_aws_profile("selected", "bucket", environment)

	assert "AWS_PROFILE" not in environment


def test_s3_info_preflight_reads_and_closes_metadata(monkeypatch):
	class Body:
		closed = False

		def read(self):
			return b'{"scales": [{"key": "700_700_700"}]}'

		def close(self):
			self.closed = True

	body = Body()
	requests = []

	class Client:
		def get_object(self, **kwargs):
			requests.append(kwargs)
			return {"Body": body}

	class Session:
		def client(self, service):
			assert service == "s3"
			return Client()

	monkeypatch.setattr(
		aws,
		"create_boto3_session",
		lambda profile: Session() if profile == "selected" else None,
	)

	info = aws.preflight_s3_info(
		"precomputed://s3://bucket/layer",
		"selected",
	)

	assert info["scales"][0]["key"] == "700_700_700"
	assert requests == [{"Bucket": "bucket", "Key": "layer/info"}]
	assert body.closed


def test_s3_mesh_adds_aws_dependency_without_upload(load_module):
	module = load_module("mctutil/ng/publish.py")
	stages = ("mesh",)

	assert module.mesh_uses_s3(stages, "s3", 0, True)
	assert module.required_extras(stages, s3_mesh=True) == ("mesh", "aws")
	assert not module.mesh_uses_s3(stages, "local", 0, True)
	assert module.required_extras(stages) == ("mesh",)


@pytest.mark.parametrize(
	(
		"stages",
		"mesh_at",
		"mesh_mip",
		"include_mip0",
		"expected_s3",
	),
	(
		(("mesh",), "s3", 0, True, True),
		(("upload", "mesh"), "auto", 0, True, True),
		(("mesh",), "auto", 0, True, False),
		(("upload", "mesh"), "auto", 0, False, False),
		(("upload", "mesh"), "s3", 1, False, True),
	),
)
def test_mesh_target_matches_s3_dependency_planning(
	load_module,
	tmp_path,
	stages,
	mesh_at,
	mesh_mip,
	include_mip0,
	expected_s3,
):
	module = load_module("mctutil/ng/publish.py")
	plan = types.SimpleNamespace(
		dataset=tmp_path / "sample",
		staged=tmp_path / "sample_precomputed_sharded_local",
	)
	options = {
		"effective_stages": stages,
		"mesh_at": mesh_at,
		"mesh_mip": mesh_mip,
		"upload_include_mip0": include_mip0,
		"s3_prefix": "s3://bucket/prefix",
	}

	target = module.mesh_target(plan, options)

	assert module.mesh_uses_s3(
		stages,
		mesh_at,
		mesh_mip,
		include_mip0,
	) is expected_s3
	assert target.startswith("precomputed://s3://") is expected_s3


def test_publish_reports_aws_for_s3_mesh_without_upload(
	load_module,
	monkeypatch,
	tmp_path,
):
	module = load_module("mctutil/ng/publish.py")
	root = tmp_path / "root"
	dataset = root / "cell_labels"
	dataset.mkdir(parents=True)
	precomputed = root / "cell_labels_precomputed"
	precomputed.mkdir()
	(precomputed / "info").write_text('{"scales":[{}]}', encoding="utf-8")
	monkeypatch.setenv("HOME", str(tmp_path))
	for variable in aws.RAW_AWS_CREDENTIAL_VARIABLES:
		monkeypatch.delenv(variable, raising=False)
	monkeypatch.setattr(module, "module_available", lambda _name: True)

	result = CliRunner().invoke(
		module.publish,
		[
			str(root),
			"--start-at", "mesh",
			"--no-upload",
			"--mesh-at", "s3",
			"--s3-prefix", "s3://bucket/prefix",
			"--aws-profile", "selected",
			"--dry-run",
		],
	)

	assert result.exit_code == 0, result.output
	assert "Required extras: [mesh], [aws]" in result.output
	assert "AWS profile: selected" in result.output


def test_local_mesh_never_resolves_an_aws_profile(load_module, monkeypatch):
	module = load_module("mctutil/shared/mesh.py")
	monkeypatch.setattr(
		module,
		"configure_aws_profile",
		lambda *_args: (_ for _ in ()).throw(
			AssertionError("AWS profile resolved for a local mesh")
		),
	)
	monkeypatch.setattr(module.log, "write", lambda *_args, **_kwargs: None)

	module.build_mesh(
		"precomputed://file:///tmp/layer",
		parallel=1,
		execute=False,
		aws_profile="unused",
	)


def test_s3_mesh_authentication_fails_before_loading_igneous(
	load_module,
	monkeypatch,
):
	module = load_module("mctutil/shared/mesh.py")
	monkeypatch.setattr(
		module,
		"configure_aws_profile",
		lambda _profile, _bucket: "selected",
	)
	monkeypatch.setattr(
		module,
		"preflight_s3_info",
		lambda *_args: (_ for _ in ()).throw(
			click.ClickException("authentication failed")
		),
	)
	monkeypatch.setattr(
		module,
		"_require_mesh_dependencies",
		lambda: (_ for _ in ()).throw(
			AssertionError("Igneous loaded before S3 preflight")
		),
	)
	monkeypatch.setattr(module.log, "write", lambda *_args, **_kwargs: None)

	with pytest.raises(click.ClickException, match="authentication failed"):
		module.build_mesh(
			"precomputed://s3://bucket/layer",
			parallel=1,
		)


def test_s3_mesh_queue_workers_inherit_selected_profile(
	load_module,
	monkeypatch,
	tmp_path,
):
	module = load_module("mctutil/shared/mesh.py")
	monkeypatch.setenv("HOME", str(tmp_path))
	for variable in aws.RAW_AWS_CREDENTIAL_VARIABLES:
		monkeypatch.delenv(variable, raising=False)
	monkeypatch.setattr(
		module,
		"preflight_s3_info",
		lambda *_args: {"scales": [{}]},
	)
	task_creation = types.SimpleNamespace(
		create_meshing_tasks=lambda *_args, **_kwargs: ["forge"],
		create_unsharded_multires_mesh_tasks=lambda *_args, **_kwargs: ["merge"],
	)
	monkeypatch.setattr(
		module,
		"_require_mesh_dependencies",
		lambda: (object, task_creation),
	)
	worker_profiles = []

	def run_tasks(_queue, _fingerprint, tasks_factory, *_args):
		worker_profiles.append(os.environ.get("AWS_PROFILE"))
		assert list(tasks_factory())

	monkeypatch.setattr(module, "run_persistent_tasks", run_tasks)
	monkeypatch.setattr(module.log, "write", lambda *_args, **_kwargs: None)

	module.build_mesh(
		"precomputed://s3://bucket/layer",
		parallel=1,
		queue_dir=tmp_path / "queue",
		aws_profile="selected",
	)

	assert worker_profiles == ["selected", "selected"]


def test_legacy_upload_processes_receive_selected_profile(
	load_module,
	monkeypatch,
	tmp_path,
):
	module = load_module("mctutil/transport/s3upload.py")
	source = tmp_path / "source"
	source.mkdir()
	(source / "info").write_text("{}", encoding="utf-8")
	submissions = []

	class RecordingExecutor:
		def __init__(self, max_workers):
			assert max_workers == 2

		def __enter__(self):
			return self

		def __exit__(self, *_args):
			return False

		def submit(self, function, *args):
			submissions.append(
				(function, args, os.environ.get("AWS_PROFILE"))
			)

	monkeypatch.setenv("HOME", str(tmp_path))
	for variable in aws.RAW_AWS_CREDENTIAL_VARIABLES:
		monkeypatch.delenv(variable, raising=False)
	monkeypatch.setattr(module, "ProcessPoolExecutor", RecordingExecutor)
	monkeypatch.setattr(
		module,
		"_get_session",
		lambda _profile: (_ for _ in ()).throw(
			AssertionError("dry-run constructed an S3 session")
		),
	)

	result = CliRunner().invoke(
		module.s3upload,
		[
			"--bucket-prefix", "prefix",
			"--bucket-name", "bucket",
			"--aws-profile", "selected",
			"--process-count", "2",
			"--dry-run",
			str(source),
			"target",
		],
	)

	assert result.exit_code == 0, result.output
	assert len(submissions) == 1
	_function, arguments, inherited_profile = submissions[0]
	assert arguments[-1] == "selected"
	assert inherited_profile == "selected"
