from pathlib import Path

import click
import pytest

from mctutil.shared import deps


def test_require_returns_imported_modules(monkeypatch):
	loaded = object()
	monkeypatch.setattr(
		deps.importlib,
		"import_module",
		lambda name: loaded if name == "h5py" else None,
	)

	assert deps.require("h5py", "transform") is loaded


def test_require_uses_canonical_hint(monkeypatch):
	def missing(_name):
		raise ImportError("missing")

	monkeypatch.setattr(deps.importlib, "import_module", missing)

	with pytest.raises(click.ClickException) as error:
		deps.require(
			"tifffile",
			"als832",
			purpose="TIFF support is required",
			error_type=click.ClickException,
		)

	assert str(error.value) == (
		"TIFF support is required; install with pip install -e '.[als832]'"
	)


def test_require_rejects_module_extra_drift():
	with pytest.raises(ValueError, match="not declared"):
		deps.require("boto3", "ng")


def test_publish_consumes_shared_dependency_table(load_module):
	publish = load_module("mctutil/ng/publish.py")

	assert publish.EXTRA_MODULES is deps.EXTRA_MODULES
	assert publish.install_command(("ng", "mesh")) == (
		"pip install -e '.[ng,mesh]'"
	)


def test_commands_do_not_embed_install_commands():
	offenders = []
	for path in Path("mctutil").rglob("*.py"):
		if path == Path("mctutil/shared/deps.py"):
			continue
		if "pip install" in path.read_text(encoding="utf-8"):
			offenders.append(str(path))

	assert offenders == []
