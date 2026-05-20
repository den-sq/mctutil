from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

import pytest


@dataclass
class WorkspacePaths:
	root: Path
	input_dir: Path
	output_dir: Path


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspacePaths:
	input_dir = tmp_path / "input"
	output_dir = tmp_path / "output"
	input_dir.mkdir()
	output_dir.mkdir()
	return WorkspacePaths(root=tmp_path, input_dir=input_dir, output_dir=output_dir)


class _DummySession:
	def client(self, *_args, **_kwargs):
		return types.SimpleNamespace(put_object=lambda **_kw: None, upload_file=lambda *_a, **_kw: None)


def _stub_modules() -> dict[str, types.ModuleType]:
	boto3 = types.ModuleType("boto3")
	boto3.Session = lambda *args, **kwargs: _DummySession()

	botocore = types.ModuleType("botocore")
	botocore_exceptions = types.ModuleType("botocore.exceptions")
	botocore_exceptions.ClientError = type("ClientError", (Exception,), {})
	botocore.exceptions = botocore_exceptions

	cloudvolume = types.ModuleType("cloudvolume")
	cloudvolume.CloudVolume = type("CloudVolume", (), {})

	config = types.ModuleType("config")
	python_console = types.ModuleType("config.pythonConsoleAutoImport")
	python_console.List = type("List", (), {})
	python_console.Managed = type(
		"Managed",
		(),
		{
			"getAllObjectsOfClassAndTitle": staticmethod(
				lambda *_args, **_kwargs: [types.SimpleNamespace(getAsNDArray=lambda *_a: None)]
			),
		},
	)
	python_console.orsObj = lambda *_args, **_kwargs: types.SimpleNamespace(getAsNDArray=lambda *_a: None)
	python_console.roi = types.SimpleNamespace(getTitle=lambda: "roi")
	python_console.Progress = type("Progress", (), {})
	config.pythonConsoleAutoImport = python_console

	dicom2jpg = types.ModuleType("dicom2jpg")
	dicom2jpg.dicom2tiff = lambda *_args, **_kwargs: None

	brotli = types.ModuleType("brotli")

	cv2 = types.ModuleType("cv2")
	tomopy = types.ModuleType("tomopy")
	tomopy.circ_mask = lambda *args, **kwargs: None

	igneous = types.ModuleType("igneous")
	igneous_task_creation = types.ModuleType("igneous.task_creation")
	igneous_task_creation.create_meshing_tasks = lambda *_args, **_kwargs: []
	igneous_task_creation.create_unsharded_multires_mesh_tasks = lambda *_args, **_kwargs: []
	igneous_task_creation.create_mesh_manifest_tasks = lambda *_args, **_kwargs: []
	igneous.task_creation = igneous_task_creation

	gdal = types.SimpleNamespace(UseExceptions=lambda: None)

	google = types.ModuleType("google")
	google_auth_oauthlib = types.ModuleType("google_auth_oauthlib")
	google_auth_oauthlib_flow = types.ModuleType("google_auth_oauthlib.flow")
	google_auth_oauthlib.flow = google_auth_oauthlib_flow
	googleapiclient = types.ModuleType("googleapiclient")
	googleapiclient_discovery = types.ModuleType("googleapiclient.discovery")
	googleapiclient.discovery = googleapiclient_discovery

	ipyslurm = types.ModuleType("ipyslurm")
	ipyslurm.Slurm = type(
		"Slurm",
		(),
		{
			"login": lambda *_args, **_kwargs: None,
			"command": lambda *_args, **_kwargs: "",
			"sbatch": lambda *_args, **_kwargs: "job",
		},
	)

	neuroglancer_scripts = types.ModuleType("neuroglancer_scripts")
	neuroglancer_scripts_scripts = types.ModuleType("neuroglancer_scripts.scripts")
	neuroglancer_generate = types.ModuleType("neuroglancer_scripts.scripts.generate_scales_info")
	neuroglancer_generate.generate_scales_info = lambda *_args, **_kwargs: None
	neuroglancer_slices = types.ModuleType("neuroglancer_scripts.scripts.slices_to_precomputed")
	neuroglancer_slices.convert_slices_in_directory = lambda *_args, **_kwargs: None
	neuroglancer_compute = types.ModuleType("neuroglancer_scripts.scripts.compute_scales")
	neuroglancer_compute.compute_scales = lambda *_args, **_kwargs: None
	neuroglancer_scripts.scripts = neuroglancer_scripts_scripts

	osgeo = types.ModuleType("osgeo")
	osgeo.gdal = gdal
	osgeo_gdal = types.ModuleType("osgeo.gdal")
	osgeo_gdal.UseExceptions = lambda: None

	skimage = types.ModuleType("skimage")
	skimage_restoration = types.ModuleType("skimage.restoration")
	skimage_restoration.denoise_nl_means = lambda *_args, **_kwargs: None
	skimage_restoration.estimate_sigma = lambda *_args, **_kwargs: None
	skimage.restoration = skimage_restoration

	taskqueue = types.ModuleType("taskqueue")
	taskqueue.LocalTaskQueue = type("LocalTaskQueue", (), {})

	return {
		"boto3": boto3,
		"brotli": brotli,
		"botocore": botocore,
		"botocore.exceptions": botocore_exceptions,
		"cloudvolume": cloudvolume,
		"config": config,
		"config.pythonConsoleAutoImport": python_console,
		"cv2": cv2,
		"dicom2jpg": dicom2jpg,
		"google": google,
		"google_auth_oauthlib": google_auth_oauthlib,
		"google_auth_oauthlib.flow": google_auth_oauthlib_flow,
		"googleapiclient": googleapiclient,
		"googleapiclient.discovery": googleapiclient_discovery,
		"igneous": igneous,
		"igneous.task_creation": igneous_task_creation,
		"ipyslurm": ipyslurm,
		"neuroglancer_scripts": neuroglancer_scripts,
		"neuroglancer_scripts.scripts": neuroglancer_scripts_scripts,
		"neuroglancer_scripts.scripts.compute_scales": neuroglancer_compute,
		"neuroglancer_scripts.scripts.generate_scales_info": neuroglancer_generate,
		"neuroglancer_scripts.scripts.slices_to_precomputed": neuroglancer_slices,
		"osgeo": osgeo,
		"osgeo.gdal": osgeo_gdal,
		"skimage": skimage,
		"skimage.restoration": skimage_restoration,
		"taskqueue": taskqueue,
		"tomopy": tomopy,
	}


@pytest.fixture()
def load_module(monkeypatch: pytest.MonkeyPatch):
	for name, module in _stub_modules().items():
		monkeypatch.setitem(sys.modules, name, module)

	loaded: list[str] = []

	def _load(module_path: str):
		module_name = module_path.replace('/', '.').removesuffix('.py') + '.__smoke__'
		spec = importlib.util.spec_from_file_location(module_name, Path(module_path))
		module = importlib.util.module_from_spec(spec)
		assert spec.loader is not None
		sys.modules[module_name] = module
		loaded.append(module_name)
		spec.loader.exec_module(module)
		return module

	yield _load

	for module_name in loaded:
		sys.modules.pop(module_name, None)
