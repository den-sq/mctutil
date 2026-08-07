from pathlib import Path
import re
import tomllib

from mctutil.shared.deps import EXTRA_MODULES


MODULE_DISTRIBUTIONS = {
	"RangeHTTPServer": "rangehttpserver",
	"boto3": "boto3",
	"cloudfiles": "cloud-files",
	"cloudvolume": "cloud-volume",
	"flask": "flask",
	"flask_cors": "flask-cors",
	"google": "google-auth",
	"google_auth_oauthlib": "google-auth-oauthlib",
	"googleapiclient": "google-api-python-client",
	"h5py": "h5py",
	"igneous": "igneous-pipeline",
	"neuroglancer": "neuroglancer",
	"neuroglancer_scripts": "neuroglancer-scripts",
	"qrcode": "qrcode",
	"scipy": "scipy",
	"skimage": "scikit-image",
	"taskqueue": "task-queue",
	"tifffile": "tifffile",
	"zarr": "zarr",
}


def project_metadata():
	with Path("pyproject.toml").open("rb") as source:
		return tomllib.load(source)["project"]


def environment_requirements() -> list[str]:
	contents = Path("environment.yml").read_text(encoding="utf-8")
	dependency_lines = contents.split("dependencies:\n", 1)[1].splitlines()
	return [
		line.strip()[2:]
		for line in dependency_lines
		if line.strip().startswith("- ")
		and line.strip() != "- pip:"
	]


def environment_pip_requirements() -> list[str]:
	contents = Path("environment.yml").read_text(encoding="utf-8")
	pip_lines = contents.split("  - pip:\n", 1)[1].splitlines()
	return [
		line.strip()[2:]
		for line in pip_lines
		if line.startswith("      - ")
	]


def requirement_name(requirement: str) -> str:
	return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].lower()


def requirements_by_name(requirements: list[str]) -> dict[str, str]:
	return {
		requirement_name(requirement): requirement
		for requirement in requirements
	}


def test_guard_modules_are_installed_by_their_named_extra():
	extras = project_metadata()["optional-dependencies"]

	assert set(extras) == set(EXTRA_MODULES)
	for extra, modules in EXTRA_MODULES.items():
		packages = requirements_by_name(extras[extra])
		for module_name in modules:
			root_module = module_name.split(".", 1)[0]
			assert MODULE_DISTRIBUTIONS[root_module] in packages, (
				extra,
				module_name,
			)


def test_every_pip_requirement_has_a_compatibility_range():
	project = project_metadata()
	requirements = list(project["dependencies"])
	for extra_requirements in project["optional-dependencies"].values():
		requirements.extend(extra_requirements)

	assert all(
		">=" in requirement and "<" in requirement
		for requirement in requirements
	)


def test_conda_environment_contains_the_same_pip_contract():
	project = project_metadata()
	project_requirements = list(project["dependencies"])
	for extra_requirements in project["optional-dependencies"].values():
		project_requirements.extend(extra_requirements)
	environment = requirements_by_name(environment_requirements())

	for requirement in project_requirements:
		assert environment[requirement_name(requirement)] == requirement


def test_coupled_requirements_are_identical_across_extras():
	extras = project_metadata()["optional-dependencies"]

	assert {
		requirements_by_name(extras[extra])["zarr"]
		for extra in ("transform", "ng")
	} == {"zarr>=2.18,<3"}
	assert {
		requirements_by_name(requirements)["tifffile"]
		for requirements in extras.values()
		if "tifffile" in requirements_by_name(requirements)
	} == {"tifffile>=2024.8.30,<2025.5.21"}
	assert {
		requirements_by_name(requirements)["cloud-volume"]
		for requirements in extras.values()
		if "cloud-volume" in requirements_by_name(requirements)
	} == {"cloud-volume>=12.13,<13"}


def test_numpy_tomopy_and_tiff_zarr_contract_matches_conda_environment():
	project = project_metadata()
	base = requirements_by_name(project["dependencies"])
	environment = requirements_by_name(environment_requirements())

	assert base["numpy"] == "numpy>=1.24,<2"
	assert environment["numpy"] == "numpy>=1.24,<2"
	assert environment["tifffile"] == "tifffile>=2024.8.30,<2025.5.21"
	assert environment["tomopy"] == "tomopy>=1.15,<2"
	assert environment["zarr"] == "zarr>=2.18,<3"
	assert requirements_by_name(environment_pip_requirements())["numpy"] == (
		"numpy>=1.24,<2"
	)


def test_ci_checks_resolved_dependencies_and_lazy_imports():
	workflow = Path(".github/workflows/phase-0.yml").read_text(
		encoding="utf-8"
	)

	assert "python -m pip check" in workflow
	assert "python scripts/check_optional_dependencies.py --all" in workflow
