"""Stage-aware, resumable orchestration for sharded Neuroglancer publishing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlparse

import click

from mctutil.ng.completeness import check_mip0_completeness
from mctutil.shared.aws import configure_aws_profile
from mctutil.shared.cli import XYZ
from mctutil.shared.persistent_queue import (
	read_state,
	stable_fingerprint,
	write_state,
)


STAGES = ("prep", "precompute", "downsample", "shard", "upload", "mesh")
STAGE_EXTRAS = {
	"prep": ("ng",),
	"precompute": ("ng",),
	"downsample": ("mesh",),
	"shard": ("mesh",),
	"upload": ("aws",),
	"mesh": ("mesh",),
}
EXTRA_MODULES = {
	"ng": ("cloudvolume", "cloudfiles", "zarr"),
	"mesh": ("cloudvolume", "igneous.task_creation", "taskqueue"),
	"aws": ("boto3",),
}
DERIVED_SUFFIXES = ("_precomputed", "_precomputed_sharded_local")


@dataclass(frozen=True)
class DatasetPlan:
	dataset: Path
	layer_type: str
	tiff_paths: tuple[Path, ...]
	precompute_input: Path | None
	prep_input: Path | None
	prep_output: Path | None
	precomputed: Path
	staged: Path
	state_path: Path
	input_fingerprint: str


def utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def module_available(module_name: str) -> bool:
	try:
		return importlib.util.find_spec(module_name) is not None
	except (ImportError, ModuleNotFoundError, ValueError):
		return False


def resolve_stage_range(
	start_at: str,
	stop_after: str | None,
) -> tuple[str, ...]:
	"""Resolve a contiguous range, including the AWS-only upload resume case."""
	if stop_after is None:
		stop_after = "upload" if start_at == "upload" else "mesh"
	start_index = STAGES.index(start_at)
	stop_index = STAGES.index(stop_after)
	if start_index > stop_index:
		raise ValueError(
			f"--start-at {start_at} comes after --stop-after {stop_after}"
		)
	return STAGES[start_index:stop_index + 1]


def effective_stages(
	selected_stages: tuple[str, ...],
	no_upload: bool,
) -> tuple[str, ...]:
	return tuple(
		stage
		for stage in selected_stages
		if not (stage == "upload" and no_upload)
	)


def required_extras(
	stages: tuple[str, ...],
	s3_mesh: bool = False,
) -> tuple[str, ...]:
	required = {
		extra
		for stage in stages
		for extra in STAGE_EXTRAS[stage]
	}
	if s3_mesh:
		required.add("aws")
	return tuple(extra for extra in ("ng", "mesh", "aws") if extra in required)


def mesh_uses_s3(
	stages: tuple[str, ...],
	mesh_at: str,
	mesh_mip: int,
	upload_include_mip0: bool,
) -> bool:
	"""Whether the selected mesh stage will use an S3 layer target."""
	if "mesh" not in stages:
		return False
	use_s3 = mesh_at == "s3" or (
		mesh_at == "auto"
		and "upload" in stages
	)
	if use_s3 and mesh_mip == 0 and not upload_include_mip0:
		return False
	return use_s3


def missing_dependencies(
	extras: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
	return {
		extra: tuple(
			module
			for module in EXTRA_MODULES[extra]
			if not module_available(module)
		)
		for extra in extras
		if any(
			not module_available(module)
			for module in EXTRA_MODULES[extra]
		)
	}


def install_command(extras: tuple[str, ...]) -> str:
	return f"pip install -e '.[{','.join(extras)}]'"


def guess_layer_type(dataset: Path) -> str:
	name = dataset.name.lower()
	if (
		"segmentation" in name
		or "labels" in name
		or name.endswith("_seg")
	):
		return "segmentation"
	return "image"


def discover_datasets(root: Path) -> tuple[Path, ...]:
	return tuple(
		child.resolve()
		for child in sorted(root.iterdir(), key=lambda path: path.name.lower())
		if (
			child.is_dir()
			and not child.name.startswith(".")
			and not child.name.endswith(DERIVED_SUFFIXES)
		)
	)


def fingerprint_paths(
	dataset: Path,
	paths: tuple[Path, ...],
) -> str:
	entries = []
	for path in paths:
		stat = path.stat()
		try:
			name = path.relative_to(dataset).as_posix()
		except ValueError:
			name = str(path)
		entries.append((name, stat.st_size, stat.st_mtime_ns))
	return hashlib.sha256(
		json.dumps(entries, separators=(",", ":")).encode("utf-8")
	).hexdigest()


def fingerprint_precomputed_info(info_path: Path) -> str:
	try:
		info = json.loads(info_path.read_text(encoding="utf-8"))
		scale = info["scales"][0]
	except (FileNotFoundError, IndexError, KeyError, json.JSONDecodeError) as exc:
		raise ValueError(f"invalid precomputed metadata: {info_path}") from exc
	identity = {
		"type": info.get("type"),
		"data_type": info.get("data_type"),
		"num_channels": info.get("num_channels"),
		"scale0": {
			key: scale.get(key)
			for key in (
				"key",
				"resolution",
				"voxel_offset",
				"size",
				"encoding",
			)
		},
	}
	return hashlib.sha256(
		json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
	).hexdigest()


def build_dataset_plan(
	dataset: Path,
	needs_tiff: bool,
) -> DatasetPlan:
	all_tiffs = tuple(
		path.resolve()
		for path in sorted(dataset.iterdir(), key=lambda path: path.name.lower())
		if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
	)
	raw_tiffs = tuple(
		path
		for path in all_tiffs
		if "_memmap_" not in path.name.lower()
	)
	memmap_tiffs = tuple(
		path
		for path in all_tiffs
		if "_memmap_" in path.name.lower()
	)
	precomputed = dataset.with_name(f"{dataset.name}_precomputed")
	staged = precomputed.with_name(f"{precomputed.name}_sharded_local")

	prep_input = None
	prep_output = None
	precompute_input = None
	fingerprint_sources = raw_tiffs or memmap_tiffs
	if len(raw_tiffs) == 1:
		prep_input = raw_tiffs[0]
		prep_output = dataset / f"{prep_input.stem}_MEMMAP_original.tif"
		precompute_input = prep_output
	elif len(raw_tiffs) > 1:
		if memmap_tiffs:
			raise ValueError(
				f"{dataset} mixes slice TIFFs with generated memmap TIFFs"
			)
		precompute_input = dataset
	elif len(memmap_tiffs) == 1:
		precompute_input = memmap_tiffs[0]
	elif len(memmap_tiffs) > 1:
		raise ValueError(f"{dataset} has multiple candidate memmap TIFFs")
	elif needs_tiff:
		raise ValueError(f"{dataset} contains no TIFF input")

	if fingerprint_sources:
		input_fingerprint = fingerprint_paths(dataset, fingerprint_sources)
	else:
		fallbacks = tuple(
			path
			for path in (precomputed / "info", staged / "info")
			if path.is_file()
		)
		if not fallbacks:
			raise ValueError(
				f"{dataset} has no TIFF input or existing precomputed metadata"
			)
		input_fingerprint = fingerprint_precomputed_info(fallbacks[0])

	return DatasetPlan(
		dataset=dataset,
		layer_type=guess_layer_type(dataset),
		tiff_paths=fingerprint_sources,
		precompute_input=precompute_input,
		prep_input=prep_input,
		prep_output=prep_output,
		precomputed=precomputed,
		staged=staged,
		state_path=dataset / ".mctutil_ng_publish.json",
		input_fingerprint=input_fingerprint,
	)


def parse_s3_prefix(value: str) -> tuple[str, str]:
	parsed = urlparse(value)
	if parsed.scheme != "s3" or not parsed.netloc:
		raise ValueError("--s3-prefix must have the form s3://BUCKET/optional/prefix")
	return parsed.netloc, parsed.path.strip("/")


def resolve_controls(
	start_at: str,
	stop_after: str | None,
	no_upload: bool,
	mesh_at: str,
	s3_prefix: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
	selected_stages = resolve_stage_range(start_at, stop_after)
	if no_upload and start_at == "upload":
		raise ValueError("--start-at upload contradicts --no-upload")
	effective = effective_stages(selected_stages, no_upload)
	if "upload" in effective and not s3_prefix:
		raise ValueError("--s3-prefix is required when upload is selected")
	if mesh_at == "s3" and "mesh" in selected_stages and not s3_prefix:
		raise ValueError("--mesh-at s3 requires --s3-prefix")
	if s3_prefix:
		parse_s3_prefix(s3_prefix)
	return selected_stages, effective


def dataset_s3_target(plan: DatasetPlan, s3_prefix: str) -> tuple[str, str]:
	bucket, prefix = parse_s3_prefix(s3_prefix)
	key = "/".join(
		part
		for part in (prefix, f"{plan.dataset.name}_precomputed_sharded")
		if part
	)
	return bucket, key


def load_dataset_state(plan: DatasetPlan) -> dict:
	state = read_state(plan.state_path)
	if state is None:
		return {
			"version": 1,
			"input_fingerprint": plan.input_fingerprint,
			"stages": {},
		}
	if state.get("input_fingerprint") != plan.input_fingerprint:
		raise ValueError(
			f"input changed since publish state was recorded: {plan.dataset}"
		)
	if not isinstance(state.get("stages"), dict):
		raise ValueError(f"invalid publish state: {plan.state_path}")
	return state


def _read_info(path: Path) -> dict | None:
	try:
		return json.loads((path / "info").read_text(encoding="utf-8"))
	except (FileNotFoundError, json.JSONDecodeError):
		return None


def valid_memmap(plan: DatasetPlan) -> bool:
	if plan.prep_input is None or plan.prep_output is None:
		return False
	if not plan.prep_output.is_file():
		return False
	try:
		import tifffile

		with tifffile.TiffFile(plan.prep_input) as source:
			expected_shape = tuple(source.series[0].shape)
			expected_dtype = source.series[0].dtype
		mapped = tifffile.memmap(plan.prep_output)
		try:
			return (
				tuple(mapped.shape) == expected_shape
				and mapped.dtype == expected_dtype
			)
		finally:
			del mapped
	except Exception:
		return False


def stage_artifact_valid(stage: str, plan: DatasetPlan) -> bool:
	if stage == "prep":
		return valid_memmap(plan)
	if stage == "precompute":
		return check_mip0_completeness(plan.precomputed).complete
	if stage == "downsample":
		info = _read_info(plan.precomputed)
		return bool(info and len(info.get("scales", [])) > 1)
	if stage == "shard":
		info = _read_info(plan.staged)
		if not info or not info.get("scales"):
			return False
		sharded_scales = [
			scale
			for scale in info["scales"]
			if scale.get("sharding")
		]
		return bool(
			sharded_scales
			and all(
				(plan.staged / scale["key"]).is_dir()
				and any((plan.staged / scale["key"]).glob("*.shard"))
				for scale in sharded_scales
			)
		)
	return True


def resolved_precompute_input(plan: DatasetPlan, options: dict) -> Path | None:
	if (
		plan.prep_input is not None
		and "prep" not in options["selected_stages"]
		and (
			plan.prep_output is None
			or not plan.prep_output.is_file()
		)
	):
		return plan.prep_input
	return plan.precompute_input


def stage_configuration(
	stage: str,
	plan: DatasetPlan,
	options: dict,
) -> str:
	configuration = {
		"stage": stage,
		"layer_type": plan.layer_type,
	}
	if stage == "prep":
		configuration.update(
			input=str(plan.prep_input),
			output=str(plan.prep_output),
		)
	elif stage == "precompute":
		configuration.update(
			input=str(resolved_precompute_input(plan, options)),
			output=str(plan.precomputed),
			voxel_resolution=options["voxel_resolution"],
			voxel_offset=options["voxel_offset"],
			segmentation_encoding=options["segmentation_encoding"],
		)
	elif stage == "downsample":
		configuration.update(
			layer=str(plan.precomputed),
			memory=options["memory"],
			initial_chunk=(64, 64, 64),
			extend_chunk=(16, 16, 16),
		)
	elif stage == "shard":
		configuration.update(
			source=str(plan.precomputed),
			destination=str(plan.staged),
			include_mip0=options["stage_include_mip0"],
			memory=options["memory"],
		)
	elif stage == "upload":
		configuration.update(
			destination=options["s3_prefix"],
			include_mip0=options["upload_include_mip0"],
		)
	elif stage == "mesh":
		configuration.update(
			target=mesh_target(plan, options),
			mip=options["mesh_mip"],
			num_lod=options["mesh_num_lod"],
		)
	return stable_fingerprint(configuration)


def omitted_reason(stage: str, plan: DatasetPlan, options: dict) -> str | None:
	if stage == "prep" and plan.prep_input is None:
		return "input is already memmappable or is a TIFF-slice directory"
	if stage == "upload" and options["no_upload"]:
		return "disabled by --no-upload"
	if stage == "mesh" and plan.layer_type != "segmentation":
		return "image layers do not require meshes"
	return None


def stage_decision(
	stage: str,
	plan: DatasetPlan,
	state: dict,
	options: dict,
) -> tuple[str, str | None]:
	reason = omitted_reason(stage, plan, options)
	if reason is not None:
		return "omitted", reason
	if stage == "prep" and options["overwrite_prep"]:
		return "pending", "forced by --overwrite-prep"
	configuration = stage_configuration(stage, plan, options)
	record = state["stages"].get(stage, {})
	if (
		record.get("status") == "complete"
		and record.get("configuration") == configuration
		and stage_artifact_valid(stage, plan)
	):
		return "complete", "recorded completion and output are valid"
	if stage == "prep" and stage_artifact_valid(stage, plan):
		return "complete", "valid memmappable output already exists"
	return "pending", None


def mesh_target(plan: DatasetPlan, options: dict) -> str:
	use_s3 = options["mesh_at"] == "s3" or (
		options["mesh_at"] == "auto"
		and "upload" in options["effective_stages"]
	)
	if (
		use_s3
		and options["mesh_mip"] == 0
		and not options["upload_include_mip0"]
	):
		use_s3 = False
	if use_s3:
		bucket, key = dataset_s3_target(plan, options["s3_prefix"])
		return f"precomputed://s3://{bucket}/{key}"
	return f"precomputed://{plan.staged.resolve().as_uri()}"


def local_mesh_upload_warning(
	plan: DatasetPlan,
	options: dict,
) -> str | None:
	if (
		plan.layer_type != "segmentation"
		or "upload" not in options["effective_stages"]
		or "mesh" not in options["selected_stages"]
		or mesh_target(plan, options).startswith("precomputed://s3://")
	):
		return None
	return (
		f"local mesh for {plan.dataset.name} runs after upload and will not be "
		"present in S3; rerun upload afterward or use --mesh-at s3"
	)


def validate_prerequisites(
	plan: DatasetPlan,
	selected_stages: tuple[str, ...],
	options: dict,
) -> None:
	first = STAGES.index(selected_stages[0])
	precompute_input = resolved_precompute_input(plan, options)
	if first <= STAGES.index("precompute") and precompute_input is None:
		raise ValueError(f"no precompute input for {plan.dataset}")
	if (
		selected_stages[0] == "precompute"
		and not precompute_input.exists()
	):
		raise ValueError(f"precompute input is missing: {precompute_input}")
	if (
		STAGES.index("precompute") < first <= STAGES.index("shard")
		and not (plan.precomputed / "info").is_file()
	):
		raise ValueError(f"precomputed input is missing: {plan.precomputed}")
	if (
		first > STAGES.index("shard")
		and ("upload" in selected_stages or options["mesh_at"] != "s3")
		and not (plan.staged / "info").is_file()
	):
		raise ValueError(f"sharded staging input is missing: {plan.staged}")


def run_stage(stage: str, plan: DatasetPlan, options: dict) -> None:
	queue_root = plan.precomputed / ".mctutil-queues"
	encoding = (
		options["segmentation_encoding"]
		if plan.layer_type == "segmentation"
		else "raw"
	)
	if stage == "prep":
		module = importlib.import_module("mctutil.transform.memmap_prep")
		module.memmap_prep.callback(
			input_tif=plan.prep_input,
			output=plan.prep_output,
			output_dir=None,
			out_dtypes="original",
			normalize_mode="none",
			norm_min=None,
			norm_max=None,
			pct_low=0.1,
			pct_high=99.9,
			sample_slices=32,
			sample_pixels=200_000,
			rng_seed=0,
			bigtiff_mode="auto",
			contiguous=True,
			overwrite=options["overwrite_prep"],
			verify=True,
			execute=True,
		)
	elif stage == "precompute":
		module = importlib.import_module("mctutil.ng.precompute")
		module.precompute.callback(
			input_path=resolved_precompute_input(plan, options),
			output_path=plan.precomputed,
			workers=options["workers"],
			layer_type=plan.layer_type,
			segmentation_encoding=options["segmentation_encoding"],
			dtype_override=None,
			chunk_size=None,
			segmentation_block=(8, 8, 8),
			voxel_resolution=options["voxel_resolution"],
			voxel_offset=options["voxel_offset"],
			execute=True,
		)
	elif stage == "downsample":
		module = importlib.import_module("mctutil.ng.downsample_pyramid")
		module.downsample_pyramid.callback(
			layer_path=str(plan.precomputed),
			queue_dir=queue_root,
			initial_chunk=(64, 64, 64),
			extend_chunk=(16, 16, 16),
			max_extend_passes=3,
			initial_parallel=16,
			extend_parallel=16,
			memory=options["memory"],
			encoding=encoding,
			lease_seconds=3600,
			release_leases=options["release_queue_leases"],
			force=False,
			execute=True,
		)
	elif stage == "shard":
		module = importlib.import_module("mctutil.ng.shard")
		module.shard.callback(
			source=str(plan.precomputed),
			destination=str(plan.staged),
			mips=None,
			low_chunk=(96, 96, 96),
			mid_chunk=(64, 64, 64),
			high_chunk=(16, 16, 16),
			memory=options["memory"],
			parallel=8,
			include_mip0=options["stage_include_mip0"],
			encoding=encoding,
			queue_dir=queue_root,
			lease_seconds=3600,
			release_leases=options["release_queue_leases"],
			execute=True,
		)
	elif stage == "upload":
		module = importlib.import_module("mctutil.transport.s3upload")
		bucket, key = dataset_s3_target(plan, options["s3_prefix"])
		module.upload_sharded_tree(
			source_folder=plan.staged,
			target_folder=key,
			bucket_name=bucket,
			jobs=options["upload_jobs"],
			include_mip0=options["upload_include_mip0"],
			execute=True,
			aws_profile=options["aws_profile"],
		)
	elif stage == "mesh":
		module = importlib.import_module("mctutil.shared.mesh")
		module.build_mesh(
			layer_path=mesh_target(plan, options),
			mip=options["mesh_mip"],
			num_lod=options["mesh_num_lod"],
			parallel=options["mesh_parallel"],
			fill_missing=True,
			queue_dir=queue_root / "mesh",
			execute=True,
			aws_profile=options["aws_profile"],
		)


def print_dependency_plan(
	extras: tuple[str, ...],
	missing: dict[str, tuple[str, ...]],
) -> None:
	click.echo(f"Required extras: {', '.join(f'[{extra}]' for extra in extras) or 'none'}")
	if missing:
		for extra, modules in missing.items():
			click.echo(f"Missing [{extra}]: {', '.join(modules)}")
		click.echo(f"Install with: {install_command(extras)}")
	else:
		click.echo("Dependency preflight: satisfied")


def publish_datasets(
	plans: tuple[DatasetPlan, ...],
	selected_stages: tuple[str, ...],
	options: dict,
	execute: bool,
) -> None:
	for plan in plans:
		state = load_dataset_state(plan)
		click.echo(f"\nDataset: {plan.dataset.name} ({plan.layer_type})")
		click.echo(f"  state: {plan.state_path}")
		for stage in STAGES:
			if stage not in selected_stages:
				click.echo(f"  {stage}: not-run (outside selected range)")
				continue
			decision, reason = stage_decision(
				stage,
				plan,
				state,
				options,
			)
			suffix = f" — {reason}" if reason else ""
			click.echo(f"  {stage}: {decision}{suffix}")

		if not execute:
			continue
		state["requested_stages"] = list(selected_stages)
		state["updated_at"] = utc_now()
		write_state(plan.state_path, state)
		for stage in selected_stages:
			decision, reason = stage_decision(stage, plan, state, options)
			configuration = stage_configuration(stage, plan, options)
			if decision == "omitted":
				state["stages"][stage] = {
					"status": "omitted",
					"reason": reason,
					"configuration": configuration,
					"updated_at": utc_now(),
				}
				write_state(plan.state_path, state)
				continue
			if decision == "complete":
				record = state["stages"].get(stage, {})
				if (
					record.get("status") != "complete"
					or record.get("configuration") != configuration
				):
					state["stages"][stage] = {
						"status": "complete",
						"configuration": configuration,
						"validated_at": utc_now(),
					}
					state["updated_at"] = utc_now()
					write_state(plan.state_path, state)
				continue
			click.echo(f"Running {stage} for {plan.dataset.name}.")
			run_stage(stage, plan, options)
			state["stages"][stage] = {
				"status": "complete",
				"configuration": configuration,
				"completed_at": utc_now(),
			}
			state["updated_at"] = utc_now()
			write_state(plan.state_path, state)


@click.command("publish")
@click.argument(
	"root",
	type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--s3-prefix", help="Destination prefix, e.g. s3://bucket/path.")
@click.option("--aws-profile", help="Named AWS profile for S3 upload and meshing.")
@click.option("--start-at", type=click.Choice(STAGES), default="prep", show_default=True)
@click.option(
	"--stop-after",
	type=click.Choice(STAGES),
	help="Last selected stage; starting at upload defaults to upload only.",
)
@click.option("--no-upload", is_flag=True, help="Explicitly omit upload from the range.")
@click.option("--workers", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--memory", type=click.IntRange(min=1), default=10_000_000_000, show_default=True)
@click.option(
	"--release-queue-leases/--preserve-queue-leases",
	default=True,
	show_default=True,
	help="Release existing Igneous FileQueue leases when resuming.",
)
@click.option("--upload-jobs", type=click.IntRange(min=1), default=6, show_default=True)
@click.option("--mesh-parallel", type=click.IntRange(min=1), default=16, show_default=True)
@click.option("--mesh-mip", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--mesh-num-lod", type=click.IntRange(min=0), default=4, show_default=True)
@click.option(
	"--mesh-at",
	type=click.Choice(("auto", "local", "s3")),
	default="auto",
	show_default=True,
)
@click.option(
	"--seg-encoding",
	"segmentation_encoding",
	type=click.Choice(("compressed_segmentation", "compresso")),
	default="compressed_segmentation",
	show_default=True,
)
@click.option(
	"--voxel-resolution",
	type=XYZ,
	default="700,700,700",
	show_default=True,
	help="Voxel resolution in nanometers as X,Y,Z.",
)
@click.option(
	"--voxel-offset",
	type=XYZ,
	default="0,0,0",
	show_default=True,
	help="Voxel-coordinate offset as X,Y,Z.",
)
@click.option("--stage-include-mip0/--stage-exclude-mip0", default=True, show_default=True)
@click.option("--upload-include-mip0/--upload-exclude-mip0", default=True, show_default=True)
@click.option("--overwrite-prep", is_flag=True)
@click.option("--execute/--dry-run", default=True, show_default=True)
def publish(
	root: Path,
	s3_prefix: str | None,
	aws_profile: str | None,
	start_at: str,
	stop_after: str | None,
	no_upload: bool,
	workers: int,
	memory: int,
	release_queue_leases: bool,
	upload_jobs: int,
	mesh_parallel: int,
	mesh_mip: int,
	mesh_num_lod: int,
	mesh_at: str,
	segmentation_encoding: str,
	voxel_resolution: tuple[int, int, int],
	voxel_offset: tuple[int, int, int],
	stage_include_mip0: bool,
	upload_include_mip0: bool,
	overwrite_prep: bool,
	execute: bool,
) -> None:
	"""Publish each child dataset as a resumable sharded Neuroglancer layer."""
	try:
		selected_stages, effective = resolve_controls(
			start_at,
			stop_after,
			no_upload,
			mesh_at,
			s3_prefix,
		)

		s3_mesh = mesh_uses_s3(
			effective,
			mesh_at,
			mesh_mip,
			upload_include_mip0,
		)
		uses_s3 = "upload" in effective or s3_mesh
		if uses_s3:
			bucket, _prefix = parse_s3_prefix(s3_prefix)
			aws_profile = configure_aws_profile(aws_profile, bucket)
		else:
			aws_profile = None

		extras = required_extras(effective, s3_mesh=s3_mesh)
		missing = missing_dependencies(extras)
		click.echo(f"Root: {root.resolve()}")
		click.echo(f"Selected stages: {', '.join(selected_stages)}")
		if no_upload and "upload" in selected_stages:
			click.echo("Upload is explicitly omitted by --no-upload.")
		print_dependency_plan(extras, missing)
		if aws_profile is not None:
			click.echo(f"AWS profile: {aws_profile}")
		click.echo(f"Voxel resolution (nm): {voxel_resolution}")
		click.echo(f"Voxel offset: {voxel_offset}")

		datasets = discover_datasets(root)
		if not datasets:
			raise ValueError(f"no dataset directories found in {root}")
		needs_tiff = STAGES.index(selected_stages[0]) <= STAGES.index("precompute")
		plans = tuple(
			build_dataset_plan(dataset, needs_tiff)
			for dataset in datasets
		)
		options = {
			"s3_prefix": s3_prefix,
			"selected_stages": selected_stages,
			"effective_stages": effective,
			"no_upload": no_upload,
			"aws_profile": aws_profile,
			"workers": workers,
			"memory": memory,
			"release_queue_leases": release_queue_leases,
			"upload_jobs": upload_jobs,
			"mesh_parallel": mesh_parallel,
			"mesh_mip": mesh_mip,
			"mesh_num_lod": mesh_num_lod,
			"mesh_at": mesh_at,
			"segmentation_encoding": segmentation_encoding,
			"voxel_resolution": voxel_resolution,
			"voxel_offset": voxel_offset,
			"stage_include_mip0": stage_include_mip0,
			"upload_include_mip0": upload_include_mip0,
			"overwrite_prep": overwrite_prep,
		}
		for plan in plans:
			warning = local_mesh_upload_warning(plan, options)
			if warning is not None:
				click.echo(f"Warning: {warning}", err=True)
		for plan in plans:
			load_dataset_state(plan)
			validate_prerequisites(plan, selected_stages, options)

		if execute and missing:
			raise ValueError(
				"missing dependencies; "
				f"install with {install_command(extras)}"
			)
		publish_datasets(plans, selected_stages, options, execute)
	except click.ClickException:
		raise
	except Exception as exc:
		raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
	publish()
