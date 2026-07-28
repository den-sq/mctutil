"""Cheng-Lab micro-CT schema for `mctutil parse meta-shift`.

Holds the chenglab-specific bits that were previously inline in
parsing/meta_shift.py: the STATUS enum, folder-name conventions, sbatch
parameter extraction, Google Sheets row layout, and the *V.yaml discovery
glob.
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Iterable

from ruamel.yaml import YAML

yaml = YAML()


class STATUS(Enum):
	EMPTY = auto()
	FLATS_GENERATED_UC = auto()
	FLATS_GENERATED = auto()
	CENTERS_DUMPED_UC = auto()
	CENTERS_DUMPED = auto()
	INCOMPLETE_RECONSTRUCTION = auto()
	RECONSTRUCTED_UC = auto()
	RECONSTRUCTED = auto()

	def __str__(self):
		parts = [f'{part[:1].upper()}{part[1:].lower()}' for part in self.name.split("_")]
		return " ".join(parts)

	@staticmethod
	def from_step(step, uc=True):
		if step == "flats":
			return STATUS.FLATS_GENERATED_UC if uc else STATUS.FLATS_GENERATED
		if step == "centerfind":
			return STATUS.CENTERS_DUMPED_UC if uc else STATUS.CENTERS_DUMPED
		if step == "recon":
			return STATUS.RECONSTRUCTED_UC if uc else STATUS.RECONSTRUCTED
		return STATUS.EMPTY


def get_outer_comment(folder):
	components = folder.name.split("_")
	outer_comment = []
	energy = components[-1]
	for part in components:
		if len(part) == 6 and part[-3:].isnumeric() and not part[:3].isnumeric():
			continue
		if part != energy:
			outer_comment.append(part)
	return "_".join(outer_comment) if len(outer_comment) > 0 else ""


def get_updated_config(old_config, empty_id):
	with open(old_config) as conf:
		conf_dict = yaml.load(conf)
	if "comment" in conf_dict["storage"]:
		base_comment = conf_dict["storage"].pop("comment").removesuffix("scans").removesuffix("_")
		conf_dict["storage"]["has_scan"] = base_comment[-4:] == "scan"
		conf_dict["storage"]["inner_comment"] = base_comment.removesuffix("scan").removesuffix("_")
		conf_dict["storage"]["outer_comment"] = get_outer_comment(
			Path(conf_dict["storage"].pop("override_input"))
		)
	scan_id = "_".join([
		str(conf_dict["storage"]["outer_comment"]),
		str(conf_dict["storage"]["inner_comment"]),
	]).strip("_")
	if scan_id == '':
		scan_id = f"SS{empty_id}"
		empty_id += 1
	conf_dict["scan"]["id"] = scan_id
	return conf_dict


def get_paths(conf: dict, job_id):
	steps = ["flats", "centerfind", "recon"]
	old_paths = {}
	new_paths = {}
	data_path = Path(conf["storage"]["project"], conf["sample"]["id"], "data")

	for step in steps:
		if conf["storage"]["has_scan"]:
			old_folder = (
				f'{conf["scan"]["energy"]}kV_{conf["storage"]["inner_comment"]}_scan_{step}_p{job_id}'
			).replace("__", "_")
		else:
			old_folder = f'{conf["scan"]["energy"]}kV_{conf["storage"]["inner_comment"]}_{step}_p{job_id}'
		new_folder = f'{conf["scan"]["energy"]}kV_{conf["scan"]["id"]}_{step}_p{job_id}'
		old_paths[step] = data_path.joinpath(old_folder)
		new_paths[step] = data_path.joinpath(new_folder)

	old_paths["script"] = data_path.joinpath(
		"history",
		f'{conf["scan"]["energy"]}kV_{conf["storage"]["inner_comment"]}',
		job_id,
	)
	new_paths["script"] = data_path.joinpath(
		"history",
		f'{conf["scan"]["energy"]}kV_{conf["scan"]["id"]}',
		job_id,
	)
	return old_paths, new_paths


def get_run_params(batch_file: Path):
	start_param = ""
	stop_param = ""
	base = ""
	phase_alpha = None
	with open(batch_file) as batch:
		all_lines = batch.readlines()
		run_command = all_lines[-1]
		for line in all_lines:
			if line[:5] == "start":
				start_param = line.split(" ")[1]
			if line[:4] == "stop":
				stop_param = line.split(" ")[3]
			if line[:11] == "phase_alpha":
				phase_alpha = line.split("=")[1]
			if line[:4] == "base":
				base = line.split(" ")[2]

	run_params = {
		command_pair.split("=")[0][2:]: command_pair.split("=")[1]
		for command_pair in run_command.split(".py")[1].strip("\n").strip().split(" ")
		if "=" in command_pair
	}

	if "phase_tomo" not in run_params:
		run_params["phase_tomo"] = "phase_alpha" not in run_params
	if "stripe_removal" not in run_params:
		run_params["stripe_removal"] = "VO (all)"
	if "phase_alpha" not in run_params:
		run_params["phase_alpha"] = 0.03
	if phase_alpha is not None:
		run_params["phase_alpha"] = str(run_params["phase_alpha"]).replace("$phase_alpha", phase_alpha)

	run_params["slice_range"] = (
		run_params["slice_range"]
		.replace("$start", start_param)
		.replace("$stop", stop_param)
		.replace("base", base)
	)
	return run_params


def build_sheet_row(conf_dict, run_params, drive, move_pairs, scan_num, status):
	recon_line = []
	remote_params = [
		"machine_conf",
		"sample_conf",
		"old_file_naming",
		"window_size",
		"slice_range",
		"recon_algorithm",
		"job_id",
		"phase_alpha",
		"stripe_removal",
		"phase_tomo",
		"rot_center",
		"tilt",
	]
	folder_fields = ["flats", "centerfind", "recon", "script"]
	folders = [
		str(drive.joinpath(conf_dict["storage"]["project"], conf_dict["sample"]["id"], "data"))
	]

	recon_line += [
		'',
		conf_dict["sample"]["id"],
		conf_dict["storage"]["trip_dir"],
		scan_num,
		conf_dict["scan"]["id"],
		conf_dict["storage"]["project"],
		'',
	]
	recon_line += ['' if param not in run_params else str(run_params[param]) for param in remote_params]
	recon_line += ['', str(status), '', '', '']

	for field in folder_fields:
		folders.append('' if field not in move_pairs else str(move_pairs[field]["to"].relative_to(folders[0])))

	recon_line += folders
	return recon_line


class ChenglabMicroCTAdapter:
	"""Default adapter for `mctutil parse meta-shift`.

	Encodes Cheng-Lab micro-CT folder layout, status pipeline, sbatch parameter
	extraction, and Google Sheets row format.
	"""

	name = "chenglab"
	default_spreadsheet = "1RiCh3kjJhmUKZ5Y4UTwtKeiEag8jS_bU6jBeGJnru18"
	default_sheet = "GPFS (DEN)"
	steps = ("flats", "centerfind", "recon", "script")

	def discover_sample_configs(self, drive: Path) -> list[Path]:
		return list(drive.rglob("*V.yaml")) if drive.exists() else []

	def load_sample_config(self, conf_path: Path, empty_id: int) -> dict:
		return get_updated_config(conf_path, empty_id)

	def compute_paths(self, conf: dict, job_id: str) -> tuple[dict, dict]:
		return get_paths(conf, job_id)

	def extract_run_params(self, sample_conf: Path) -> dict:
		batch_list = [fname for fname in sample_conf.parent.iterdir() if fname.suffix == ".sbatch"]
		if len(batch_list) > 0:
			return get_run_params(batch_list[0])
		with open(sample_conf.parent.joinpath("reconstruct_parameters.yaml")) as rp:
			return yaml.load(rp)["reconstruct"]

	def scan_num_from_config(self, sample_conf: Path) -> str:
		return sample_conf.name.split("_")[0]

	def status_for_step(self, step: str, uncorrected: bool = True):
		return STATUS.from_step(step, uc=uncorrected)

	def empty_status(self):
		return STATUS.EMPTY

	def step_order(self) -> Iterable[str]:
		return self.steps

	def build_sheet_row(self, conf_dict, run_params, drive, move_pairs, scan_num, status):
		return build_sheet_row(conf_dict, run_params, drive, move_pairs, scan_num, status)

	def transform_sample_config(self, conf_path: Path, empty_id: int) -> dict:
		return get_updated_config(conf_path, empty_id)
