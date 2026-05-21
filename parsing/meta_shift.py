from enum import Enum, auto
from pathlib import Path
from time import sleep
import os
import traceback

import click
from ruamel.yaml import YAML

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

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


def shift_old_new(drive, conf_dict, old_paths, new_paths, run_params, scan_num, execute=False):
	status = STATUS.EMPTY
	keep_pairs = {}
	move_pairs = {}
	can_move = True

	for step in old_paths:
		old_target = drive.joinpath(old_paths[step])
		new_target = drive.joinpath(new_paths[step])
		if old_target.exists() and STATUS.from_step(step) is not STATUS.EMPTY:
			status = STATUS.from_step(step)

		if old_paths[step] == new_paths[step] and old_target.exists():
			keep_pairs[step] = {"from": old_target, "to": new_target}
		elif old_target.exists():
			if not new_target.exists():
				move_pairs[step] = {"from": old_target, "to": new_target}
			else:
				can_move = False
		else:
			can_move = False

	if can_move and execute:
		for step in move_pairs:
			if not move_pairs[step]["to"].parent.exists():
				move_pairs[step]["to"].parent.mkdir(parents=True)
			move_pairs[step]["from"].rename(move_pairs[step]["to"])

	if can_move and execute:
		hist_dir = drive.joinpath(new_paths["script"])
		hist_dir.mkdir(parents=True, exist_ok=True)
		conf_name = f'{scan_num}_{conf_dict["sample"]["id"]}_{conf_dict["scan"]["id"]}.yaml'
		with open(hist_dir.joinpath(conf_name), "w") as new_conf:
			yaml.dump(conf_dict, new_conf)
		with open(hist_dir.joinpath("batch_extracted.yaml"), "w") as new_params:
			yaml.dump(run_params, new_params)

	return status, can_move, move_pairs, keep_pairs


def establish_google_creds(creds, google_conf, gscopes, cred_name="gsheets"):
	token_path = Path(google_conf, f"{cred_name}_token.json")
	cred_path = Path(google_conf, f'{cred_name}_credentials.json')

	if token_path.exists():
		creds[cred_name] = Credentials.from_authorized_user_file(token_path, gscopes)

	if cred_name not in creds or not creds[cred_name].valid:
		if cred_name in creds and creds[cred_name].expired and creds[cred_name].refresh_token:
			creds[cred_name].refresh(Request())
		else:
			if cred_path.exists():
				flow = InstalledAppFlow.from_client_secrets_file(cred_path, gscopes)
				creds[cred_name] = flow.run_local_server(port=0)
			else:
				raise FileNotFoundError(cred_path)
		with open(token_path, "w") as gtoken:
			gtoken.write(creds[cred_name].to_json())


def write_sheets_fields(creds, google_conf, gscopes, spreadsheet, sheet, values):
	cred_name = "gsheets"
	if cred_name not in creds:
		establish_google_creds(creds, google_conf, gscopes, cred_name)
	service = build('sheets', 'v4', credentials=creds[cred_name]).spreadsheets()
	body_val = {"range": f'{sheet}!A1:A2', "values": [values]}
	return service.values().append(
		spreadsheetId=spreadsheet,
		range=f'{sheet}!A1:A2',
		valueInputOption="RAW",
		insertDataOption="INSERT_ROWS",
		body=body_val,
	).execute()


def update_gsheet(
		creds,
		google_conf,
		gscopes,
		conf_dict,
		run_params,
		drive,
		move_pairs,
		scan_num,
		status,
		spreadsheet,
		sheet,
):
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
	return write_sheets_fields(creds, google_conf, gscopes, spreadsheet, sheet, recon_line)


def parse_sample_list(sample_list_file):
	return [
		Path(line.strip())
		for line in sample_list_file.read_text().splitlines()
		if line.strip() != ''
	]


def discover_sample_configs(drive):
	return list(drive.rglob("*V.yaml")) if drive.exists() else []


def write_updated_configs(config_dir, empty_id=0):
	for conf_path in [conf_path for conf_path in Path(config_dir).iterdir() if conf_path.suffix == ".yaml"]:
		conf_dict = get_updated_config(conf_path, empty_id)
		new_conf_path = Path(conf_path.parent, "new", conf_path.name)
		new_conf_path.parent.mkdir(parents=True, exist_ok=True)
		with open(new_conf_path, "w") as out:
			yaml.dump(conf_dict, out)


def cleanup_empty_history(drive):
	for history in list(drive.rglob("*history")):
		for inner_dir in history.iterdir():
			if inner_dir.is_dir() and not bool(len(list(inner_dir.iterdir()))):
				click.echo(f"Removing Empty {inner_dir}")
				inner_dir.rmdir()


def load_machine_config(machine_conf):
	with open(machine_conf) as mc:
		return yaml.load(mc)


def run_meta_shift(
		sample_conf_list,
		drive,
		spreadsheet,
		sheet,
		google_conf,
		sleep_seconds=0.0,
		execute=False,
		update_sheet=True,
):
	creds = {}
	gscopes = ['https://www.googleapis.com/auth/spreadsheets']
	empty_id = 0

	for sample_conf in sample_conf_list:
		try:
			job_id = sample_conf.parent.stem
			conf_dict = get_updated_config(sample_conf, empty_id)
			old_paths, new_paths = get_paths(conf_dict, job_id)
			batch_list = [fname for fname in sample_conf.parent.iterdir() if fname.suffix == ".sbatch"]
			if len(batch_list) > 0:
				run_params = get_run_params(batch_list[0])
			else:
				with open(sample_conf.parent.joinpath("reconstruct_parameters.yaml")) as rp:
					run_params = yaml.load(rp)["reconstruct"]

			scan_num = sample_conf.name.split("_")[0]
			if "job_id" not in run_params:
				run_params["job_id"] = job_id

			status, written, move_pairs, keep_pairs = shift_old_new(
				drive,
				conf_dict,
				old_paths,
				new_paths,
				run_params,
				scan_num,
				execute=execute,
			)
			if written and update_sheet:
				update_gsheet(
					creds,
					google_conf,
					gscopes,
					conf_dict,
					run_params,
					drive,
					move_pairs | keep_pairs,
					scan_num,
					status,
					spreadsheet,
					sheet,
				)
		except IndexError as ex:
			click.echo(f"{sample_conf}: {ex}: {traceback.format_exc()}")
		if sleep_seconds > 0:
			sleep(sleep_seconds)


@click.command()
@click.option(
	"--machine-conf",
	type=click.Path(exists=True, path_type=Path),
	default=Path("mac.yaml"),
	show_default=True,
	help="Machine configuration containing the storage drive root.",
)
@click.option(
	"--config-dir",
	type=click.Path(exists=True, file_okay=False, path_type=Path),
	default=Path("samples"),
	show_default=True,
	help="Directory containing sample config YAML files.",
)
@click.option(
	"--sample-config",
	"sample_configs",
	multiple=True,
	type=click.Path(exists=True, dir_okay=False, path_type=Path),
	help="Explicit sample config(s) to process.",
)
@click.option(
	"--sample-list-file",
	type=click.Path(exists=True, dir_okay=False, path_type=Path),
	help="File containing newline-separated sample config paths to process.",
)
@click.option(
	"--google-conf",
	type=click.Path(path_type=Path),
	default=lambda: Path(os.environ.get("MCTUTIL_GOOGLE_CONF", "conf")),
	show_default="conf",
	help="Directory containing Google Sheets credential files.",
)
@click.option(
	"--spreadsheet",
	default=lambda: os.environ.get(
		"MCTUTIL_GSHEET_ID",
		"1RiCh3kjJhmUKZ5Y4UTwtKeiEag8jS_bU6jBeGJnru18",
	),
	show_default=True,
)
@click.option(
	"--sheet",
	default=lambda: os.environ.get("MCTUTIL_GSHEET_SHEET", "GPFS (DEN)"),
	show_default=True,
)
@click.option(
	"--write-configs",
	is_flag=True,
	help="Write updated sample YAML files into <config-dir>/new before processing.",
)
@click.option(
	"--cleanup-empty-history",
	is_flag=True,
	help="Remove empty history subdirectories after processing.",
)
@click.option(
	"--discover/--no-discover",
	default=True,
	show_default=True,
	help="Discover sample configs from the machine-config drive when no explicit list is provided.",
)
@click.option(
	"--update-sheet/--no-update-sheet",
	default=True,
	show_default=True,
	help="Whether to append processed rows to Google Sheets.",
)
@click.option(
	"--execute/--dry-run",
	default=False,
	show_default=True,
	help="Whether to move/write files or only simulate path checks.",
)
@click.option("--sleep-seconds", type=click.FLOAT, default=0.0, show_default=True)
def meta_shift(
		machine_conf,
		config_dir,
		sample_configs,
		sample_list_file,
		google_conf,
		spreadsheet,
		sheet,
		write_configs,
		cleanup_empty_history,
		discover,
		update_sheet,
		execute,
		sleep_seconds,
):
	if write_configs:
		write_updated_configs(config_dir)

	mac = load_machine_config(machine_conf)
	drive = Path(mac["storage"]["drive"])

	if len(sample_configs) > 0:
		resolved_configs = list(sample_configs)
	elif sample_list_file is not None:
		resolved_configs = parse_sample_list(sample_list_file)
	elif discover:
		resolved_configs = discover_sample_configs(drive)
	else:
		resolved_configs = []

	run_meta_shift(
		resolved_configs,
		drive,
		spreadsheet,
		sheet,
		google_conf,
		sleep_seconds=sleep_seconds,
		execute=execute,
		update_sheet=update_sheet,
	)

	if cleanup_empty_history:
		cleanup_empty_history(drive)


if __name__ == "__main__":
	meta_shift()
