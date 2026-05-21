"""Generic meta-shift engine.

Drives a per-sample loop that delegates schema knowledge (folder conventions,
status enum, sbatch parsing, sheet row layout) to a pluggable adapter. The
default adapter is `chenglab.meta_shift:ChenglabMicroCTAdapter`; additional
adapters can be registered by adding an entry to `ADAPTER_REGISTRY`.
"""

from __future__ import annotations

from importlib import import_module
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


from mctutil.shared import log

yaml = YAML()


# Schema registry. Keys are the values accepted for --schema; values are
# "module:Class" import paths that resolve lazily so optional schema-specific
# dependencies are not pulled in just to list --help.
ADAPTER_REGISTRY = {
	"chenglab": "chenglab.meta_shift:ChenglabMicroCTAdapter",
}


def load_adapter(name: str):
	if name not in ADAPTER_REGISTRY:
		raise click.BadParameter(
			f"Unknown schema '{name}'. Known schemas: {', '.join(sorted(ADAPTER_REGISTRY))}.",
			param_hint="--schema",
		)
	module_name, attr_name = ADAPTER_REGISTRY[name].split(":", 1)
	module = import_module(module_name)
	return getattr(module, attr_name)()


def shift_old_new(adapter, drive, conf_dict, old_paths, new_paths, run_params, scan_num, execute=False):
	status = adapter.empty_status()
	keep_pairs = {}
	move_pairs = {}
	can_move = True

	for step in old_paths:
		old_target = drive.joinpath(old_paths[step])
		new_target = drive.joinpath(new_paths[step])
		step_status = adapter.status_for_step(step)
		if old_target.exists() and step_status is not adapter.empty_status():
			status = step_status

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
		adapter,
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
	row = adapter.build_sheet_row(conf_dict, run_params, drive, move_pairs, scan_num, status)
	return write_sheets_fields(creds, google_conf, gscopes, spreadsheet, sheet, row)


def parse_sample_list(sample_list_file):
	return [
		Path(line.strip())
		for line in sample_list_file.read_text().splitlines()
		if line.strip() != ''
	]


def write_updated_configs(adapter, config_dir, empty_id=0):
	for conf_path in [conf_path for conf_path in Path(config_dir).iterdir() if conf_path.suffix == ".yaml"]:
		conf_dict = adapter.transform_sample_config(conf_path, empty_id)
		new_conf_path = Path(conf_path.parent, "new", conf_path.name)
		new_conf_path.parent.mkdir(parents=True, exist_ok=True)
		with open(new_conf_path, "w") as out:
			yaml.dump(conf_dict, out)


def cleanup_empty_history(drive):
	for history in list(drive.rglob("*history")):
		for inner_dir in history.iterdir():
			if inner_dir.is_dir() and not bool(len(list(inner_dir.iterdir()))):
				log.log("Meta Shift", f"Removing empty {inner_dir}", log_level=log.DEBUG.STATUS)
				inner_dir.rmdir()


def load_machine_config(machine_conf):
	with open(machine_conf) as mc:
		return yaml.load(mc)


def run_meta_shift(
		adapter,
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
			conf_dict = adapter.load_sample_config(sample_conf, empty_id)
			old_paths, new_paths = adapter.compute_paths(conf_dict, job_id)
			run_params = adapter.extract_run_params(sample_conf)

			scan_num = adapter.scan_num_from_config(sample_conf)
			if "job_id" not in run_params:
				run_params["job_id"] = job_id

			status, written, move_pairs, keep_pairs = shift_old_new(
				adapter,
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
					adapter,
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
			log.log("Meta Shift", f"{sample_conf}: {ex}: {traceback.format_exc()}",
					log_level=log.DEBUG.ERROR)
		if sleep_seconds > 0:
			sleep(sleep_seconds)


@click.command()
@click.option(
	"--schema",
	type=click.Choice(sorted(ADAPTER_REGISTRY), case_sensitive=False),
	default=lambda: os.environ.get("MCTUTIL_META_SHIFT_SCHEMA", "chenglab"),
	show_default="chenglab",
	help="Adapter selecting per-lab folder conventions, status enum, sbatch format, and sheet layout.",
)
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
	default=lambda: os.environ.get("MCTUTIL_GSHEET_ID"),
	help="Google Sheets spreadsheet ID. Defaults to the selected schema's default if set, otherwise required.",
)
@click.option(
	"--sheet",
	default=lambda: os.environ.get("MCTUTIL_GSHEET_SHEET"),
	help="Google Sheets sheet/tab name. Defaults to the selected schema's default if set, otherwise required.",
)
@click.option(
	"--write-configs",
	is_flag=True,
	help="Write updated sample YAML files into <config-dir>/new before processing.",
)
@click.option(
	"--cleanup-empty-history",
	"cleanup_empty",
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
		schema,
		machine_conf,
		config_dir,
		sample_configs,
		sample_list_file,
		google_conf,
		spreadsheet,
		sheet,
		write_configs,
		cleanup_empty,
		discover,
		update_sheet,
		execute,
		sleep_seconds,
):
	adapter = load_adapter(schema)

	if spreadsheet is None:
		spreadsheet = getattr(adapter, "default_spreadsheet", None)
	if sheet is None:
		sheet = getattr(adapter, "default_sheet", None)

	if update_sheet and (spreadsheet is None or sheet is None):
		raise click.UsageError(
			"--spreadsheet and --sheet are required when --update-sheet is on and the selected "
			"schema has no defaults (set MCTUTIL_GSHEET_ID / MCTUTIL_GSHEET_SHEET or pass them explicitly)."
		)

	if write_configs:
		write_updated_configs(adapter, config_dir)

	mac = load_machine_config(machine_conf)
	drive = Path(mac["storage"]["drive"])

	if len(sample_configs) > 0:
		resolved_configs = list(sample_configs)
	elif sample_list_file is not None:
		resolved_configs = parse_sample_list(sample_list_file)
	elif discover:
		resolved_configs = adapter.discover_sample_configs(drive)
	else:
		resolved_configs = []

	run_meta_shift(
		adapter,
		resolved_configs,
		drive,
		spreadsheet,
		sheet,
		google_conf,
		sleep_seconds=sleep_seconds,
		execute=execute,
		update_sheet=update_sheet,
	)

	if cleanup_empty:
		cleanup_empty_history(drive)


if __name__ == "__main__":
	meta_shift()
