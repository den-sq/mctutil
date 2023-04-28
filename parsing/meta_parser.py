# %% [markdown]
# # Metadata Transition
# These scripts transition existing folders from old (inner_comment) to new (outer_inner) id scheme,
# gets updated configs for samples and parameters, and sends the information to google sheets.
# #### Library Import

# %%
from enum import Enum, auto
from pathlib import Path
from ruamel.yaml import YAML
from time import sleep
import traceback

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

yaml = YAML()

# %% [markdown]
# #### Recon Status Tracking.

# %%


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

	def from_step(step, uc=True):
		if step == "flats":
			return STATUS.FLATS_GENERATED_UC if uc else STATUS.FLATS_GENERATED
		elif step == "centerfind":
			return STATUS.CENTERS_DUMPED_UC if uc else STATUS.CENTERS_DUMPED
		elif step == "recon":
			return STATUS.RECONSTRUCTED_UC if uc else STATUS.RECONSTRUCTED
		else:
			return STATUS.EMPTY

# %% [markdown]
# #### Metadata Handling and Printing Functions

# %%


# Parses comment from override folder name (end of path)
def get_outer_comment(folder):
	components = folder.name.split("_")
	outer_comment = []
	energy = components[-1]
	for part in components:
		if len(part) == 6 and part[-3:].isnumeric() and not part[:3].isnumeric():
			continue
		elif not part == energy:
			outer_comment.append(part)
	return "_".join(outer_comment) if len(outer_comment) > 0 else ""


# Returns a config file with new values (inner/outer comment, it)
def get_updated_config(old_config, empty_id):
	with open(old_config) as conf:
		conf_dict = yaml.load(conf)
	if "comment" in conf_dict["storage"]:
		base_comment = conf_dict["storage"].pop("comment").removesuffix("scans").removesuffix("_")
		conf_dict["storage"]["has_scan"] = base_comment[-4:] == "scan"
		conf_dict["storage"]["inner_comment"] = base_comment.removesuffix("scan").removesuffix("_")
		conf_dict["storage"]["outer_comment"] = get_outer_comment(Path(conf_dict["storage"].pop("override_input")))
	scan_id = ("_".join([str(conf_dict["storage"]["outer_comment"]), str(conf_dict["storage"]["inner_comment"])])).strip("_")
	if scan_id == '':
		scan_id = f"SS{empty_id}"
		empty_id += 1
	conf_dict["scan"]["id"] = scan_id
	return conf_dict


# Gets the paths that need to change (from/to) with new id.
def get_paths(conf: dict, job_id):
	steps = ["flats", "centerfind", "recon"]
	old_paths = {}
	new_paths = {}
	data_path = Path(conf["storage"]["project"], conf["sample"]["id"], "data")

	for step in steps:
		if conf["storage"]["has_scan"]:
			old_folder = f'{conf["scan"]["energy"]}kV_{conf["storage"]["inner_comment"]}_scan_{step}_p{job_id}'.replace("__", "_")
		else:
			old_folder = f'{conf["scan"]["energy"]}kV_{conf["storage"]["inner_comment"]}_{step}_p{job_id}'
		new_folder = f'{conf["scan"]["energy"]}kV_{conf["scan"]["id"]}_{step}_p{job_id}'

		old_paths[step] = data_path.joinpath(old_folder)
		new_paths[step] = data_path.joinpath(new_folder)

	old_paths["script"] = data_path.joinpath("history", f'{conf["scan"]["energy"]}kV_{conf["storage"]["inner_comment"]}',
												job_id)
	new_paths["script"] = data_path.joinpath("history", f'{conf["scan"]["energy"]}kV_{conf["scan"]["id"]}', job_id)

	return old_paths, new_paths


# Parses batch file for correct set of parameters
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
			if line[:4] == "phase_alpha":
				phase_alpha = line.split("=")[1]
			if line[:4] == "base":
				base = line.split(" ")[2]

	run_params = {command_pair.split("=")[0][2:]: command_pair.split("=")[1]
				for command_pair in run_command.split(".py")[1].strip("\n").strip().split(" ") if "=" in command_pair}

	if "phase_tomo" not in run_params:
		if "phase_alpha" in run_params:
			run_params["phase_tomo"] = False
		else:
			run_params["phase_tomo"] = True
	if "stripe_removal" not in run_params:
		run_params["stripe_removal"] = "VO (all)"
	if "phase_alpha" not in run_params:
		run_params["phase_alpha"] = 0.03

	if phase_alpha is not None:
		run_params["phase_alpha"].replace("$phase_alpha", phase_alpha)

	run_params["slice_range"] = run_params["slice_range"].replace("$start", start_param).replace("$stop", stop_param).replace("base", base)

	return run_params


# Executes (or simulates - default) actual shift from old to new folder paths.
def shift_old_new(drive, conf_dict, old_paths, new_paths, run_params, scan_num, execute=False):
	status = STATUS.EMPTY
	keep_pairs = {}
	move_pairs = {}
	fail_pairs = {}
	can_move = True

	for step in old_paths:
		if drive.joinpath(old_paths[step]).exists() and STATUS.from_step(step) is not STATUS.EMPTY:
			status = STATUS.from_step(step) 	# Update Status due to source existence.

		if old_paths[step] == new_paths[step] and drive.joinpath(old_paths[step]).exists():
			keep_pairs[step] = {"from": drive.joinpath(old_paths[step]),
								"to": drive.joinpath(new_paths[step])}
		elif drive.joinpath(old_paths[step]).exists():
			# Target existence check
			if not drive.joinpath(new_paths[step]).exists():
				move_pairs[step] = {"from": drive.joinpath(old_paths[step]),
									"to": drive.joinpath(new_paths[step])}
			else:
				fail_pairs[step] = {"from": drive.joinpath(old_paths[step]),
									"to": drive.joinpath(new_paths[step])}
				can_move = False
		else:
			fail_pairs[step] = {"from": drive.joinpath(old_paths[step]),
								"to": drive.joinpath(new_paths[step])}

	# If all targets can move, move.
	if can_move and execute:
		for step in move_pairs:
			if not move_pairs[step]["to"].parent.exists():
				move_pairs[step]["to"].parent.mkdir(parents=True)
			move_pairs[step]["from"].rename(move_pairs[step]["to"])
	if can_move:
		print(f"Can Move For {scan_num}")
	else:
		print(f"Cannot Move For {scan_num}")

	if can_move and execute:
		hist_dir = drive.joinpath(new_paths["script"])
		conf_name = f'{scan_num}_{conf_dict["sample"]["id"]}_{conf_dict["scan"]["id"]}.yaml'

		# Write new config yaml
		with open(hist_dir.joinpath(conf_name), "w") as new_conf:
			yaml.dump(conf_dict, new_conf)

		# Write new parameters yaml.
		with open(hist_dir.joinpath("batch_extracted.yaml"), "w") as new_params:
			yaml.dump(run_params, new_params)

	# Return if wrote and current status
	return status, can_move, move_pairs, keep_pairs


# %% [markdown]
# #### Google Sheets Interaction Functions

# %%
def establish_google_creds(cred_name):
	token_path = Path(google_conf, f"{cred_name}_token.json")
	cred_path = Path(google_conf, f'{cred_name}_credentials.json')

	if token_path.exists():
		creds[cred_name] = Credentials.from_authorized_user_file(token_path, GSCOPES)

	if cred_name not in creds or not creds[cred_name].valid:
		if cred_name in creds and creds[cred_name].expired and creds[cred_name].refresh_token:
			creds[cred_name].refresh(Request())
		else:
			if cred_path.exists():
				flow = InstalledAppFlow.from_client_secrets_file(
						Path(google_conf, f'{cred_name}_credentials.json'), GSCOPES)
				creds[cred_name] = flow.run_local_server(port=0)
			else:
				raise FileNotFoundError(cred_path)
		with open(token_path, "w") as gtoken:
			gtoken.write(creds[cred_name].to_json())


def write_sheets_fields(spreadsheet, sheet, values):
	cred_name = "gsheets"
	if cred_name not in creds:
		establish_google_creds(cred_name)
	try:
		service = build('sheets', 'v4', credentials=creds[cred_name]).spreadsheets()

		body_val = {"range": f'{sheet}!A1:A2', "values": [values]}

		result = service.values().append(
			spreadsheetId=spreadsheet, range=f'{sheet}!A1:A2', valueInputOption="RAW", insertDataOption="INSERT_ROWS",
			body=body_val).execute()
		print(result)

	except HttpError as err:
		print(f"Google Sheets Error {err}")
	except Exception as err:
		print(f"Other Error {err}")


# Updates google sheets with values.
def update_gsheet(conf_dict, run_params, drive, move_pairs, scan_num, status, spreadsheet, sheet):
	# generate fields components
	recon_line = []
	remote_params = ["machine_conf", "sample_conf", "old_file_naming", "window_size", "slice_range", "recon_algorithm",
					"job_id", "phase_alpha", "stripe_removal", "phase_tomo", "rot_center", "tilt"]
	folder_fields = ["flats", "centerfind", "recon", "script"]

	folders = [str(drive.joinpath(conf_dict["storage"]["project"], conf_dict["sample"]["id"], "data"))]

	# f'{sheet}!B:F'
	recon_line += ['',
		conf_dict["sample"]["id"], conf_dict["storage"]["trip_dir"], scan_num, conf_dict["scan"]["id"],
		conf_dict["storage"]["project"], ''
	]

	# f'{sheet}!H:S'
	recon_line += ['' if param not in run_params else str(run_params[param]) for param in remote_params]
	# f'{sheet}!U1:U2'
	recon_line += ['', str(status), '', '', '']

	for field in folder_fields:
		folders.append('' if field not in move_pairs else str(move_pairs[field]["to"].relative_to(folders[0])))

	# f'{sheet}!Y:AC'
	recon_line += folders

	# write call
	write_sheets_fields(spreadsheet, sheet, recon_line)


# %% [markdown]
# #### Setup: Machine and Google Variables

# %%
# Google Configuration Information.,
creds = {}
google_conf = "conf"
GSCOPES = ['https://www.googleapis.com/auth/spreadsheets']
spreadsheet = "1RiCh3kjJhmUKZ5Y4UTwtKeiEag8jS_bU6jBeGJnru18"
sheet = "GPFS (DEN)"

# Local Machine & Sample Configuration
conf_folder = "samples"
machine_conf = Path("mac.yaml")

config_list = [conf_path for conf_path in Path(conf_folder).iterdir() if conf_path.suffix == ".yaml"]

with open(machine_conf) as mc:
	mac = yaml.load(mc)

drive = Path(mac["storage"]["drive"])

# %% [markdown]
# #### Configuration Updates

# %%
empty_id = 0

if False:
	for conf_path in config_list:
		conf_dict = get_updated_config(conf_path, empty_id)
		new_conf_path = Path(conf_path.parent, "new", conf_path.name)
		with open(new_conf_path, "w") as out:
			yaml.dump(conf_dict, out)

# %% [markdown]
# #### Execution

# %%
empty_id = 0

if drive.exists():
	# Has to be separate because we're moving folders will change.
	sample_conf_list = list(drive.rglob("*V.yaml"))

	for sample_conf in sample_conf_list:
		try:
			job_id = sample_conf.parent.stem

			conf_dict = get_updated_config(sample_conf, empty_id)

			# Temporary Skip
			if conf_dict["sample"]["id"] == "AAA480":
				continue

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

			status, written, move_pairs, keep_pairs = shift_old_new(drive, conf_dict, old_paths, new_paths, run_params, scan_num,
																	execute=True)

			if written:
				update_gsheet(conf_dict, run_params, drive, move_pairs | keep_pairs, scan_num, status, spreadsheet, sheet)
		except IndexError as iex:
			print(f"{sample_conf}: {iex}: {traceback.format_exc()}")
		sleep(1.1)
