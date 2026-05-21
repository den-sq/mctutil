import os
from pathlib import Path
import sys

# Needed to run script from subfolder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared import log 	# noqa::E402


def _env_path(name: str, default: str) -> Path:
	return Path(os.environ.get(name, default))


def set_df_environment():
	df_dir = _env_path("DRAGONFLY_DIR", "C:\\Program Files\\Dragonfly")
	ors_dir = _env_path("DRAGONFLY_ORS_DIR", "C:\\ProgramData\\ORS\\Dragonfly2024.1")
	user_dir = _env_path("DRAGONFLY_USER_DIR", "C:\\Users\\dnorthover\\AppData\\Local\\ORS\\Dragonfly2024.1")
	ana_dir = df_dir.joinpath("Anaconda3")

	os.environ["orspath"] = str(df_dir)
	os.environ["orspython"] = str(ors_dir.joinpath("python"))
	os.environ["orspythonhome"] = str(df_dir.joinpath("Anaconda3"))
	os.environ["pythonhome"] = str(df_dir.joinpath("Anaconda3"))
	os.environ["pythonpath"] = "%orspython%"
	os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "%orspath%\\platforms"
	os.environ["GMT_SHAREDIR"] = "%orspath%\\libs"

	sys.path.extend([str(x) for x in
		[df_dir, df_dir.joinpath("libs"), df_dir.joinpath("plugins"), ors_dir.joinpath("python"),
		ana_dir, ana_dir.joinpath("scripts"), ana_dir.joinpath("library\\bin"),
		ana_dir.joinpath("Lib\\site-packages\\pywin32_system32"),
		ors_dir.joinpath("pythonAllUsersExtensions"), user_dir.joinpath("pythonUserExtensions")]])

	log.log("DF Write TIFF", f"Python {sys.version} on {sys.platform}", log_level=log.DEBUG.INFO)


set_df_environment()

import click 	# noqa:E402
from config.pythonConsoleAutoImport import List, Managed, orsObj, roi, Progress 	# noqa:E402
import tifffile as tf 	# noqa:E402


@click.command()
@click.option("-s", "--df-source", type=click.Path(exists=True, dir_okay=False, path_type=Path),
				help="Path to ORSObjects or Sessions to load data from.")
@click.option("-o", "--df-object", type=click.STRING, help="Type of object to output, if done by class and title.")
@click.option("-t", "--df-title", type=click.STRING, help="Title of object to output, if done by class and title.")
@click.option("-i", "--df-id", type=click.STRING, help="ID of object to output, if done by id.")
@click.option('--execute/--dry-run', default=True,
				help="Whether to actually write the TIFF or just describe the planned write.")
@click.argument("OUTPUTDIR", type=click.Path(exists=False, writable=True, dir_okay=True, path_type=Path))
def df_write_tiff(df_source, df_object, df_title, df_id, execute, outputdir):
	roi_set = List()
	roi_set.loadFromFileFiltered(df_source, False, ['CxvLabeledMultiROI'], Progress())

	if (df_object is not None) and (df_title is not None):
		source = Managed.getAllObjectsOfClassAndTitle(df_object, df_title)[0]
	else:
		source = orsObj(df_id)

	target = outputdir.joinpath(f"{source.getTitle()}.tif")
	if execute:
		tf.imwrite(target, source.getAsNDArray(0))
		log.log("DF Write TIFF", f"Wrote {target}", log_level=log.DEBUG.STATUS)
	else:
		log.log("DF Write TIFF", f"Would write {target}", log_level=log.DEBUG.INFO)


if __name__ == "__main__":
	df_write_tiff()
