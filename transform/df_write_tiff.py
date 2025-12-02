import os
from pathlib import Path
import sys


def set_df_environment():
	df_dir = Path("C:\\Program Files\\Dragonfly")
	ors_dir = Path("C:\\ProgramData\\ORS\\Dragonfly2024.1")
	user_dir = Path("C:\\Users\\dnorthover\\AppData\\Local\\ORS\\Dragonfly2024.1")
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

	print('Python %s on %s' % (sys.version, sys.platform), flush=True)


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
@click.argument("OUTPUTDIR", type=click.Path(exists=False, writable=True, dir_okay=True, path_type=Path))
def df_write_tiff(df_source, df_object, df_title, df_id, outputdir):
	roi_set = List()
	roi_set.loadFromFileFiltered(df_source, False, ['CxvLabeledMultiROI'], Progress())

	if (df_object is not None) and (df_title is not None):
		source = Managed.getAllObjectsOfClassAndTitle(df_object, df_title)[0]
	else:
		source = orsObj(df_id)

	tf.imwrite(outputdir.joinpath(f"{roi.getTitle()}.tif"), source.getAsNDArray(0))


if __name__ == "__main__":
	df_write_tiff()
