from multiprocessing import cpu_count
from pathlib import Path
from taskqueue import LocalTaskQueue

import click
import igneous.task_creation as tc


@click.command()
@click.option("-p", "--proc-count", type=click.INT, default=cpu_count,
				help="# of processes.")
@click.argument("REMOTE_FOLDER", nargs=1, type=click.Path(path_type=Path))
def mesh_ig(proc_count, remote_folder):
	tq = LocalTaskQueue(parallel=proc_count)
	tq.insert(tc.create_meshing_tasks(remote_folder, mip=0))
	tq.execute()
	tq.insert(tc.create_mesh_manifest_tasks(remote_folder))
	tq.execute()



if __name__ == "__main__":
	mesh_ig()
