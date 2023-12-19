from datetime import datetime, timedelta
from os import walk
from pathlib import Path
import click

@click.command()
@click.option("--scan_root", type=click.Path(), required=True, help="Root to search for projection start/stop from images.")
def timecheck(scan_root):
	scandatalist = []
	for root, dirs, files in walk(scan_root):
		if Path(root).name == 'projections':
			base = Path(root).parent
			if len(files) > 0:
				files.sort()
				try:
					stop = int(Path(files[-1]).stem.split('_')[-1])
					start = int(Path(files[0]).stem.split('_')[-1])
					duration = stop - start
					scan_duration = timedelta(milliseconds=duration / 1000000)
					with Path(base, 'scanlog.txt').open('r') as handle:
						timestring = handle.readline()[-27:-1]
					scan_start = datetime.strptime(timestring, '%Y-%m-%d %H:%M:%S.%f')
					projection_size = Path(root, files[-1]).stat().st_size / 1000000
					scandatalist.append(f"{scan_start},{scan_start+scan_duration},{base.parent.parent.name},{base.parent.name},{'_'.join(base.name.split('_')[3:-1])},{projection_size:.1f}MB")
				except Exception as ex:
					print(ex)
	scandatalist.sort()
	for line in scandatalist:
		print(line)
if __name__ == '__main__':
	timecheck()
