from datetime import datetime, timedelta
from pathlib import Path

import click


from mctutil.shared import log


@click.command()
@click.option("--scan_root", type=click.Path(path_type=Path), required=True,
				help="Root to search for projection start/stop from images.")
def timecheck(scan_root):
	scandatalist = []
	for projections in scan_root.rglob("projections"):
		if not projections.is_dir():
			continue
		files = sorted(p for p in projections.iterdir() if p.is_file())
		if not files:
			continue
		base = projections.parent
		try:
			stop = int(files[-1].stem.split('_')[-1])
			start = int(files[0].stem.split('_')[-1])
			duration = stop - start
			scan_duration = timedelta(milliseconds=duration / 1000000)
			with (base / 'scanlog.txt').open('r') as handle:
				timestring = handle.readline()[-27:-1]
			scan_start = datetime.strptime(timestring, '%Y-%m-%d %H:%M:%S.%f')
			projection_size = files[-1].stat().st_size / 1000000
			scan_label = '_'.join(base.name.split('_')[3:-1])
			scandatalist.append(
				f"{scan_start},{scan_start + scan_duration},"
				f"{base.parent.parent.name},{base.parent.name},"
				f"{scan_label},{projection_size:.1f}MB"
			)
		except Exception as ex:
			log.log("Time Check", f"{ex}", log_level=log.DEBUG.ERROR)
	scandatalist.sort()
	for line in scandatalist:
		log.log("Time Check", line, log_level=log.DEBUG.INFO)


if __name__ == '__main__':
	timecheck()
