from os import walk
from sys import argv
from pathlib import Path

def is_empty(path):
	return path.stat().st_size == 0


job_set = set()
no_error_set = set()

with open(argv[2], "w") as output, open(argv[3], "w") as ne_output:
	for root, dirs, files in walk(argv[1]):
		for file in files:
			if file[:3] == "err":
				err_path = Path(root, file)
				if not is_empty(err_path):
					job_set.add(f'{err_path.parent}\n')
				else:
					no_error_set.add(f'{err_path.parent}\n')

	no_error_set = no_error_set - job_set
	output.writelines(job_set)
	ne_output.writelines(no_error_set)

