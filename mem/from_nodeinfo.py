
from subprocess import run

node_list = []

with open("node_list.txt") as node_info:
	next(node_info)
	for line in node_info:
		status = line[-8:].strip()
		if status == "idle" or status == "mix":
			# Add node to list of those to clear.
			node_list.append(line[:16].strip())
		else:
			print(status)

	print(node_list)

	for node in node_list:
		run(["sbatch", "-w", node, "memclean.sbatch"])

