from pathlib import Path

drive = Path("/gpfs/Labs/Cheng/phenome/")

history_list = list(drive.rglob("*history"))

for history in history_list:
	for inner_dir in history.iterdir():
		if not bool(len(list(inner_dir.iterdir()))):
			print(f"Removing Empty {inner_dir}")
			inner_dir.rmdir()
