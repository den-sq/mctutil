# DO NOT USE UNLESS YOU ARE ABSOLUTELY CERTAIN YOU WON'T BE NUKING ANYTHING IMPORTANTs

from os import listdir
from multiprocessing import shared_memory
from sys import argv

shm = {							# Names of shared memory to be script-specific (for now)
	"rsm": "rot",
	"psm": "proj",
	"fsm": "flats",
	"ysm": "y_rot",
	"pfm": "phase",
	"log": "last_step",
	"ssm": "sino",
	"wsm": "work",
	"csm": "center",
	"inp": "input",
}

other_shm = "__KMP_REGISTERED_LIB"

apply = eval(argv[1]) 	# Yes This Is Very Stupid Do Not Make This Accessible Generally

base = listdir("/dev/shm")

for mem_name in base:
	for prefix in shm.values():
		if mem_name[:len(prefix)] == prefix:
			print(mem_name)
			if apply:
				clean_target = shared_memory.SharedMemory(name=mem_name)
				clean_target.close()
				clean_target.unlink()
