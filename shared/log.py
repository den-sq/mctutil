from datetime import datetime
from enum import Enum
from sys import stdout
from typing import TextIO

import click
import psutil

script_start = datetime.now()

__attached_funcs = []


# More specific debug-level logging.
class DEBUG(Enum):
	SILENT = ("SILENT", 0, "black")
	ERROR = ("ERROR", 1, "red")
	STATUS = ("STATUS", 2, "green")
	TIME = ("TIME", 3, "cyan")
	WARN = ("WARN", 4, "yellow")
	INFO = ("INFO", 5, "white")

	def __str__(self):
		return str(self.value[0])

	def __le__(self, other):
		return self.value[1] <= other.value[1]

	@property
	def color(self):
		return self.value[2]


def __log_message(step: str, statement: str = '', log_level: DEBUG = DEBUG.TIME):
	styled_type = click.style(f'{log_level.name:6}', log_level.color)
	message = (f'{styled_type}'
			f'|{step[:20]:20}'
			f'|{str(datetime.now() - script_start).zfill(15)}'
			f'|{psutil.Process().memory_info().vms // 1024 ** 2:09.2f}MB'
			f'|{psutil.virtual_memory().available // 1024 ** 2:09.2f}MB'
			f'|"{statement}"')
	return message


def start():
	click.echo(f'{"TYPE":6}|{"STEP":20}|   TIMESTAMP   | MEM USAGE | MEM FREE  | STATEMENT ')
	log("Script Start", script_start)


def log(step: str, statement: str = '', log_level: DEBUG = DEBUG.TIME, out: TextIO = stdout,
		pid: int = psutil.Process().pid):
	for func in __attached_funcs:
		func(step, pid)
	click.echo(__log_message(step, statement, log_level), file=out, err=(log_level == DEBUG.ERROR))


def log_progress(step: str, items, length=None, disp=None, out: TextIO = stdout):
	styled_type = click.style(f'{DEBUG.STATUS.name:6}', DEBUG.STATUS.color)
	return click.progressbar(items, length=length, item_show_func=disp, file=out, show_eta=True, show_pos=True,
				label=f'{styled_type}|{step[:20]:20}', info_sep='|', width=39,
				bar_template="%(label)s|%(bar)s|%(info)s|")


def log_confirm(step: str, statement: str = '', log_level: DEBUG = DEBUG.TIME, out: TextIO = stdout,
				pid: int = psutil.Process().pid):
	for func in __attached_funcs:
		func(step, pid)
	return click.confirm(__log_message(step, statement, log_level), err=(log_level == DEBUG.ERROR))


def log_prompt(step: str, statement: str = '', log_level: DEBUG = DEBUG.TIME, out: TextIO = stdout,
				pid: int = psutil.Process().pid, default=None):
	for func in __attached_funcs:
		func(step, pid)
	return click.prompt(__log_message(step, statement, log_level), err=(log_level == DEBUG.ERROR), default=default)


def attach_func(func: callable):
	""" Attaches a function to be called during a logging step.
		e.g. can be used to pass data to other processes for more granular logging.

		:param func:  Callable object to be called on each log.
		"""
	if func not in __attached_funcs:
		__attached_funcs.append(func)


def cleanup_mem(*shm_objects):
	""" Close and unlink shared memory objects.

		:param shm_objects: Shared memory objects to shut down.
	"""
	for shm in shm_objects:
		if shm is not None:
			shm.close()
			shm.unlink()


def exit_cleanly(step: str, *shm_objects, return_code: int = 0, statement: str = '', log_level: DEBUG = DEBUG.TIME,
					out: TextIO = stdout, throw: Exception = None):
	""" Exit while cleaning up shared memory.

		:param step: Step of reconstruction process we are exiting during.
		:param shm_objects: Shared memory objects to shut down.
		:param return_code: Process return code to send.
	"""
	log(step, statement, log_level, out)
	cleanup_mem(*shm_objects)

	if throw is not None:
		raise throw
	exit(return_code)
