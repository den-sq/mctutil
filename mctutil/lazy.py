"""Lazy Click command loader for the unified Phase 4 CLI."""

from importlib import import_module

import click
from click.formatting import measure_table


class LazyGroup(click.Group):
	"""Resolve commands only when they are requested."""

	def __init__(self, *args, lazy_subcommands=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.lazy_subcommands = dict(lazy_subcommands or {})

	def list_commands(self, ctx):
		commands = set(super().list_commands(ctx))
		commands.update(self.lazy_subcommands.keys())
		return sorted(commands)

	def get_command(self, ctx, cmd_name):
		command = super().get_command(ctx, cmd_name)
		if command is not None:
			return command
		import_path = self.lazy_subcommands.get(cmd_name)
		if import_path is None:
			return None
		module_name, attr_name = import_path.split(":", 1)
		module = import_module(module_name)
		return getattr(module, attr_name)

	def format_commands(self, ctx, formatter):
		rows = [(name, "") for name in self.list_commands(ctx)]
		if len(rows) == 0:
			return
		with formatter.section("Commands"):
			formatter.write_dl(rows, col_max=measure_table(rows)[0])
