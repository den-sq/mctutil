"""Expose the legacy mem mark subcommand under the unified CLI."""

from mem.clean import memclean

command = memclean.commands["mark"]
