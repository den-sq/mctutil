"""Expose the legacy mem clean subcommand under the unified CLI."""

from mem.clean import memclean

command = memclean.commands["clean"]
