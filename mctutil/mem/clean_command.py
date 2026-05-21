"""Expose the legacy mem clean subcommand under the unified CLI."""

from mctutil.mem.clean import memclean

command = memclean.commands["clean"]
