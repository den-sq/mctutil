"""Unified top-level CLI entrypoint for mctutil."""

import click

from mctutil.als832 import main as als832_group
from mctutil.flats import main as flats_group
from mctutil.hpc import main as hpc_group
from mctutil.mem import main as mem_group
from mctutil.mesh import main as mesh_group
from mctutil.ng import main as ng_group
from mctutil.parse import main as parse_group
from mctutil.serve import main as serve_group
from mctutil.shared.log import log, LOG_MASK_QUIET, LOG_MASK_DEFAULT, LOG_MASK_VERBOSE, LOG_MASK_ALL
from mctutil.sino import main as sino_group
from mctutil.transform import main as transform_group
from mctutil.transport import main as transport_group


_LOG_LEVEL_MASKS = {
	"quiet": LOG_MASK_QUIET,
	"default": LOG_MASK_DEFAULT,
	"verbose": LOG_MASK_VERBOSE,
	"debug": LOG_MASK_ALL,
}


@click.group(help="Unified mctutil command surface.")
@click.option(
	"--log-level",
	type=click.Choice(list(_LOG_LEVEL_MASKS), case_sensitive=False),
	default="default",
	show_default=True,
	help=(
		"Verbosity for stdout. quiet=ERROR only; default adds STATUS/TIME/WARN; "
		"verbose adds INFO; debug adds DEBUG."
	),
)
@click.option("--quiet", "-q", "quiet_flag", is_flag=True,
				help="Shorthand for --log-level quiet.")
@click.option("--verbose", "-v", "verbose_flag", is_flag=True,
				help="Shorthand for --log-level verbose.")
def main(log_level, quiet_flag, verbose_flag):
	"""Root command for the Phase 4 unified CLI."""
	if quiet_flag:
		log_level = "quiet"
	elif verbose_flag:
		log_level = "verbose"
	log.set_threshold(_LOG_LEVEL_MASKS[log_level.lower()])


main.add_command(transform_group, "transform")
main.add_command(sino_group, "sino")
main.add_command(als832_group, "als832")
main.add_command(flats_group, "flats")
main.add_command(ng_group, "ng")
main.add_command(serve_group, "serve")
main.add_command(mesh_group, "mesh")
main.add_command(transport_group, "transport")
main.add_command(mem_group, "mem")
main.add_command(parse_group, "parse")
main.add_command(hpc_group, "hpc")
