"""Unified top-level CLI entrypoint for mctutil."""

import click

from mctutil.hpc import main as hpc_group
from mctutil.mem import main as mem_group
from mctutil.mesh import main as mesh_group
from mctutil.ng import main as ng_group
from mctutil.parse import main as parse_group
from mctutil.sino import main as sino_group
from mctutil.transform import main as transform_group
from mctutil.transport import main as transport_group


@click.group(help="Unified mctutil command surface.")
def main():
	"""Root command for the Phase 4 unified CLI."""


main.add_command(transform_group, "transform")
main.add_command(sino_group, "sino")
main.add_command(ng_group, "ng")
main.add_command(mesh_group, "mesh")
main.add_command(transport_group, "transport")
main.add_command(mem_group, "mem")
main.add_command(parse_group, "parse")
main.add_command(hpc_group, "hpc")
