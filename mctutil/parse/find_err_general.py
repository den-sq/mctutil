"""Find job directories with non-empty or clean scheduler error files."""

from __future__ import annotations

from pathlib import Path

import click


def classify_error_directories(
	root: Path,
	pattern: str = "err*",
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
	"""Partition matching file parents, with non-empty errors taking priority."""
	errored = set()
	clean = set()
	try:
		for error_path in root.rglob(pattern):
			if not error_path.is_file():
				continue
			if error_path.stat().st_size:
				errored.add(error_path.parent)
			else:
				clean.add(error_path.parent)
	except OSError as exc:
		raise click.ClickException(
			f"Could not scan error files under {root}: {exc}"
		) from exc

	return (
		tuple(sorted(errored, key=lambda path: path.as_posix())),
		tuple(sorted(clean - errored, key=lambda path: path.as_posix())),
	)


def write_path_list(output_path: Path, paths: tuple[Path, ...]) -> None:
	"""Write one directory per line, preserving an empty result as an empty file."""
	try:
		output_path.write_text(
			"".join(f"{path}\n" for path in paths),
			encoding="utf-8",
		)
	except OSError as exc:
		raise click.ClickException(
			f"Could not write directory list {output_path}: {exc}"
		) from exc


def print_summary(
	errored: tuple[Path, ...],
	clean: tuple[Path, ...],
) -> None:
	click.echo(f"Errored directories ({len(errored)}):")
	for path in errored:
		click.echo(f"  {path}")
	click.echo(f"Clean directories ({len(clean)}):")
	for path in clean:
		click.echo(f"  {path}")


@click.command("find-errs")
@click.option(
	"--pattern",
	default="err*",
	show_default=True,
	help="rglob pattern used to select scheduler error files beneath ROOT.",
)
@click.option(
	"--errors-out",
	type=click.Path(dir_okay=False, path_type=Path),
	help="Optional file receiving directories with non-empty error files.",
)
@click.option(
	"--clean-out",
	type=click.Path(dir_okay=False, path_type=Path),
	help="Optional file receiving directories whose matching error files are empty.",
)
@click.argument(
	"root",
	type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def find_errs(
	pattern: str,
	errors_out: Path | None,
	clean_out: Path | None,
	root: Path,
) -> None:
	"""Classify job directories by the contents of matching error files."""
	if (
		errors_out is not None
		and clean_out is not None
		and errors_out.resolve() == clean_out.resolve()
	):
		raise click.UsageError(
			"--errors-out and --clean-out must name different files."
		)

	errored, clean = classify_error_directories(root, pattern)
	print_summary(errored, clean)
	if errors_out is not None:
		write_path_list(errors_out, errored)
	if clean_out is not None:
		write_path_list(clean_out, clean)


if __name__ == "__main__":
	find_errs()
