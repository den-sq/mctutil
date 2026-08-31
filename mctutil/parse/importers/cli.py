"""Generic scan-log and reconstruction-log command frontends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import click

from mctutil.parse import tabular_log

from mctutil.parse.importers.audit import AuditItem, write_audit_jsonl
from mctutil.parse.importers.diagnostics import Severity, StrictnessMode, evaluate_write
from mctutil.parse.importers.engine import ImportEngine
from mctutil.parse.importers.export import (
	EXPORT_PROFILE_REGISTRY,
	ExportProfile,
	append_records,
	write_records_csv,
)
from mctutil.parse.importers.mapping import load_builtin_mapping, load_mapping
from mctutil.parse.importers.model import ImportRecord, RecordKind
from mctutil.parse.importers.readers import open_artifacts
from mctutil.parse.importers.registry import (
	SOURCE_REGISTRY,
	SourceAdapter,
	SourceBundle,
	discover_homogeneous,
	select_adapter,
)
from mctutil.parse.importers.schema import (
	RECONSTRUCTION_SCHEMA_REGISTRY,
	SCAN_SCHEMA_REGISTRY,
)


@dataclass(frozen=True)
class ImportRunResult:
	adapter: SourceAdapter
	profile: ExportProfile
	items: tuple[AuditItem, ...]

	@property
	def records(self) -> tuple[ImportRecord, ...]:
		return tuple(item.record for item in self.items)

	@property
	def bundles(self) -> tuple[SourceBundle, ...]:
		return tuple(item.bundle for item in self.items)


def import_inputs(
	kind: RecordKind,
	inputs: tuple[Path, ...],
	*,
	source_id: str,
	profile_id: str,
	mapping_path: Path | None = None,
) -> ImportRunResult:
	"""Run the source-independent import pipeline without writing destinations."""

	profile = EXPORT_PROFILE_REGISTRY.get(kind, profile_id)
	adapter = select_adapter(
		kind, source_id, inputs, registry=SOURCE_REGISTRY
	)
	mapping = (
		load_mapping(
			mapping_path,
			schema=profile.schema,
			expected_source=adapter.source_id,
		)
		if mapping_path is not None
		else load_builtin_mapping(kind, adapter.source_id, schema=profile.schema)
	)
	bundles = discover_homogeneous(adapter, inputs)
	if not bundles:
		raise ValueError("no source bundles were discovered")
	engine = ImportEngine(profile.schema)
	items = []
	for bundle in bundles:
		with open_artifacts(bundle.all_artifacts) as context:
			record = engine.process_bundle(
				bundle=bundle,
				adapter=adapter,
				mapping=mapping,
				context=context,
			)
		items.append(AuditItem(record, bundle))
	return ImportRunResult(adapter, profile, tuple(items))


def _echo_diagnostics(result: ImportRunResult) -> None:
	for item in result.items:
		record = item.record
		click.echo(
			f"{item.bundle.identity_hint}: {record.diagnostic_status}",
			err=record.diagnostic_status != "valid",
		)
		for diagnostic in record.diagnostics:
			field = f" [{diagnostic.field}]" if diagnostic.field else ""
			click.echo(
				f"  {diagnostic.severity.value} {diagnostic.code}{field}: "
				f"{diagnostic.message}",
				err=diagnostic.severity is not Severity.INFO,
			)


def _strictness_mode(*, strict: bool, validate_only: bool) -> StrictnessMode:
	if strict and validate_only:
		raise click.UsageError("--strict and --validate-only are separate modes.")
	if validate_only:
		return StrictnessMode.VALIDATE_ONLY
	if strict:
		return StrictnessMode.STRICT
	return StrictnessMode.DEFAULT


def _schema_registry(kind: RecordKind):
	return (
		SCAN_SCHEMA_REGISTRY
		if kind == "scan"
		else RECONSTRUCTION_SCHEMA_REGISTRY
	)


def _preflight_local_outputs(
	*,
	output: Path | None,
	audit_output: Path | None,
	force: bool,
) -> None:
	if output is not None and audit_output is not None:
		if output.absolute() == audit_output.absolute():
			raise click.UsageError("--output and --audit-output must be different paths.")
	for path in (output, audit_output):
		if path is not None and path.exists() and not force:
			raise click.ClickException(
				f"Output already exists: {path} (pass --force to replace it)"
			)


def _write_destinations(
	result: ImportRunResult,
	*,
	output: Path | None,
	upload: bool,
	spreadsheet: str | None,
	sheet: str | None,
	create_tab: bool,
	google_conf: Path,
	verify_header: bool,
	strict_header_order: bool,
	audit_output: Path | None,
	force: bool,
) -> None:
	if output is not None:
		write_records_csv(output, result.records, result.profile, force=force)
		click.echo(f"Wrote {len(result.records)} record(s): {output}")
	elif upload:
		try:
			service = tabular_log.build_google_sheets_service(google_conf)
			if create_tab:
				tabular_log.create_tab_if_missing(
					service,
					spreadsheet,
					sheet,
					result.profile.headers,
				)
			response = append_records(
				service,
				spreadsheet,
				sheet,
				result.records,
				result.profile,
				verify=verify_header,
				strict_header_order=strict_header_order,
			)
		except Exception as exc:
			raise click.ClickException(f"Google Sheets upload failed: {exc}") from exc
		updated = response.get("updates", {}).get("updatedRange") or response.get(
			"updatedRange", "unknown range"
		)
		click.echo(f"Uploaded {len(result.records)} record(s) to {updated}")
	if audit_output is not None:
		write_audit_jsonl(
			audit_output,
			result.items,
			profile=result.profile,
			source_version=result.adapter.version,
			schema_registry=_schema_registry(result.profile.kind),
			force=force,
		)
		click.echo(f"Wrote audit sidecar: {audit_output}")


def _run_command(
	kind: RecordKind,
	inputs: tuple[Path, ...],
	*,
	source: str,
	mapping_path: Path | None,
	profile: str,
	output: Path | None,
	upload: bool,
	spreadsheet: str | None,
	sheet: str | None,
	create_tab: bool,
	google_conf: Path,
	verify_header: bool,
	strict_header_order: bool,
	strict: bool,
	validate_only: bool,
	audit_output: Path | None,
	force: bool,
) -> None:
	if not inputs:
		raise click.UsageError("At least one input file or directory is required.")
	mode = _strictness_mode(strict=strict, validate_only=validate_only)
	tabular_log.validate_destination_options(
		upload=upload,
		create_tab=create_tab,
		output=output,
		force=force,
		spreadsheet=spreadsheet,
		sheet=sheet,
		verify=verify_header,
		strict_header_order=strict_header_order,
	)
	if mode is not StrictnessMode.VALIDATE_ONLY:
		_preflight_local_outputs(output=output, audit_output=audit_output, force=force)
	try:
		result = import_inputs(
			kind,
			inputs,
			source_id=source,
			profile_id=profile,
			mapping_path=mapping_path,
		)
	except Exception as exc:
		if isinstance(exc, click.ClickException):
			raise
		raise click.ClickException(str(exc)) from exc
	_echo_diagnostics(result)
	decision = evaluate_write(result.records, mode=mode)
	if decision.reason == "validate_only":
		click.echo(f"Validated {len(result.records)} {kind} record(s); no output written.")
		return
	if not decision.should_write:
		raise click.ClickException(
			"Strict validation failed; no destination output was written."
		)
	_write_destinations(
		result,
		output=output,
		upload=upload,
		spreadsheet=spreadsheet,
		sheet=sheet,
		create_tab=create_tab,
		google_conf=google_conf,
		verify_header=verify_header,
		strict_header_order=strict_header_order,
		audit_output=audit_output,
		force=force,
	)
	if output is None and not upload and audit_output is None:
		click.echo(f"Processed {len(result.records)} {kind} record(s); no output selected.")


def _importer_options(default_profile: str) -> Callable:
	def decorate(function: Callable) -> Callable:
		decorators = (
			click.argument(
				"inputs",
				nargs=-1,
				type=click.Path(exists=True, path_type=Path),
			),
			click.option(
				"--source",
				default="auto",
				show_default=True,
				help="Source adapter ID, or auto for fail-closed detection.",
			),
			click.option(
				"--mapping",
				"mapping_path",
				type=click.Path(exists=True, dir_okay=False, path_type=Path),
				help="Complete replacement YAML mapping.",
			),
			click.option(
				"--profile", default=default_profile, show_default=True
			),
			click.option(
				"--output",
				"-o",
				type=click.Path(dir_okay=False, path_type=Path),
				help="Optional canonical CSV output.",
			),
			click.option(
				"--upload",
				is_flag=True,
				help="Append canonical rows to Google Sheets.",
			),
			click.option(
				"--spreadsheet",
				default=lambda: os.environ.get("MCTUTIL_GSHEET_ID"),
				help="Destination spreadsheet ID for --upload.",
			),
			click.option(
				"--sheet",
				default=lambda: os.environ.get("MCTUTIL_GSHEET_SHEET"),
				help="Destination tab for --upload.",
			),
			click.option(
				"--create-tab",
				is_flag=True,
				help="Create a missing destination tab with profile headers.",
			),
			click.option(
				"--google-conf",
				type=click.Path(path_type=Path),
				default=lambda: Path(os.environ.get("MCTUTIL_GOOGLE_CONF", "conf")),
				show_default="conf",
			),
			click.option(
				"--verify-header/--no-verify-header",
				default=True,
				show_default=True,
			),
			click.option(
				"--strict-header-order",
				is_flag=True,
				help="Require exact profile header ordering.",
			),
			click.option(
				"--strict",
				is_flag=True,
				help="Write nothing if any record has an error.",
			),
			click.option(
				"--validate-only",
				is_flag=True,
				help="Run the full import and diagnostics without output.",
			),
			click.option(
				"--audit-output",
				type=click.Path(dir_okay=False, path_type=Path),
				help="Optional one-way JSON Lines audit sidecar.",
			),
			click.option(
				"--force", is_flag=True, help="Replace existing local output files."
			),
		)
		for option in reversed(decorators):
			function = option(function)
		return function

	return decorate


@click.command("scan-log")
@_importer_options("scan-v1")
def scan_log(**kwargs) -> None:
	"""Import scan metadata into the canonical scan record model."""

	_run_command("scan", **kwargs)


@click.command("reconstruction-log")
@_importer_options("reconstruction-v1")
def reconstruction_log(**kwargs) -> None:
	"""Import reconstruction metadata into the canonical reconstruction model."""

	_run_command("reconstruction", **kwargs)
