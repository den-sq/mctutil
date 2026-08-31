from __future__ import annotations

from pathlib import Path

import pytest

from mctutil.parse.importers.diagnostics import (
	BASE_DIAGNOSTIC_CODES,
	Diagnostic,
	DiagnosticCodeRegistry,
	Severity,
	StrictnessMode,
	diagnostic_status,
	evaluate_write,
)
from mctutil.parse.importers.model import (
	CanonicalValue,
	ImportRecord,
	SourceEvidence,
)


def _evidence() -> SourceEvidence:
	return SourceEvidence(
		Path("fixture.json"),
		"primary",
		"/scan_id",
		"scan-1",
		None,
		"observed",
	)


def _diagnostic(
	severity: Severity, code: str = "value.type_invalid"
) -> Diagnostic:
	return Diagnostic(severity, code, "test diagnostic", "scan_id", (_evidence(),))


def _record(*diagnostics: Diagnostic) -> ImportRecord:
	value = CanonicalValue("scan-1", "observed", (_evidence(),))
	return ImportRecord(
		"scan",
		"scan/1.0",
		"fixture-scan",
		{"scan_id": value},
		diagnostics=diagnostics,
	)


def test_diagnostic_model_validates_shape_and_immutable_evidence() -> None:
	diagnostic = _diagnostic(Severity.WARNING)
	assert diagnostic.evidence == (_evidence(),)
	with pytest.raises(ValueError, match="invalid diagnostic code"):
		Diagnostic(Severity.ERROR, "not-namespaced", "bad")
	with pytest.raises(TypeError, match="Severity"):
		Diagnostic("error", "value.type_invalid", "bad")  # type: ignore[arg-type]


def test_source_adds_codes_without_mutating_shared_definitions() -> None:
	base_snapshot = dict(BASE_DIAGNOSTIC_CODES)
	shared = DiagnosticCodeRegistry.shared()
	extended = shared.with_source(
		"chenglab_camera", {"formula_row_rejected", "timezone_unknown"}
	)
	diagnostic = Diagnostic(
		Severity.WARNING,
		"chenglab_camera.formula_row_rejected",
		"formula-filled empty row was rejected",
	)

	assert not shared.contains(diagnostic.code)
	assert extended.contains(diagnostic.code)
	extended.validate(diagnostic)
	assert dict(BASE_DIAGNOSTIC_CODES) == base_snapshot
	with pytest.raises(ValueError, match="reserved"):
		shared.with_source("value", {"source_owned"})


def test_record_status_and_summary_are_derived_from_diagnostics() -> None:
	record = _record(
		_diagnostic(Severity.INFO),
		_diagnostic(Severity.WARNING, "scan.frame_gap"),
	)

	assert diagnostic_status(record.diagnostics) == "warning"
	assert record.diagnostic_status == "warning"
	assert record.diagnostic_summary == "scan.frame_gap: test diagnostic"


def test_default_mode_writes_warnings_and_record_errors() -> None:
	records = (
		_record(_diagnostic(Severity.WARNING)),
		_record(_diagnostic(Severity.ERROR)),
	)
	decision = evaluate_write(records)

	assert decision.should_write is True
	assert decision.record_statuses == ("warning", "error")


def test_strict_mode_blocks_any_record_error() -> None:
	decision = evaluate_write(
		(_record(_diagnostic(Severity.ERROR)),), mode=StrictnessMode.STRICT
	)

	assert decision.should_write is False
	assert decision.reason == "strict_record_error"


def test_validate_only_runs_policy_without_output() -> None:
	decision = evaluate_write((_record(),), mode=StrictnessMode.VALIDATE_ONLY)

	assert decision.should_write is False
	assert decision.reason == "validate_only"


@pytest.mark.parametrize("mode", tuple(StrictnessMode))
def test_invocation_level_uncertainty_always_blocks(mode: StrictnessMode) -> None:
	decision = evaluate_write(
		(_record(),),
		mode=mode,
		invocation_diagnostics=(
			Diagnostic(Severity.ERROR, "source.ambiguous", "two sources matched"),
		),
	)

	assert decision.should_write is False
	assert decision.reason == "invocation_error"
