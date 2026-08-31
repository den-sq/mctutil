from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

import mctutil.parse.importers.engine as engine_module
from mctutil.parse.importers.diagnostics import Severity
from mctutil.parse.importers.engine import CandidateCollector, ImportEngine
from mctutil.parse.importers.mapping import load_builtin_mapping, load_mapping_text
from mctutil.parse.importers.model import PrecedencePolicy, SourceEvidence
from mctutil.parse.importers.readers import OpenedArtifact, ReaderContext
from mctutil.parse.importers.registry import SOURCE_REGISTRY, SourceArtifact, SourceBundle
from mctutil.parse.importers.schema import RECONSTRUCTION_SCHEMA_V1, SCAN_SCHEMA_V1


@dataclass
class Adapter:
	source_id: str = "fixture-scan"
	kind: str = "scan"
	policies: tuple[PrecedencePolicy, ...] = ()
	derived: tuple[tuple[str, object, SourceEvidence], ...] = ()
	validate_calls: int = 0

	def probe(self, inputs):
		raise NotImplementedError

	def discover(self, inputs):
		return ()

	def precedence_policies(self):
		return self.policies

	def derive(self, bundle, candidates: CandidateCollector) -> None:
		for field_key, value, evidence in self.derived:
			candidates.add_candidate(field_key, value, evidence)

	def validate(self, record, bundle):
		self.validate_calls += 1
		return ()


def _bundle(kind: str = "scan", source_id: str = "fixture-scan") -> SourceBundle:
	return SourceBundle(
		kind,
		source_id,
		SourceArtifact(Path("fixture.json"), "primary", "application/json"),
		identity_hint="fixture",
	)


def _context(payload: dict) -> ReaderContext:
	return ReaderContext(
		(
			OpenedArtifact(
				"primary", Path("fixture.json"), "application/json", payload
			),
		)
	)


def _evidence(
	value: object,
	state: str,
	locator: str,
	unit: str | None = None,
) -> SourceEvidence:
	return SourceEvidence(
		Path("derived.json"), "derived", locator, value, unit, state
	)


def test_fixture_mapping_runs_through_engine_to_canonical_record() -> None:
	adapter = SOURCE_REGISTRY.get("scan", "fixture-scan")
	mapping = load_builtin_mapping("scan", "fixture-scan", schema=SCAN_SCHEMA_V1)
	record = ImportEngine(SCAN_SCHEMA_V1).process_bundle(
		bundle=_bundle(),
		adapter=adapter,
		mapping=mapping,
		context=_context({"scan_id": " scan-1 ", "projection_count": "1200"}),
	)

	assert record.values["scan_id"].value == "scan-1"
	assert record.values["projection_count_acquired"].value == 1200
	assert record.values["projection_count_acquired"].state == "observed"
	assert record.diagnostic_status == "valid"


def test_configured_only_candidate_retains_configured_state() -> None:
	adapter = SOURCE_REGISTRY.get("reconstruction", "fixture-reconstruction")
	mapping = load_builtin_mapping(
		"reconstruction",
		"fixture-reconstruction",
		schema=RECONSTRUCTION_SCHEMA_V1,
	)
	record = ImportEngine(RECONSTRUCTION_SCHEMA_V1).process_bundle(
		bundle=_bundle("reconstruction", "fixture-reconstruction"),
		adapter=adapter,
		mapping=mapping,
		context=_context({"reconstruction_id": "r1", "algorithm": "gridrec"}),
	)

	assert record.values["reconstruction_algorithm"].state == "configured"


def test_default_precedence_selects_observed_then_effective_then_configured() -> None:
	configured = _evidence("configured", "configured", "/configured")
	effective = _evidence("effective", "effective", "/effective")
	observed = _evidence("observed", "observed", "/observed")
	adapter = Adapter(
		derived=(
			("scan_id", "configured", configured),
			("scan_id", "effective", effective),
			("scan_id", "observed", observed),
		)
	)
	empty_mapping = load_mapping_text(
		"mapping_format: 1\nschema: scan/1.0\nsource: fixture-scan\nfields: {}\n",
		schema=SCAN_SCHEMA_V1,
		expected_source="fixture-scan",
	)
	record = ImportEngine(SCAN_SCHEMA_V1).process_bundle(
		bundle=_bundle(), adapter=adapter, mapping=empty_mapping, context=_context({})
	)

	assert record.values["scan_id"].value == "observed"
	assert record.values["scan_id"].state == "observed"
	conflict = next(item for item in record.diagnostics if item.code == "value.conflict")
	assert conflict.severity is Severity.WARNING
	assert {item.locator for item in conflict.evidence} == {
		"/configured",
		"/effective",
		"/observed",
	}


def test_adapter_policy_is_data_applied_by_engine() -> None:
	adapter = Adapter(
		policies=(PrecedencePolicy("scan_id", ("configured", "observed")),),
		derived=(
			("scan_id", "configured", _evidence("configured", "configured", "/c")),
			("scan_id", "observed", _evidence("observed", "observed", "/o")),
		),
	)
	mapping = load_mapping_text(
		"mapping_format: 1\nschema: scan/1.0\nsource: fixture-scan\nfields: {}\n",
		schema=SCAN_SCHEMA_V1,
		expected_source="fixture-scan",
	)
	record = ImportEngine(SCAN_SCHEMA_V1).process_bundle(
		bundle=_bundle(), adapter=adapter, mapping=mapping, context=_context({})
	)

	assert record.values["scan_id"].value == "configured"
	assert adapter.validate_calls == 1


def test_equal_rank_conflict_fails_closed_with_all_locations() -> None:
	adapter = Adapter(
		derived=(
			("scan_id", "one", _evidence("one", "observed", "/one")),
			("scan_id", "two", _evidence("two", "observed", "/two")),
		)
	)
	mapping = load_mapping_text(
		"mapping_format: 1\nschema: scan/1.0\nsource: fixture-scan\nfields: {}\n",
		schema=SCAN_SCHEMA_V1,
		expected_source="fixture-scan",
	)
	record = ImportEngine(SCAN_SCHEMA_V1).process_bundle(
		bundle=_bundle(), adapter=adapter, mapping=mapping, context=_context({})
	)

	assert "scan_id" not in record.values
	conflict = next(item for item in record.diagnostics if item.code == "value.conflict")
	assert conflict.severity is Severity.ERROR
	assert {item.locator for item in conflict.evidence} == {"/one", "/two"}
	assert any(item.code == "value.missing_required" for item in record.diagnostics)


def test_planned_and_acquired_scan_counts_are_independent_fields() -> None:
	mapping = load_mapping_text(
		(
			"mapping_format: 1\n"
			"schema: scan/1.0\n"
			"source: fixture-scan\n"
			"fields:\n"
			"  scan_id:\n"
			"    read: {kind: json_pointer, pointer: /scan_id, state: observed}\n"
			"  projection_count_planned:\n"
			"    read: {kind: json_pointer, pointer: /planned, source_unit: count, state: configured}\n"
			"  projection_count_acquired:\n"
			"    read: {kind: json_pointer, pointer: /actual, source_unit: count, state: observed}\n"
		),
		schema=SCAN_SCHEMA_V1,
		expected_source="fixture-scan",
	)
	record = ImportEngine(SCAN_SCHEMA_V1).process_bundle(
		bundle=_bundle(),
		adapter=Adapter(),
		mapping=mapping,
		context=_context({"scan_id": "s1", "planned": 1000, "actual": 990}),
	)

	assert record.values["projection_count_planned"].value == 1000
	assert record.values["projection_count_planned"].state == "configured"
	assert record.values["projection_count_acquired"].value == 990
	assert record.values["projection_count_acquired"].state == "observed"


def test_unknown_units_and_nonfinite_values_become_diagnostics() -> None:
	collector = CandidateCollector(SCAN_SCHEMA_V1)
	collector.add_candidate(
		"exposure_us",
		1.0,
		_evidence(1.0, "observed", "/exposure", "furlong"),
	)
	collector.add_candidate(
		"rotation_start_deg",
		float("inf"),
		_evidence(float("inf"), "observed", "/theta", "deg"),
	)

	assert collector.candidates == {}
	assert {item.code for item in collector.diagnostics} == {
		"value.unit_unknown",
		"value.normalization_failed",
	}


def test_engine_has_one_canonical_value_construction_site() -> None:
	tree = ast.parse(inspect.getsource(engine_module))
	constructor_calls = [
		node
		for node in ast.walk(tree)
		if isinstance(node, ast.Call)
		and isinstance(node.func, ast.Name)
		and node.func.id == "CanonicalValue"
	]

	assert len(constructor_calls) == 1
	assert not hasattr(Adapter(), "resolve")
