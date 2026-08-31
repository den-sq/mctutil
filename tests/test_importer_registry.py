from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mctutil.parse.importers.mapping import load_builtin_mapping
from mctutil.parse.importers.model import PrecedencePolicy
from mctutil.parse.importers.registry import (
	SOURCE_REGISTRY,
	ProbeConfidence,
	ProbeEvidence,
	ProbeEvidenceKind,
	ProbeResult,
	SourceArtifact,
	SourceBundle,
	SourceDetectionError,
	SourceRegistry,
	detect_adapter,
	discover_homogeneous,
)
from mctutil.parse.importers.schema import (
	RECONSTRUCTION_SCHEMA_V1,
	SCAN_SCHEMA_V1,
)


@dataclass(frozen=True)
class Adapter:
	source_id: str
	kind: str = "scan"
	confidence: ProbeConfidence = ProbeConfidence.NONE
	evidence_kind: ProbeEvidenceKind = ProbeEvidenceKind.CONTENT
	bundles: tuple[SourceBundle, ...] = ()

	def probe(self, inputs):
		return ProbeResult(
			self.source_id,
			self.kind,
			self.confidence,
			(
				ProbeEvidence(
					inputs[0], self.evidence_kind, f"signature for {self.source_id}"
				),
			),
		)

	def discover(self, inputs):
		return self.bundles

	def precedence_policies(self) -> tuple[PrecedencePolicy, ...]:
		return ()

	def derive(self, bundle, candidates) -> None:
		return None

	def validate(self, record, bundle):
		return ()


def _bundle(path: Path, *, source_id: str = "one") -> SourceBundle:
	return SourceBundle(
		"scan",
		source_id,
		SourceArtifact(path, "primary", "application/json"),
		identity_hint=path.stem,
	)


def test_registry_is_keyed_by_kind_and_source_and_rejects_duplicates() -> None:
	first = Adapter("same", "scan")
	second_kind = Adapter("same", "reconstruction")
	registry = SourceRegistry((first, second_kind))

	assert registry.get("scan", "same") is first
	assert registry.get("reconstruction", "same") is second_kind
	with pytest.raises(ValueError, match="duplicate source"):
		SourceRegistry((first, first))


def test_single_substantive_high_confidence_probe_selects() -> None:
	selected = Adapter("selected", confidence=ProbeConfidence.HIGH)
	registry = SourceRegistry((Adapter("weak", confidence=ProbeConfidence.WEAK), selected))

	assert detect_adapter("scan", (Path("input.h5"),), registry=registry) is selected


def test_weak_tie_and_filename_only_detection_fail_closed_with_evidence() -> None:
	weak_registry = SourceRegistry((Adapter("weak", confidence=ProbeConfidence.WEAK),))
	with pytest.raises(SourceDetectionError, match="no_high_confidence_match") as weak:
		detect_adapter("scan", (Path("input.h5"),), registry=weak_registry)
	assert "signature for weak" in str(weak.value)

	tie_registry = SourceRegistry(
		(
			Adapter("one", confidence=ProbeConfidence.HIGH),
			Adapter("two", confidence=ProbeConfidence.HIGH),
		)
	)
	with pytest.raises(SourceDetectionError, match="tie") as tie:
		detect_adapter("scan", (Path("input.h5"),), registry=tie_registry)
	assert {result.source_id for result in tie.value.results} == {"one", "two"}

	filename_only = SourceRegistry(
		(
			Adapter(
				"hdf5-name",
				confidence=ProbeConfidence.HIGH,
				evidence_kind=ProbeEvidenceKind.FILENAME,
			),
		)
	)
	with pytest.raises(SourceDetectionError, match="no_high_confidence_match"):
		detect_adapter("scan", (Path("looks-right.h5"),), registry=filename_only)


def test_discovery_is_natural_deduplicated_and_bounded(tmp_path: Path) -> None:
	(tmp_path / "scan10.json").write_text(
		'{"mctutil_fixture_kind":"scan","scan_id":"10"}', encoding="utf-8"
	)
	(tmp_path / "scan2.json").write_text(
		'{"mctutil_fixture_kind":"scan","scan_id":"2"}', encoding="utf-8"
	)
	nested = tmp_path / "nested"
	nested.mkdir()
	(nested / "scan1.json").write_text(
		'{"mctutil_fixture_kind":"scan","scan_id":"1"}', encoding="utf-8"
	)
	adapter = SOURCE_REGISTRY.get("scan", "fixture-scan")

	bundles = discover_homogeneous(
		adapter, (tmp_path, tmp_path / "scan2.json")
	)

	assert [bundle.primary.path.name for bundle in bundles] == [
		"scan2.json",
		"scan10.json",
	]


def test_discovery_rejects_a_mixed_source_bundle() -> None:
	selected = Adapter(
		"one", bundles=(_bundle(Path("input.json"), source_id="other"),)
	)
	with pytest.raises(ValueError, match="mixed source discovery"):
		discover_homogeneous(selected, (Path("input.json"),))


def test_every_registered_fixture_has_matching_packaged_mapping() -> None:
	for adapter in SOURCE_REGISTRY.adapters:
		schema = (
			SCAN_SCHEMA_V1
			if adapter.kind == "scan"
			else RECONSTRUCTION_SCHEMA_V1
		)
		mapping = load_builtin_mapping(adapter.kind, adapter.source_id, schema=schema)
		assert mapping.source == adapter.source_id
		assert mapping.schema == schema.identifier
