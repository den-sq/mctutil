"""Synthetic Phase-1 adapters used to exercise the shared core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..diagnostics import Diagnostic
from ..model import ImportRecord, PrecedencePolicy, RecordKind


@dataclass(frozen=True)
class FixtureAdapter:
	kind: RecordKind
	source_id: str
	id_field: str

	def probe(self, inputs: tuple[Path, ...]):
		from ..registry import (
			ProbeConfidence,
			ProbeEvidence,
			ProbeEvidenceKind,
			ProbeResult,
			bounded_input_files,
		)

		evidence = []
		confidence = ProbeConfidence.NONE
		for path in bounded_input_files(inputs, suffixes=frozenset({".json"})):
			try:
				payload = json.loads(path.read_text(encoding="utf-8"))
			except (OSError, ValueError):
				continue
			if payload.get("mctutil_fixture_kind") == self.kind:
				evidence.append(
					ProbeEvidence(
						path,
						ProbeEvidenceKind.CONTENT,
						f"mctutil_fixture_kind={self.kind}",
					)
				)
				match_confidence = (
					ProbeConfidence.HIGH
					if self.id_field in payload
					else ProbeConfidence.WEAK
				)
				confidence = max(confidence, match_confidence)
		return ProbeResult(self.source_id, self.kind, confidence, tuple(evidence))

	def discover(self, inputs: tuple[Path, ...]):
		from ..registry import SourceArtifact, SourceBundle, bounded_input_files

		for path in bounded_input_files(inputs, suffixes=frozenset({".json"})):
			try:
				payload = json.loads(path.read_text(encoding="utf-8"))
			except (OSError, ValueError):
				continue
			if payload.get("mctutil_fixture_kind") != self.kind:
				continue
			identity = str(payload.get(self.id_field, path.stem))
			yield SourceBundle(
				self.kind,
				self.source_id,
				SourceArtifact(path, "primary", "application/json"),
				identity_hint=identity,
			)

	def precedence_policies(self) -> tuple[PrecedencePolicy, ...]:
		return ()

	def derive(self, bundle, candidates) -> None:
		return None

	def validate(
		self, record: ImportRecord, bundle
	) -> tuple[Diagnostic, ...]:
		return ()


FIXTURE_SCAN_ADAPTER = FixtureAdapter("scan", "fixture-scan", "scan_id")
FIXTURE_RECONSTRUCTION_ADAPTER = FixtureAdapter(
	"reconstruction", "fixture-reconstruction", "reconstruction_id"
)
