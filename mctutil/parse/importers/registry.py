"""Code-owned source registry, probing, and bounded discovery contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol, Sequence, runtime_checkable

from natsort import natsorted

from .diagnostics import Diagnostic
from .model import ImportRecord, PrecedencePolicy, RecordKind, SourceEvidence
from .sources.fixture import FIXTURE_RECONSTRUCTION_ADAPTER, FIXTURE_SCAN_ADAPTER


class ProbeConfidence(IntEnum):
	NONE = 0
	WEAK = 1
	HIGH = 2


class ProbeEvidenceKind(str, Enum):
	FILENAME = "filename"
	CONTENT = "content"
	STRUCTURE = "structure"
	METADATA = "metadata"


@dataclass(frozen=True)
class ProbeEvidence:
	path: Path
	kind: ProbeEvidenceKind
	signature: str

	def __post_init__(self) -> None:
		if not isinstance(self.path, Path):
			raise TypeError("probe evidence path must be a pathlib.Path")
		if not isinstance(self.kind, ProbeEvidenceKind):
			raise TypeError("probe evidence kind must be ProbeEvidenceKind")
		if not self.signature.strip():
			raise ValueError("probe signature must be non-empty")


@dataclass(frozen=True)
class ProbeResult:
	source_id: str
	kind: RecordKind
	confidence: ProbeConfidence
	evidence: tuple[ProbeEvidence, ...] = ()

	def __post_init__(self) -> None:
		if not self.source_id.strip() or self.kind not in {"scan", "reconstruction"}:
			raise ValueError("probe source_id and kind must be valid")
		if not isinstance(self.confidence, ProbeConfidence):
			raise TypeError("probe confidence must be ProbeConfidence")
		object.__setattr__(self, "evidence", tuple(self.evidence))

	@property
	def has_substantive_evidence(self) -> bool:
		return any(
			item.kind is not ProbeEvidenceKind.FILENAME for item in self.evidence
		)

	@property
	def is_high_confidence(self) -> bool:
		# Applying this to every source is stricter than the HDF5-specific minimum
		# and guarantees filename-only HDF5 probes can never select an adapter.
		return self.confidence is ProbeConfidence.HIGH and self.has_substantive_evidence


@dataclass(frozen=True)
class SourceArtifact:
	path: Path
	role: str
	media_type: str

	def __post_init__(self) -> None:
		if not isinstance(self.path, Path):
			raise TypeError("source artifact path must be a pathlib.Path")
		if not self.role.strip() or not self.media_type.strip():
			raise ValueError("source artifact role and media_type must be non-empty")


@dataclass(frozen=True)
class SourceBundle:
	kind: RecordKind
	source_id: str
	primary: SourceArtifact
	artifacts: tuple[SourceArtifact, ...] = ()
	identity_hint: str = ""

	def __post_init__(self) -> None:
		if self.kind not in {"scan", "reconstruction"} or not self.source_id.strip():
			raise ValueError("bundle kind and source_id must be valid")
		object.__setattr__(self, "artifacts", tuple(self.artifacts))
		if not self.identity_hint.strip():
			raise ValueError("bundle identity_hint must be non-empty")
		all_artifacts = self.all_artifacts
		identities = [(item.role, item.path) for item in all_artifacts]
		if len(identities) != len(set(identities)):
			raise ValueError("bundle artifacts must be de-duplicated")
		roles = [item.role for item in all_artifacts]
		if len(roles) != len(set(roles)):
			raise ValueError("bundle artifact roles must be unique")

	@property
	def all_artifacts(self) -> tuple[SourceArtifact, ...]:
		return (self.primary,) + self.artifacts


class CandidateSink(Protocol):
	"""The only adapter surface for adding derived pre-resolution candidates."""

	def add_candidate(
		self,
		field_key: str,
		raw_value: object,
		evidence: SourceEvidence,
		*,
		source_unit: str | None = None,
	) -> None:
		...


@runtime_checkable
class SourceAdapter(Protocol):
	source_id: str
	kind: RecordKind
	version: str

	def probe(self, inputs: tuple[Path, ...]) -> ProbeResult:
		...

	def discover(self, inputs: tuple[Path, ...]) -> Iterable[SourceBundle]:
		...

	def precedence_policies(self) -> tuple[PrecedencePolicy, ...]:
		...

	def derive(self, bundle: SourceBundle, candidates: CandidateSink) -> None:
		...

	def validate(
		self, record: ImportRecord, bundle: SourceBundle
	) -> tuple[Diagnostic, ...]:
		...


class SourceDetectionError(ValueError):
	def __init__(self, reason: str, results: Sequence[ProbeResult]) -> None:
		self.reason = reason
		self.results = tuple(results)
		details = []
		for result in self.results:
			signatures = ", ".join(
				f"{item.path}:{item.kind.value}:{item.signature}"
				for item in result.evidence
			) or "no evidence"
			details.append(
				f"{result.source_id}={result.confidence.name.lower()} ({signatures})"
			)
		suffix = "; ".join(details) if details else "no registered adapters"
		super().__init__(f"source detection failed ({reason}): {suffix}")


@dataclass(frozen=True)
class SourceRegistry:
	adapters: tuple[SourceAdapter, ...]

	def __post_init__(self) -> None:
		object.__setattr__(self, "adapters", tuple(self.adapters))
		keys = [(adapter.kind, adapter.source_id) for adapter in self.adapters]
		if len(keys) != len(set(keys)):
			raise ValueError("duplicate source adapter registration")

	@property
	def by_key(self) -> Mapping[tuple[RecordKind, str], SourceAdapter]:
		return MappingProxyType(
			{(adapter.kind, adapter.source_id): adapter for adapter in self.adapters}
		)

	def get(self, kind: RecordKind, source_id: str) -> SourceAdapter:
		try:
			return self.by_key[(kind, source_id)]
		except KeyError as exc:
			raise KeyError(f"unknown {kind} source: {source_id}") from exc

	def for_kind(self, kind: RecordKind) -> tuple[SourceAdapter, ...]:
		return tuple(
			sorted(
				(adapter for adapter in self.adapters if adapter.kind == kind),
				key=lambda adapter: adapter.source_id,
			)
		)


def _failed_probe(adapter: SourceAdapter, inputs: tuple[Path, ...], exc: Exception) -> ProbeResult:
	path = inputs[0] if inputs else Path(".")
	return ProbeResult(
		adapter.source_id,
		adapter.kind,
		ProbeConfidence.NONE,
		(
			ProbeEvidence(
				path,
				ProbeEvidenceKind.METADATA,
				f"probe error {type(exc).__name__}: {exc}",
			),
		),
	)


def detect_adapter(
	kind: RecordKind,
	inputs: tuple[Path, ...],
	*,
	registry: SourceRegistry,
) -> SourceAdapter:
	results = []
	adapters = registry.for_kind(kind)
	for adapter in adapters:
		try:
			result = adapter.probe(inputs)
		except Exception as exc:
			result = _failed_probe(adapter, inputs, exc)
		if result.kind != kind or result.source_id != adapter.source_id:
			raise ValueError(
				f"adapter {adapter.source_id!r} returned a probe for another identity"
			)
		results.append(result)
	high = [
		(adapter, result)
		for adapter, result in zip(adapters, results)
		if result.is_high_confidence
	]
	if len(high) == 1:
		return high[0][0]
	reason = "tie" if len(high) > 1 else "no_high_confidence_match"
	raise SourceDetectionError(reason, results)


def select_adapter(
	kind: RecordKind,
	source_id: str,
	inputs: tuple[Path, ...],
	*,
	registry: SourceRegistry,
) -> SourceAdapter:
	if source_id == "auto":
		return detect_adapter(kind, inputs, registry=registry)
	return registry.get(kind, source_id)


def _path_identity(path: Path) -> str:
	return str(path.absolute())


def bounded_input_files(
	inputs: tuple[Path, ...], *, suffixes: frozenset[str]
) -> tuple[Path, ...]:
	"""Return explicit files and immediate directory children; never recurse."""

	candidates: dict[str, Path] = {}
	for input_path in inputs:
		if input_path.is_dir():
			children = (
				child
				for child in input_path.iterdir()
				if child.is_file() and child.suffix.lower() in suffixes
			)
		else:
			children = (input_path,)
		for child in children:
			if child.suffix.lower() in suffixes:
				candidates.setdefault(_path_identity(child), child)
	return tuple(natsorted(candidates.values(), key=lambda path: str(path)))


def discover_homogeneous(
	adapter: SourceAdapter, inputs: tuple[Path, ...]
) -> tuple[SourceBundle, ...]:
	discovered = tuple(adapter.discover(inputs))
	by_primary: dict[str, SourceBundle] = {}
	for bundle in discovered:
		if bundle.kind != adapter.kind or bundle.source_id != adapter.source_id:
			raise ValueError(
				"mixed source discovery is not supported: "
				f"selected {(adapter.kind, adapter.source_id)!r}, "
				f"found {(bundle.kind, bundle.source_id)!r}"
			)
		by_primary.setdefault(_path_identity(bundle.primary.path), bundle)
	return tuple(
		natsorted(
			by_primary.values(),
			key=lambda bundle: (str(bundle.primary.path), bundle.identity_hint),
		)
	)


# Synthetic adapters make the provisional Phase-1 core executable without
# claiming support for any real acquisition or reconstruction source.
SOURCE_REGISTRY = SourceRegistry(
	(FIXTURE_SCAN_ADAPTER, FIXTURE_RECONSTRUCTION_ADAPTER)
)
