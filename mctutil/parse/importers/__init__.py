"""Shared scan and reconstruction import infrastructure.

The package owns canonical importer records.  Source representations enter
through adapters and mappings; tabular and audit representations are derived
from canonical records at output boundaries.
"""

from .model import (
	CanonicalValue,
	FieldSpec,
	FieldType,
	ImportRecord,
	PrecedencePolicy,
	SourceEvidence,
)
from .mapping import (
	MappingDefinition,
	MappingValidationError,
	load_builtin_mapping,
	load_mapping,
	load_mapping_text,
)
from .diagnostics import (
	BASE_DIAGNOSTIC_CODES,
	Diagnostic,
	DiagnosticCodeRegistry,
	StrictnessMode,
	WriteDecision,
	evaluate_write,
)
from .registry import (
	SOURCE_REGISTRY,
	ProbeConfidence,
	ProbeEvidence,
	ProbeEvidenceKind,
	ProbeResult,
	SourceAdapter,
	SourceArtifact,
	SourceBundle,
	SourceDetectionError,
	SourceRegistry,
	detect_adapter,
	discover_homogeneous,
	select_adapter,
)
from .engine import (
	DEFAULT_PRECEDENCE_ORDER,
	CandidateCollector,
	ImportEngine,
	ValueCandidate,
)
from .schema import (
	RECONSTRUCTION_SCHEMA_REGISTRY,
	SCAN_SCHEMA_REGISTRY,
	CanonicalSchema,
	ReconstructionSchema,
	ScanSchema,
	SchemaRegistry,
	SchemaVersion,
	promote_extension,
)

__all__ = [
	"CanonicalSchema",
	"CanonicalValue",
	"CandidateCollector",
	"DEFAULT_PRECEDENCE_ORDER",
	"BASE_DIAGNOSTIC_CODES",
	"Diagnostic",
	"DiagnosticCodeRegistry",
	"FieldSpec",
	"FieldType",
	"ImportRecord",
	"ImportEngine",
	"MappingDefinition",
	"MappingValidationError",
	"PrecedencePolicy",
	"ProbeConfidence",
	"ProbeEvidence",
	"ProbeEvidenceKind",
	"ProbeResult",
	"RECONSTRUCTION_SCHEMA_REGISTRY",
	"ReconstructionSchema",
	"SCAN_SCHEMA_REGISTRY",
	"SOURCE_REGISTRY",
	"ScanSchema",
	"SchemaRegistry",
	"SchemaVersion",
	"SourceEvidence",
	"SourceAdapter",
	"SourceArtifact",
	"SourceBundle",
	"SourceDetectionError",
	"SourceRegistry",
	"StrictnessMode",
	"WriteDecision",
	"ValueCandidate",
	"evaluate_write",
	"detect_adapter",
	"discover_homogeneous",
	"load_builtin_mapping",
	"load_mapping",
	"load_mapping_text",
	"promote_extension",
	"select_adapter",
]
