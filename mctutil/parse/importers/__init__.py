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
	"FieldSpec",
	"FieldType",
	"ImportRecord",
	"MappingDefinition",
	"MappingValidationError",
	"PrecedencePolicy",
	"RECONSTRUCTION_SCHEMA_REGISTRY",
	"ReconstructionSchema",
	"SCAN_SCHEMA_REGISTRY",
	"ScanSchema",
	"SchemaRegistry",
	"SchemaVersion",
	"SourceEvidence",
	"load_builtin_mapping",
	"load_mapping",
	"load_mapping_text",
	"promote_extension",
]
