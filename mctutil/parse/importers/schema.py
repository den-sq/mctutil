"""Immutable, versioned canonical scan and reconstruction schemas."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

from .model import CanonicalValue, FieldSpec, FieldType, ImportRecord


VERSION_PATTERN = re.compile(r"^(scan|reconstruction)/(\d+)\.(\d+)$")


@dataclass(frozen=True, order=True)
class SchemaVersion:
	kind: str = field(compare=False)
	major: int
	minor: int

	def __post_init__(self) -> None:
		if self.kind not in {"scan", "reconstruction"}:
			raise ValueError(f"unknown schema kind: {self.kind!r}")
		if self.major < 1 or self.minor < 0:
			raise ValueError("schema versions require major >= 1 and minor >= 0")

	@classmethod
	def parse(cls, value: str) -> "SchemaVersion":
		match = VERSION_PATTERN.fullmatch(value)
		if not match:
			raise ValueError(f"invalid schema version: {value!r}")
		kind, major, minor = match.groups()
		return cls(kind=kind, major=int(major), minor=int(minor))

	def next_minor(self) -> "SchemaVersion":
		return SchemaVersion(self.kind, self.major, self.minor + 1)

	def __str__(self) -> str:
		return f"{self.kind}/{self.major}.{self.minor}"


@dataclass(frozen=True)
class FieldAlias:
	key: str
	canonical_key: str
	valid_through_minor: int


@dataclass(frozen=True)
class CanonicalSchema:
	"""One immutable version of a canonical schema."""

	version: SchemaVersion
	fields: tuple[FieldSpec, ...]
	aliases: tuple[FieldAlias, ...] = ()

	def __post_init__(self) -> None:
		object.__setattr__(self, "fields", tuple(self.fields))
		object.__setattr__(self, "aliases", tuple(self.aliases))
		keys = [field_spec.key for field_spec in self.fields]
		if len(keys) != len(set(keys)):
			duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
			raise ValueError(f"duplicate canonical field keys: {duplicates!r}")
		alias_keys = [alias.key for alias in self.aliases]
		if len(alias_keys) != len(set(alias_keys)):
			raise ValueError("duplicate schema aliases")
		for alias in self.aliases:
			if alias.key in keys:
				raise ValueError(f"alias collides with canonical field: {alias.key}")
			if alias.canonical_key not in keys:
				raise ValueError(f"alias target is not canonical: {alias.canonical_key}")
			if alias.valid_through_minor < self.version.minor:
				raise ValueError(f"expired alias present in schema: {alias.key}")

	@property
	def kind(self) -> str:
		return self.version.kind

	@property
	def identifier(self) -> str:
		return str(self.version)

	@property
	def by_key(self) -> Mapping[str, FieldSpec]:
		return MappingProxyType({field_spec.key: field_spec for field_spec in self.fields})

	def resolve_key(self, key: str) -> str:
		if key in self.by_key:
			return key
		for alias in self.aliases:
			if alias.key == key:
				return alias.canonical_key
		raise KeyError(f"unknown canonical field for {self.identifier}: {key}")

	def field(self, key: str) -> FieldSpec:
		return self.by_key[self.resolve_key(key)]

	def validate_values(self, values: Mapping[str, CanonicalValue]) -> None:
		resolved_keys: set[str] = set()
		for key, canonical_value in values.items():
			resolved_key = self.resolve_key(key)
			if resolved_key in resolved_keys:
				raise ValueError(f"multiple values resolve to {resolved_key!r}")
			resolved_keys.add(resolved_key)
			self.field(resolved_key).validate(canonical_value.value)

		missing = [
			spec.key
			for spec in self.fields
			if not spec.nullable and spec.key not in resolved_keys
		]
		if missing:
			raise ValueError(f"missing required canonical fields: {missing!r}")

	def validate_record(self, record: ImportRecord) -> None:
		if record.kind != self.kind:
			raise ValueError(
				f"record kind {record.kind!r} does not match schema {self.kind!r}"
			)
		if record.schema_version != self.identifier:
			raise ValueError(
				f"record schema {record.schema_version!r} does not match "
				f"{self.identifier!r}"
			)
		self.validate_values(record.values)


class ScanSchema(CanonicalSchema):
	def __post_init__(self) -> None:
		super().__post_init__()
		if self.kind != "scan":
			raise ValueError("ScanSchema requires a scan version")


class ReconstructionSchema(CanonicalSchema):
	def __post_init__(self) -> None:
		super().__post_init__()
		if self.kind != "reconstruction":
			raise ValueError("ReconstructionSchema requires a reconstruction version")


@dataclass(frozen=True)
class SchemaRegistry:
	"""Immutable lookup of every supported version for one record kind."""

	kind: str
	schemas: tuple[CanonicalSchema, ...]

	def __post_init__(self) -> None:
		object.__setattr__(self, "schemas", tuple(self.schemas))
		if self.kind not in {"scan", "reconstruction"}:
			raise ValueError(f"unknown registry kind: {self.kind!r}")
		identifiers = [schema.identifier for schema in self.schemas]
		if not identifiers:
			raise ValueError("a schema registry cannot be empty")
		if len(identifiers) != len(set(identifiers)):
			raise ValueError("duplicate schema version")
		if any(schema.kind != self.kind for schema in self.schemas):
			raise ValueError("registry contains a schema of another kind")

	@property
	def versions(self) -> tuple[str, ...]:
		return tuple(schema.identifier for schema in self.schemas)

	@property
	def latest(self) -> CanonicalSchema:
		return max(self.schemas, key=lambda schema: schema.version)

	def get(self, identifier: str) -> CanonicalSchema:
		for schema in self.schemas:
			if schema.identifier == identifier:
				return schema
		raise KeyError(f"unsupported {self.kind} schema: {identifier}")

	def validate_versioned_values(
		self, identifier: str, values: Mapping[str, CanonicalValue]
	) -> None:
		"""Validate historical boundary data against its own schema version."""

		self.get(identifier).validate_values(values)


def promote_extension(
	schema: CanonicalSchema,
	*,
	extension_key: str,
	field_spec: FieldSpec,
) -> CanonicalSchema:
	"""Promote an extension for one minor version with a read-only alias."""

	if "." not in extension_key:
		raise ValueError("a promoted extension key must be source-namespaced")
	if field_spec.key in schema.by_key:
		raise ValueError(f"canonical field already exists: {field_spec.key}")

	version = schema.version.next_minor()
	retained_aliases = tuple(
		alias for alias in schema.aliases if alias.valid_through_minor >= version.minor
	)
	aliases = retained_aliases + (
		FieldAlias(extension_key, field_spec.key, version.minor),
	)
	schema_type = type(schema)
	return schema_type(version, schema.fields + (field_spec,), aliases)


def evolve_minor(
	schema: CanonicalSchema, added_fields: Iterable[FieldSpec] = ()
) -> CanonicalSchema:
	"""Create the next minor version and discard expired promotion aliases."""

	version = schema.version.next_minor()
	aliases = tuple(
		alias for alias in schema.aliases if alias.valid_through_minor >= version.minor
	)
	schema_type = type(schema)
	return schema_type(version, schema.fields + tuple(added_fields), aliases)


def _field(
	key: str,
	value_type: FieldType,
	*,
	unit: str | None = None,
	nullable: bool = True,
	label: str,
	description: str,
	min_items: int | None = None,
	max_items: int | None = None,
) -> FieldSpec:
	return FieldSpec(
		key=key,
		value_type=value_type,
		unit=unit,
		nullable=nullable,
		label=label,
		description=description,
		min_items=min_items,
		max_items=max_items,
	)


# The concrete field set is intentionally provisional pending design OQ1.  The
# registries, validation behavior, and version transitions are the settled
# contract delivered by SI-01.
SCAN_SCHEMA_V1 = ScanSchema(
	SchemaVersion.parse("scan/1.0"),
	(
		_field(
			"scan_id",
			FieldType.STRING,
			nullable=False,
			label="Scan ID",
			description="Stable acquisition identifier.",
		),
		_field(
			"project_id",
			FieldType.STRING,
			label="Project ID",
			description="Project or proposal identifier.",
		),
		_field(
			"sample_id",
			FieldType.STRING,
			label="Sample ID",
			description="Sample identifier.",
		),
		_field(
			"scan_method",
			FieldType.STRING,
			label="Scan Method",
			description="Acquisition method reported by the source.",
		),
		_field(
			"projection_count_planned",
			FieldType.INTEGER,
			unit="count",
			label="Planned Projections",
			description="Projection count requested before acquisition.",
		),
		_field(
			"projection_count_acquired",
			FieldType.INTEGER,
			unit="count",
			label="Acquired Projections",
			description="Projection count observed in acquired data.",
		),
		_field(
			"rotation_start_deg",
			FieldType.FLOAT,
			unit="deg",
			label="Rotation Start (deg)",
			description="Absolute first projection angle.",
		),
		_field(
			"rotation_stop_deg",
			FieldType.FLOAT,
			unit="deg",
			label="Rotation Stop (deg)",
			description="Absolute final projection angle.",
		),
		_field(
			"exposure_us",
			FieldType.FLOAT,
			unit="us",
			label="Exposure (us)",
			description="Normalized projection exposure time.",
		),
		_field(
			"detector_shape_px",
			FieldType.INTEGER_TUPLE,
			unit="px",
			label="Detector Shape (px)",
			description="Detector image height and width.",
			min_items=2,
			max_items=2,
		),
		_field(
			"effective_pixel_size_um",
			FieldType.FLOAT,
			unit="um",
			label="Effective Pixel Size (um)",
			description="Effective object-plane sampling.",
		),
		_field(
			"scan_start",
			FieldType.DATETIME,
			label="Scan Start",
			description="Timezone-aware acquisition start time.",
		),
		_field(
			"scan_end",
			FieldType.DATETIME,
			label="Scan End",
			description="Timezone-aware acquisition end time.",
		),
		_field(
			"scan_duration_s",
			FieldType.FLOAT,
			unit="s",
			label="Scan Duration (s)",
			description="Elapsed acquisition duration.",
		),
		_field(
			"source_files",
			FieldType.STRING_LIST,
			label="Source Files",
			description="Source files contributing to the record.",
			min_items=1,
			max_items=256,
		),
	),
)

RECONSTRUCTION_SCHEMA_V1 = ReconstructionSchema(
	SchemaVersion.parse("reconstruction/1.0"),
	(
		_field(
			"reconstruction_id",
			FieldType.STRING,
			nullable=False,
			label="Reconstruction ID",
			description="Stable reconstruction or configuration identifier.",
		),
		_field(
			"scan_id",
			FieldType.STRING,
			label="Scan ID",
			description="Input acquisition identifier.",
		),
		_field(
			"input_projection_file",
			FieldType.STRING,
			label="Input Projection File",
			description="Input projection file or URI.",
		),
		_field(
			"software_name",
			FieldType.STRING,
			label="Software",
			description="Reconstruction application name.",
		),
		_field(
			"software_version",
			FieldType.STRING,
			label="Software Version",
			description="Reconstruction application version.",
		),
		_field(
			"rotation_axis_coordinate_px",
			FieldType.FLOAT,
			unit="px",
			label="Rotation Axis Coordinate (px)",
			description="Absolute detector-coordinate rotation center.",
		),
		_field(
			"rotation_axis_offset_px",
			FieldType.FLOAT,
			unit="px",
			label="Rotation Axis Offset (px)",
			description="Offset from an explicitly declared detector center.",
		),
		_field(
			"output_voxel_size_mm",
			FieldType.FLOAT,
			unit="mm",
			label="Output Voxel Size (mm)",
			description="Reconstructed output voxel sampling.",
		),
		_field(
			"reconstruction_algorithm",
			FieldType.STRING,
			label="Reconstruction Algorithm",
			description="Selected reconstruction algorithm.",
		),
		_field(
			"reconstruction_filter",
			FieldType.STRING,
			label="Reconstruction Filter",
			description="Selected reconstruction filter.",
		),
		_field(
			"ring_stripe_method",
			FieldType.STRING,
			label="Ring/Stripe Method",
			description="Selected ring or stripe removal method.",
		),
		_field(
			"phase_retrieval_enabled",
			FieldType.BOOLEAN,
			label="Phase Retrieval Enabled",
			description="Whether phase retrieval is enabled.",
		),
		_field(
			"output_path",
			FieldType.STRING,
			label="Output Path",
			description="Reconstruction output directory or template.",
		),
		_field(
			"output_file_count",
			FieldType.INTEGER,
			unit="count",
			label="Output File Count",
			description="Observed valid reconstructed output count.",
		),
		_field(
			"output_image_shape",
			FieldType.INTEGER_TUPLE,
			unit="px",
			label="Output Image Shape",
			description="Observed reconstructed slice height and width.",
			min_items=2,
			max_items=2,
		),
		_field(
			"source_metadata_files",
			FieldType.STRING_LIST,
			label="Source Metadata Files",
			description="Metadata files contributing to the reconstruction record.",
			min_items=1,
			max_items=256,
		),
	),
)

SCAN_SCHEMA_REGISTRY = SchemaRegistry("scan", (SCAN_SCHEMA_V1,))
RECONSTRUCTION_SCHEMA_REGISTRY = SchemaRegistry(
	"reconstruction", (RECONSTRUCTION_SCHEMA_V1,)
)
