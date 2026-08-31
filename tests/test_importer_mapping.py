from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from mctutil.parse.importers.mapping import (
	MappingValidationError,
	SelectorRule,
	TransformRule,
	apply_selector,
	apply_transform,
	evaluate_field_rule,
	load_builtin_mapping,
	load_mapping_text,
)
from mctutil.parse.importers.readers import (
	OpenedArtifact,
	ReaderContext,
	cli_option_raw,
)
from mctutil.parse.importers.schema import (
	RECONSTRUCTION_SCHEMA_V1,
	SCAN_SCHEMA_V1,
)


SCAN_MAPPING = (
	"mapping_format: 1\n"
	"schema: scan/1.0\n"
	"source: test-scan\n"
	"fields:\n"
	"  scan_id:\n"
	"    read:\n"
	"      kind: json_pointer\n"
	"      pointer: /scan_id\n"
	"      state: observed\n"
	"    transforms:\n"
	"      - whitespace: strip\n"
)


def test_builtin_mappings_load_as_package_resources() -> None:
	scan = load_builtin_mapping(
		"scan", "fixture-scan", schema=SCAN_SCHEMA_V1
	)
	reconstruction = load_builtin_mapping(
		"reconstruction",
		"fixture-reconstruction",
		schema=RECONSTRUCTION_SCHEMA_V1,
	)

	assert scan.source == "fixture-scan"
	assert reconstruction.source == "fixture-reconstruction"
	package_data = Path("pyproject.toml").read_text(encoding="utf-8")
	assert '"mappings/*/*.yaml"' in package_data


@pytest.mark.parametrize(
	("replacement", "message"),
	[
		("scan_id:", "unknown canonical field"),
		("bogus", "unknown selector"),
		("transforms:\n      - bogus", "unknown transform"),
		("kind: made_up", "unknown reader"),
		("missing_pointer", "missing locator properties"),
	],
)
def test_validator_rejects_unknown_vocabulary(
	replacement: str, message: str
) -> None:
	if replacement == "scan_id:":
		text = SCAN_MAPPING.replace("scan_id:", "unknown_field:", 1)
	elif replacement == "bogus":
		text = SCAN_MAPPING.replace(
			"    transforms:", "    select: bogus\n    transforms:"
		)
	elif replacement.startswith("transforms"):
		text = SCAN_MAPPING.replace(
			"transforms:\n      - whitespace: strip", replacement
		)
	elif replacement.startswith("kind"):
		text = SCAN_MAPPING.replace("kind: json_pointer", replacement)
	else:
		text = SCAN_MAPPING.replace("      pointer: /scan_id\n", "")

	with pytest.raises(MappingValidationError, match=message):
		load_mapping_text(text, schema=SCAN_SCHEMA_V1, expected_source="test-scan")


def test_validator_rejects_schema_source_unit_and_duplicate_keys() -> None:
	with pytest.raises(MappingValidationError, match="mapping schema"):
		load_mapping_text(
			SCAN_MAPPING.replace("scan/1.0", "scan/2.0"),
			schema=SCAN_SCHEMA_V1,
			expected_source="test-scan",
		)
	with pytest.raises(MappingValidationError, match="mapping source"):
		load_mapping_text(
			SCAN_MAPPING,
			schema=SCAN_SCHEMA_V1,
			expected_source="another-source",
		)

	unit_mapping = SCAN_MAPPING.replace(
		"pointer: /scan_id",
		"pointer: /scan_id\n      source_unit: furlong",
	).replace("scan_id:", "exposure_us:", 1)
	with pytest.raises(MappingValidationError, match="unknown unit conversion"):
		load_mapping_text(
			unit_mapping, schema=SCAN_SCHEMA_V1, expected_source="test-scan"
		)

	duplicate = SCAN_MAPPING + "\n  scan_id:\n    read:\n      kind: artifact\n      property: stem\n"
	with pytest.raises(MappingValidationError, match="invalid mapping YAML"):
		load_mapping_text(
			duplicate, schema=SCAN_SCHEMA_V1, expected_source="test-scan"
		)


def test_mapping_order_does_not_affect_canonical_rule_order() -> None:
	first = load_builtin_mapping("scan", "fixture-scan", schema=SCAN_SCHEMA_V1)
	source_text = Path(
		"mctutil/parse/importers/mappings/scan/fixture-scan.yaml"
	).read_text(encoding="utf-8")
	before, scan_rule = source_text.split("  scan_id:", 1)
	header, projection_rule = before.split("  projection_count_acquired:", 1)
	reordered = header + "  scan_id:" + scan_rule + "  projection_count_acquired:" + projection_rule
	second = load_mapping_text(
		reordered, schema=SCAN_SCHEMA_V1, expected_source="fixture-scan"
	)

	assert tuple(first.by_field) == tuple(second.by_field) == (
		"projection_count_acquired",
		"scan_id",
	)


def test_cli_option_reader_returns_raw_exact_token_only() -> None:
	command = "tomocupy recon --rotation-axis-auto vo --rotation-axis=1616.25"
	assert cli_option_raw(command, "--rotation-axis") == "1616.25"
	assert cli_option_raw(command, "--center") is None
	assert cli_option_raw(command, "--rotation-axis-auto") == "vo"


def test_readers_use_only_opened_artifacts() -> None:
	parser = configparser.ConfigParser()
	parser.read_string("[General]\nname = configured\n")
	artifacts = (
		OpenedArtifact("primary", Path("does-not-exist.json"), "application/json", {"a/b": [7]}),
		OpenedArtifact("ini", Path("does-not-exist.ini"), "text/ini", parser),
		OpenedArtifact("cli", Path("does-not-exist.txt"), "text/plain", "--value raw"),
		OpenedArtifact(
			"xlsx",
			Path("does-not-exist.xlsx"),
			"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
			{"ScanLog": [{"Scan ID": "scan-1"}]},
			{"row_index": 0},
		),
	)
	context = ReaderContext(artifacts)

	assert context.read("json_pointer", "primary", {"pointer": "/a~1b/0"}).value == 7
	assert context.read("ini", "ini", {"section": "General", "key": "name"}).value == "configured"
	assert context.read("cli_option", "cli", {"option": "--value"}).value == "raw"
	assert context.read(
		"xlsx_column", "xlsx", {"sheet": "ScanLog", "header": "Scan ID"}
	).value == "scan-1"


def test_artifact_and_hdf5_metadata_readers_do_not_open_or_load_arrays() -> None:
	class Dataset:
		shape = (1200, 2048, 2048)
		dtype = "uint16"
		reads = 0

		def __getitem__(self, key):
			self.reads += 1
			return 42

	dataset = Dataset()
	context = ReaderContext(
		(
			OpenedArtifact(
				"primary",
				Path("missing/scan.h5"),
				"application/x-hdf5",
				{"/exchange/data": dataset},
				{"size": 1234},
			),
		)
	)

	assert context.read("artifact", "primary", {"property": "stem"}).value == "scan"
	assert context.read("artifact", "primary", {"property": "size"}).value == 1234
	assert context.read(
		"hdf5_shape", "primary", {"path": "/exchange/data", "property": "shape"}
	).value == (1200, 2048, 2048)
	assert dataset.reads == 0
	assert context.read("hdf5", "primary", {"path": "/exchange/data"}).value == 42
	assert dataset.reads == 1


def test_selectors_and_transforms_are_allowlisted_and_typed() -> None:
	mapping = load_mapping_text(
		SCAN_MAPPING,
		schema=SCAN_SCHEMA_V1,
		expected_source="test-scan",
	)
	context = ReaderContext(
		(
			OpenedArtifact(
				"primary",
				Path("fixture.json"),
				"application/json",
				{"scan_id": "  scan-1  "},
			),
		)
	)
	result = evaluate_field_rule(
		mapping.by_field["scan_id"],
		context=context,
		field_spec=SCAN_SCHEMA_V1.field("scan_id"),
	)

	assert result.value == "scan-1"
	assert apply_selector(mapping.by_field["scan_id"].selector, ["only"]) == "only"


def test_unit_conversion_is_explicit_and_registered() -> None:
	text = (
		"mapping_format: 1\n"
		"schema: scan/1.0\n"
		"source: test-scan\n"
		"fields:\n"
		"  exposure_us:\n"
		"    read:\n"
		"      kind: json_pointer\n"
		"      pointer: /exposure\n"
		"      source_unit: s\n"
		"      state: observed\n"
		"    transforms:\n"
		"      - unit_convert\n"
	)
	mapping = load_mapping_text(
		text, schema=SCAN_SCHEMA_V1, expected_source="test-scan"
	)
	context = ReaderContext(
		(
			OpenedArtifact(
				"primary", Path("fixture.json"), "application/json", {"exposure": 0.002}
			),
		)
	)
	result = evaluate_field_rule(
		mapping.by_field["exposure_us"],
		context=context,
		field_spec=SCAN_SCHEMA_V1.field("exposure_us"),
	)

	assert result.value == 2000.0
	assert result.source_unit == "us"


@pytest.mark.parametrize(
	("selector", "value", "expected"),
	[
		(SelectorRule("first"), [3, 2, 1], 3),
		(SelectorRule("last"), [3, 2, 1], 1),
		(SelectorRule("first_nonempty"), ["", None, "x"], "x"),
		(SelectorRule("median"), [3, 1, 2], 2),
		(SelectorRule("minimum"), [3, 1, 2], 1),
		(SelectorRule("maximum"), [3, 1, 2], 3),
		(SelectorRule("count"), [3, 1, 2], 3),
		(SelectorRule("shape_axis", {"axis": 1}), [1200, 2048, 2048], 2048),
		(SelectorRule("dtype_bits"), "uint16", 16),
		(SelectorRule("ordered_coalesce"), [None, "", "chosen"], "chosen"),
	],
)
def test_selector_vocabulary(
	selector: SelectorRule, value: object, expected: object
) -> None:
	assert apply_selector(selector, value) == expected


def test_transform_vocabulary() -> None:
	assert apply_transform(
		TransformRule("scale", {"value": 2}), 3, source_unit=None, target_unit=None
	) == 6.0
	assert apply_transform(
		TransformRule("offset", {"value": 2}), 3, source_unit=None, target_unit=None
	) == 5.0
	assert apply_transform(
		TransformRule("bool_parse"), "yes", source_unit=None, target_unit=None
	) is True
	assert apply_transform(
		TransformRule("whitespace", {"value": "collapse"}),
		" a   b ",
		source_unit=None,
		target_unit=None,
	) == "a b"
	assert apply_transform(
		TransformRule("timestamp", {"value": "iso8601"}),
		"2026-08-31T12:00:00Z",
		source_unit=None,
		target_unit=None,
	).utcoffset() is not None


def test_safe_yaml_rejects_python_object_construction() -> None:
	malicious = "!!python/object/apply:os.system ['echo mapping-ran']\n"
	with pytest.raises(MappingValidationError, match="invalid mapping YAML"):
		load_mapping_text(
			malicious, schema=SCAN_SCHEMA_V1, expected_source="test-scan"
		)
