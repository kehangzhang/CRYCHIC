import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from crychic.core import canonical_digest
from crychic.results import RESULT_SCHEMA_VERSION, TABLE_NAMES, table_contract

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "filename",
    [
        "run_manifest.schema.json",
        "bootstrap_support.schema.json",
        "edge_evidence.schema.json",
        "selection_frequency.schema.json",
        "scoring_collections.schema.json",
        "scoring_collections_v2.schema.json",
        "scoring_collections_v3.schema.json",
        "scoring_collections_v4.schema.json",
        "interactions.schema.json",
        "differential.schema.json",
        "responses.schema.json",
        "sample_scores.schema.json",
        "signatures.schema.json",
        "specificity_support.schema.json",
    ],
)
def test_v0_1_result_schema_is_valid_draft_2020_12(filename: str) -> None:
    value = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(value)


def test_table_contracts_define_keys_types_and_null_semantics() -> None:
    for name in TABLE_NAMES:
        contract = table_contract(name)
        assert contract.primary_key
        assert set(contract.primary_key).issubset(contract.columns)
        assert all(not contract.columns[key].nullable for key in contract.primary_key)
        assert set(contract.inferential_null_columns).issubset(contract.columns)


def test_run_manifest_schema_accepts_v0_1_shape() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "run_manifest.schema.json").read_text(encoding="utf-8")
    )
    tables = {
        name: {
            "filename": table_contract(name).filename,
            "rows": 0,
            "sha256": "a" * 64,
            "schema": table_contract(name).schema_filename,
        }
        for name in TABLE_NAMES
    }
    manifest: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_id": "run-1",
        "status": "complete",
        "mode": "exploratory",
        "created_at": "2026-07-12T00:00:00Z",
        "code_version": "0.1.0",
        "config_digest": "b" * 64,
        "workflow_parameters": {"min_cells": 10},
        "workflow_digest": canonical_digest({"min_cells": 10}),
        "provenance_digest": "c" * 64,
        "input_digest": None,
        "resource_digests": {},
        "tables": tables,
        "stages": [],
        "warnings": [],
    }

    Draft202012Validator(schema, format_checker=FormatChecker()).validate(manifest)

    manifest["extensions"] = {
        "scoring_collections": {
            "extension_schema_version": "1.0.0",
            "filename": "scoring_collections.json",
            "collections": 1,
            "sha256": "e" * 64,
            "schema": "scoring_collections.schema.json",
            "linked_tables": {
                "interactions": tables["interactions"]["sha256"],
                "sample_scores": tables["sample_scores"]["sha256"],
            },
        }
    }
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(manifest)

    scoring_record = manifest["extensions"]["scoring_collections"]
    for version, schema_filename in (
        ("2.0.0", "scoring_collections_v2.schema.json"),
        ("3.0.0", "scoring_collections_v3.schema.json"),
        ("4.0.0", "scoring_collections_v4.schema.json"),
    ):
        scoring_record["extension_schema_version"] = version
        scoring_record["schema"] = schema_filename
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(
            manifest
        )
    scoring_record["schema"] = "scoring_collections_v2.schema.json"
    assert not Draft202012Validator(
        schema, format_checker=FormatChecker()
    ).is_valid(manifest)

    manifest["extensions"] = {
        "edge_evidence": {
            "extension_schema_version": "1.0.0",
            "filename": "edge_evidence.parquet",
            "rows": 0,
            "sha256": "d" * 64,
            "schema": "edge_evidence.schema.json",
            "linked_tables": {
                "sample_scores": tables["sample_scores"]["sha256"]
            },
        },
        "scoring_collections": {
            "extension_schema_version": "1.0.0",
            "filename": "scoring_collections.json",
            "collections": 1,
            "sha256": "e" * 64,
            "schema": "scoring_collections.schema.json",
            "linked_tables": {
                "interactions": tables["interactions"]["sha256"],
                "sample_scores": tables["sample_scores"]["sha256"],
            },
        },
    }
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(manifest)

    manifest["extensions"] = {
        "bootstrap_support": {
            "extension_schema_version": "1.0.0",
            "specificity_support": {
                "filename": "specificity_support.parquet",
                "rows": 1,
                "sha256": "1" * 64,
                "schema": "specificity_support.schema.json",
            },
            "selection_frequency": {
                "filename": "selection_frequency.parquet",
                "rows": 1,
                "sha256": "2" * 64,
                "schema": "selection_frequency.schema.json",
            },
            "registry": {
                "filename": "bootstrap_support_registry.json",
                "records": 2,
                "sha256": "3" * 64,
                "schema": "bootstrap_support.schema.json",
            },
            "linked_tables": {"differential": tables["differential"]["sha256"]},
        }
    }
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(manifest)
