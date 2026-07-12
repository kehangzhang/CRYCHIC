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
        "interactions.schema.json",
        "differential.schema.json",
        "responses.schema.json",
        "sample_scores.schema.json",
        "signatures.schema.json",
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
