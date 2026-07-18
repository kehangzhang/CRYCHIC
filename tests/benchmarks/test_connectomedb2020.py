from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.literature.connectomedb2020 import (
    REQUIRED_COLUMNS,
    add_adapter_identifiers,
    sha256_file,
    validate_resource_table,
)


def _resource() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                "Z",
                "R2",
                "Z R2",
                "source",
                "1",
                "HGNC:2",
                "secreted",
                "HGNC:4",
                "HGNC:2 HGNC:4",
                "",
            ],
            [
                "A",
                "R1",
                "A R1",
                "source",
                "2",
                "HGNC:1",
                "secreted",
                "HGNC:3",
                "HGNC:1 HGNC:3",
                "",
            ],
        ],
        columns=REQUIRED_COLUMNS,
    )


def test_validate_resource_table_preserves_direction_and_sorts() -> None:
    result = validate_resource_table(_resource(), expected_rows=2)

    assert result[["ligand", "receptor"]].to_records(index=False).tolist() == [
        ("A", "R1"),
        ("Z", "R2"),
    ]


def test_adapter_identifiers_are_stable_and_covered() -> None:
    resource = validate_resource_table(_resource(), expected_rows=2)

    first = add_adapter_identifiers(resource)
    second = add_adapter_identifiers(resource)

    pd.testing.assert_frame_equal(first, second)
    assert first["harmonized_interaction_id"].is_unique
    assert first["liana_covered"].all()
    assert first["cellchat_source_interaction_id"].equals(
        first["harmonized_interaction_id"]
    )


def test_validate_resource_table_rejects_duplicate_directional_pairs() -> None:
    table = pd.concat([_resource(), _resource().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate directed"):
        validate_resource_table(table, expected_rows=3)


@pytest.mark.parametrize("column", ["ligand", "receptor"])
def test_validate_resource_table_rejects_missing_or_noncanonical_symbols(
    column: str,
) -> None:
    table = _resource()
    table.loc[0, column] = None
    with pytest.raises(ValueError, match=column):
        validate_resource_table(table, expected_rows=2)

    table = _resource()
    table.loc[0, column] = f" {table.loc[0, column]}"
    with pytest.raises(ValueError, match=column):
        validate_resource_table(table, expected_rows=2)


def test_sha256_file_is_streaming_and_deterministic(tmp_path: Path) -> None:
    payload = tmp_path / "payload.tsv"
    payload.write_bytes(b"ligand\treceptor\nA\tB\n")

    assert sha256_file(payload) == (
        "2b1b2175ffe665cc7ef90f789c4ce5f2eb4c6703156a2a09fef63a383b192f85"
    )
