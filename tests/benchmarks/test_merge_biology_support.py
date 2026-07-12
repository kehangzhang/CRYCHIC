from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.metrics.merge_biology_support import (
    BIOLOGY_IDENTITY_COLUMNS,
    merge_biology_support,
    read_source_list,
)


def _row(*, method: str, support_status: str = "supported") -> dict[str, object]:
    identity: dict[str, object] = {
        "dataset": "dataset_a",
        "observation_id": "observation_a",
        "method": method,
        "method_version": "1.0",
        "analysis_track": "lr_stlr",
        "resource": "resource_a",
        "resource_version": "2026-07-12",
        "resource_mode": "native",
        "score_semantics": "strength",
        "universe_id": "universe_a",
    }
    return identity | {
        "support_status": support_status,
        "status": "observed",
        "reason_code": "supportive_only",
        "source_biology_file": f"{method}.tsv",
    }


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_merge_deduplicates_only_exact_rows(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.tsv", [_row(method="m1")])
    second = _write(
        tmp_path / "second.tsv", [_row(method="m1"), _row(method="m2")]
    )

    merged = merge_biology_support([first, second], tmp_path / "merged.tsv")

    assert len(merged) == 2
    assert list(merged["method"]) == ["m1", "m2"]
    assert (tmp_path / "merged.tsv").is_file()


def test_merge_rejects_identity_conflicts(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.tsv", [_row(method="m1")])
    second = _write(
        tmp_path / "second.tsv",
        [_row(method="m1", support_status="discordant")],
    )

    with pytest.raises(ValueError, match="identity conflicts"):
        merge_biology_support([first, second], tmp_path / "merged.tsv")


def test_merge_requires_exact_schema_and_source_lineage(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.tsv", [_row(method="m1")])
    original = pd.read_csv(first, sep="\t")
    reordered_columns = [
        "status",
        *[column for column in original.columns if column != "status"],
    ]
    reordered = original.loc[:, reordered_columns]
    second = tmp_path / "second.tsv"
    reordered.to_csv(second, sep="\t", index=False)

    with pytest.raises(ValueError, match="schemas must match exactly"):
        merge_biology_support([first, second], tmp_path / "merged.tsv")

    without_lineage = pd.DataFrame([_row(method="m2")]).drop(
        columns="source_biology_file"
    )
    third = tmp_path / "third.tsv"
    without_lineage.to_csv(third, sep="\t", index=False)
    with pytest.raises(ValueError, match="source_biology_file"):
        merge_biology_support([third], tmp_path / "merged.tsv")


def test_read_source_list_resolves_relative_paths(tmp_path: Path) -> None:
    source = _write(tmp_path / "source.tsv", [_row(method="m1")])
    source_list = tmp_path / "sources.txt"
    source_list.write_text("# frozen inputs\nsource.tsv\n", encoding="utf-8")

    assert read_source_list(source_list) == (source.resolve(),)
    assert set(BIOLOGY_IDENTITY_COLUMNS).issubset(_row(method="m1"))
