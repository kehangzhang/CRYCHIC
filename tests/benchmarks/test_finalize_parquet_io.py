from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from benchmarks.metrics import finalize_multicondition as module


def _long_table(n_per_view: int = 6) -> pd.DataFrame:
    run_ids: np.ndarray[Any, np.dtype[np.str_]] = np.repeat(
        ["view-a", "view-b", "view-c"], n_per_view
    )
    rows = len(run_ids)
    return pd.DataFrame(
        {
            "run_id": run_ids,
            "dataset_id": "fixture",
            "method_id": "crychic",
            "method_version": "1.0",
            "analysis_track": "lr_stlr",
            "resource_id": "fixture-common",
            "resource_version": "2026-07-13",
            "resource_mode": "H-common",
            "universe_id": "fixture-universe-v1",
            "score_name": "fixture_strength",
            "score": np.arange(rows, dtype=float) / max(rows, 1),
            "unused_payload": [
                f"payload-{index:08d}-" + "x" * 96 for index in range(rows)
            ],
        }
    )


def test_unique_metadata_scan_matches_full_table_across_batches(
    tmp_path: Path,
) -> None:
    source = _long_table()
    path = tmp_path / "views.parquet"
    source.to_parquet(path, index=False, row_group_size=4)

    observed = module._read_unique_parquet_metadata(
        path,
        module.TRACK_METADATA_COLUMNS,
        batch_size=3,
    )
    expected = source.loc[:, list(module.TRACK_METADATA_COLUMNS)].drop_duplicates(
        ignore_index=True
    )

    pd.testing.assert_frame_equal(observed, expected)
    assert len(observed) == 3


def test_score_view_read_is_predicate_filtered_projected_and_exact(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = _long_table()
    path = tmp_path / "views.parquet"
    source.to_parquet(path, index=False, row_group_size=4)
    projection = ("run_id", "score")
    calls: list[dict[str, object]] = []
    read_parquet = pd.read_parquet

    def recording_read_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
        calls.append(dict(kwargs))
        return read_parquet(path, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read_parquet)
    observed = module._read_parquet_score_view(
        path,
        "view-b",
        columns=projection,
    )
    expected = source.loc[source["run_id"].eq("view-b"), list(projection)]

    pd.testing.assert_frame_equal(
        observed.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
    assert calls == [
        {
            "columns": list(projection),
            "filters": [("run_id", "==", "view-b")],
            "dtype_backend": "pyarrow",
        }
    ]


@pytest.mark.parametrize("run_id", [1, 1.0])
def test_metadata_scan_rejects_non_string_run_id_schema(
    tmp_path: Path,
    run_id: object,
) -> None:
    source = _long_table()
    source["run_id"] = run_id
    path = tmp_path / "invalid-run-id.parquet"
    source.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="run_id must use an Arrow string type"):
        module._read_unique_parquet_metadata(path, module.TRACK_METADATA_COLUMNS)


@pytest.mark.parametrize("run_id", ["", " view-a", "view-a "])
def test_metadata_scan_rejects_noncanonical_run_id_values(
    tmp_path: Path,
    run_id: str,
) -> None:
    source = _long_table()
    source["run_id"] = run_id
    path = tmp_path / "invalid-run-id.parquet"
    source.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="canonical non-empty strings"):
        module._read_unique_parquet_metadata(path, module.TRACK_METADATA_COLUMNS)


@pytest.mark.parametrize(
    "values",
    [
        ["True", "False", "True", "False", "True", "False"],
        [1, 0, 1, 0, 1, 0],
    ],
)
def test_score_view_rejects_non_boolean_universe_member_schema(
    tmp_path: Path,
    values: list[object],
) -> None:
    source = _long_table(n_per_view=2)
    source["universe_member"] = values
    path = tmp_path / "invalid-membership.parquet"
    source.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="must use an Arrow boolean type"):
        module._read_parquet_score_view(
            path,
            "view-a",
            columns=("run_id", "score", "universe_member"),
        )


def test_score_view_rejects_null_boolean_universe_member_values(
    tmp_path: Path,
) -> None:
    source = _long_table(n_per_view=2)
    source["universe_member"] = pd.Series(
        [True, None, True, False, True, False],
        dtype="boolean",
    )
    path = tmp_path / "null-membership.parquet"
    source.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="null-free boolean values"):
        module._read_parquet_score_view(
            path,
            "view-a",
            columns=("run_id", "score", "universe_member"),
        )


def test_score_view_parquet_io_performance_smoke(tmp_path: Path) -> None:
    source = _long_table(n_per_view=25_000)
    path = tmp_path / "views.parquet"
    source.to_parquet(path, index=False, row_group_size=2_500)

    started = time.perf_counter()
    metadata = module._read_unique_parquet_metadata(
        path,
        module.TRACK_METADATA_COLUMNS,
        batch_size=4_096,
    )
    selected = module._read_parquet_score_view(
        path,
        "view-b",
        columns=("run_id", "score"),
    )
    elapsed = time.perf_counter() - started

    full_projection_bytes = int(
        source.loc[:, ["run_id", "score"]].memory_usage(deep=True).sum()
    )
    selected_bytes = int(selected.memory_usage(deep=True).sum())
    assert len(metadata) == 3
    assert len(selected) == 25_000
    assert selected_bytes < full_projection_bytes * 0.4
    assert elapsed < 10.0
