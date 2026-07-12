from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd

from crychic.results import write_result


def test_rank_and_signature_queries_use_persisted_values(
    tmp_path: Path, result_payload: dict[str, Any]
) -> None:
    result = write_result(tmp_path / "result", **result_payload)

    ranked = result.rank_interactions(
        context={"condition": "stim"},
        receiver="B cell",
        contrast="stim_vs_ctrl",
        top_n=1,
    )
    signature = result.get_signature(
        context={"condition": "stim"},
        sender="Monocyte",
        receiver="B cell",
        interaction="CXCL10_CXCR3",
    )

    assert ranked["interaction_id"].tolist() == ["CXCL10_CXCR3"]
    assert ranked["comm_strength"].tolist() == [0.9]
    assert signature["gene"].tolist() == ["ISG15", "STAT1"]


def test_read_table_pushes_lazy_filters_to_parquet(
    tmp_path: Path,
    result_payload: dict[str, Any],
    monkeypatch: Any,
) -> None:
    result = write_result(tmp_path / "result", **result_payload)
    original = cast(Callable[..., pd.DataFrame], pd.read_parquet)
    calls: list[object] = []

    def spy(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs.get("filters"))
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    selected = result.read_table(
        "interactions",
        filters={"sender": "T cell", "receiver": "B cell"},
    )

    assert selected["interaction_id"].tolist() == ["CD40LG_CD40"]
    assert calls == [[("sender", "==", "T cell"), ("receiver", "==", "B cell")]]
