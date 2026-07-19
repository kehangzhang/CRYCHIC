from __future__ import annotations

import numpy as np
import pandas as pd

import crychic.workflow.crossfit as crossfit_module
from crychic.core import canonical_json, stable_id


def test_crossfit_table_digest_streams_and_preserves_legacy_digest() -> None:
    table = pd.DataFrame(
        [
            ("row-10", np.float32(1.25), None),
            ('row-2 "quoted"', np.int64(2), pd.NA),
            ("row-1", -0.0, pd.NaT),
        ],
        columns=("identifier", "value", "missing"),
    )
    normalized = [
        [crossfit_module._table_cell_token(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    expected = stable_id(
        "crossfit_table",
        {
            "columns": list(table.columns),
            "rows": sorted(normalized, key=canonical_json),
            "table_name": "digest_fixture",
        },
        schema_version="1",
        digest_length=64,
    )

    assert crossfit_module._table_digest("digest_fixture", table) == expected
    assert (
        crossfit_module._table_digest("digest_fixture", table.iloc[::-1])
        == expected
    )
