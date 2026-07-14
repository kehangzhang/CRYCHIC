from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import crychic.workflow.repeated_crossfit as repeated_crossfit_module
from crychic.core import ContractError, stable_id


def _legacy_table_digest(
    name: str,
    table: pd.DataFrame,
    columns: tuple[str, ...],
) -> str:
    rows = [
        [repeated_crossfit_module._canonical_cell(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    result: str = stable_id(
        name,
        {"columns": list(columns), "rows": rows},
        schema_version="1",
        digest_length=64,
    )
    return result


def _canonical_fixture() -> tuple[pd.DataFrame, tuple[str, ...]]:
    columns = (
        "tuple_value",
        "numpy_scalar",
        "none_value",
        "na_value",
        "nat_value",
        "finite_float",
    )
    table = pd.DataFrame(
        {
            "tuple_value": pd.Series(
                [
                    ("ligand", np.int64(2), (None, pd.NA)),
                    ('\u53d7\u4f53 "rows":[]', np.float32(1.25)),
                ],
                dtype=object,
            ),
            "numpy_scalar": pd.Series(
                [np.int64(7), np.float32(2.5)],
                dtype=object,
            ),
            "none_value": pd.Series([None, None], dtype=object),
            "na_value": pd.Series([pd.NA, pd.NA], dtype=object),
            "nat_value": pd.Series([pd.NaT, pd.NaT], dtype=object),
            "finite_float": pd.Series([-0.0, 3.5], dtype=float),
        },
        columns=columns,
    )
    return table, columns


def test_streamed_table_digest_matches_legacy_canonical_stable_id() -> None:
    table, columns = _canonical_fixture()
    name = "repeated_crossfit_digest_fixture"

    assert repeated_crossfit_module._table_digest(name, table, columns) == (
        _legacy_table_digest(name, table, columns)
    )
    assert repeated_crossfit_module._table_digest(
        name,
        table.iloc[0:0],
        columns,
    ) == _legacy_table_digest(name, table.iloc[0:0], columns)


def test_streamed_table_digest_rejects_column_mismatch() -> None:
    table, columns = _canonical_fixture()

    with pytest.raises(ValueError, match="columns do not match"):
        repeated_crossfit_module._table_digest(
            "repeated_crossfit_digest_fixture",
            table,
            (*columns[:-1], "wrong_column"),
        )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "value",
    [float("inf"), float("-inf")],
)
def test_streamed_table_digest_rejects_infinity(value: float) -> None:
    table = pd.DataFrame({"value": [value]})

    with pytest.raises(ValueError, match="cannot contain infinity"):
        repeated_crossfit_module._table_digest(
            "repeated_crossfit_digest_fixture",
            table,
            ("value",),
        )


def test_streamed_table_digest_preserves_stable_id_kind_validation() -> None:
    table = pd.DataFrame({"value": [1]})

    with pytest.raises(ContractError) as error:
        repeated_crossfit_module._table_digest(
            "Invalid Kind",
            table,
            ("value",),
        )
    assert error.value.details.code == "invalid_id_kind"
