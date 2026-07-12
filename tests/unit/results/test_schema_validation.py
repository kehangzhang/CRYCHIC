from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from crychic.results import ResultValidationError, validate_table


def test_duplicate_primary_key_is_rejected(result_payload: dict[str, Any]) -> None:
    frame = result_payload["tables"]["interactions"]
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ResultValidationError, match="duplicate primary keys"):
        validate_table("interactions", duplicated)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("table_name", "column"),
    [
        ("interactions", "comm_probability"),
        ("differential", "p_value"),
        ("differential", "q_value"),
        ("differential", "specificity_support"),
        ("responses", "p_value"),
    ],
)
def test_v0_1_inferential_fields_must_be_null(
    result_payload: dict[str, Any], table_name: str, column: str
) -> None:
    frame = result_payload["tables"][table_name].copy()
    frame.loc[0, column] = 0.5

    with pytest.raises(ResultValidationError, match="entirely null"):
        validate_table(table_name, frame)


def test_null_value_requires_reason_code(result_payload: dict[str, Any]) -> None:
    frame = deepcopy(result_payload["tables"]["sample_scores"])
    frame.loc[0, "comm_strength"] = None

    with pytest.raises(ResultValidationError, match="requires reason_code"):
        validate_table("sample_scores", frame)


def test_wrong_dtype_is_rejected(result_payload: dict[str, Any]) -> None:
    frame = result_payload["tables"]["responses"].copy()
    frame["n_subjects"] = frame["n_subjects"].astype(float)

    with pytest.raises(ResultValidationError, match="dtype integer"):
        validate_table("responses", frame)


def test_context_id_must_match_canonical_context_json(
    result_payload: dict[str, Any],
) -> None:
    frame = result_payload["tables"]["interactions"].copy()
    frame["context_id"] = "wrong-but-nonempty"

    with pytest.raises(ResultValidationError, match="does not match context_json"):
        validate_table("interactions", frame)
