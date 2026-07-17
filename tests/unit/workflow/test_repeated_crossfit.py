from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import crychic.workflow.repeated_crossfit as repeated_crossfit_module
from crychic.core import ContractError, stable_id
from crychic.design import balanced_contrast
from crychic.workflow import CrossFitSpec, RepeatedCrossFitSpec


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


@pytest.mark.parametrize("n_jobs", (0, -1, True))  # type: ignore[untyped-decorator]
def test_repeated_crossfit_rejects_invalid_n_jobs(n_jobs: int) -> None:
    spec = RepeatedCrossFitSpec(
        crossfit_spec=CrossFitSpec(
            contrasts=(
                balanced_contrast(
                    ("treated",),
                    ("control",),
                    name="treated_vs_control",
                ),
            ),
        ),
        n_repeats=2,
    )

    with pytest.raises(ValueError, match="n_jobs must be an integer >= 1"):
        repeated_crossfit_module.run_repeated_subject_crossfit(
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            spec=spec,
            n_jobs=n_jobs,
        )


def _point_estimate_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        (0, "s1"): ("observed", 1.0, 0.2),
        (1, "s1"): ("observed", 3.0, 0.4),
        (0, "s2"): ("structural_zero", 0.0, 0.0),
        (1, "s2"): ("observed", 2.0, 0.2),
    }
    for (repeat_index, subject_id), (status, effect, gain) in values.items():
        rows.append(
            {
                "repeat_index": repeat_index,
                "repeat_id": f"repeat-{repeat_index}",
                "fold_id": f"fold-{repeat_index}-{subject_id}",
                "subject_id": subject_id,
                "contrast_name": "treated-vs-control",
                "receiver": "Receiver",
                "family_id": "family-1",
                "driver_ids": ("driver-1", "driver-2"),
                "family_available": True,
                "family_estimable": True,
                "family_selected": True,
                "selection_status": "observed",
                "selection_reason_code": None,
                "null_loss": 1.0,
                "family_loss": 1.0 - gain,
                "differential_effect": effect,
                "bounded_incremental_gain": gain,
                "effect_status": status,
                "effect_reason_code": None,
            }
        )
    return pd.DataFrame(
        rows,
        columns=repeated_crossfit_module._SUBJECT_FAMILY_REPEAT_COLUMNS,
    )


def test_repeat_point_estimates_use_equal_subject_aggregation() -> None:
    subjects, families = (
        repeated_crossfit_module._build_repeat_aggregated_point_estimates(
            _point_estimate_fixture(),
            n_repeats=2,
            subject_ids=("s1", "s2"),
            partition_by_repeat={0: "partition-a", 1: "partition-b"},
        )
    )

    assert list(subjects["subject_id"]) == ["s1", "s2"]
    s1, s2 = (subjects.iloc[index] for index in range(2))
    assert s1["repeat_mean_differential_effect"] == pytest.approx(2.0)
    assert s1["repeat_mean_bounded_incremental_gain"] == pytest.approx(0.3)
    assert s1["repeat_sd_differential_effect"] == pytest.approx(1.0)
    assert s2["n_structural_zero_repeats"] == 1
    assert s2["repeat_mean_differential_effect"] == pytest.approx(1.0)
    assert s2["repeat_mean_bounded_incremental_gain"] == pytest.approx(0.1)
    assert set(subjects["formal_inference_status"]) == {
        "not_computed_repeated_crossfit_diagnostic_only"
    }

    assert len(families) == 1
    family = families.iloc[0]
    assert family["status"] == "observed"
    assert family["subject_coverage_fraction"] == pytest.approx(1.0)
    assert family["equal_subject_mean_differential_effect"] == pytest.approx(1.5)
    assert family["equal_subject_mean_bounded_incremental_gain"] == pytest.approx(0.2)
    assert family["between_subject_sd_differential_effect"] == pytest.approx(0.5)
    assert family["between_subject_sd_bounded_incremental_gain"] == pytest.approx(0.1)


def test_repeat_point_estimates_fail_closed_on_incomplete_subject_coverage() -> None:
    values = _point_estimate_fixture()
    missing = values["repeat_index"].eq(1) & values["subject_id"].eq("s2")
    values.loc[missing, "differential_effect"] = None
    values.loc[missing, "bounded_incremental_gain"] = None
    values.loc[missing, "effect_status"] = "not_estimable"
    values.loc[missing, "effect_reason_code"] = "receiver_absent_in_outer_training"

    subjects, families = (
        repeated_crossfit_module._build_repeat_aggregated_point_estimates(
            values,
            n_repeats=2,
            subject_ids=("s1", "s2"),
            partition_by_repeat={0: "partition-a", 1: "partition-b"},
        )
    )

    s2 = subjects.loc[subjects["subject_id"].eq("s2")].iloc[0]
    assert s2["status"] == "not_estimable"
    assert s2["n_not_estimable_repeats"] == 1
    assert pd.isna(s2["repeat_mean_differential_effect"])
    family = families.iloc[0]
    assert family["status"] == "not_estimable"
    assert family["reason_code"] == "incomplete_subject_repeat_effect_coverage"
    assert family["subject_coverage_fraction"] == pytest.approx(0.5)
    assert pd.isna(family["equal_subject_mean_differential_effect"])


def test_repeat_point_estimates_require_distinct_partitions_and_consistent_values() -> (
    None
):
    values = _point_estimate_fixture()
    _, families = repeated_crossfit_module._build_repeat_aggregated_point_estimates(
        values,
        n_repeats=2,
        subject_ids=("s1", "s2"),
        partition_by_repeat={0: "same", 1: "same"},
    )
    assert families.iloc[0]["status"] == "not_estimable"
    assert families.iloc[0]["reason_code"] == (
        "insufficient_distinct_repeat_partitions"
    )

    inconsistent = values.copy(deep=True)
    inconsistent.loc[inconsistent.index[0], "bounded_incremental_gain"] = None
    with pytest.raises(ValueError, match="status and repeat point-estimate"):
        repeated_crossfit_module._build_repeat_aggregated_point_estimates(
            inconsistent,
            n_repeats=2,
            subject_ids=("s1", "s2"),
            partition_by_repeat={0: "partition-a", 1: "partition-b"},
        )
