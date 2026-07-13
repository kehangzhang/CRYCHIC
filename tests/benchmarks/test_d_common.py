from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.metrics.d_common import DCommonSpec, d_common_edge_effects
from benchmarks.metrics.multicondition import EDGE_KEYS, METHOD_IDENTITY_KEYS


def _prepared_scores(
    design: pd.DataFrame,
    *,
    methods: tuple[str, ...] = ("m1", "m2"),
    values: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    supplied = values or {}
    rows: list[dict[str, object]] = []
    for method in methods:
        for sample in design.itertuples(index=False):
            for edge_index in range(2):
                strength = supplied.get(
                    (str(sample.sample_id), f"i{edge_index}"),
                    0.2 + 0.1 * edge_index,
                )
                rows.append(
                    {
                        "dataset": "d1",
                        "method": method,
                        "method_version": "1",
                        "analysis_track": "lr_stlr",
                        "resource": "r1",
                        "resource_version": "1",
                        "resource_mode": "H-common",
                        "score_semantics": "native",
                        "universe_id": "u1",
                        "contrast": "case-vs-control",
                        "sample_id": sample.sample_id,
                        "subject_id": sample.subject_id,
                        "context": sample.context,
                        "sender": "S",
                        "receiver": "R",
                        "interaction_id": f"i{edge_index}",
                        "ligand": f"L{edge_index}",
                        "receptor": f"R{edge_index}",
                        "score": strength,
                        "score_direction": "higher",
                        "status": "observed",
                        "universe_member": True,
                        "universe_size": 2,
                        "comparison_eligible": True,
                        "comparison_rank": 1.0,
                        "comparison_strength": strength,
                        "eligible_universe_size": 2,
                        "comparison_eligible_size": 2,
                    }
                )
    columns = [
        *METHOD_IDENTITY_KEYS,
        "contrast",
        "sample_id",
        "subject_id",
        "context",
        *EDGE_KEYS,
    ]
    result = pd.DataFrame(rows)
    return result.sort_values(columns, kind="stable", ignore_index=True)


def _unpaired_design(*, confounded: bool = False) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for context, prefix in (("control", "c"), ("case", "t")):
        for index in range(5):
            rows.append(
                {
                    "sample_id": f"{prefix}{index}",
                    "subject_id": f"{prefix}{index}",
                    "context": context,
                    "batch": (
                        "b1"
                        if confounded and context == "control"
                        else "b2"
                        if confounded
                        else f"b{1 + index % 2}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_d_common_uses_one_frozen_batch_adjusted_model_for_all_methods() -> None:
    design = _unpaired_design()
    values: dict[tuple[str, str], float] = {}
    for row in design.itertuples(index=False):
        batch = 0.30 if row.batch == "b2" else 0.0
        treatment = 0.20 if row.context == "case" else 0.0
        subject_noise = int(str(row.subject_id)[1:]) * 0.01
        values[(row.sample_id, "i0")] = 0.1 + batch + treatment + subject_noise
        values[(row.sample_id, "i1")] = 0.3 + subject_noise

    result = d_common_edge_effects(
        _prepared_scores(design, values=values),
        design,
        spec=DCommonSpec(
            reference="control",
            target="case",
            batch_keys=("batch",),
            min_subjects_per_group=3,
        ),
        validated=True,
    )

    selected = result.loc[result["interaction_id"].eq("i0")]
    assert set(selected["status"]) == {"exploratory"}
    assert selected["d_common_effect_model_id"].nunique() == 1
    np.testing.assert_allclose(selected["effect"], 0.2, atol=1e-12)
    assert set(selected["design_kind"]) == {"unpaired_subject_ols"}


def test_d_common_fails_closed_when_batch_is_fully_confounded() -> None:
    design = _unpaired_design(confounded=True)
    result = d_common_edge_effects(
        _prepared_scores(design),
        design,
        spec=DCommonSpec(
            reference="control",
            target="case",
            batch_keys=("batch",),
        ),
        validated=True,
    )

    assert set(result["status"]) == {"not_estimable"}
    assert set(result["reason_code"]) == {"target_not_estimable_after_batch_adjustment"}
    assert result["effect"].isna().all()


def test_d_common_paired_difference_adjusts_sample_varying_batch() -> None:
    design = pd.DataFrame(
        [
            {
                "sample_id": f"{subject}-{context}",
                "subject_id": subject,
                "context": context,
                "batch": "b2" if context == "case" and index % 2 else "b1",
            }
            for index, subject in enumerate(("s1", "s2", "s3", "s4", "s5"))
            for context in ("control", "case")
        ]
    )
    values: dict[tuple[str, str], float] = {}
    for row in design.itertuples(index=False):
        batch = 0.4 if row.batch == "b2" else 0.0
        treatment = 0.15 if row.context == "case" else 0.0
        values[(row.sample_id, "i0")] = 0.2 + batch + treatment
        values[(row.sample_id, "i1")] = 0.1

    result = d_common_edge_effects(
        _prepared_scores(design, methods=("m1",), values=values),
        design,
        spec=DCommonSpec(
            reference="control",
            target="case",
            batch_keys=("batch",),
            min_subjects_per_group=3,
        ),
        validated=True,
    )

    row = result.loc[result["interaction_id"].eq("i0")].iloc[0]
    assert row["status"] == "exploratory"
    assert row["design_kind"] == "paired_difference_ols"
    assert row["n_paired_subjects"] == 5
    assert row["effect"] == pytest.approx(0.15)


def test_d_common_marks_mixed_paired_unpaired_design_not_estimable() -> None:
    design = pd.DataFrame(
        [
            {"sample_id": "s1-c", "subject_id": "s1", "context": "control"},
            {"sample_id": "s1-t", "subject_id": "s1", "context": "case"},
            {"sample_id": "s2-c", "subject_id": "s2", "context": "control"},
            {"sample_id": "s3-t", "subject_id": "s3", "context": "case"},
        ]
    )
    result = d_common_edge_effects(
        _prepared_scores(design, methods=("m1",)),
        design,
        spec=DCommonSpec(
            reference="control",
            target="case",
            min_subjects_per_group=2,
        ),
        validated=True,
    )

    assert set(result["status"]) == {"not_estimable"}
    assert set(result["reason_code"]) == {
        "mixed_paired_unpaired_requires_repeated_measures_backend"
    }
