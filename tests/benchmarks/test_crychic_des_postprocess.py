from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.crychic.des_postprocess import (
    adjusted_subject_directed_lr_effects,
    condition_ranking_diagnostics,
    score_reason_waterfall,
    sender_specific_direct_scores,
    stable_breadth_unordered_cell_pair_rankings,
)

EDGE_COLUMNS = ("sender", "receiver", "interaction_id")


def test_sender_specific_direct_scores_never_imputes_missing_semantic_rows() -> None:
    score_layers = pd.DataFrame(
        {
            "fold_id": ["f1", "f1", "f1"],
            "sample_id": ["s1", "s2", "s3"],
            "subject_id": ["p1", "p2", "p3"],
            "condition": ["A", "A", "B"],
            "sender": ["S", "S", "S"],
            "receiver": ["R", "R", "R"],
            "interaction_id": ["lr1", "lr1", "lr1"],
            "mechanistic_status": [
                "observed",
                "structural_zero",
                "not_estimable",
            ],
            "mechanistic_reason_code": [None, "parent_zero", "parent_missing"],
        }
    )
    semantic = pd.DataFrame(
        {
            "fold_id": ["f1"],
            "sample_id": ["s1"],
            "subject_id": ["p1"],
            "sender": ["S"],
            "receiver": ["R"],
            "interaction_id": ["lr1"],
            "mode": ["state"],
            "availability_score": [0.4],
            "status": ["observed"],
            "reason_code": [None],
        }
    )

    direct = sender_specific_direct_scores(
        score_layers,
        semantic,
        condition_column="condition",
        edge_columns=EDGE_COLUMNS,
    )

    assert direct["direct_response_status"].tolist() == [
        "observed",
        "not_estimable",
        "not_estimable",
    ]
    assert direct["direct_response_score"].iloc[0] == 0.4
    assert pd.isna(direct["direct_response_score"].iloc[1])
    assert pd.isna(direct["direct_response_score"].iloc[2])
    assert direct["direct_response_reason_code"].iloc[1] == (
        "sender_specific_availability_not_estimable"
    )


def _bidirectional_direct_scores() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    values = {
        ("reference", "lr_target_up"): 0.1,
        ("target", "lr_target_up"): 0.3,
        ("reference", "lr_reference_up"): 0.4,
        ("target", "lr_reference_up"): 0.2,
    }
    for condition in ("reference", "target"):
        for subject_index in range(3):
            subject = f"{condition}_{subject_index}"
            for interaction_id in ("lr_target_up", "lr_reference_up"):
                records.append(
                    {
                        "sample_id": subject,
                        "subject_id": subject,
                        "condition": condition,
                        "sender": "A",
                        "receiver": "B",
                        "interaction_id": interaction_id,
                        "direct_response_score": values[(condition, interaction_id)],
                        "direct_response_status": "observed",
                        "direct_response_transform": "identity",
                    }
                )
    return pd.DataFrame.from_records(records)


def test_adjusted_contrast_is_exactly_antisymmetric_and_bidirectional() -> None:
    direct = _bidirectional_direct_scores()
    forward = adjusted_subject_directed_lr_effects(
        direct,
        reference="reference",
        target="target",
        condition_column="condition",
        edge_columns=EDGE_COLUMNS,
    )
    reverse = adjusted_subject_directed_lr_effects(
        direct,
        reference="target",
        target="reference",
        condition_column="condition",
        edge_columns=EDGE_COLUMNS,
    )

    forward_effect = forward["effect_target_minus_reference"].to_numpy()
    reverse_effect = reverse["effect_target_minus_reference"].to_numpy()
    assert np.array_equal(forward_effect, -reverse_effect)
    assert forward.set_index("interaction_id").loc[
        "lr_target_up", "effect_target_minus_reference"
    ] > 0
    assert forward.set_index("interaction_id").loc[
        "lr_reference_up", "effect_target_minus_reference"
    ] < 0
    assert forward["one_standard_error_stable"].all()
    assert forward["mean_strength_reference"].equals(
        reverse["mean_strength_target"]
    )
    assert forward["mean_strength_target"].equals(
        reverse["mean_strength_reference"]
    )


def test_batch_adjustment_removes_a_pure_batch_shift() -> None:
    records: list[dict[str, object]] = []
    design = (
        ("r1", "reference", "A"),
        ("r2", "reference", "A"),
        ("r3", "reference", "A"),
        ("r4", "reference", "B"),
        ("t1", "target", "A"),
        ("t2", "target", "B"),
        ("t3", "target", "B"),
        ("t4", "target", "B"),
    )
    for subject, condition, batch in design:
        records.append(
            {
                "sample_id": subject,
                "subject_id": subject,
                "condition": condition,
                "batch": batch,
                "sender": "A",
                "receiver": "B",
                "interaction_id": "lr1",
                "direct_response_score": 0.1 if batch == "A" else 0.9,
                "direct_response_status": "observed",
                "direct_response_transform": "identity",
            }
        )
    effects = adjusted_subject_directed_lr_effects(
        pd.DataFrame.from_records(records),
        reference="reference",
        target="target",
        condition_column="condition",
        categorical_covariates=("batch",),
        edge_columns=EDGE_COLUMNS,
    )

    row = effects.iloc[0]
    assert abs(float(row["effect_target_minus_reference"])) < 1e-12
    assert not bool(row["one_standard_error_stable"])
    assert row["covariate_adjustment"] == "categorical_fixed_effects:batch"


def test_contrast_specific_hc2_ignores_saturated_nuisance_only_rows() -> None:
    design = (
        ("r1", "reference", "1", 0.01),
        ("r2", "reference", "2", -0.02),
        ("r3", "reference", "3", 0.03),
        ("r4", "reference", "4", -0.01),
        ("r5", "reference", "4", 0.02),
        ("r6", "reference", "4", -0.03),
        ("t1", "target", "1", 0.02),
        ("t2", "target", "1", -0.01),
        ("t3", "target", "4", 0.03),
        ("t4", "target", "4", -0.02),
        ("t5", "target", "4", 0.01),
    )
    batch_offset = {"1": 0.1, "2": 0.5, "3": 0.8, "4": 0.3}
    records = [
        {
            "sample_id": subject,
            "subject_id": subject,
            "condition": condition,
            "batch": batch,
            "sender": "A",
            "receiver": "B",
            "interaction_id": "lr1",
            "direct_response_score": (
                batch_offset[batch]
                + (0.2 if condition == "target" else 0.0)
                + residual
            ),
            "direct_response_status": "observed",
            "direct_response_transform": "identity",
        }
        for subject, condition, batch, residual in design
    ]
    direct = pd.DataFrame.from_records(records)
    full = adjusted_subject_directed_lr_effects(
        direct,
        reference="reference",
        target="target",
        condition_column="condition",
        categorical_covariates=("batch",),
        edge_columns=EDGE_COLUMNS,
    ).iloc[0]
    reduced = adjusted_subject_directed_lr_effects(
        direct.loc[direct["batch"].isin(("1", "4"))],
        reference="reference",
        target="target",
        condition_column="condition",
        categorical_covariates=("batch",),
        edge_columns=EDGE_COLUMNS,
    ).iloc[0]

    assert full["status"] == "observed"
    assert full["n_model_subjects"] == 11
    assert full["n_contrast_informative_subjects"] == 9
    assert full["effect_target_minus_reference"] == pytest.approx(
        reduced["effect_target_minus_reference"], abs=1e-12
    )
    assert full["effect_standard_error_hc2"] == pytest.approx(
        reduced["effect_standard_error_hc2"], abs=1e-12
    )


def test_stable_breadth_recovers_both_conditions_and_reports_opportunity() -> None:
    effects = adjusted_subject_directed_lr_effects(
        _bidirectional_direct_scores(),
        reference="reference",
        target="target",
        condition_column="condition",
        edge_columns=EDGE_COLUMNS,
    )
    rankings, opportunity = stable_breadth_unordered_cell_pair_rankings(
        effects,
        dataset="synthetic",
        method="crychic",
        method_version="test",
        resource="synthetic_resource",
    )

    by_condition = rankings.set_index("condition")
    assert by_condition.loc["target", "ranked_strength"] == 1
    assert by_condition.loc["reference", "ranked_strength"] == 1
    assert opportunity.loc[0, "n_target_up_directed_lr"] == 1
    assert opportunity.loc[0, "n_reference_up_directed_lr"] == 1
    assert opportunity.loc[0, "n_one_se_stable_target_up_directed_lr"] == 1
    assert opportunity.loc[0, "n_one_se_stable_reference_up_directed_lr"] == 1


def test_condition_degeneracy_is_checked_per_condition() -> None:
    rankings = pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "ranked_strength": [0.0, 0.0, 1.0, 2.0],
            "status": ["observed"] * 4,
        }
    )
    diagnostics = condition_ranking_diagnostics(rankings)

    assert diagnostics["any_degenerate_condition"] is True
    assert diagnostics["conditions"]["A"]["degenerate_ranking"] is True
    assert diagnostics["conditions"]["B"]["degenerate_ranking"] is False


def test_condition_diagnostics_report_opportunity_correlation() -> None:
    rankings = pd.DataFrame(
        {
            "condition": ["A", "A", "A"],
            "ranked_strength": [1.0, 2.0, 3.0],
            "estimable_directed_lr": [10, 20, 30],
            "status": ["observed"] * 3,
        }
    )
    diagnostics = condition_ranking_diagnostics(rankings)

    condition = diagnostics["conditions"]["A"]
    assert condition[
        "spearman_ranked_strength_vs_estimable_directed_lr"
    ] == pytest.approx(1.0)
    assert condition["high_opportunity_correlation"] is True
    assert diagnostics["any_high_opportunity_correlation"] is True


def test_reason_waterfall_includes_base_gate_reasons() -> None:
    score_layers = pd.DataFrame(
        {
            "status": ["structural_zero", "observed"],
            "reason_code": ["ligand_contrast_not_supported", None],
            "mechanistic_status": ["structural_zero", "observed"],
            "mechanistic_reason_code": [
                "mechanistic_parent_structural_zero",
                None,
            ],
            "downstream_status": ["not_estimable", "observed"],
            "downstream_reason_code": ["downstream_missing", None],
            "selected_score_status": ["structural_zero", "observed"],
            "selected_score_reason_code": [
                "mechanistic_parent_structural_zero",
                None,
            ],
        }
    )
    direct = pd.DataFrame(
        {
            "direct_response_status": ["not_estimable", "observed"],
            "direct_response_reason_code": [
                "sender_specific_availability_not_estimable",
                None,
            ],
        }
    )
    waterfall = score_reason_waterfall(score_layers, direct)

    base = waterfall.loc[waterfall["score_layer"].eq("base")]
    assert "ligand_contrast_not_supported" in set(base["reason_code"].dropna())
