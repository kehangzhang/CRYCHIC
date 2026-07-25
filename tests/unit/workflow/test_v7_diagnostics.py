from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from tests.unit.workflow.test_full_pipeline_resampling import (
    _adata,
    _bundle,
    _config,
    _prior,
    _spec,
)

from crychic.scoring import AbsoluteActivityV2Spec
from crychic.workflow import (
    CANDIDATE_SENDER_BIAS_COLUMNS,
    GATE_ATTRITION_COLUMNS,
    GATE_STAGES,
    RESOLUTION_PERFORMANCE_COLUMNS,
    SCORE_GEOMETRY_COLUMNS,
    CrossFitArtifacts,
    V7DiagnosticsResult,
    build_v7_diagnostics,
    run_subject_crossfit,
    summarize_v7_resolution_performance,
)


@pytest.fixture(scope="module")
def crossfit() -> CrossFitArtifacts:
    return run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=replace(
            _spec(),
            absolute_activity_v2_spec=AbsoluteActivityV2Spec(),
        ),
    )


def _truth(crossfit: CrossFitArtifacts) -> pd.DataFrame:
    truth = (
        crossfit.oof_sample_edge_scores_v2.loc[
            :, ["sender", "receiver", "interaction_id"]
        ]
        .drop_duplicates(ignore_index=True)
        .copy()
    )
    truth["contrast_name"] = "stim_vs_control"
    truth["truth"] = truth["sender"].eq("Sender") & truth["receiver"].eq("Receiver")
    truth["truth_effect"] = truth["truth"].astype(float)
    truth["mechanism_class"] = "secreted"
    truth["pathway"] = "synthetic_pathway"
    return truth


def _cell_counts(adata: AnnData) -> pd.DataFrame:
    return (
        adata.obs.groupby(["sample_id", "cell_type"], observed=True)
        .size()
        .rename("cell_count")
        .reset_index()
    )


def _annotated_key(crossfit: CrossFitArtifacts) -> dict[str, str]:
    row = crossfit.oof_sample_edge_scores_v2.iloc[0]
    return {
        "contrast_name": "stim_vs_control",
        **{
            column: str(row[column])
            for column in (
                "fold_id",
                "sample_id",
                "context_id",
                "sender",
                "receiver",
                "interaction_id",
            )
        },
    }


def _gate_annotations(crossfit: CrossFitArtifacts) -> pd.DataFrame:
    key = _annotated_key(crossfit)
    return pd.DataFrame.from_records(
        [
            {
                **key,
                "stage": stage,
                "stage_status": "passed",
                "reason_code": None,
            }
            for stage in GATE_STAGES[2:-1]
        ]
    )


def _significance(crossfit: CrossFitArtifacts) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [{**_annotated_key(crossfit), "significant": True, "reason_code": None}]
    )


def _resolution_evaluation() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level in (
        "lr_only",
        "lr_receiver_parent",
        "sender_lr_receiver_child",
        "pathway",
        "exact_hyperedge",
    ):
        rows.extend(
            [
                {
                    "contrast_name": "stim_vs_control",
                    "score_head": "parent_mean_raw",
                    "resolution_level": level,
                    "unit_id": f"{level}:positive",
                    "score": 2.0,
                    "truth": True,
                    "truth_effect": 1.5,
                },
                {
                    "contrast_name": "stim_vs_control",
                    "score_head": "parent_mean_raw",
                    "resolution_level": level,
                    "unit_id": f"{level}:negative",
                    "score": 0.1,
                    "truth": False,
                    "truth_effect": 0.0,
                },
            ]
        )
    return pd.DataFrame.from_records(rows)


def _report(crossfit: CrossFitArtifacts) -> V7DiagnosticsResult:
    return build_v7_diagnostics(
        crossfit,
        dataset_id="tiny-v7",
        truth=_truth(crossfit),
        cell_counts=_cell_counts(_adata()),
        gate_annotations=_gate_annotations(crossfit),
        significance=_significance(crossfit),
        resolution_evaluation=_resolution_evaluation(),
    )


def test_v7_diagnostics_cover_all_requested_outputs(
    crossfit: CrossFitArtifacts,
) -> None:
    result = _report(crossfit)

    assert tuple(result.gate_attrition.columns) == GATE_ATTRITION_COLUMNS
    assert tuple(result.score_geometry.columns) == SCORE_GEOMETRY_COLUMNS
    assert tuple(result.candidate_sender_bias.columns) == (
        CANDIDATE_SENDER_BIAS_COLUMNS
    )
    assert tuple(result.resolution_performance.columns) == (
        RESOLUTION_PERFORMANCE_COLUMNS
    )
    overall = result.gate_attrition.loc[
        result.gate_attrition["fold_id"].eq("__all__")
        & result.gate_attrition["stratum_kind"].eq("overall")
    ]
    assert tuple(overall["stage"]) == GATE_STAGES
    assert tuple(overall["n_cumulative_retained"]) == (
        32,
        32,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert set(result.gate_attrition["stratum_kind"]) == {
        "overall",
        "truth_class",
        "mechanism_class",
        "candidate_sender_bin",
        "cell_count_bin",
        "receptor_complex_cardinality",
    }
    assert {"truth_positive", "truth_negative"}.issubset(
        result.gate_attrition["stratum_value"]
    )
    assert "secreted" in set(result.gate_attrition["stratum_value"])
    assert "0-19" in set(result.gate_attrition["stratum_value"])
    assert "1" in set(result.gate_attrition["stratum_value"])

    geometry = result.score_geometry.loc[
        result.score_geometry["fold_id"].eq("__all__")
    ].set_index("score_head")
    assert int(geometry.loc["sender_detection_raw", "n_rows"]) == 32
    assert int(geometry.loc["parent_mean_raw", "n_rows"]) == 16
    assert geometry.loc["sender_detection_raw", "status"] == "observed"
    assert geometry.loc["program_signed", "status"] == "not_estimable"
    assert pd.notna(geometry.loc["parent_mean_raw", "truth_positive_q50"])
    assert pd.notna(geometry.loc["parent_mean_raw", "truth_negative_q50"])

    sender_bias = result.candidate_sender_bias.loc[
        result.candidate_sender_bias["fold_id"].eq("__all__")
    ].iloc[0]
    assert sender_bias["candidate_sender_bin"] == "2"
    assert int(sender_bias["n_truth_positive_rows"]) == 8
    assert int(sender_bias["n_truth_negative_rows"]) == 24
    assert float(sender_bias["sender_auprc"]) == pytest.approx(1.0)
    assert float(sender_bias["sender_auroc"]) == pytest.approx(1.0)

    assert set(result.resolution_performance["resolution_level"]) == {
        "lr_only",
        "lr_receiver_parent",
        "sender_lr_receiver_child",
        "pathway",
        "exact_hyperedge",
    }
    assert np.allclose(result.resolution_performance["auprc"], 1.0)
    assert np.allclose(result.resolution_performance["auroc"], 1.0)
    manifest = result.to_manifest()
    assert manifest["result_id"] == result.result_id
    assert manifest["formal_inference_allowed"] is False
    assert manifest["unknown_truth_treated_as_negative"] is False
    assert manifest["uncomputed_gate_treated_as_failure"] is False
    result._require_intact()


def test_uncomputed_legacy_gates_are_not_called_failures(
    crossfit: CrossFitArtifacts,
) -> None:
    result = build_v7_diagnostics(crossfit, dataset_id="tiny-v7")
    overall = result.gate_attrition.loc[
        result.gate_attrition["fold_id"].eq("__all__")
        & result.gate_attrition["stratum_kind"].eq("overall")
    ].set_index("stage")
    for stage in (*GATE_STAGES[2:-1], "significant"):
        assert int(overall.loc[stage, "n_stage_failed"]) == 0
        assert int(overall.loc[stage, "n_not_computed"]) == 32
    assert int(overall.loc["measurable_ligand_receptor", "n_stage_passed"]) == 32


def test_diagnostic_identity_is_input_order_independent_and_tamper_evident(
    crossfit: CrossFitArtifacts,
) -> None:
    ordered = _report(crossfit)
    shuffled = build_v7_diagnostics(
        crossfit,
        dataset_id="tiny-v7",
        truth=_truth(crossfit).sample(frac=1.0, random_state=1),
        cell_counts=_cell_counts(_adata()).sample(frac=1.0, random_state=2),
        gate_annotations=_gate_annotations(crossfit).sample(frac=1.0, random_state=3),
        significance=_significance(crossfit),
        resolution_evaluation=_resolution_evaluation().sample(frac=1.0, random_state=4),
    )
    assert ordered.input_digest == shuffled.input_digest
    assert ordered.output_digest == shuffled.output_digest
    assert ordered.result_id == shuffled.result_id

    ordered.score_geometry.loc[0, "n_rows"] += 1
    with pytest.raises(ValueError, match="integrity violation"):
        ordered._require_intact()


def test_resolution_metrics_require_explicit_truth_and_preserve_unknown_scores() -> (
    None
):
    evaluation = pd.DataFrame.from_records(
        [
            {
                "contrast_name": "b_vs_a",
                "score_head": "sender_detection_raw",
                "resolution_level": "exact_hyperedge",
                "unit_id": "positive",
                "score": 0.9,
                "truth": True,
                "truth_effect": 1.0,
            },
            {
                "contrast_name": "b_vs_a",
                "score_head": "sender_detection_raw",
                "resolution_level": "exact_hyperedge",
                "unit_id": "negative",
                "score": 0.2,
                "truth": False,
                "truth_effect": 0.0,
            },
            {
                "contrast_name": "b_vs_a",
                "score_head": "sender_detection_raw",
                "resolution_level": "exact_hyperedge",
                "unit_id": "missing-score",
                "score": np.nan,
                "truth": False,
                "truth_effect": 0.0,
            },
        ]
    )
    result = summarize_v7_resolution_performance(evaluation, dataset_id="toy")
    row = result.iloc[0]
    assert int(row["n_units"]) == 3
    assert int(row["n_scored_units"]) == 2
    assert float(row["auprc"]) == pytest.approx(1.0)
    assert float(row["auroc"]) == pytest.approx(1.0)

    invalid = evaluation.drop(columns="truth")
    with pytest.raises(ValueError, match="missing columns"):
        summarize_v7_resolution_performance(invalid, dataset_id="toy")

    invalid_score = evaluation.astype({"score": "object"})
    invalid_score.loc[0, "score"] = "not-a-number"
    with pytest.raises(ValueError, match="score must be numeric"):
        summarize_v7_resolution_performance(invalid_score, dataset_id="toy")


def test_unknown_sender_truth_is_not_removed_before_top1_evaluation(
    crossfit: CrossFitArtifacts,
) -> None:
    truth = _truth(crossfit)
    unknown_competitor = truth["sender"].eq("Receiver") & truth["receiver"].eq(
        "Receiver"
    )
    result = build_v7_diagnostics(
        crossfit,
        dataset_id="tiny-v7",
        truth=truth.loc[~unknown_competitor],
    )
    combined = result.candidate_sender_bias.loc[
        result.candidate_sender_bias["fold_id"].eq("__all__")
    ].iloc[0]
    assert pd.isna(combined["true_sender_top1_accuracy"])


def test_truth_and_gate_contracts_fail_closed(crossfit: CrossFitArtifacts) -> None:
    duplicate_truth = pd.concat([_truth(crossfit), _truth(crossfit).iloc[[0]]])
    with pytest.raises(ValueError, match="keys must be unique"):
        build_v7_diagnostics(
            crossfit,
            dataset_id="tiny-v7",
            truth=duplicate_truth,
        )

    invalid_gate = _gate_annotations(crossfit)
    invalid_gate.loc[0, "stage"] = "common_candidate_universe"
    with pytest.raises(ValueError, match="legacy intermediate"):
        build_v7_diagnostics(
            crossfit,
            dataset_id="tiny-v7",
            gate_annotations=invalid_gate,
        )
