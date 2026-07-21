from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.literature.component_swap_benchmark import (
    EDGE_COLUMNS,
    _build_pair_rankings,
    _effect_arm,
    _pair_universe,
    _pairwise_diagnostics,
    _wilcoxon_arm,
)


def _effects() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sender": ["A", "A", "B", "B"],
            "receiver": ["A", "B", "A", "B"],
            "interaction_id": ["i1", "i1", "i1", "i1"],
            "ligand": ["L"] * 4,
            "receptor": ["R"] * 4,
            "family_id": ["f"] * 4,
            "driver_id": ["L"] * 4,
            "reference_condition": ["ref"] * 4,
            "target_condition": ["target"] * 4,
            "mean_strength_reference": [0.1, 0.3, 0.5, 0.2],
            "mean_strength_target": [0.4, 0.1, 0.5, 0.5],
            "n_samples_reference": [4] * 4,
            "n_samples_target": [4] * 4,
            "n_subjects_reference": [4] * 4,
            "n_subjects_target": [4] * 4,
            "effect_target_minus_reference": [0.3, -0.2, 0.0, 0.3],
            "effect_standard_error_hc2": [0.1, 0.3, 0.0, 0.6],
            "status": ["observed"] * 4,
            "reason_code": [None] * 4,
        }
    )


def test_effect_arms_separate_hard_wald_and_continuous_heads() -> None:
    effects = _effects()
    arm_a = _effect_arm(
        effects,
        dataset_id="dataset",
        arm="A",
        mode="one_se",
        threshold=1.0,
    )
    assert arm_a["selected_target"].tolist() == [True, False, False, False]
    assert arm_a["selected_reference"].tolist() == [False, False, False, False]

    arm_c = _effect_arm(
        effects,
        dataset_id="dataset",
        arm="C",
        mode="wald",
        threshold=0.05,
    )
    assert arm_c["selected_target"].tolist() == [True, False, False, False]
    assert arm_c["p_value"].between(0.0, 1.0).all()

    arm_f = _effect_arm(
        effects,
        dataset_id="dataset",
        arm="F",
        mode="continuous",
    )
    assert 0.0 < arm_f.loc[0, "target_weight"] < 1.0
    assert 0.0 < arm_f.loc[1, "reference_weight"] < 1.0
    assert arm_f.loc[2, "target_weight"] == 0.0
    assert arm_f.loc[2, "reference_weight"] == 0.0


def test_pair_ranking_collapses_directions_and_tracks_opportunity() -> None:
    effects = _effects()
    edges = _effect_arm(
        effects,
        dataset_id="dataset",
        arm="A",
        mode="one_se",
        threshold=1.0,
    )
    universe = _pair_universe(effects)
    rankings = _build_pair_rankings(
        edges,
        universe,
        dataset_id="dataset",
        method="A",
        semantics="test",
    )
    ab_target = rankings.loc[
        rankings["condition"].eq("target")
        & rankings["sender"].eq("A")
        & rankings["receiver"].eq("B")
    ].iloc[0]
    assert ab_target["estimable_directed_lr"] == 2
    assert ab_target["ranked_strength"] == 0.0
    assert ab_target["condition_specific_directed_lr"] == 0


def test_wilcoxon_arm_uses_four_units_and_raw_threshold() -> None:
    effects = _effects().iloc[:2].copy()
    records: list[dict[str, object]] = []
    for edge_index, edge in effects.iterrows():
        for index in range(8):
            condition = "ref" if index < 4 else "target"
            if edge_index == 0:
                score = float(index) if condition == "target" else float(index - 10)
            else:
                score = 1.0
            records.append(
                {
                    **{column: edge[column] for column in EDGE_COLUMNS},
                    "sample_id": f"s{index}",
                    "condition": condition,
                    "direct_response_score": score,
                }
            )
    direct = pd.DataFrame.from_records(records)
    result = _wilcoxon_arm(
        direct,
        effects,
        dataset_id="dataset",
        condition_column="condition",
        reference="ref",
        target="target",
    )
    assert result["status"].eq("observed").all()
    assert bool(result.loc[0, "selected_target"])
    assert not bool(result.loc[1, "selected_target"])
    assert np.isclose(result.loc[1, "p_value"], 1.0)


def test_pairwise_diagnostics_reports_edge_pair_and_top10_metrics() -> None:
    effects = _effects()
    universe = _pair_universe(effects)
    edges_a = _effect_arm(
        effects,
        dataset_id="dataset",
        arm="A",
        mode="one_se",
        threshold=1.0,
    )
    edges_f = _effect_arm(
        effects,
        dataset_id="dataset",
        arm="F",
        mode="continuous",
    )
    rankings = pd.concat(
        [
            _build_pair_rankings(
                edges_a,
                universe,
                dataset_id="dataset",
                method="A",
                semantics="hard",
            ),
            _build_pair_rankings(
                edges_f,
                universe,
                dataset_id="dataset",
                method="F",
                semantics="continuous",
            ),
        ],
        ignore_index=True,
    )
    result = _pairwise_diagnostics({"A": edges_a, "F": edges_f}, rankings)
    assert len(result) == 2
    assert result["edge_sign_concordance"].eq(1.0).all()
    assert result["top10_pair_overlap_jaccard"].eq(1.0).all()
