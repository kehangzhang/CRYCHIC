from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.literature.signed_expected_cardinality import (
    build_signed_expected_cardinality_head,
    fit_spike_normal_working_prior,
    signed_working_probabilities,
)
from benchmarks.simulation.signed_cardinality_benchmark import (
    _select_candidate,
    simulate_effect_summary,
)


def _effect_table() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    effect = np.r_[rng.normal(0.0, 0.04, 450), rng.normal(0.3, 0.04, 50)]
    return pd.DataFrame(
        {
            "sender": ["A"] * 500,
            "receiver": ["B"] * 500,
            "interaction_id": [f"i{index}" for index in range(500)],
            "ligand": ["L"] * 500,
            "receptor": ["R"] * 500,
            "reference_condition": ["ref"] * 500,
            "target_condition": ["target"] * 500,
            "effect_target_minus_reference": effect,
            "effect_standard_error_hc2": [0.04] * 500,
            "n_samples_reference": [5] * 500,
            "n_samples_target": [6] * 500,
            "status": ["observed"] * 500,
            "reason_code": [None] * 500,
        }
    )


def test_spike_normal_probabilities_are_directional_and_scale_invariant() -> None:
    table = _effect_table()
    effect = table["effect_target_minus_reference"].to_numpy()
    se = table["effect_standard_error_hc2"].to_numpy()
    fit = fit_spike_normal_working_prior(effect, se)
    first = signed_working_probabilities(effect, se, fit, delta_fraction=0.25)
    scaled_fit = fit_spike_normal_working_prior(effect * 10.0, se * 10.0)
    second = signed_working_probabilities(
        effect * 10.0, se * 10.0, scaled_fit, delta_fraction=0.25
    )
    np.testing.assert_allclose(
        first["working_p_target_active"],
        second["working_p_target_active"],
        rtol=1e-5,
        atol=1e-7,
    )
    assert first["working_p_target_active"][-50:].mean() > 0.95
    assert first["working_p_reference_active"][-50:].mean() < 0.01
    assert not fit.formal_release_allowed


def test_head_preserves_unreleased_boundary_and_soft_weights() -> None:
    head, fit = build_signed_expected_cardinality_head(
        _effect_table(),
        dataset_id="synthetic",
        arm="G",
        delta_fraction=0.25,
    )
    assert head["probability_status"].eq("candidate_unreleased").all()
    assert not head["formal_release_allowed"].any()
    assert head["target_weight"].between(0.0, 1.0).all()
    assert head["reference_weight"].between(0.0, 1.0).all()
    assert fit.n_fit_edges == 500


def test_nonzero_effect_with_zero_se_fails_closed() -> None:
    with pytest.raises(ValueError, match="nonzero effects with zero standard error"):
        fit_spike_normal_working_prior(
            np.r_[np.ones(200), 1.0], np.r_[np.ones(200), 0.0]
        )


def test_simulation_is_deterministic_and_selection_ignores_holdout() -> None:
    config = {
        "simulation": {
            "pair_count": 6,
            "edge_opportunities": [40, 80],
            "effect_floor": 0.06,
            "effect_scale": 0.08,
        },
        "selection": {"eligible_scenarios": ["sparse_low_n"]},
    }
    first = simulate_effect_summary(config, scenario="sparse_low_n", seed=3)
    second = simulate_effect_summary(config, scenario="sparse_low_n", seed=3)
    pd.testing.assert_frame_equal(first, second)

    rows = []
    for split in ("development", "holdout"):
        for candidate, value in (("a", 0.8), ("b", 0.7)):
            if split == "holdout":
                value = 1.0 - value
            rows.append(
                {
                    "split": split,
                    "scenario": "sparse_low_n",
                    "candidate": candidate,
                    "pair_rank_spearman": value,
                    "top_quartile_auroc": value,
                    "tie_fraction": 0.0,
                }
            )
    selected, _ = _select_candidate(pd.DataFrame(rows), config)
    assert selected == "a"
