from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from crychic.inference import (
    DifferentialContrastSpec,
    DifferentialDesignSpec,
    fit_design_aware_differential,
    fit_design_aware_hypergraph_shrinkage_v2,
)
from crychic.scoring import (
    HYPERGRAPH_UNCERTAINTY_SHRINKAGE_V2_VERSION,
    UncertaintyAwareHypergraphShrinkageV2Spec,
    fit_uncertainty_aware_hypergraph_shrinkage_v2,
    freeze_hypergraph_prior,
    permute_hypergraph_prior_degree_matched,
    sparse_hypergraph_incidence_v2,
)


def _edge_table() -> pd.DataFrame:
    rows = [
        {
            "edge_id": f"edge-{sender}-{ligand}-{receptor}",
            "sender": sender,
            "ligand": ligand,
            "receptor": receptor,
        }
        for sender in ("s0", "s1", "s2")
        for ligand in ("l0", "l1", "l2", "l3")
        for receptor in ("r0", "r1", "r2")
    ]
    return pd.DataFrame(rows)


def _prior():
    return freeze_hypergraph_prior(
        _edge_table(),
        view_columns=("sender", "ligand", "receptor"),
    )


def _truth(prior) -> np.ndarray:
    sender = {"s0": -0.8, "s1": 0.3, "s2": 1.0}
    ligand = {"l0": -0.5, "l1": 0.1, "l2": 0.6, "l3": 1.1}
    receptor = {"r0": -0.3, "r1": 0.2, "r2": 0.7}
    return np.asarray(
        [
            sender[edge.memberships[0]]
            + ligand[edge.memberships[1]]
            + receptor[edge.memberships[2]]
            for edge in prior.edges
        ],
        dtype=float,
    )


def _estimates(*, seed: int = 19) -> tuple[pd.DataFrame, np.ndarray]:
    prior = _prior()
    truth = _truth(prior)
    rng = np.random.default_rng(seed)
    standard_error = np.where(np.arange(len(truth)) % 3 == 0, 0.15, 0.9)
    observed = truth + rng.normal(scale=standard_error)
    return (
        pd.DataFrame(
            {
                "edge_id": [edge.edge_id for edge in prior.edges],
                "effect": observed,
                "standard_error": standard_error,
            }
        ),
        truth,
    )


def test_sparse_incidence_has_exact_intercept_and_view_nnz() -> None:
    prior = _prior()
    incidence, columns = sparse_hypergraph_incidence_v2(prior)

    assert sparse.isspmatrix_csr(incidence)
    assert incidence.shape == (36, 1 + 3 + 4 + 3)
    assert incidence.nnz == 36 * 4
    assert columns[0] == "intercept"
    assert np.asarray(incidence.sum(axis=1)).ravel().tolist() == [4.0] * 36


def test_shrinkage_factor_and_conditional_posterior_se_are_exact() -> None:
    prior = _prior()
    estimates, _ = _estimates()
    result, fit = fit_uncertainty_aware_hypergraph_shrinkage_v2(
        estimates,
        prior=prior,
        spec=UncertaintyAwareHypergraphShrinkageV2Spec(
            node_ridge_penalty=0.5,
            minimum_observed_edges=4,
        ),
    )

    expected_kappa = fit.prior_variance / (
        fit.prior_variance + np.square(result["raw_standard_error"])
    )
    expected_effect = (
        expected_kappa * result["raw_effect"]
        + (1.0 - expected_kappa) * result["topology_mean"]
    )
    expected_se = np.sqrt(expected_kappa * np.square(result["raw_standard_error"]))
    assert result["shrinkage_factor"].to_numpy() == pytest.approx(expected_kappa)
    assert result["posterior_effect"].to_numpy() == pytest.approx(expected_effect)
    assert result["posterior_standard_error"].to_numpy() == pytest.approx(expected_se)
    assert result.loc[0, "shrinkage_factor"] > result.loc[1, "shrinkage_factor"]
    assert fit.incidence_nnz == 36 * 4
    assert result["score_version"].unique().tolist() == [
        HYPERGRAPH_UNCERTAINTY_SHRINKAGE_V2_VERSION
    ]
    assert not result["formal_inference_allowed"].any()
    assert not {"p_value", "q_value"}.intersection(result.columns)


def test_correct_topology_reduces_mse_more_than_degree_matched_permutation() -> None:
    prior = _prior()
    estimates, truth = _estimates(seed=31)
    spec = UncertaintyAwareHypergraphShrinkageV2Spec(
        node_ridge_penalty=0.5,
        minimum_observed_edges=4,
    )
    correct, _ = fit_uncertainty_aware_hypergraph_shrinkage_v2(
        estimates,
        prior=prior,
        spec=spec,
    )
    wrong, _ = fit_uncertainty_aware_hypergraph_shrinkage_v2(
        estimates,
        prior=permute_hypergraph_prior_degree_matched(prior, seed=29),
        spec=spec,
    )

    raw_mse = float(np.mean(np.square(estimates["effect"] - truth)))
    correct_mse = float(np.mean(np.square(correct["posterior_effect"] - truth)))
    wrong_mse = float(np.mean(np.square(wrong["posterior_effect"] - truth)))
    assert correct_mse < raw_mse
    assert correct_mse < wrong_mse


def test_missing_effect_remains_not_estimable_without_poisoning_other_edges() -> None:
    prior = _prior()
    estimates, _ = _estimates()
    estimates.loc[0, ["effect", "standard_error"]] = np.nan

    result, fit = fit_uncertainty_aware_hypergraph_shrinkage_v2(
        estimates,
        prior=prior,
    )

    assert fit.observed_edge_count == 35
    assert result.loc[0, "status"] == "not_estimable"
    assert result.loc[0, "reason_code"] == "input_effect_not_estimable"
    assert pd.isna(result.loc[0, "posterior_effect"])
    assert result.loc[1:, "posterior_effect"].notna().all()
    assert result["topology_mean"].notna().all()


def test_fit_is_row_order_deterministic_and_rejects_bad_uncertainty() -> None:
    prior = _prior()
    estimates, _ = _estimates()
    spec = UncertaintyAwareHypergraphShrinkageV2Spec(minimum_observed_edges=4)

    first, first_fit = fit_uncertainty_aware_hypergraph_shrinkage_v2(
        estimates,
        prior=prior,
        spec=spec,
    )
    second, second_fit = fit_uncertainty_aware_hypergraph_shrinkage_v2(
        estimates.sample(frac=1.0, random_state=11),
        prior=prior,
        spec=spec,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_fit.fit_id == second_fit.fit_id
    mismatched = estimates.copy()
    mismatched.loc[0, "standard_error"] = np.nan
    with pytest.raises(ValueError, match="jointly observed"):
        fit_uncertainty_aware_hypergraph_shrinkage_v2(
            mismatched,
            prior=prior,
            spec=spec,
        )
    nonpositive = estimates.copy()
    nonpositive.loc[0, "standard_error"] = 0.0
    with pytest.raises(ValueError, match="positive SEs"):
        fit_uncertainty_aware_hypergraph_shrinkage_v2(
            nonpositive,
            prior=prior,
            spec=spec,
        )


def test_design_aware_adapter_binds_one_exact_contrast_lineage() -> None:
    prior = _prior()
    truth = _truth(prior)
    residual = np.asarray([-0.12, -0.04, 0.03, 0.1, -0.08, 0.06, 0.02, 0.03])
    rows = [
        {
            "event_id": edge.edge_id,
            "sample_id": f"{condition}-{index}",
            "subject_id": f"{condition}-{index}",
            "condition": condition,
            "score": (
                residual[index]
                if condition == "control"
                else truth[edge_index] + residual[::-1][index]
            ),
            "score_status": "observed",
            "out_of_fold": True,
        }
        for edge_index, edge in enumerate(prior.edges)
        for condition in ("control", "treated")
        for index in range(8)
    ]
    design = DifferentialDesignSpec(
        design_kind="independent_two_group",
        condition_column="condition",
        condition_levels=("control", "treated"),
        contrasts=(
            DifferentialContrastSpec(
                name="treated-vs-control",
                weights=(("control", -1.0), ("treated", 1.0)),
            ),
        ),
        precision_weight_column=None,
        minimum_subjects_per_level=4,
    )
    differential = fit_design_aware_differential(pd.DataFrame(rows), design)

    result = fit_design_aware_hypergraph_shrinkage_v2(
        differential,
        prior=prior,
        contrast_name="treated-vs-control",
        spec=UncertaintyAwareHypergraphShrinkageV2Spec(minimum_observed_edges=4),
    )

    assert len(result.shrinkage) == len(prior.edges)
    assert result.design_spec_id == design.spec_id
    assert result.contrast_name == "treated-vs-control"
    assert result.to_manifest()["required_formal_next_stage"] == (
        "full_pipeline_subject_resampling"
    )
