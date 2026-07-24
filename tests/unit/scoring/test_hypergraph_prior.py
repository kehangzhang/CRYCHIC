from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crychic.scoring import (
    HypergraphShrinkageSpec,
    fit_hypergraph_prior_shrinkage,
    freeze_hypergraph_prior,
    permute_hypergraph_prior_degree_matched,
    rewire_hypergraph_prior_degree_matched,
    select_hypergraph_prior_views,
)


def _edges() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "edge_id": f"e{index:02d}",
                "sender": f"s{index % 3}",
                "ligand": f"l{index % 4}",
                "receptor": f"r{(index // 2) % 4}",
                "receiver": f"c{(index // 3) % 3}",
                "pathway": f"p{index % 2}",
            }
            for index in range(24)
        ]
    )


def _prior():
    return freeze_hypergraph_prior(
        _edges(),
        view_columns=("sender", "ligand", "receptor", "receiver", "pathway"),
    )


def test_degree_matched_permutation_is_stable_and_preserves_every_view_degree() -> None:
    prior = _prior()
    first = permute_hypergraph_prior_degree_matched(prior, seed=11)
    second = permute_hypergraph_prior_degree_matched(prior, seed=11)

    assert first.prior_id == second.prior_id
    assert first.to_dict() == second.to_dict()
    assert first.prior_id != prior.prior_id
    assert first.degree_profile() == prior.degree_profile()
    assert first.parent_prior_id == prior.prior_id


def test_partial_rewiring_is_stable_bounded_and_degree_matched() -> None:
    prior = _prior()
    first = rewire_hypergraph_prior_degree_matched(
        prior, fraction=0.25, seed=17
    )
    second = rewire_hypergraph_prior_degree_matched(
        prior, fraction=0.25, seed=17
    )

    changed = sum(
        source.memberships != target.memberships
        for source, target in zip(prior.edges, first.edges, strict=True)
    )
    assert first.to_dict() == second.to_dict()
    assert first.prior_id != prior.prior_id
    assert first.parent_prior_id == prior.prior_id
    assert first.degree_profile() == prior.degree_profile()
    assert first.topology_kind == "degree_matched_partial_rewire:fraction=0.25"
    assert 0 < changed <= 6


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.0, 1.1, float("nan")])
def test_partial_rewiring_rejects_invalid_fractions(fraction: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        rewire_hypergraph_prior_degree_matched(_prior(), fraction=fraction, seed=3)


def test_view_subset_preserves_edges_and_declares_only_selected_memberships() -> None:
    prior = _prior()
    subset = select_hypergraph_prior_views(prior, view_names=("ligand", "receptor"))

    assert subset.view_names == ("ligand", "receptor")
    assert tuple(edge.edge_id for edge in subset.edges) == tuple(
        edge.edge_id for edge in prior.edges
    )
    assert subset.prior_id != prior.prior_id


def test_zero_penalty_has_exact_raw_parity_and_no_inferential_fields() -> None:
    prior = _prior()
    values = np.linspace(-1.0, 1.0, len(prior.edges))
    result, fit = fit_hypergraph_prior_shrinkage(
        pd.DataFrame(
            {"edge_id": [edge.edge_id for edge in prior.edges], "estimate": values}
        ),
        prior=prior,
        spec=HypergraphShrinkageSpec(edge_residual_penalty=0.0),
    )

    assert np.array_equal(result["raw_estimate"], result["shrunk_estimate"])
    assert not result["formal_inference_allowed"].any()
    assert fit.formal_inference_allowed is False
    assert not {"p_value", "q_value", "probability"}.intersection(result.columns)


def test_correct_topology_shrinkage_reduces_mse_more_than_permutation() -> None:
    prior = _prior()
    rng = np.random.default_rng(19)
    sender_effect = {f"s{index}": rng.normal() for index in range(3)}
    ligand_effect = {f"l{index}": rng.normal() for index in range(4)}
    truth = np.asarray(
        [
            sender_effect[edge.memberships[0]] + ligand_effect[edge.memberships[1]]
            for edge in prior.edges
        ]
    )
    observed = truth + rng.normal(scale=1.2, size=len(truth))
    estimates = pd.DataFrame(
        {"edge_id": [edge.edge_id for edge in prior.edges], "estimate": observed}
    )
    spec = HypergraphShrinkageSpec(node_ridge_penalty=1.0, edge_residual_penalty=1.0)
    correct, _ = fit_hypergraph_prior_shrinkage(
        estimates,
        prior=select_hypergraph_prior_views(prior, view_names=("sender", "ligand")),
        spec=spec,
    )
    wrong, _ = fit_hypergraph_prior_shrinkage(
        estimates,
        prior=select_hypergraph_prior_views(
            permute_hypergraph_prior_degree_matched(prior, seed=23),
            view_names=("sender", "ligand"),
        ),
        spec=spec,
    )

    raw_mse = float(np.mean((observed - truth) ** 2))
    correct_mse = float(np.mean((correct["shrunk_estimate"] - truth) ** 2))
    wrong_mse = float(np.mean((wrong["shrunk_estimate"] - truth) ** 2))
    assert correct_mse < raw_mse
    assert correct_mse < wrong_mse


def test_prior_and_fit_reject_incomplete_or_mismatched_universes() -> None:
    edges = _edges()
    edges.loc[0, "ligand"] = None
    with pytest.raises(ValueError, match="complete"):
        freeze_hypergraph_prior(edges, view_columns=("sender", "ligand"))

    prior = _prior()
    estimates = pd.DataFrame(
        {
            "edge_id": [edge.edge_id for edge in prior.edges[:-1]],
            "estimate": 1.0,
        }
    )
    with pytest.raises(ValueError, match="universe differs"):
        fit_hypergraph_prior_shrinkage(estimates, prior=prior)
