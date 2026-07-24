from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks.comprehensive.evaluate_m5_hprior import (
    FULL_METHOD,
    PERMUTED_METHOD,
    RAW_METHOD,
    _evaluate_dataset,
    _method_summary,
)
from benchmarks.comprehensive.generate_m5_hprior_fixture import (
    SCHEMA_VERSION,
    VIEW_COLUMNS,
    _dataset_seed,
    generate,
)
from crychic.scoring import (
    HypergraphShrinkageSpec,
    freeze_hypergraph_prior,
    permute_hypergraph_prior_degree_matched,
    select_hypergraph_prior_views,
)


def test_m5_generator_reuses_one_outcome_blind_prior_and_is_reproducible(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = generate(first, seeds=(20290101,), n_edges=240)
    second_manifest = generate(second, seeds=(20290101,), n_edges=240)

    assert first_manifest["schema_version"] == SCHEMA_VERSION
    assert first_manifest["hypergraph_prior"]["outcome_blind"]
    assert (
        first_manifest["hypergraph_prior"]["sha256"]
        == second_manifest["hypergraph_prior"]["sha256"]
    )
    assert (
        first_manifest["records"][0]["input_sha256"]
        == second_manifest["records"][0]["input_sha256"]
    )
    estimates = pd.read_csv(first / first_manifest["records"][0]["input"], sep="\t")
    assert "true_effect" not in estimates.columns


def test_m5_single_seed_full_prior_beats_raw_and_degree_matched_permutation(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    manifest = generate(fixture, seeds=(20290101,), n_edges=600)
    topology = pd.read_csv(fixture / manifest["hypergraph_prior"]["filename"], sep="\t")
    full = freeze_hypergraph_prior(topology, view_columns=VIEW_COLUMNS)
    permuted = permute_hypergraph_prior_degree_matched(full, seed=8675309)
    priors = {
        FULL_METHOD: full,
        PERMUTED_METHOD: permuted,
        "ligand_only_hypergraph_prior": select_hypergraph_prior_views(
            full, view_names=("ligand",)
        ),
        "receptor_only_hypergraph_prior": select_hypergraph_prior_views(
            full, view_names=("receptor",)
        ),
        "pathway_only_hypergraph_prior": select_hypergraph_prior_views(
            full, view_names=("pathway",)
        ),
    }
    record = manifest["records"][0]
    truth = pd.read_csv(fixture / manifest["truth"]["filename"], sep="\t").drop(
        columns=["dataset_id", "root_seed"]
    )
    estimates = pd.read_csv(fixture / record["input"], sep="\t")
    metrics, _, fits = _evaluate_dataset(
        dataset_id=str(record["dataset_id"]),
        root_seed=int(record["root_seed"]),
        estimates=estimates,
        truth=truth,
        priors=priors,
        spec=HypergraphShrinkageSpec(node_ridge_penalty=1.0, edge_residual_penalty=1.0),
    )
    summary = _method_summary(metrics).set_index("method")

    assert (
        summary.loc[FULL_METHOD, "effect_mse"] < summary.loc[RAW_METHOD, "effect_mse"]
    )
    assert (
        summary.loc[FULL_METHOD, "effect_mse"]
        < summary.loc[PERMUTED_METHOD, "effect_mse"]
    )
    assert (
        summary.loc[FULL_METHOD, "average_precision"]
        > summary.loc[PERMUTED_METHOD, "average_precision"]
    )
    assert fits["converged"].all()
    assert full.degree_profile() == permuted.degree_profile()


def test_m5_dataset_seed_is_stable_and_seed_specific() -> None:
    assert _dataset_seed(11) == _dataset_seed(11)
    assert _dataset_seed(11) != _dataset_seed(12)
