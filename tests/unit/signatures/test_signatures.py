from __future__ import annotations

import numpy as np
import pandas as pd

from crychic.attribution import attribute_target_prior
from crychic.response import ResponseStatus
from crychic.signatures import (
    DirectionAgreement,
    SignatureStatus,
    build_signature_table,
)


def test_gene_reconstruction_and_driver_contribution_sum(
    prior_factory, response_factory
) -> None:
    response = response_factory([1.0, 2.0])
    prior = prior_factory({"D1": {"G1": 1.0}, "D2": {"G2": 1.0}})
    basis, attribution = attribute_target_prior(
        prior,
        response.feature_ids,
        {"D1": 1.0, "D2": 1.0},
        np.asarray([1.0, 2.0]),
        tolerance=1e-12,
        kkt_tolerance=1e-12,
    )
    context = (("condition", "treated"),)
    table = build_signature_table(
        response,
        attribution,
        basis,
        context=context,
        receiver="R",
        contrast="treated-v-control",
        gene_namespace="HGNC symbol",
        fold_id="fold-1",
    )

    np.testing.assert_allclose(
        table.genes["predicted"] + table.genes["residual"],
        table.genes["observed"],
    )
    contribution_sum = table.contributions.groupby("gene")["contribution"].sum()
    predicted = table.genes.set_index("gene")["predicted"]
    pd.testing.assert_series_equal(
        contribution_sum.sort_index(), predicted.sort_index(), check_names=False
    )
    assert table.genes["signature_id"].is_unique
    assert table.contributions["contribution_id"].is_unique
    assert not {"p", "p_value", "q", "q_value"}.intersection(table.genes)
    assert not table.inference_eligible

    queried = table.query_genes(
        context=context, receiver="R", contrast="treated-v-control"
    )
    assert len(queried) == 2
    assert len(table.query_contributions(driver="D1")) == 1
    queried.loc[0, "observed"] = 999.0
    assert table.genes.loc[0, "observed"] != 999.0


def test_negative_observed_response_is_unexplained_and_direction_filtered(
    prior_factory, response_factory
) -> None:
    response = response_factory([-1.0, 2.0])
    prior = prior_factory({"L": {"G1": 1.0, "G2": 1.0}})
    basis, attribution = attribute_target_prior(
        prior,
        response.feature_ids,
        {"L": 1.0},
        np.asarray([-1.0, 2.0]),
        lambda2=0.1,
        tolerance=1e-12,
        kkt_tolerance=1e-10,
    )
    table = build_signature_table(
        response,
        attribution,
        basis,
        context="treated",
        receiver="R",
        contrast="treated-v-control",
        gene_namespace="HGNC symbol",
    )

    negative = table.genes.set_index("gene").loc["G1"]
    assert negative["predicted"] > 0
    assert negative["residual"] < negative["observed"]
    assert negative["direction_consistent_predicted"] == 0.0
    assert negative["direction_agreement"] == DirectionAgreement.DISCORDANT.value
    negative_contribution = table.contributions.loc[
        table.contributions["gene"] == "G1"
    ].iloc[0]
    assert negative_contribution["contribution"] > 0
    assert negative_contribution["direction_consistent_contribution"] == 0.0
    np.testing.assert_allclose(
        table.genes["predicted"] + table.genes["residual"],
        table.genes["observed"],
    )


def test_empty_attribution_keeps_all_observed_signal_as_residual(
    response_factory,
) -> None:
    response = response_factory([-2.0, 3.0])
    table = build_signature_table(
        response,
        None,
        None,
        context="treated",
        receiver="R",
        contrast="treated-v-control",
        gene_namespace="HGNC symbol",
    )

    assert table.contributions.empty
    assert set(table.genes["status"]) == {SignatureStatus.EMPTY_ATTRIBUTION.value}
    np.testing.assert_allclose(table.genes["predicted"], 0.0)
    np.testing.assert_allclose(table.genes["residual"], table.genes["observed"])
    assert table.genes.set_index("gene").loc["G1", "residual"] == -2.0


def test_failed_attribution_does_not_emit_modeled_signature(
    prior_factory, response_factory
) -> None:
    response = response_factory([2.0])
    prior = prior_factory({"L": {"G1": 1.0}})
    basis, attribution = attribute_target_prior(
        prior,
        response.feature_ids,
        {"L": 1.0},
        np.asarray([2.0]),
        tolerance=1e-15,
        max_iterations=1,
    )
    assert not attribution.succeeded

    table = build_signature_table(
        response,
        attribution,
        basis,
        context="treated",
        receiver="R",
        contrast="treated-v-control",
        gene_namespace="HGNC symbol",
    )

    row = table.genes.iloc[0]
    assert row["observed"] == 2.0
    assert np.isnan(row["predicted"])
    assert np.isnan(row["residual"])
    assert row["status"] == SignatureStatus.ATTRIBUTION_FAILED.value
    assert row["reason_code"] == "coordinate_or_kkt_tolerance_not_met"
    assert table.contributions.empty


def test_non_estimable_response_retains_reason_without_fake_zero(
    response_factory,
) -> None:
    response = response_factory(
        [np.nan],
        statuses=[ResponseStatus.NOT_ESTIMABLE.value],
        reasons=["insufficient_subject_support"],
    )
    table = build_signature_table(
        response,
        None,
        None,
        context="treated",
        receiver="R",
        contrast="treated-v-control",
        gene_namespace="HGNC symbol",
    )

    row = table.genes.iloc[0]
    assert row["status"] == SignatureStatus.RESPONSE_NOT_ESTIMABLE.value
    assert row["reason_code"] == "insufficient_subject_support"
    assert np.isnan(row["observed"])
    assert np.isnan(row["predicted"])
    assert np.isnan(row["residual"])
    assert table.contributions.empty


def test_signature_stable_keys_and_ranks_are_deterministic(
    prior_factory, response_factory
) -> None:
    response = response_factory([1.0, 3.0])
    prior = prior_factory({"D": {"G1": 1.0, "G2": 1.0}})
    basis, attribution = attribute_target_prior(
        prior,
        response.feature_ids,
        {"D": 1.0},
        np.asarray([1.0, 3.0]),
    )
    kwargs = {
        "context": "treated",
        "receiver": "R",
        "contrast": "treated-v-control",
        "gene_namespace": "HGNC symbol",
    }
    first = build_signature_table(response, attribution, basis, **kwargs)
    second = build_signature_table(response, attribution, basis, **kwargs)

    assert first.genes["signature_id"].tolist() == second.genes[
        "signature_id"
    ].tolist()
    assert first.genes["observed_rank"].tolist() == [2, 1]
    assert first.contributions["contribution_rank"].tolist() == second.contributions[
        "contribution_rank"
    ].tolist()
