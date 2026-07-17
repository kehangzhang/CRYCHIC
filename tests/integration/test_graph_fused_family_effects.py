from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.integration.test_graph_fused_crossfit_registry import _run_registry
from tests.integration.test_subject_crossfit import (
    _adata,
    _bundle,
    _config,
    _prior,
    _spec,
)

from crychic.core import ContractError
from crychic.workflow import (
    CrossFitArtifacts,
    GraphFusedCrossFitRegistry,
    derive_graph_fused_family_effects,
    run_subject_crossfit,
)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def observed_sources() -> tuple[CrossFitArtifacts, GraphFusedCrossFitRegistry]:
    adata = _adata(tuple(f"p{index}" for index in range(1, 9)))
    crossfit = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
    )
    return crossfit, _run_registry(adata, crossfit=crossfit)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def ne_sources() -> tuple[CrossFitArtifacts, GraphFusedCrossFitRegistry]:
    adata = _adata()
    crossfit = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
    )
    return crossfit, _run_registry(adata, crossfit=crossfit)


def _outer_parent(crossfit: CrossFitArtifacts, fold_id: str, receiver: str) -> Any:
    fold = next(item for item in crossfit.folds if item.fold_id == fold_id)
    return next(
        model.receiver_family_artifact
        for model in fold.receiver_family_models
        if model.receiver_family_artifact.receiver == receiver
    )


def test_observed_family_effects_recompute_exact_heldout_ablation(
    observed_sources: tuple[CrossFitArtifacts, GraphFusedCrossFitRegistry],
) -> None:
    crossfit, registry = observed_sources
    collection = derive_graph_fused_family_effects(crossfit, registry)
    table = collection.family_effects

    expected_rows = sum(
        len(record.heldout_subject_ids)
        * len(record.application.context_ids)  # type: ignore[union-attr]
        * len(record.application.family_ids)  # type: ignore[union-attr]
        for record in registry.records
    )
    assert len(table) == expected_rows
    assert not table.duplicated(
        ["fold_id", "receiver", "subject_id", "context_id", "family_id"]
    ).any()
    assert set(table["status"]).issubset({"observed", "structural_zero"})
    assert set(table["semantics"]) == {"graph_joint_conditional_family_gain_v1"}
    assert not table["formal_inference_allowed"].any()
    assert (
        table[
            [
                "coefficient_digest",
                "fixed_precision_digest",
                "heldout_response_digest",
                "context_loss_digest",
            ]
        ]
        .notna()
        .all()
        .all()
    )
    negative = table.loc[table["raw_conditional_gain"].lt(0)]
    if not negative.empty:
        assert set(negative["status"]) == {"observed"}
        assert negative["bounded_conditional_gain"].eq(0.0).all()

    by_record = {record.record_id: record for record in registry.records}
    for row in table.itertuples(index=False):
        record = by_record[row.record_id]
        assert record.workflow is not None and record.application is not None
        assert record.workflow.fit is not None
        problem = record.workflow.problem
        alpha = record.workflow.fit.attribution.coefficients
        subject_index = record.heldout_subject_ids.index(row.subject_id)
        context_index = problem.context_ids.index(row.context_id)
        family_index = problem.family_ids.index(row.family_id)
        basis = problem.family_bases[context_index]
        coefficient = float(alpha[context_index, family_index])
        full_prediction = np.asarray(basis.matrix.dot(alpha[context_index])).ravel()
        contribution = (
            np.asarray(basis.matrix.getcol(family_index).toarray()).ravel()
            * coefficient
        )
        response = record.application.heldout_positive_responses[
            subject_index, context_index
        ]
        precision = record.application.fixed_precision_weights[context_index]
        loss_without = float(
            np.dot(
                precision,
                np.square(response - (full_prediction - contribution)),
            )
            / precision.sum()
        )
        full_loss = float(
            record.application.context_losses[subject_index, context_index]
        )
        raw_gain = (loss_without - full_loss) / (loss_without + 1.0e-12)

        assert row.family_coefficient == pytest.approx(coefficient)
        assert row.full_loss == pytest.approx(full_loss)
        assert row.loss_without == pytest.approx(loss_without)
        if coefficient == 0.0:
            assert row.status == "structural_zero"
            assert row.raw_conditional_gain == 0.0
            assert row.bounded_conditional_gain == 0.0
        else:
            assert row.status == "observed"
            assert row.raw_conditional_gain == pytest.approx(raw_gain)
            assert row.bounded_conditional_gain == pytest.approx(
                np.clip(raw_gain, 0.0, 1.0)
            )


def test_not_estimable_records_expand_the_outer_receiver_family_axis(
    ne_sources: tuple[CrossFitArtifacts, GraphFusedCrossFitRegistry],
) -> None:
    crossfit, registry = ne_sources
    collection = derive_graph_fused_family_effects(crossfit, registry)
    table = collection.family_effects
    numeric = [
        "family_coefficient",
        "full_loss",
        "loss_without",
        "raw_conditional_gain",
        "bounded_conditional_gain",
    ]

    assert set(table["status"]) == {"not_estimable"}
    assert table[numeric].isna().all().all()
    assert (
        table[
            [
                "coefficient_digest",
                "fixed_precision_digest",
                "heldout_response_digest",
                "context_loss_digest",
            ]
        ]
        .isna()
        .all()
        .all()
    )
    assert table["reason_code"].notna().all()
    for record in registry.records:
        parent = _outer_parent(crossfit, record.fold_id, record.receiver)
        selected = table.loc[table["record_id"].eq(record.record_id)]
        expected = pd.MultiIndex.from_product(
            (
                record.heldout_subject_ids,
                tuple(selected["context_id"].drop_duplicates().astype(str)),
                parent.family_basis.family_ids,
            ),
            names=("subject_id", "context_id", "family_id"),
        )
        observed = pd.MultiIndex.from_frame(
            selected.loc[:, ["subject_id", "context_id", "family_id"]]
        )
        assert set(observed) == set(expected)
        assert set(selected["family_basis_id"]) == {parent.family_basis.family_basis_id}
        assert set(selected["receiver_family_training_artifact_id"]) == {
            parent.training_artifact_id
        }


def test_family_effect_collection_rejects_private_table_mutation(
    observed_sources: tuple[CrossFitArtifacts, GraphFusedCrossFitRegistry],
) -> None:
    crossfit, registry = observed_sources
    collection = derive_graph_fused_family_effects(crossfit, registry)
    collection._family_effects.loc[0, "bounded_conditional_gain"] = 0.123

    with pytest.raises(ContractError, match="integrity") as caught:
        _ = collection.family_effects
    assert caught.value.details.code == (
        "graph_fused_family_effect_collection_integrity_violation"
    )
