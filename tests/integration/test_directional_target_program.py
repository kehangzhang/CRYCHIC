from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.crychic import (
    adapt_directional_target_program_scores_to_signed_track_b,
)
from benchmarks.metrics.signed_track_b import (
    evaluate_signed_track_b,
    signed_track_b_truth_sha256,
)
from tests.integration.test_subject_crossfit import (
    _bundle,
    _config,
    _directional_spec,
    _independent_adata,
    _prior,
)

from crychic import (
    DirectionalTargetProgramScoreCollection,
    FrozenDirectionalTargetProgramUniverse,
    OOFEffectSpec,
    fit_oof_context_effect,
    freeze_crossfit_directional_target_program_universe,
    freeze_directional_target_program_universe,
    run_subject_crossfit,
    score_crossfit_directional_target_programs,
)
from crychic.core import ContractError
from crychic.design import node_context_fields
from crychic.resources import (
    GeneNamespace,
    MappingReport,
    Species,
    TargetPrior,
)
from crychic.workflow import CrossFitArtifacts
from crychic.workflow import directional_target_program as target_program_module


@pytest.fixture(scope="module")
def directional_scores() -> tuple[
    CrossFitArtifacts,
    FrozenDirectionalTargetProgramUniverse,
    DirectionalTargetProgramScoreCollection,
]:
    artifacts = run_subject_crossfit(
        _independent_adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )
    universe = freeze_crossfit_directional_target_program_universe(artifacts)
    collection = score_crossfit_directional_target_programs(
        artifacts,
        universe,
        pair_spec_id=artifacts.spec.directional_pairs[0].pair_spec_id,
    )
    return artifacts, universe, collection


def test_common_scale_scores_are_complete_source_agnostic_and_directional(
    directional_scores: tuple[
        CrossFitArtifacts,
        FrozenDirectionalTargetProgramUniverse,
        DirectionalTargetProgramScoreCollection,
    ],
) -> None:
    artifacts, universe, collection = directional_scores
    table = collection.scores

    assert universe.program_ids == _prior().driver_ids
    assert universe.selection_policy == "all_target_prior_drivers_prior_only_v1"
    assert collection.receiver_ids == ("Receiver", "Sender")
    assert len(table) == 2 * 2 * 2
    assert set(table["analysis_track"]) == {"signed_target_program"}
    assert set(table["sender"]) == {"__source_agnostic__"}
    assert set(table["status"]) == {"observed"}
    assert set(table["score_direction"]) == {"higher"}
    assert not {
        "p_value",
        "q_value",
        "posterior_probability",
        "communication_probability",
    }.intersection(table.columns)
    assert collection.supports_active_inhibition_claim is False
    assert collection.formal_inference_allowed is False
    assert all(effect.status == "observed" for effect in collection.effects)
    assert all(
        len(effect.fold_ids) == len(artifacts.folds)
        and effect.n_subjects == 8
        and effect.value_scale == "log1p_cpm"
        for effect in collection.effects
    )

    receiver_effects = collection.gene_effects["Receiver"]
    assert receiver_effects.loc["T1"] > 0
    assert receiver_effects.loc["T2"] > 0
    receiver_scores = table.loc[table["receiver"].eq("Receiver")]
    by_channel = receiver_scores.groupby("channel")["score"].mean()
    assert by_channel["increased_activation_compatible"] > 0
    assert by_channel["reduced_activation_compatible"] == 0


@pytest.mark.parametrize("feature_id", ("T1", "T2"))
def test_vectorized_effect_matches_existing_scalar_oof_wls(
    directional_scores: tuple[
        CrossFitArtifacts,
        FrozenDirectionalTargetProgramUniverse,
        DirectionalTargetProgramScoreCollection,
    ],
    feature_id: str,
) -> None:
    artifacts, _, collection = directional_scores
    pair = artifacts.spec.directional_pairs[0]
    rows: list[dict[str, object]] = []
    contrast_weights: tuple[tuple[str, float], ...] | None = None
    for fold in artifacts.folds:
        response = next(
            item
            for item in fold.receiver_responses
            if item.receiver == "Receiver"
            and item.contrast_name == pair.forward_contrast.name
        )
        application = next(
            item
            for item in fold.receiver_response_applications
            if item.training_response_id == response.artifact_id
        )
        context_by_node = {
            node: node_context_fields(node, response.context_keys)[0]
            for node in pair.forward_contrast.weights
        }
        fold_weights = tuple(
            sorted(
                (
                    context_by_node[node],
                    float(weight),
                )
                for node, weight in pair.forward_contrast.weights.items()
            )
        )
        if contrast_weights is None:
            contrast_weights = fold_weights
        else:
            assert contrast_weights == fold_weights
        feature_index = application.feature_ids.index(feature_id)
        selected_contexts = {context for context, _ in fold_weights}
        for sample_id, subject_id, context_id, value in zip(
            application.sample_ids,
            application.sample_subject_ids,
            application.sample_context_ids,
            application.sample_values[:, feature_index],
            strict=True,
        ):
            if context_id not in selected_contexts:
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "sample_id": sample_id,
                    "fold_id": fold.fold_id,
                    "context_id": context_id,
                    "score": float(value),
                    "score_status": "observed",
                    "scoring_function_id": f"raw-common-scale:{fold.fold_id}",
                }
            )
    assert contrast_weights is not None
    scalar = fit_oof_context_effect(
        pd.DataFrame(rows),
        OOFEffectSpec(
            hypothesis_id=f"gene:{feature_id}",
            contrast_name=pair.forward_contrast.name,
            contrast_weights=contrast_weights,
            minimum_clusters_for_diagnostic_se=2,
        ),
    )

    assert scalar.effect_status == "observed"
    assert scalar.effect is not None
    assert collection.gene_effects.loc[feature_id, "Receiver"] == pytest.approx(
        scalar.effect,
        abs=2.0e-14,
    )


def _prior_with_unmatched_program() -> TargetPrior:
    return TargetPrior(
        resource_id="directional-program-prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="interaction",
        target_ids=("NOPE", "T1", "T2"),
        driver_ids=("i1", "i2", "i3"),
        indptr=(0, 1, 2, 3),
        target_indices=(1, 2, 0),
        weights=(1.0, 1.0, 1.0),
        ranks=None,
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(3, 3, 3),
        manifest_digest="e" * 64,
    )


def test_complete_program_universe_retains_unmatched_program_as_typed_ne() -> None:
    prior = _prior_with_unmatched_program()
    artifacts = run_subject_crossfit(
        _independent_adata(),
        _config(),
        _bundle(),
        prior,
        spec=_directional_spec(),
    )
    features = artifacts.folds[0].receiver_responses[0].feature_ids
    first = freeze_directional_target_program_universe(prior, features)
    repeated = freeze_directional_target_program_universe(prior, features)
    collection = score_crossfit_directional_target_programs(
        artifacts,
        first,
        pair_spec_id=artifacts.spec.directional_pairs[0].pair_spec_id,
    )

    assert first.universe_id == repeated.universe_id
    assert first.program_ids == ("i1", "i2", "i3")
    assert first.matched_target_counts == (1, 1, 0)
    unmatched = collection.scores.loc[lambda table: table["program_id"].eq("i3")]
    assert len(unmatched) == 4
    assert set(unmatched["status"]) == {"not_estimable"}
    assert unmatched["score"].isna().all()
    assert set(unmatched["reason_code"]) == {
        "directional_target_program_insufficient_matched_targets"
    }
    matched = collection.scores.loc[lambda table: table["program_id"].ne("i3")]
    assert set(matched["status"]) == {"observed"}


def test_training_absent_receiver_is_complete_ne_not_partial_oof() -> None:
    adata = _independent_adata()
    rare = adata.obs["subject_id"].eq("control-0") & adata.obs[
        "cell_type"
    ].eq("Sender")
    adata.obs.loc[rare, "cell_type"] = "Novel"
    artifacts = run_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )
    universe = freeze_crossfit_directional_target_program_universe(artifacts)
    collection = score_crossfit_directional_target_programs(
        artifacts,
        universe,
        pair_spec_id=artifacts.spec.directional_pairs[0].pair_spec_id,
    )

    assert collection.receiver_ids == artifacts.receiver_universe.receiver_ids
    effect = next(item for item in collection.effects if item.receiver == "Novel")
    supports = tuple(
        next(
            record
            for record in fold.receiver_training_support
            if record.receiver_id == "Novel"
        )
        for fold in sorted(artifacts.folds, key=lambda item: item.fold_id)
    )
    observed_supports = tuple(
        record for record in supports if record.status == "observed"
    )
    assert observed_supports
    assert any(record.status == "not_estimable" for record in supports)
    assert effect.status == "not_estimable"
    assert effect.reason_code == "receiver_absent_in_outer_training"
    assert effect.fold_ids == tuple(record.outer_fold_id for record in supports)
    assert effect.receiver_training_support_ids == tuple(
        record.support_record_id for record in supports
    )
    assert effect.response_application_fold_ids == tuple(
        record.outer_fold_id for record in observed_supports
    )
    assert len(effect.forward_response_application_ids) == len(observed_supports)
    assert len(effect.reverse_response_application_ids) == len(observed_supports)
    actual_application_ids = {
        application.application_id
        for fold in artifacts.folds
        for application in fold.receiver_response_applications
        if application.receiver == "Novel"
    }
    assert set(effect.forward_response_application_ids).issubset(
        actual_application_ids
    )
    assert set(effect.reverse_response_application_ids).issubset(
        actual_application_ids
    )
    rows = collection.scores.loc[lambda table: table["receiver"].eq("Novel")]
    assert len(rows) == 2 * len(universe.program_ids)
    assert set(rows["status"]) == {"not_estimable"}
    assert set(rows["reason_code"]) == {"receiver_absent_in_outer_training"}
    assert rows["score"].isna().all()


def test_predeclared_all_absent_receiver_retains_axis_without_fake_parents() -> None:
    spec = replace(
        _directional_spec(),
        predeclared_receiver_ids=("Ghost", "Receiver", "Sender"),
    )
    artifacts = run_subject_crossfit(
        _independent_adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )
    universe = freeze_crossfit_directional_target_program_universe(artifacts)
    collection = score_crossfit_directional_target_programs(
        artifacts,
        universe,
        pair_spec_id=artifacts.spec.directional_pairs[0].pair_spec_id,
    )

    ghost = next(item for item in collection.effects if item.receiver == "Ghost")
    assert collection.receiver_ids == ("Ghost", "Receiver", "Sender")
    assert ghost.status == "not_estimable"
    assert ghost.reason_code == "receiver_absent_in_outer_training"
    assert ghost.n_folds == len(artifacts.folds)
    assert ghost.n_response_application_folds == 0
    assert ghost.response_application_fold_ids == ()
    assert ghost.forward_response_application_ids == ()
    assert ghost.reverse_response_application_ids == ()
    assert len(ghost.receiver_training_support_ids) == len(artifacts.folds)
    rows = collection.scores.loc[lambda table: table["receiver"].eq("Ghost")]
    assert len(rows) == 2 * len(universe.program_ids)
    assert set(rows["reason_code"]) == {"receiver_absent_in_outer_training"}


def test_program_universe_must_match_exact_response_feature_order(
    directional_scores: tuple[
        CrossFitArtifacts,
        FrozenDirectionalTargetProgramUniverse,
        DirectionalTargetProgramScoreCollection,
    ],
) -> None:
    artifacts, universe, _ = directional_scores
    misaligned = freeze_directional_target_program_universe(
        _prior(), tuple(reversed(universe.feature_ids))
    )

    with pytest.raises(ContractError) as error:
        score_crossfit_directional_target_programs(
            artifacts,
            misaligned,
            pair_spec_id=artifacts.spec.directional_pairs[0].pair_spec_id,
        )
    assert error.value.details.code == "directional_target_program_feature_mismatch"


def test_whole_missing_oof_fold_is_typed_ne_not_silently_dropped(
    directional_scores: tuple[
        CrossFitArtifacts,
        FrozenDirectionalTargetProgramUniverse,
        DirectionalTargetProgramScoreCollection,
    ],
) -> None:
    artifacts, universe, _ = directional_scores
    pair = artifacts.spec.directional_pairs[0]
    registry = target_program_module._raw_views_by_receiver(
        artifacts, pair, universe
    )["Receiver"]
    missing_view = replace(
        registry.views[0],
        sample_ids=(),
        sample_subject_ids=(),
        sample_context_ids=(),
        values=np.empty((0, len(universe.feature_ids)), dtype=float),
    )
    incomplete = replace(
        registry,
        views=(missing_view, *registry.views[1:]),
    )
    spec = target_program_module.DirectionalTargetProgramScoreSpec()

    effect = target_program_module._fit_receiver_effect(
        "Receiver",
        universe.feature_ids,
        pair.pair_spec_id,
        artifacts.receiver_universe.universe_id,
        artifacts.receiver_universe.receiver_axis_id,
        incomplete,
        spec,
    )

    assert effect.status == "not_estimable"
    assert effect.reason_code == (
        "directional_target_program_fold_context_support_incomplete"
    )


def test_producer_has_no_truth_input_and_rejects_identity_tampering(
    directional_scores: tuple[
        CrossFitArtifacts,
        FrozenDirectionalTargetProgramUniverse,
        DirectionalTargetProgramScoreCollection,
    ],
) -> None:
    artifacts, universe, _ = directional_scores
    assert (
        "truth"
        not in inspect.signature(score_crossfit_directional_target_programs).parameters
    )
    collection = score_crossfit_directional_target_programs(
        artifacts,
        universe,
        pair_spec_id=artifacts.spec.directional_pairs[0].pair_spec_id,
    )
    object.__setattr__(collection, "score_method", "forged")

    with pytest.raises(ContractError) as error:
        _ = collection.scores
    assert error.value.details.code == (
        "directional_target_program_collection_integrity_violation"
    )


def test_universe_rejects_profile_lineage_tampering() -> None:
    universe = freeze_directional_target_program_universe(
        _prior(),
        ("L1", "R1", "L2", "R2", "T1", "T2"),
    )
    object.__setattr__(universe, "normalized_profile_digest", "0" * 64)

    with pytest.raises(ContractError) as error:
        universe.to_dict()
    assert error.value.details.code == (
        "directional_target_program_universe_integrity_violation"
    )


def test_observed_scores_are_finite_unit_interval(
    directional_scores: tuple[
        CrossFitArtifacts,
        FrozenDirectionalTargetProgramUniverse,
        DirectionalTargetProgramScoreCollection,
    ],
) -> None:
    _, _, collection = directional_scores
    observed = collection.scores.loc[lambda table: table["status"].eq("observed")]
    numeric = observed["score"].to_numpy(dtype=float)
    assert np.isfinite(numeric).all()
    assert ((numeric >= 0.0) & (numeric <= 1.0)).all()


def test_native_collection_feeds_strict_signed_track_b_evaluator(
    directional_scores: tuple[
        CrossFitArtifacts,
        FrozenDirectionalTargetProgramUniverse,
        DirectionalTargetProgramScoreCollection,
    ],
) -> None:
    _, _, collection = directional_scores
    truth = pd.DataFrame(
        [
            {
                "truth_set_id": "directional-integration-truth-v1",
                "truth_scope": "simulation_target_program_truth",
                "scenario_cell_id": "independent-stim-vs-control",
                "receiver": receiver,
                "program_id": program_id,
                "truth_direction": (
                    "increased_activation_compatible"
                    if receiver == "Receiver"
                    else "reduced_activation_compatible"
                ),
            }
            for receiver in collection.receiver_ids
            for program_id in collection.program_ids
        ]
    )
    predictions = adapt_directional_target_program_scores_to_signed_track_b(
        collection,
        method_version="0.1.dev",
        evaluation_phase="development",
        truth_set_id="directional-integration-truth-v1",
        truth_sha256=signed_track_b_truth_sha256(truth),
        seed=19,
        scenario_cell_id="independent-stim-vs-control",
    )
    native = collection.scores.sort_values(
        ["receiver", "program_id", "channel"],
        kind="stable",
        ignore_index=True,
    )
    pd.testing.assert_series_equal(
        predictions["score"],
        native["score"],
        check_names=False,
    )
    assert tuple(predictions["effect_result_id"]) == tuple(
        native["effect_result_id"]
    )
    evaluated = evaluate_signed_track_b(
        predictions,
        truth,
        expected_seeds=(19,),
    )

    assert len(predictions) == len(collection.scores)
    assert set(predictions["method"]) == {"crychic"}
    assert set(predictions["collection_id"]) == {collection.collection_id}
    assert set(predictions["receiver_universe_id"]) == {
        collection.receiver_universe_id
    }
    assert set(predictions["receiver_axis_id"]) == {collection.receiver_axis_id}
    assert set(predictions["analysis_track"]) == {"signed_target_program"}
    assert set(predictions["sender"]) == {"__source_agnostic__"}
    assert not predictions[
        [
            "supports_active_inhibition_claim",
            "formal_inference_allowed",
            "native_nichenet_claim",
            "sender_claim",
            "lr_edge_claim",
        ]
    ].any(axis=None)
    assert evaluated.primary.iloc[0]["status"] == "observed"
    assert np.isfinite(float(evaluated.primary.iloc[0]["estimate"]))
