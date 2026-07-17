from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from crychic.attribution import (
    GraphPenaltyTuningSpec,
    GraphTuningFold,
    build_family_first_basis,
    build_gated_target_basis,
    cluster_driver_families,
    fit_tuned_graph_fused_family_attribution,
    tune_graph_fused_penalties,
)
from crychic.core import ContractError
from crychic.design import ContextGraph


def _fold(
    fold_id: str,
    *,
    training_subjects: tuple[str, ...],
    validation_subjects: tuple[str, ...],
    validation_values: tuple[float, float],
    reverse_mappings: bool = False,
) -> GraphTuningFold:
    graph = ContextGraph.chain(("reference", "treated"))
    nodes = tuple(graph.nodes)
    matrices = {node: sparse.csc_matrix([[1.0]]) for node in nodes}
    training_responses = {
        "reference": np.asarray([1.0]),
        "treated": np.asarray([3.0]),
    }
    training_precision = {node: np.ones(1) for node in nodes}
    validation_responses = {
        subject: {
            "reference": np.asarray([validation_values[0]]),
            "treated": np.asarray([validation_values[1]]),
        }
        for subject in validation_subjects
    }
    validation_precision = {
        subject: {node: np.ones(1) for node in nodes}
        for subject in validation_subjects
    }
    if reverse_mappings:
        matrices = dict(reversed(tuple(matrices.items())))
        training_responses = dict(reversed(tuple(training_responses.items())))
        training_precision = dict(reversed(tuple(training_precision.items())))
        validation_responses = dict(reversed(tuple(validation_responses.items())))
        validation_precision = dict(reversed(tuple(validation_precision.items())))
    return GraphTuningFold.from_mappings(
        fold_id=fold_id,
        graph=graph,
        feature_ids=("target",),
        family_ids=("family",),
        training_subject_ids=training_subjects,
        matrices=matrices,
        training_responses=training_responses,
        training_precision_weights=training_precision,
        validation_responses=validation_responses,
        validation_precision_weights=validation_precision,
    )


def _folds(
    validation_values: tuple[float, float], *, reverse_second: bool = False
) -> tuple[GraphTuningFold, GraphTuningFold]:
    return (
        _fold(
            "inner-1",
            training_subjects=("s3", "s4"),
            validation_subjects=("s1", "s2"),
            validation_values=validation_values,
        ),
        _fold(
            "inner-2",
            training_subjects=("s1", "s2"),
            validation_subjects=("s3", "s4"),
            validation_values=validation_values,
            reverse_mappings=reverse_second,
        ),
    )


def _spec() -> GraphPenaltyTuningSpec:
    return GraphPenaltyTuningSpec(lambda_f_values=(0.0, 10.0))


def test_subject_blocked_tuning_selects_fusion_for_smooth_validation() -> None:
    result = tune_graph_fused_penalties(_folds((2.0, 2.0)), _spec())

    assert result.selected_candidate.lambda_f == 10.0
    assert result.best_candidate_id == result.selected_candidate_id
    assert result.training_subject_ids == ("s1", "s2", "s3", "s4")
    assert result.validation_subject_ids == ("s1", "s2", "s3", "s4")
    np.testing.assert_array_equal(result.selection_frequency, [[1.0], [1.0]])
    assert not result.selection_frequency.flags.writeable
    assert result.to_dict()["formal_inference_allowed"] is False


def test_validation_jump_prevents_oversmoothing_selection() -> None:
    result = tune_graph_fused_penalties(_folds((1.0, 3.0)), _spec())

    assert result.selected_candidate.lambda_f == 0.0
    summary_by_candidate = {
        summary.candidate_id: summary for summary in result.summaries
    }
    fused = next(
        candidate for candidate in result.spec.candidates if candidate.lambda_f > 0
    )
    assert not summary_by_candidate[fused.candidate_id].within_one_se


def test_mapping_order_does_not_change_frozen_fold_or_tuning_identity() -> None:
    ordinary = _folds((2.0, 2.0))
    reordered = _folds((2.0, 2.0), reverse_second=True)

    first = tune_graph_fused_penalties(ordinary, _spec())
    second = tune_graph_fused_penalties(reordered, _spec())

    assert ordinary[1].fold_data_id == reordered[1].fold_data_id
    assert first.tuning_id == second.tuning_id
    np.testing.assert_array_equal(first.selection_frequency, second.selection_frequency)


def test_fold_rejects_subject_overlap() -> None:
    with pytest.raises(ContractError) as error:
        _fold(
            "leaking",
            training_subjects=("s1", "s2"),
            validation_subjects=("s1",),
            validation_values=(2.0, 2.0),
        )

    assert error.value.details.code == "graph_tuning_subject_leakage"


def test_tuning_rejects_reused_validation_subject() -> None:
    first, _ = _folds((2.0, 2.0))
    reused = _fold(
        "inner-2",
        training_subjects=("s3", "s4"),
        validation_subjects=("s1", "s2"),
        validation_values=(2.0, 2.0),
    )

    with pytest.raises(ContractError) as error:
        tune_graph_fused_penalties((first, reused), _spec())

    assert error.value.details.code == "graph_tuning_validation_subject_reused"


def test_tuning_rejects_nonpartitioned_outer_subject_sets() -> None:
    first, _ = _folds((2.0, 2.0))
    inconsistent = _fold(
        "inner-2",
        training_subjects=("s1", "s5"),
        validation_subjects=("s3", "s4"),
        validation_values=(2.0, 2.0),
    )

    with pytest.raises(ContractError) as error:
        tune_graph_fused_penalties((first, inconsistent), _spec())

    assert error.value.details.code == "graph_tuning_partition_mismatch"


def test_no_converged_candidate_fails_closed() -> None:
    with pytest.raises(ContractError) as error:
        tune_graph_fused_penalties(
            _folds((2.0, 2.0)),
            _spec(),
            max_iterations=1,
        )

    assert error.value.details.code == "graph_tuning_no_estimable_candidate"


def test_tampered_fold_identity_is_rejected_before_solver_use() -> None:
    folds = _folds((2.0, 2.0))
    object.__setattr__(folds[0], "fold_data_id", "graph_tuning_fold_forged")

    with pytest.raises(ContractError) as error:
        tune_graph_fused_penalties(folds, _spec())

    assert error.value.details.code == "graph_tuning_fold_integrity_violation"


def test_candidate_grid_is_canonical_and_explicitly_absolute() -> None:
    first = GraphPenaltyTuningSpec(
        lambda1_values=(0.0, 0.1, 0.0),
        lambda2_values=(0.2, 0.0),
        lambda_f_values=(1.0, 0.0),
    )
    second = GraphPenaltyTuningSpec(
        lambda1_values=(0.1, 0.0),
        lambda2_values=(0.0, 0.2),
        lambda_f_values=(0.0, 1.0),
    )

    assert first.spec_id == second.spec_id
    assert first.candidates == second.candidates
    assert first.to_dict()["penalty_scale"] == (
        "absolute_on_normalized_family_basis_v1"
    )


def test_tuned_family_fit_refits_selected_penalty_on_full_outer_problem(
    prior_factory,
) -> None:
    prior = prior_factory({"A": {"target": 1.0}})
    source = build_gated_target_basis(prior, ("target",), {"A": 1.0})
    families = cluster_driver_families(source, cosine_threshold=0.99)
    basis = build_family_first_basis(
        source, families, strict_cosine_threshold=0.99
    )
    graph = ContextGraph.chain(("reference", "treated"))
    matrices = {node: basis.matrix for node in graph.nodes}
    training_responses = {
        "reference": np.asarray([1.0]),
        "treated": np.asarray([3.0]),
    }
    precision = {node: np.ones(1) for node in graph.nodes}

    def make_fold(
        fold_id: str,
        training: tuple[str, ...],
        validation: tuple[str, ...],
    ) -> GraphTuningFold:
        return GraphTuningFold.from_mappings(
            fold_id=fold_id,
            graph=graph,
            feature_ids=basis.feature_ids,
            family_ids=basis.family_ids,
            training_subject_ids=training,
            matrices=matrices,
            training_responses=training_responses,
            training_precision_weights=precision,
            validation_responses={
                subject: {
                    "reference": np.asarray([2.0]),
                    "treated": np.asarray([2.0]),
                }
                for subject in validation
            },
            validation_precision_weights={
                subject: {node: np.ones(1) for node in graph.nodes}
                for subject in validation
            },
        )

    result = fit_tuned_graph_fused_family_attribution(
        {node: basis for node in graph.nodes},
        training_responses,
        graph,
        (
            make_fold("inner-1", ("s3", "s4"), ("s1", "s2")),
            make_fold("inner-2", ("s1", "s2"), ("s3", "s4")),
        ),
        _spec(),
        precision_weights=precision,
    )

    assert result.tuning.selected_candidate.lambda_f == 10.0
    assert result.attribution.lambda_f == 10.0
    np.testing.assert_allclose(result.attribution.coefficients, 2.0, atol=2e-4)
    assert result.to_dict()["formal_inference_allowed"] is False
