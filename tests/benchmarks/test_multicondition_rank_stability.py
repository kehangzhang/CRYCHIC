from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import pandas.testing as pdt
from benchmarks.metrics.multicondition_rank_stability import (
    BOOTSTRAP_REPLICATES,
    CONFIDENCE_LEVEL,
    FAMILY_MAX_K,
    LR_MAX_K,
    MINIMUM_ESTIMABLE_REPLICATE_FRACTION,
    MINIMUM_SUBJECTS,
    MINIMUM_TOP_K_FREQUENCY,
    RANDOM_SEED,
    RBO_PERSISTENCE,
    SPLIT_REPEATS,
    WEIGHTED_KENDALL_POWER,
    RankStabilityParameters,
    _all_tied,
    _tie_inclusive_top,
    evaluate_multicondition_rank_stability,
    not_estimable_rank_stability,
)

Edge = tuple[str, str, str, str, str]
EDGES: tuple[Edge, ...] = (
    ("S1", "R1", "I1", "L1", "RCP1"),
    ("S1", "R1", "I2", "L2", "RCP2"),
    ("S1", "R1", "I3", "L3", "RCP3"),
    ("S2", "R2", "I4", "L4", "RCP4"),
    ("S2", "R2", "I5", "L5", "RCP5"),
    ("S2", "R2", "I6", "L6", "RCP6"),
)


def _score_table(
    subject_contexts: Sequence[tuple[str, str]],
    *,
    scores: Callable[[str, str, int], float],
    statuses: Callable[[str, str, int, int], str] | None = None,
    technical_replicates: int = 1,
    edges: Sequence[Edge] = EDGES,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    status_fn = statuses or (lambda _subject, _context, _edge, _replicate: "observed")
    for subject, context in subject_contexts:
        for replicate in range(technical_replicates):
            sample_id = f"{subject}_{context}_rep{replicate + 1}"
            for edge_index, edge in enumerate(edges):
                sender, receiver, interaction, ligand, receptor = edge
                status = status_fn(subject, context, edge_index, replicate)
                rows.append(
                    {
                        "dataset": "fixture",
                        "method": "method_a",
                        "method_version": "1",
                        "analysis_track": "lr_stlr",
                        "resource": "common",
                        "resource_version": "1",
                        "resource_mode": "H-common",
                        "score_semantics": "fixture_score",
                        "universe_id": "fixture-universe",
                        "sample_id": sample_id,
                        "subject_id": subject,
                        "context": context,
                        "contrast": "Target_vs_Reference",
                        "sender": sender,
                        "receiver": receiver,
                        "interaction_id": interaction,
                        "ligand": ligand,
                        "receptor": receptor,
                        "score": (
                            scores(subject, context, edge_index)
                            if status == "observed"
                            else np.nan
                        ),
                        "score_direction": "higher",
                        "status": status,
                        "universe_member": True,
                        "universe_size": len(edges),
                    }
                )
    return pd.DataFrame(rows)


def _paired_subject_contexts(n_subjects: int = 6) -> list[tuple[str, str]]:
    return [
        (f"P{subject}", context)
        for subject in range(1, n_subjects + 1)
        for context in ("Reference", "Target")
    ]


def _ordered_scores(_subject: str, context: str, edge: int) -> float:
    reference = (6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
    target = (6.0, 4.0, 5.0, 1.0, 3.0, 2.0)
    return (reference if context == "Reference" else target)[edge]


def _small_parameters(**overrides: object) -> RankStabilityParameters:
    values: dict[str, object] = {
        "n_bootstrap": 20,
        "n_split_repeats": 10,
        "min_subjects": 3,
        "min_subjects_per_half": 2,
        "min_observed_ranks": 2,
        "minimum_estimable_replicate_fraction": 0.5,
        "random_seed": 31,
    }
    values.update(overrides)
    return RankStabilityParameters(**values)  # type: ignore[arg-type]


def test_preregistered_ranking_parameters_match_protocol() -> None:
    parameters = RankStabilityParameters()

    assert parameters.n_bootstrap == BOOTSTRAP_REPLICATES == 2000
    assert parameters.n_split_repeats == SPLIT_REPEATS == 200
    assert parameters.confidence == CONFIDENCE_LEVEL == 0.95
    assert parameters.minimum_top_k_frequency == MINIMUM_TOP_K_FREQUENCY == 0.8
    assert (
        parameters.minimum_estimable_replicate_fraction
        == MINIMUM_ESTIMABLE_REPLICATE_FRACTION
        == 0.8
    )
    assert parameters.min_subjects == MINIMUM_SUBJECTS == 3
    assert parameters.rbo_persistence == RBO_PERSISTENCE == 0.9
    assert parameters.weighted_kendall_power == WEIGHTED_KENDALL_POWER == 1.0
    assert parameters.random_seed == RANDOM_SEED == 20260712
    assert FAMILY_MAX_K == 25
    assert LR_MAX_K == 100


def test_paired_rank_stability_is_deterministic_and_persists_design() -> None:
    table = _score_table(_paired_subject_contexts(), scores=_ordered_scores)
    parameters = _small_parameters()

    first = evaluate_multicondition_rank_stability(
        table,
        reference="Reference",
        target="Target",
        design="paired",
        parameters=parameters,
    )
    second = evaluate_multicondition_rank_stability(
        table,
        reference="Reference",
        target="Target",
        design="paired",
        parameters=parameters,
    )

    pdt.assert_frame_equal(first.agreement, second.agreement)
    pdt.assert_frame_equal(first.top_k_curve, second.top_k_curve)
    pdt.assert_frame_equal(first.rank_intervals, second.rank_intervals)
    pdt.assert_frame_equal(first.stable_tiers, second.stable_tiers)
    observed = first.agreement[first.agreement["status"].eq("observed")]
    assert not observed.empty
    assert set(observed["interval_type"]) == {"split_repeat_quantile_envelope"}
    assert set(observed["design"]) == {"paired"}
    assert set(observed["reference"]) == {"Reference"}
    assert set(observed["target"]) == {"Target"}
    assert set(observed["n_paired_subjects"]) == {6}
    assert "ci_lower" not in observed
    assert "selection_frequency" not in first.rank_intervals
    assert {"rank_availability_frequency", "top_k_frequency"}.issubset(
        first.rank_intervals
    )


def test_unpaired_groups_are_resampled_independently() -> None:
    contexts = [
        *((f"C{index}", "Reference") for index in range(1, 5)),
        *((f"T{index}", "Target") for index in range(1, 5)),
    ]
    result = evaluate_multicondition_rank_stability(
        _score_table(contexts, scores=_ordered_scores),
        reference="Reference",
        target="Target",
        design="unpaired",
        parameters=_small_parameters(),
    )

    observed = result.agreement[result.agreement["status"].eq("observed")]
    assert not observed.empty
    assert set(observed["design"]) == {"unpaired"}
    assert set(observed["n_reference_subjects"]) == {4}
    assert set(observed["n_target_subjects"]) == {4}
    assert set(observed["n_paired_subjects"]) == {0}


def test_all_tied_effects_make_the_endpoint_ne_without_id_tiebreak() -> None:
    tied = _score_table(
        _paired_subject_contexts(),
        scores=lambda _subject, _context, _edge: 1.0,
    )
    result = evaluate_multicondition_rank_stability(
        tied,
        reference="Reference",
        target="Target",
        design="paired",
        parameters=_small_parameters(),
    )

    comparable = result.agreement[
        result.agreement["ranking_level"].isin(
            {"lr", "sender", "sender_receiver_pair"}
        )
    ]
    assert set(comparable["status"]) == {"not_estimable"}
    assert set(comparable["reason_code"]) == {"all_effects_tied"}
    items = result.rank_intervals[
        result.rank_intervals["ranking_level"].eq("lr")
    ]
    assert set(items["status"]) == {"not_estimable"}
    assert set(items["reason_code"]) == {"all_effects_tied"}
    assert set(result.stable_tiers["tier"]) == {"not_estimable"}


def test_tie_inclusive_top_k_records_realized_boundary() -> None:
    result = _tie_inclusive_top(np.asarray([3.0, 2.0, 2.0, 1.0]), 2)

    assert result is not None
    selected, realized_k, boundary_size = result
    assert selected.tolist() == [True, True, True, False]
    assert realized_k == 3
    assert boundary_size == 2


def test_tie_detection_has_zero_relative_tolerance() -> None:
    values = np.asarray([1.0, 1.000005, 0.0])

    assert not _all_tied(values[:2])
    selected = _tie_inclusive_top(values, 1)
    assert selected is not None
    mask, realized_k, boundary_size = selected
    assert mask.tolist() == [False, True, False]
    assert realized_k == 1
    assert boundary_size == 1


def test_bootstrap_rank_intervals_use_average_ranks_for_tie_blocks() -> None:
    edges = EDGES[:4]

    def scores(_subject: str, context: str, edge: int) -> float:
        reference = (4.0, 3.0, 2.0, 1.0)
        target = (2.0, 1.0, 4.0, 3.0)
        return (reference if context == "Reference" else target)[edge]

    result = evaluate_multicondition_rank_stability(
        _score_table(_paired_subject_contexts(), scores=scores, edges=edges),
        reference="Reference",
        target="Target",
        design="paired",
        parameters=_small_parameters(),
    )
    lr = result.rank_intervals[
        result.rank_intervals["ranking_level"].eq("lr")
        & result.rank_intervals["status"].eq("observed")
    ]

    assert set(lr["median_rank"]) == {1.5, 3.5}
    assert lr["rank_availability_frequency"].eq(1.0).all()


def test_frozen_members_propagate_dynamic_missingness_instead_of_reaveraging() -> None:
    def statuses(_subject: str, context: str, edge: int, _replicate: int) -> str:
        if context == "Target" and edge == 1:
            return "missing"
        return "observed"

    result = evaluate_multicondition_rank_stability(
        _score_table(
            _paired_subject_contexts(),
            scores=_ordered_scores,
            statuses=statuses,
        ),
        reference="Reference",
        target="Target",
        design="paired",
        parameters=_small_parameters(),
    )
    senders = result.rank_intervals[
        result.rank_intervals["ranking_level"].eq("sender")
    ]
    s1 = senders[senders["item_sender"].eq("S1")].iloc[0]

    assert s1["frozen_member_count"] == 3
    assert s1["minimum_observed_member_count"] == 2
    assert s1["minimum_member_coverage_fraction"] == 2 / 3
    assert s1["rank_availability_frequency"] == 0.0
    assert s1["status"] == "not_estimable"


def test_two_subjects_never_produce_observed_intervals_or_tiers() -> None:
    result = evaluate_multicondition_rank_stability(
        _score_table(_paired_subject_contexts(2), scores=_ordered_scores),
        reference="Reference",
        target="Target",
        design="paired",
        parameters=_small_parameters(),
    )

    assert set(result.agreement["status"]) == {"not_estimable"}
    observed_levels = result.rank_intervals[
        result.rank_intervals["ranking_level"].ne("lr_family")
    ]
    assert set(observed_levels["status"]) == {"not_estimable"}
    assert set(result.stable_tiers["tier"]) == {"not_estimable"}
    assert observed_levels["top_k_frequency"].eq(0.0).all()
    assert observed_levels["n_replicates_requested"].eq(20).all()


def test_technical_replicates_do_not_inflate_subjects_and_missing_is_strict() -> None:
    def statuses(subject: str, context: str, edge: int, replicate: int) -> str:
        if subject == "P1" and context == "Target" and edge == 0 and replicate == 1:
            return "missing"
        return "observed"

    result = evaluate_multicondition_rank_stability(
        _score_table(
            _paired_subject_contexts(),
            scores=_ordered_scores,
            statuses=statuses,
            technical_replicates=2,
        ),
        reference="Reference",
        target="Target",
        design="paired",
        parameters=_small_parameters(),
    )

    assert set(result.agreement["n_paired_subjects"]) == {6}
    lr = result.rank_intervals[result.rank_intervals["ranking_level"].eq("lr")]
    i1 = lr[lr["item_interaction_id"].eq("I1")].iloc[0]
    assert i1["complete_subject_context_fraction"] < 1.0
    assert i1["rank_availability_frequency"] < 1.0
    lr_agreement = result.agreement[
        result.agreement["ranking_level"].eq("lr")
    ]
    assert set(lr_agreement["status"]) == {"not_estimable"}
    assert set(lr_agreement["n_repeats_estimable"]) == {0}


def test_track_b_and_unsupported_designs_remain_explicit_ne() -> None:
    tables = not_estimable_rank_stability(
        {
            "dataset": "kuppe",
            "method": "NicheNet",
            "analysis_track": "ligand_target_program",
            "contrast": "injury_vs_control",
        },
        reason_code="track_b_ligand_target_program_not_lr_stlr_comparable",
        parameters=_small_parameters(),
    )

    assert set(tables.agreement["status"]) == {"not_estimable"}
    assert set(tables.agreement["reason_code"]) == {
        "track_b_ligand_target_program_not_lr_stlr_comparable"
    }
    assert set(tables.stable_tiers["tier"]) == {"not_estimable"}
