from __future__ import annotations

import pandas as pd
import pytest

from crychic.core import ContractError
from crychic.scoring.active_edge import (
    ActiveEdgeCandidate,
    ActiveEdgePointStatus,
    FrozenActiveEdgeUniverse,
    freeze_active_edge_universe,
    summarize_subject_equal_active_edge_score,
)


def _candidate(
    *,
    context: str = "treated",
    receiver: str = "Receiver",
    interaction: str = "lr-1",
) -> ActiveEdgeCandidate:
    return ActiveEdgeCandidate(
        contrast_id="treated-vs-control",
        context_id=context,
        sender="Sender",
        receiver=receiver,
        interaction_id=interaction,
        driver_id=f"driver-{interaction}",
        mode="state",
    )


def test_active_edge_universe_derives_receiver_specific_strata() -> None:
    first = _candidate(receiver="R1")
    second = _candidate(receiver="R2")

    universe = freeze_active_edge_universe(
        (second, first),
        universe_name="active-edge-test",
        contrast_id="treated-vs-control",
        score_version="score-v1",
    )

    assert isinstance(universe, FrozenActiveEdgeUniverse)
    assert universe.candidate_edge_ids == tuple(sorted(universe.candidate_edge_ids))
    assert len(set(universe.stratum_ids)) == 2
    assert universe.candidate_for(first.candidate_edge_id) is first
    assert universe.to_dict()["complete_candidate_coverage"] is True
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenActiveEdgeUniverse()


def test_stratum_excludes_sender_and_interaction_but_candidate_identity_does_not() -> (
    None
):
    first = _candidate(interaction="lr-1")
    second = _candidate(interaction="lr-2")

    assert first.candidate_edge_id != second.candidate_edge_id
    assert first.stratum_id(score_version="score-v1") == second.stratum_id(
        score_version="score-v1"
    )


def test_duplicate_biological_edge_and_contrast_mismatch_fail_closed() -> None:
    candidate = _candidate()
    with pytest.raises(ContractError) as duplicate:
        freeze_active_edge_universe(
            (candidate, candidate),
            universe_name="duplicate",
            contrast_id=candidate.contrast_id,
            score_version="score-v1",
        )
    assert duplicate.value.details.code == "duplicate_active_edge_candidate"

    with pytest.raises(ContractError) as contrast:
        freeze_active_edge_universe(
            (candidate,),
            universe_name="wrong-contrast",
            contrast_id="other-contrast",
            score_version="score-v1",
        )
    assert contrast.value.details.code == "active_edge_universe_contrast_mismatch"


def test_candidate_and_universe_reject_tampering() -> None:
    candidate = _candidate()
    universe = freeze_active_edge_universe(
        (candidate,),
        universe_name="tamper-test",
        contrast_id=candidate.contrast_id,
        score_version="score-v1",
    )
    object.__setattr__(candidate, "receiver", "forged")

    with pytest.raises(ContractError) as error:
        universe.to_dict()

    assert error.value.details.code == "active_edge_universe_integrity_violation"


def _point_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1-a", "s1-b", "s2-a"],
            "subject_id": ["s1", "s1", "s2"],
            "repeat_id": ["repeat-0"] * 3,
            "context_id": ["treated"] * 3,
            "sender": ["Sender"] * 3,
            "receiver": ["Receiver"] * 3,
            "interaction_id": ["lr-1"] * 3,
            "mode": ["state"] * 3,
            "global_sender_lr_score": [1.0, 0.0, 1.0],
            "status": ["observed", "structural_zero", "observed"],
            "reason_code": [None, "absolute_sender_evidence_zero", None],
            "score_version": ["score-v1"] * 3,
        }
    )


def test_point_reducer_weights_subjects_not_technical_samples() -> None:
    result = summarize_subject_equal_active_edge_score(
        _candidate(),
        _point_rows(),
        source_collection_id="collection-1",
        score_version="score-v1",
    )

    assert result.status is ActiveEdgePointStatus.OBSERVED
    assert result.score == pytest.approx(0.75)
    assert result.n_source_rows == 3
    assert result.n_subjects == 2
    assert result.to_dict()["point_semantics"].startswith("technical_sample_mean")


def test_point_reducer_weights_repeats_within_subject_before_subjects() -> None:
    rows = pd.DataFrame(
        {
            "sample_id": ["s1-a", "s1-b", "s1-a", "s2-a"],
            "subject_id": ["s1", "s1", "s1", "s2"],
            "repeat_id": ["repeat-0", "repeat-0", "repeat-1", "repeat-0"],
            "context_id": ["treated"] * 4,
            "sender": ["Sender"] * 4,
            "receiver": ["Receiver"] * 4,
            "interaction_id": ["lr-1"] * 4,
            "mode": ["state"] * 4,
            "global_sender_lr_score": [1.0, 1.0, 0.0, 1.0],
            "status": ["observed"] * 4,
            "reason_code": [None] * 4,
            "score_version": ["score-v1"] * 4,
        }
    )

    result = summarize_subject_equal_active_edge_score(
        _candidate(),
        rows,
        source_collection_id="collection-1",
        score_version="score-v1",
    )

    # s1: mean(mean(1, 1), mean(0)) = 0.5; s2: 1.0; subjects equal.
    assert result.score == pytest.approx(0.75)


def test_point_reducer_is_row_order_invariant_and_propagates_ne() -> None:
    candidate = _candidate()
    rows = _point_rows()
    first = summarize_subject_equal_active_edge_score(
        candidate,
        rows,
        source_collection_id="collection-1",
        score_version="score-v1",
    )
    second = summarize_subject_equal_active_edge_score(
        candidate,
        rows.sample(frac=1.0, random_state=4),
        source_collection_id="collection-1",
        score_version="score-v1",
    )
    assert first.point_score_id == second.point_score_id

    rows.loc[0, "global_sender_lr_score"] = None
    rows.loc[0, "status"] = "not_estimable"
    rows.loc[0, "reason_code"] = "synthetic_missing"
    blocked = summarize_subject_equal_active_edge_score(
        candidate,
        rows,
        source_collection_id="collection-1",
        score_version="score-v1",
    )
    assert blocked.status is ActiveEdgePointStatus.NOT_ESTIMABLE
    assert blocked.score is None
    assert blocked.reason_code == "active_edge_point_source_not_estimable"


def test_point_reducer_retains_missing_candidate_and_rejects_wrong_edge() -> None:
    candidate = _candidate()
    missing = summarize_subject_equal_active_edge_score(
        candidate,
        _point_rows().iloc[0:0],
        source_collection_id="collection-1",
        score_version="score-v1",
    )
    assert missing.status is ActiveEdgePointStatus.NOT_ESTIMABLE
    assert missing.n_subjects == 0

    wrong = _point_rows()
    wrong["receiver"] = "Other"
    with pytest.raises(ContractError) as error:
        summarize_subject_equal_active_edge_score(
            candidate,
            wrong,
            source_collection_id="collection-1",
            score_version="score-v1",
        )
    assert error.value.details.code == "active_edge_point_source_mismatch"
