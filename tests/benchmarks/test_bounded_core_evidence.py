from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.literature.bounded_core_evidence import (
    BoundedCoreEvidencePolicy,
    bounded_core_evidence_rankings,
)


def _ranking(
    scores: list[float | None],
    statuses: list[str] | None = None,
    *,
    condition: str = "A",
) -> pd.DataFrame:
    if statuses is None:
        statuses = ["observed"] * len(scores)
    return pd.DataFrame(
        {
            "condition": [condition] * len(scores),
            "sender": [f"s{index}" for index in range(len(scores))],
            "receiver": [f"r{index}" for index in range(len(scores))],
            "ranked_strength": scores,
            "estimable_directed_lr": np.arange(1, len(scores) + 1),
            "status": statuses,
        }
    )


def test_quantile_matched_blend_preserves_base_scale() -> None:
    base = _ranking([1.0, 2.0, 3.0])
    expert = _ranking([30.0, 20.0, 10.0])
    result, diagnostics = bounded_core_evidence_rankings(
        base,
        expert,
        policy=BoundedCoreEvidencePolicy(
            expert_fraction=0.25, missing_base_penalty=0.5
        ),
    )
    np.testing.assert_allclose(result["ranked_strength"], [1.5, 2.0, 2.5])
    assert set(result["status"]) == {"observed"}
    assert diagnostics.loc[0, "common_observed"] == 3
    assert diagnostics.loc[0, "base_observed_mean"] == pytest.approx(2.0)
    assert diagnostics.loc[0, "mapped_expert_mean_on_common"] == pytest.approx(2.0)
    assert diagnostics.loc[0, "blended_mean_on_common"] == pytest.approx(2.0)


def test_missing_rows_are_typed_and_never_zero_imputed() -> None:
    base = _ranking(
        [1.0, None, 3.0, None],
        ["observed", "not_estimable", "observed", "not_estimable"],
    )
    expert = _ranking(
        [4.0, 3.0, None, None],
        ["observed", "observed", "not_estimable", "not_estimable"],
    )
    result, _ = bounded_core_evidence_rankings(
        base,
        expert,
        policy=BoundedCoreEvidencePolicy(
            expert_fraction=0.20, missing_base_penalty=0.50
        ),
    )
    assert result.loc[0, "ranked_strength"] == pytest.approx(1.0)
    assert result.loc[1, "ranked_strength"] == pytest.approx(1.0 / 30.0)
    assert result.loc[2, "ranked_strength"] == pytest.approx(3.0)
    assert pd.isna(result.loc[3, "ranked_strength"])
    assert result.loc[1, "reason_code"] == (
        "base_not_estimable_used_penalized_expert_only"
    )
    assert result.loc[3, "status"] == "not_estimable"


def test_zero_expert_fraction_does_not_promote_expert_only_rows() -> None:
    base = _ranking([1.0, None], ["observed", "not_estimable"])
    expert = _ranking([2.0, 3.0])
    result, _ = bounded_core_evidence_rankings(
        base,
        expert,
        policy=BoundedCoreEvidencePolicy(expert_fraction=0.0, missing_base_penalty=1.0),
    )
    assert result.loc[0, "ranked_strength"] == pytest.approx(1.0)
    assert pd.isna(result.loc[1, "ranked_strength"])
    assert result.loc[1, "status"] == "not_estimable"


@pytest.mark.parametrize("field", ("expert_fraction", "missing_base_penalty"))
@pytest.mark.parametrize("value", (-0.01, 1.01, float("nan")))
def test_policy_rejects_invalid_weights(field: str, value: float) -> None:
    values = {"expert_fraction": 0.2, "missing_base_penalty": 0.5}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        BoundedCoreEvidencePolicy(**values)


def test_conditions_are_quantile_matched_independently() -> None:
    base = pd.concat(
        [_ranking([1.0, 2.0], condition="A"), _ranking([10.0, 20.0], condition="B")],
        ignore_index=True,
    )
    expert = pd.concat(
        [_ranking([2.0, 1.0], condition="A"), _ranking([2.0, 1.0], condition="B")],
        ignore_index=True,
    )
    result, diagnostics = bounded_core_evidence_rankings(
        base,
        expert,
        policy=BoundedCoreEvidencePolicy(expert_fraction=1.0, missing_base_penalty=0.5),
    )
    np.testing.assert_allclose(result["ranked_strength"], [2.0, 1.0, 20.0, 10.0])
    assert list(diagnostics["condition"]) == ["A", "B"]
