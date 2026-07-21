from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from benchmarks.literature.frozen_signed_cardinality_benchmark import (
    _component_arm_a_effects,
    _validate_frozen_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repository_frozen_candidate_passes_checksum_and_release_boundary() -> None:
    candidate = _validate_frozen_candidate(
        REPO_ROOT
        / "benchmarks/results/signed_cardinality_rc1_v2/frozen_candidate.json"
    )
    assert candidate["selected_candidate"] == "eb_delta_0.5"
    assert candidate["formal_release_allowed"] is False


def test_component_arm_a_renames_only_required_effect_fields() -> None:
    table = pd.DataFrame(
        {
            "arm": ["A", "B"],
            "sender": ["S", "S"],
            "receiver": ["R", "R"],
            "interaction_id": ["i", "i"],
            "ligand": ["L", "L"],
            "receptor": ["R", "R"],
            "reference_condition": ["ref", "ref"],
            "target_condition": ["target", "target"],
            "effect": [0.2, 0.3],
            "standard_error": [0.1, 0.1],
            "n_reference": [5, 5],
            "n_target": [6, 6],
            "status": ["observed", "observed"],
            "reason_code": [None, None],
        }
    )
    result = _component_arm_a_effects(table)
    assert len(result) == 1
    assert result["effect_target_minus_reference"].iloc[0] == 0.2
    assert result["effect_standard_error_hc2"].iloc[0] == 0.1


def test_frozen_candidate_rejects_mutated_copy(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _validate_frozen_candidate(path)
