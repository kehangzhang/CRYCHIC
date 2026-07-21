from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.adapters.crychic.replay_v3_sample import (
    _sample_effects,
    _validate_sample_layer,
)


def _sample_score_layers() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, samples, subject_prefix in (
        ("CTRL", ("c1", "c2"), "p"),
        ("IZ", ("i1", "i2"), "q"),
    ):
        for index, sample in enumerate(samples):
            rows.append(
                {
                    "fold_id": "fold_a" if index == 0 else "fold_b",
                    "sample_id": sample,
                    "subject_id": f"{subject_prefix}{index}",
                    "condition": condition,
                    "sender": "A",
                    "receiver": "B",
                    "interaction_id": "lr1",
                }
            )
    return pd.DataFrame(rows)


def test_sample_layer_requires_one_fold_per_sample() -> None:
    layers = _sample_score_layers()
    result = _validate_sample_layer(
        layers,
        condition_column="condition",
        samples_by_condition={"CTRL": 2, "IZ": 2},
        subjects_by_condition={"CTRL": 2, "IZ": 2},
    )
    assert result["folds"] == ["fold_a", "fold_b"]
    assert result["sample_coverage"]["n_samples"] == 4

    duplicated = pd.concat(
        [layers, layers.loc[layers["sample_id"].eq("c1")].assign(fold_id="fold_b")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="multiple held-out outer folds"):
        _validate_sample_layer(
            duplicated,
            condition_column="condition",
            samples_by_condition={"CTRL": 2, "IZ": 2},
            subjects_by_condition={"CTRL": 2, "IZ": 2},
        )


def test_sample_effects_use_sample_counts_and_do_not_average_subject_replicates() -> (
    None
):
    records: list[dict[str, object]] = []
    for condition, samples, value in (
        ("CTRL", ("c1", "c2"), 0.1),
        ("IZ", ("i1", "i2"), 0.4),
    ):
        for index, sample in enumerate(samples):
            records.append(
                {
                    "sample_id": sample,
                    "subject_id": f"subject_{index}",
                    "condition": condition,
                    "sender": "A",
                    "receiver": "B",
                    "interaction_id": "lr1",
                    "ligand": "L",
                    "receptor": "R",
                    "family_id": "F",
                    "driver_id": "D",
                    "direct_response_score": value,
                    "direct_response_status": "observed",
                }
            )
    effects = _sample_effects(
        pd.DataFrame.from_records(records),
        reference="CTRL",
        target="IZ",
        condition_column="condition",
    )
    row = effects.iloc[0]
    assert row["effect_target_minus_reference"] == pytest.approx(0.3)
    assert row["n_subjects_reference"] == 2
    assert row["n_subjects_target"] == 2
    assert "sample_level" in row["effect_semantics"]
    assert row["covariate_adjustment"] == "none"
