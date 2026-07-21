from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.adapters.crychic.replay_v3_sample import (
    _sample_effects,
    _validate_sample_layer,
    _validate_source_contract,
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


def test_source_contract_separates_source_and_ranking_dataset_ids() -> None:
    manifest = {
        "status": "complete",
        "dataset_id": "UCSC_Lerma_Martin_MS_snRNA_CA_vs_Ctrl",
        "input": {
            "sha256": (
                "433717d9fd98e57e15a444a338a6e1002f7ca3224022321d28a386c0a8498e1c"
            ),
            "shape": [69168, 32115],
            "samples": 11,
            "subject_support": {"Ctrl": 5, "CA": 5},
        },
        "parameters": {"outer_folds": 2},
    }

    contract = _validate_source_contract(manifest, dataset="ms")
    assert contract["ranking_dataset_id"] == (
        "UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca"
    )

    manifest["dataset_id"] = "LermaMartin_MS_CA_vs_Ctrl"
    with pytest.raises(ValueError, match="dataset identity"):
        _validate_source_contract(manifest, dataset="ms")


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
