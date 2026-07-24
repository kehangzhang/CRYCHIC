from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.comprehensive.evaluate_m2_sender_coupling import (
    M2_METHOD,
    RAW_METHOD,
    _candidate_couplings,
    _method_summary,
    _seed_metrics,
    _subject_folds,
)
from benchmarks.comprehensive.generate_m2_sender_coupling_fixture import (
    CANDIDATE_ROLES,
    SCHEMA_VERSION,
    _dataset_seed,
    generate,
)


def test_subject_folds_are_stable_balanced_and_seed_specific() -> None:
    subjects = [f"S{index:02d}" for index in range(20)]
    first = _subject_folds(subjects, root_seed=11, n_folds=4)
    reordered = _subject_folds(reversed(subjects), root_seed=11, n_folds=4)
    second_seed = _subject_folds(subjects, root_seed=12, n_folds=4)

    assert first == reordered
    assert first != second_seed
    assert sorted(first.values()).count(0) == 5
    assert all(list(first.values()).count(fold) == 5 for fold in range(4))


def _confounded_effects(root_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(root_seed)
    n_subjects = 24
    z = rng.normal(size=n_subjects)
    batch = np.resize(np.asarray([-1.0, 1.0]), n_subjects)
    rng.shuffle(batch)
    composition = rng.normal(size=n_subjects)
    receiver = (
        0.6 * z
        + 1.5 * batch
        + 1.5 * composition
        + rng.normal(scale=0.03, size=n_subjects)
    )
    sender_effects = {
        "TrueSender": z + rng.normal(scale=0.03, size=n_subjects),
        "BatchDecoy": batch + rng.normal(scale=0.10, size=n_subjects),
        "CompositionDecoy": composition + rng.normal(scale=0.10, size=n_subjects),
        "IndependentDecoy": rng.normal(size=n_subjects),
    }
    sender_proportions = {
        sender: rng.normal(size=n_subjects) for sender in sender_effects
    }
    rows = []
    for sender, values in sender_effects.items():
        for index in range(n_subjects):
            rows.append(
                {
                    "dataset_id": f"fixture-{root_seed}",
                    "root_seed": root_seed,
                    "sender": sender,
                    "subject_id": f"S{index + 1:02d}",
                    "sender_effect": values[index],
                    "receiver_effect": receiver[index],
                    "batch_contrast": batch[index],
                    "composition_design_contrast": composition[index],
                    "sender_proportion_effect": sender_proportions[sender][index],
                }
            )
    return pd.DataFrame.from_records(rows)


def _truth() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "sender": sender,
                "sender_role": role,
                "expected_coupled_sender": sender == "TrueSender",
                "receiver": "Receiver",
                "interaction_id": "CXCL10_CXCR3",
            }
            for sender, role in CANDIDATE_ROLES.items()
        ]
    )


def test_m2_crossfit_suppresses_confounders_and_preserves_true_sender() -> None:
    effects = pd.concat(
        [_confounded_effects(seed) for seed in range(101, 111)], ignore_index=True
    )
    scores, folds = _candidate_couplings(
        effects,
        _truth(),
        n_folds=4,
        minimum_training_subjects=10,
    )
    metrics = _seed_metrics(scores)
    summary = _method_summary(metrics).set_index("method")

    assert (
        summary.loc[M2_METHOD, "average_precision"]
        > summary.loc[RAW_METHOD, "average_precision"]
    )
    assert summary.loc[M2_METHOD, "auroc"] > summary.loc[RAW_METHOD, "auroc"]
    assert summary.loc[M2_METHOD, "median_true_sender_rank"] == pytest.approx(1.0)
    assert scores["m2_fold_coverage"].eq(1.0).all()
    for record in folds.itertuples(index=False):
        training = set(str(record.training_subject_ids).split("|"))
        held_out = set(str(record.held_out_subject_ids).split("|"))
        assert training.isdisjoint(held_out)


def _resource(path: Path) -> Path:
    table = pd.DataFrame.from_records(
        [
            {
                "harmonized_interaction_id": f"lr-{index}",
                "ligand": ligand,
                "receptor": receptor,
                "scseqcommdiff_source_interaction_id": f"source-{index}",
                "scseqcommdiff_covered": True,
            }
            for index, (ligand, receptor) in enumerate(
                (
                    ("CXCL10", "CXCR3"),
                    ("CCL5", "CCR5"),
                    ("VEGFA", "FLT1"),
                    ("CXCL12", "CXCR4"),
                    ("EGF", "EGFR"),
                ),
                start=1,
            )
        ]
    )
    table.to_csv(path, sep="\t", index=False)
    return path


def test_generator_freezes_masked_truth_and_withholds_latents(tmp_path: Path) -> None:
    output = tmp_path / "fixture"
    manifest = generate(
        output,
        _resource(tmp_path / "resource.tsv"),
        seeds=(20270801,),
        n_subjects=12,
        mean_cells_per_sample=100,
        n_jobs=1,
    )

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert manifest["total_cells"] >= 2_400
    assert manifest["leakage_controls"]["generation_latents_unavailable_to_methods"]
    assert manifest["leakage_controls"][
        "composition_design_index_is_observed_method_input"
    ]
    record = manifest["records"][0]
    assert "TrueSender" not in record["dataset_id"]
    input_manifest = json.loads(
        (output / "inputs" / record["dataset_id"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert not input_manifest["generation_latents"]["benchmark_method_access_allowed"]
    config = json.loads((output / "crychic_config.json").read_text(encoding="utf-8"))
    assert "generation_latents" not in json.dumps(config)
    truth = pd.read_csv(output / "sender_truth.tsv", sep="\t")
    assert truth["expected_coupled_sender"].sum() == 1


def test_dataset_seed_is_stable_and_root_specific() -> None:
    assert _dataset_seed(11) == _dataset_seed(11)
    assert _dataset_seed(11) != _dataset_seed(12)
