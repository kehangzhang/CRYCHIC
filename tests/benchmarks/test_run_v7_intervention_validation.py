from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from benchmarks.literature import run_v7_intervention_validation as runner
from benchmarks.literature.run_v7_intervention_validation import (
    DEFAULT_CONFIG,
    InterventionDatasetContract,
    build_intervention_crossfit_spec,
    build_intervention_design,
    load_intervention_config,
    paired_gene_effects,
    summarize_kang_truth,
    summarize_mechanism_effects,
    validate_paired_input,
)
from benchmarks.literature.summarize_v7_brca_response import (
    build_descriptive_response_interaction,
    summarize_response_interaction,
)


def _small_contract(*, subjects: int = 2) -> InterventionDatasetContract:
    return InterventionDatasetContract(
        slug="kang",
        dataset_id="fixture",
        role="fixture",
        condition_column="condition",
        reference="ctrl",
        target="stim",
        contrast_name="stim_vs_ctrl",
        seed=7,
        outer_fold_partition_seed=7,
        allowed_outer_folds=(2,),
        input_schema="fixture",
        input_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        cells=subjects * 4,
        genes=2,
        samples=subjects * 2,
        subjects=subjects,
        cell_types=("B", "T"),
        truth_schema="fixture_truth",
        truth_sha256="c" * 64,
    )


def _paired_data(subjects: int = 2) -> ad.AnnData:
    records: list[dict[str, str]] = []
    for subject in range(subjects):
        for condition in ("ctrl", "stim"):
            for cell_type in ("B", "T"):
                records.append(
                    {
                        "sample_id": f"s{subject}:{condition}",
                        "subject_id": f"s{subject}",
                        "condition": condition,
                        "cell_type": cell_type,
                    }
                )
    counts = sparse.csr_matrix(np.ones((len(records), 2), dtype=np.int32))
    return ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame.from_records(records),
        var=pd.DataFrame(index=["G1", "G2"]),
        layers={"counts": counts},
    )


def _mechanism_effects() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for view, values in (
        ("m0_parent_mean_component", (1.0, -2.0, 3.0)),
        ("program_signed_component", (2.0, -1.0, 4.0)),
    ):
        for index, value in enumerate(values):
            rows.append(
                {
                    "generator_id": "G4",
                    "score_view": view,
                    "inference_id": "I1",
                    "contrast_name": "stim_vs_ctrl",
                    "event_id": f"event-{index}",
                    "receiver": "T",
                    "interaction_id": f"lr-{index}",
                    "effect": value,
                    "status": "observed",
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "formal_inference_allowed": False,
                }
            )
    return pd.DataFrame.from_records(rows)


def _brca_effects(values: tuple[float, ...]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        {
            "generator_id": "G3",
            "score_view": "primary_sender_detection",
            "estimand": "sample_comparable_lr_intensity",
            "resolution": "sender_lr_receiver_child",
            "contrast_scope": "all_contrasts",
            "sender": "S",
            "receiver": "R",
            "interaction_id": f"lr-{index}",
            "inference_id": "I1",
            "effect": value,
            "standard_error": 0.5,
            "statistic": value / 0.5,
            "status": "observed",
            "reason_code": None,
            "p_value": np.nan,
            "q_value": np.nan,
            "formal_inference_allowed": False,
        }
        for index, value in enumerate(values)
    )


def test_frozen_intervention_protocol_loads_and_rejects_release_drift(
    tmp_path: Path,
) -> None:
    protocol, contracts = load_intervention_config(DEFAULT_CONFIG)

    assert tuple(contracts) == ("kang", "brca_e", "brca_ne")
    assert contracts["kang"].subjects == 8
    assert contracts["brca_e"].subjects == 9
    assert contracts["brca_ne"].subjects == 20
    assert protocol["release"]["real_data_edge_auroc_forbidden"] is True

    changed = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed["release"]["real_data_edge_auroc_forbidden"] = False
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="release boundary changed"):
        load_intervention_config(path)


def test_brca_contract_cannot_acquire_truth_fields(tmp_path: Path) -> None:
    changed = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed["datasets"]["brca_e"]["truth_schema"] = "invented_truth"
    changed["datasets"]["brca_e"]["truth_sha256"] = "d" * 64
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot declare edge-level truth"):
        load_intervention_config(path)


def test_intervention_spec_and_design_preserve_paired_subject_estimand() -> None:
    protocol, contracts = load_intervention_config(DEFAULT_CONFIG)
    contract = contracts["kang"]
    config, crossfit = build_intervention_crossfit_spec(contract, protocol["estimator"])
    design = build_intervention_design(contract, protocol["estimator"])

    assert tuple(config.context_keys) == ("condition",)
    assert crossfit.allowed_n_splits == (4, 2)
    assert crossfit.predeclared_receiver_ids == tuple(sorted(contract.cell_types))
    assert crossfit.training_spec.sender_parameters.contrast_unit == "paired_subject"
    assert crossfit.training_spec.sender_parameters.min_subjects == 4
    assert crossfit.training_spec.min_pooled_availability == 0.0
    assert crossfit.absolute_activity_v2_spec is not None
    assert crossfit.signed_program_v2_spec is not None
    assert crossfit.eb_shrunken_coupling_v2_spec is not None
    assert design.design_kind.value == "paired"
    assert design.contrasts[0].weights == (("ctrl", -1.0), ("stim", 1.0))


def test_validate_paired_input_rejects_incomplete_subject_pair() -> None:
    contract = _small_contract()
    data = _paired_data()

    metadata, audit = validate_paired_input(data, contract)
    assert len(metadata) == 4
    assert audit["paired_subjects"] == 2

    broken = data.copy()
    broken.obs.loc[
        (broken.obs["subject_id"] == "s1") & (broken.obs["condition"] == "stim"),
        "condition",
    ] = "ctrl"
    with pytest.raises(ValueError, match="complete pair"):
        validate_paired_input(broken, contract)


def test_inference_view_whitelist_excludes_annotation_only_scores() -> None:
    protocol, _ = load_intervention_config(DEFAULT_CONFIG)
    rows = [
        {"generator_id": generator, "score_view": view}
        for generator, views in runner.INFERENCE_SCORE_VIEWS.items()
        for view in views
    ]
    rows.append({"generator_id": "G5", "score_view": "coupling_prior_annotation"})

    selected = runner._inference_score_views(
        pd.DataFrame.from_records(rows), protocol["estimator"]
    )

    assert len(selected) == 6
    assert "coupling_prior_annotation" not in set(selected["score_view"])


def test_mechanism_summary_is_directional_and_rejects_formal_fields() -> None:
    effects = _mechanism_effects()
    summary = summarize_mechanism_effects(effects, _small_contract())

    metrics = summary.set_index(["receiver", "metric"])["value"]
    assert metrics.loc[
        ("all_receivers", "g4_signed_program_positive_fraction")
    ] == pytest.approx(2 / 3)
    assert metrics.loc[("all_receivers", "m0_program_sign_concordance")] == 1.0
    assert metrics.loc[("all_receivers", "m0_program_effect_spearman")] == 1.0

    effects.loc[0, "p_value"] = 0.01
    with pytest.raises(ValueError, match="withheld p/q"):
        summarize_mechanism_effects(effects, _small_contract())


def test_kang_gene_response_uses_complete_paired_raw_count_effects() -> None:
    subjects = 8
    obs: list[dict[str, str]] = []
    counts: list[list[int]] = []
    for index in range(subjects):
        for condition in ("ctrl", "stim"):
            obs.append(
                {
                    "sample_id": f"d{index}:{condition}",
                    "subject_id": f"d{index}",
                    "condition": condition,
                    "cell_type": "T",
                }
            )
            counts.append([1, 99] if condition == "ctrl" else [10, 90])
    matrix = sparse.csr_matrix(np.asarray(counts, dtype=np.int32))
    data = ad.AnnData(
        X=matrix.copy(),
        obs=pd.DataFrame.from_records(obs),
        var=pd.DataFrame(index=["ISG", "HOUSEKEEPING"]),
        layers={"counts": matrix},
    )
    contract = InterventionDatasetContract(
        slug="kang",
        dataset_id="fixture",
        role="fixture",
        condition_column="condition",
        reference="ctrl",
        target="stim",
        contrast_name="stim_vs_ctrl",
        seed=7,
        outer_fold_partition_seed=7,
        allowed_outer_folds=(2,),
        input_schema="fixture",
        input_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        cells=16,
        genes=2,
        samples=16,
        subjects=8,
        cell_types=("T",),
        truth_schema="fixture_truth",
        truth_sha256="c" * 64,
    )
    truth = {
        "truth_set_id": "fixture_truth",
        "expected_observations": [
            {
                "id": "paired_design_recovered",
                "level": "data_contract",
                "direction": "exact",
            },
            {
                "id": "isg_response",
                "level": "receiver_gene_response",
                "direction": "stim_up",
                "genes": ["ISG"],
                "receiver_cell_types": ["T"],
                "acceptance": {
                    "eligible_gene_celltype_positive_fraction_min": 0.75,
                    "donor_direction_consistency_median_min": 0.75,
                },
            },
            {
                "id": "ifnb_guardrail",
                "level": "interpretation_guardrail",
                "direction": "no_required_recovery",
            },
        ],
    }

    effects = paired_gene_effects(data, contract, truth)
    summary = summarize_kang_truth(
        truth,
        effects,
        pd.DataFrame(),
        contract,
    )

    assert effects.loc[0, "n_complete_pairs"] == 8
    assert effects.loc[0, "mean_effect"] > 0.0
    assert effects.loc[0, "positive_pair_fraction"] == 1.0
    assert summary["passed"].all()


def test_brca_response_interaction_is_descriptive_difference_of_paired_effects() -> (
    None
):
    response = build_descriptive_response_interaction(
        _brca_effects((2.0, 0.0, -2.0)),
        _brca_effects((1.0, -1.0, -1.0)),
    )
    summary = summarize_response_interaction(response)

    assert response["descriptive_response_interaction"].tolist() == [1.0, 1.0, -1.0]
    assert response["comparison_status"].eq("observed").all()
    assert not response["formal_inference_allowed"].any()
    assert response[["p_value", "q_value"]].isna().all(axis=None)
    assert summary.loc[0, "rows_observed_both"] == 3
    assert summary.loc[0, "positive_response_interaction_fraction"] == pytest.approx(
        2 / 3
    )
    assert "descriptive_difference" in summary.loc[0, "claim_scope"]


def test_brca_response_interaction_rejects_formal_input() -> None:
    expander = _brca_effects((1.0,))
    expander.loc[0, "q_value"] = 0.05

    with pytest.raises(ValueError, match="cannot consume formal tests"):
        build_descriptive_response_interaction(expander, _brca_effects((0.0,)))


def test_target_prior_requires_native_manifest_path(tmp_path: Path) -> None:
    protocol, _ = load_intervention_config(DEFAULT_CONFIG)
    copied = tmp_path / "manifest.json"
    copied.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="native NicheNet manifest path"):
        runner._load_target_prior(
            tmp_path / "database",
            copied,
            protocol["target_prior"],
        )
