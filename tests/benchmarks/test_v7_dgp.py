from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from benchmarks.simulation.v7_dgp import (
    CANDIDATE_SENDER_COUNTS,
    INTERACTIONS,
    generate_v7_dgp,
)
from benchmarks.simulation.v7_protocol import load_v7_benchmark_protocol

from crychic.workflow import (
    V7EstimatorSpec,
    build_v7_diagnostics,
    fit_crossfit_v7_estimator,
    run_subject_crossfit,
)


def test_every_registered_family_design_generates_raw_counts_and_truth() -> None:
    protocol = load_v7_benchmark_protocol()
    generated: set[tuple[str, str]] = set()
    seed = 100
    for families in protocol.family_roles.values():
        for family, designs in families.items():
            for design in designs:
                fixture = generate_v7_dgp(
                    dataset_id=f"{family}-{design}",
                    dgp_family=family,
                    design_kind=design,
                    seed=seed,
                    candidate_sender_count=2,
                    cells_per_type=1,
                    subjects_per_level=4,
                )
                seed += 1
                assert fixture.adata.n_obs > 0
                assert fixture.adata.n_vars == 25
                assert "counts" in fixture.adata.layers
                assert not fixture.truth.empty
                assert fixture.to_manifest()["truth_rows"] == len(fixture.truth)
                generated.add((family, design))
    assert len(generated) == 45


def test_expression_truth_separates_parent_measurement_from_causal_sender() -> None:
    fixture = generate_v7_dgp(
        dataset_id="expression",
        dgp_family="expression_joint",
        design_kind="paired",
        seed=1,
        candidate_sender_count=2,
        cells_per_type=1,
        subjects_per_level=4,
    )
    truth = fixture.truth.set_index(["contrast_name", "sender", "interaction_id"])
    true_sender = truth.loc[("B_vs_A", "S01", "lr_signal")]
    receiver_as_sender = truth.loc[("B_vs_A", "Receiver", "lr_signal")]
    decoy_event = truth.loc[("B_vs_A", "S01", "lr_decoy")]

    assert bool(true_sender["truth_causal_sender"])
    assert float(true_sender["truth_score_effect"]) == pytest.approx(1.25)
    assert not bool(receiver_as_sender["truth_causal_sender"])
    assert float(receiver_as_sender["truth_score_effect"]) == pytest.approx(0.5)
    assert float(true_sender["truth_parent_effect"]) == pytest.approx(0.875)
    assert float(receiver_as_sender["truth_parent_effect"]) == pytest.approx(0.875)
    assert not bool(decoy_event["truth_causal_parent"])
    assert float(decoy_event["truth_score_effect"]) == 0.0


def test_program_only_and_inhibitory_truth_keep_estimands_separate() -> None:
    program = generate_v7_dgp(
        dataset_id="program",
        dgp_family="program_only",
        design_kind="paired",
        seed=2,
        candidate_sender_count=2,
        cells_per_type=1,
        subjects_per_level=4,
    )
    active_program = program.truth.loc[program.truth["interaction_id"].eq("lr_signal")]
    assert np.allclose(active_program["truth_score_effect"], 0.0)
    assert active_program["truth_program_effect"].gt(0.0).all()
    assert not active_program["truth_causal_sender"].any()
    assert active_program["truth_causal_parent"].all()

    inhibitory = generate_v7_dgp(
        dataset_id="inhibitory",
        dgp_family="inhibitory_program",
        design_kind="paired",
        seed=3,
        candidate_sender_count=2,
        cells_per_type=1,
        subjects_per_level=4,
    )
    active_inhibitory = inhibitory.truth.loc[
        inhibitory.truth["interaction_id"].eq("lr_inhibitory")
    ]
    assert active_inhibitory["truth_program_effect"].gt(0.0).all()
    assert inhibitory.target_prior.direction == 1
    signed_spec = inhibitory.crossfit_spec.signed_program_v2_spec
    assert signed_spec is not None
    assert signed_spec.to_dict()["mechanism_direction_overrides"] == [
        ["lr_inhibitory", "attenuation"]
    ]


def test_null_families_do_not_gain_truth_from_nuisance_expression() -> None:
    for family in (
        "global_null",
        "generic_state_null",
        "batch_context_confounding_null",
        "composition_only_null",
        "receiver_autonomous_null",
        "abundance_only_null",
    ):
        fixture = generate_v7_dgp(
            dataset_id=family,
            dgp_family=family,
            design_kind="independent_two_group",
            seed=10,
            candidate_sender_count=2,
            cells_per_type=1,
            subjects_per_level=4,
        )
        assert not fixture.truth["truth_causal_sender"].any()
        assert not fixture.truth["truth_causal_parent"].any()
        assert np.allclose(fixture.truth["truth_score_effect"], 0.0)
        assert np.allclose(fixture.truth["truth_program_effect"], 0.0)


def test_candidate_cardinality_and_structural_absence_are_explicit() -> None:
    for candidate_count in CANDIDATE_SENDER_COUNTS:
        fixture = generate_v7_dgp(
            dataset_id=f"cardinality-{candidate_count}",
            dgp_family="candidate_cardinality",
            design_kind="paired",
            seed=candidate_count,
            candidate_sender_count=candidate_count,
            cells_per_type=1,
            subjects_per_level=4,
        )
        assert fixture.adata.obs["cell_type"].astype(str).nunique() == candidate_count
        expected_truth_rows = candidate_count * len(INTERACTIONS)
        assert len(fixture.truth) == expected_truth_rows

    absent = generate_v7_dgp(
        dataset_id="absence",
        dgp_family="structural_absence",
        design_kind="paired",
        seed=11,
        candidate_sender_count=2,
        cells_per_type=2,
        subjects_per_level=4,
    )
    absent_counts = absent.cell_counts.loc[
        absent.cell_counts["cell_type"].eq("S01")
        & absent.cell_counts["sample_id"].str.endswith(":B"),
        "cell_count",
    ]
    assert not absent_counts.empty
    assert absent_counts.eq(0).all()
    observed = absent.adata.obs
    assert not (
        observed["cell_type"].astype(str).eq("S01")
        & observed["condition"].astype(str).eq("B")
    ).any()


def test_resource_distinguishes_mandatory_and_from_alternative_or() -> None:
    fixture = generate_v7_dgp(
        dataset_id="and-or",
        dgp_family="alternative_or",
        design_kind="independent_two_group",
        seed=12,
        candidate_sender_count=2,
        cells_per_type=1,
        subjects_per_level=4,
    )
    interactions = {item.interaction_id: item for item in fixture.resource.interactions}
    mandatory = interactions["lr_complex"]
    alternative = interactions["lr_alternative"]
    assert mandatory.ligand_is_complex
    assert mandatory.receptor_is_complex
    assert len(mandatory.ligand_subunits) == 2
    assert not alternative.ligand_is_complex
    assert alternative.ligand_subunits == ("L4A", "L4B")
    assert alternative.pathway == "alternative_pathway"


def test_raw_dgp_runs_current_crossfit_estimator_and_diagnostics() -> None:
    fixture = generate_v7_dgp(
        dataset_id="pipeline-smoke",
        dgp_family="expression_joint",
        design_kind="paired",
        seed=7,
        candidate_sender_count=2,
        cells_per_type=2,
        subjects_per_level=8,
    )
    crossfit = run_subject_crossfit(
        fixture.adata,
        fixture.config,
        fixture.resource,
        fixture.target_prior,
        spec=fixture.crossfit_spec,
    )
    scores = crossfit.oof_sample_edge_scores_v2
    assert not scores.empty
    assert set(scores["candidate_sender_count"]) == {2}
    assert set(scores["out_of_fold"]) == {True}

    with warnings.catch_warnings():
        setting_with_copy = getattr(pd.errors, "SettingWithCopyWarning", None)
        if setting_with_copy is not None:
            warnings.simplefilter("error", setting_with_copy)
        estimator = fit_crossfit_v7_estimator(
            crossfit,
            V7EstimatorSpec(
                design=fixture.differential_design,
                score_head="parent_mean_raw",
            ),
            sample_metadata=fixture.sample_metadata,
        )
    assert estimator.differential.effects["status"].eq("observed").any()

    truth = fixture.truth.loc[
        :,
        [
            "contrast_name",
            "sender",
            "receiver",
            "interaction_id",
            "truth_causal_sender",
            "truth_score_effect",
            "mechanism_class",
            "pathway",
        ],
    ].rename(
        columns={
            "truth_causal_sender": "truth",
            "truth_score_effect": "truth_effect",
        }
    )
    diagnostics = build_v7_diagnostics(
        crossfit,
        dataset_id=fixture.dataset_id,
        truth=truth,
        cell_counts=fixture.cell_counts,
    )
    combined = diagnostics.candidate_sender_bias.loc[
        diagnostics.candidate_sender_bias["fold_id"].eq("__all__")
    ].iloc[0]
    assert combined["candidate_sender_bin"] == "2"
    assert float(combined["sender_auprc"]) > 0.5
    assert float(combined["sender_auroc"]) > 0.5


def test_generation_is_seed_deterministic_and_content_bound() -> None:
    arguments = {
        "dataset_id": "deterministic",
        "dgp_family": "sender_decoy",
        "design_kind": "paired",
        "candidate_sender_count": 5,
        "cells_per_type": 1,
        "subjects_per_level": 4,
    }
    first = generate_v7_dgp(seed=19, **arguments)
    second = generate_v7_dgp(seed=19, **arguments)
    changed = generate_v7_dgp(seed=20, **arguments)

    assert first.fixture_id == second.fixture_id
    assert first.fixture_id != changed.fixture_id
    assert first.raw_input_digest == second.raw_input_digest
    assert first.raw_input_digest != changed.raw_input_digest
    assert (first.adata.layers["counts"] != second.adata.layers["counts"]).nnz == 0
    pd.testing.assert_frame_equal(first.truth, second.truth)


def test_dgp_freezes_the_preregistered_legacy_g1_tuning_grid() -> None:
    fixture = generate_v7_dgp(
        dataset_id="legacy-grid",
        dgp_family="expression_joint",
        design_kind="independent_two_group",
        seed=91,
        candidate_sender_count=2,
        cells_per_type=1,
        subjects_per_level=4,
    )
    tuning = fixture.crossfit_spec.penalty_tuning_spec
    assert tuning is not None
    assert tuning.lambda1_fractions == (1.0, 0.1)
    assert tuning.lambda2_fractions == (0.0,)
    assert tuning.inner_allowed_n_splits == (2,)
    assert tuning.min_inner_train_subjects_per_context == 1
    assert tuning.min_inner_validation_subjects_per_context == 1
    assert tuning.root_seed == fixture.seed


@pytest.mark.parametrize(
    ("family", "design", "subjects_per_level"),
    (
        ("expression_joint", "independent_multi_group", 4),
        ("topology_jump", "repeated", 8),
        ("inhibitory_program", "paired", 8),
    ),
)
def test_crossfit_supports_multigroup_repeated_and_inhibitory_v7_families(
    family: str,
    design: str,
    subjects_per_level: int,
) -> None:
    fixture = generate_v7_dgp(
        dataset_id=f"pipeline-{family}",
        dgp_family=family,
        design_kind=design,
        seed=301,
        candidate_sender_count=2,
        cells_per_type=1,
        subjects_per_level=subjects_per_level,
    )
    crossfit = run_subject_crossfit(
        fixture.adata,
        fixture.config,
        fixture.resource,
        fixture.target_prior,
        spec=fixture.crossfit_spec,
    )
    assert not crossfit.oof_sample_edge_scores_v2.empty
    assert all(fold.family_common_applications for fold in crossfit.folds)
