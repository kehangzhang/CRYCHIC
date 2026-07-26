from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.literature import run_v7_real_multigroup as real_runner
from benchmarks.literature.run_v7_real_multigroup import (
    DEFAULT_CONFIG,
    PRIMARY_SCORE_VIEW,
    RealDatasetContract,
    build_sender_resolved_event_ledger,
    build_v7_real_design,
    evaluate_v7_real_spatial_des,
    load_real_e1_config,
)
from benchmarks.simulation.v7_integrated import (
    build_legacy_g1_score_view_from_components,
)
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
)


def _contract() -> RealDatasetContract:
    return RealDatasetContract(
        slug="kuppe",
        dataset_id="Kuppe_MI_CTRL_vs_IZ",
        role="development_non_independent",
        condition_column="condition",
        reference="CTRL",
        target="IZ",
        contrast_name="IZ_vs_CTRL",
        batch_columns=(),
        seed=20260717,
        outer_fold_partition_seed=20260717,
    )


def _interaction(interaction_id: str, ligand: str, receptor: str) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=f"source_{interaction_id}",
        ligand_name=ligand,
        receptor_name=receptor,
        ligand_subunits=(ligand,),
        receptor_subunits=(receptor,),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _bundle() -> ResourceBundle:
    return ResourceBundle(
        resource_id="fixture_resource",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(
            _interaction("i1", "L1", "R1"),
            _interaction("i2", "L2", "R2"),
        ),
        mapping_report=MappingReport(2, 2, 4),
        manifest_digest="d" * 64,
        source_files=("fixture.tsv",),
        license="CC0",
        citation="Fixture resource",
    )


def _effects() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for generator in ("G0", "G2", "G3", "G4", "G5"):
        for interaction, sender, receiver, effect, standard_error in (
            ("i1", "X", "Y", 2.0, 1.0),
            ("i2", "X", "Z", -0.5, 0.25),
        ):
            rows.append(
                {
                    "event_id": f"{generator}-{interaction}",
                    "generator_id": generator,
                    "score_view": PRIMARY_SCORE_VIEW,
                    "inference_id": "I1",
                    "contrast_name": "IZ_vs_CTRL",
                    "sender": sender,
                    "receiver": receiver,
                    "interaction_id": interaction,
                    "effect": effect,
                    "standard_error": standard_error,
                    "statistic": effect / standard_error,
                    "diagnostic_p_value": 0.1,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "formal_inference_allowed": False,
                    "status": "observed",
                    "reason_code": None,
                }
            )
    return pd.DataFrame.from_records(rows)


def _expected() -> pd.DataFrame:
    pairs = (("X", "Y"), ("X", "Z"), ("Y", "Z"))
    return pd.DataFrame.from_records(
        {
            "dataset": "Kuppe_MI_spatial_CTRL_vs_IZ",
            "variant": "spatial_neighbor_max",
            "scenario": "multi_sample",
            "condition": condition,
            "top_fraction": fraction,
            "sender": sender,
            "receiver": receiver,
            "is_expected": (condition, sender, receiver)
            in {("IZ", "X", "Y"), ("CTRL", "X", "Z")},
        }
        for condition in ("CTRL", "IZ")
        for fraction in (0.1, 0.2, 0.3, 0.4)
        for sender, receiver in pairs
    )


def _legacy_provenance() -> dict[str, object]:
    return {
        "rows": 2,
        "crossfit_id": "legacy_crossfit",
        "spec_id": "legacy_spec",
        "repeat_id": "legacy_repeat",
        "score_version": "family_first_mechanistic_ligand_contrast_gated_softmin_v2",
        "certification_status": "verified_train_only_oof_partial_pipeline",
        "is_oof_certified": False,
        "formal_inference_status": "not_available_descriptive_only",
        "claim_scope": "heldout_family_common_diagnostic_not_complete_oof_certified",
    }


def _legacy_components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crossfit_id": ["legacy_crossfit"] * 2,
            "spec_id": ["legacy_spec"] * 2,
            "repeat_id": ["legacy_repeat"] * 2,
            "fold_id": ["fold-1", "fold-2"],
            "contrast": ["IZ_vs_CTRL"] * 2,
            "sample_id": ["sample-ctrl", "sample-iz"],
            "subject_id": ["subject-ctrl", "subject-iz"],
            "context_id": ["CTRL", "IZ"],
            "sender": ["X", "X"],
            "receiver": ["Y", "Y"],
            "interaction_id": ["i1", "i1"],
            "mode": ["state"] * 2,
            "component": ["sender_resolved_strength"] * 2,
            "component_value": [0.25, 0.75],
            "status": ["observed"] * 2,
            "score_version": [
                "family_first_mechanistic_ligand_contrast_gated_softmin_v2"
            ]
            * 2,
            "certification_status": ["verified_train_only_oof_partial_pipeline"] * 2,
            "is_oof_certified": [False] * 2,
            "formal_inference_status": ["not_available_descriptive_only"] * 2,
            "claim_scope": [
                "heldout_family_common_diagnostic_not_complete_oof_certified"
            ]
            * 2,
            "source_table": ["sender_scores"] * 2,
        }
    )


def test_frozen_real_protocol_loads_roles_and_rejects_endpoint_drift(
    tmp_path: Path,
) -> None:
    protocol, contracts = load_real_e1_config(DEFAULT_CONFIG)

    assert contracts["kuppe"].role == "development_non_independent"
    assert contracts["ms"].role == "reused_locked_external_not_fresh_independent"
    assert protocol["estimator"]["minimum_cells_per_sample_cell_type"] == 10

    changed = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed["spatial_des"]["tie_policy"] = "simultaneous"
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="release boundary changed"):
        load_real_e1_config(changed_path)


def test_real_design_is_subject_level_target_minus_reference() -> None:
    protocol, _ = load_real_e1_config(DEFAULT_CONFIG)
    design = build_v7_real_design(_contract(), protocol["estimator"])

    assert design.design_kind.value == "independent_two_group"
    assert design.condition_levels == ("CTRL", "IZ")
    assert design.subject_column == "subject_id"
    assert design.contrasts[0].weights == (("CTRL", -1.0), ("IZ", 1.0))


def test_legacy_g1_component_projection_preserves_heldout_sample_scores() -> None:
    metadata = pd.DataFrame(
        {
            "sample_id": ["sample-ctrl", "sample-iz"],
            "subject_id": ["subject-ctrl", "subject-iz"],
            "condition": ["CTRL", "IZ"],
        }
    )

    scores = build_legacy_g1_score_view_from_components(
        _legacy_components(),
        metadata,
        dataset_id="Kuppe_MI_CTRL_vs_IZ",
        condition_column="condition",
        contrast_name="IZ_vs_CTRL",
        expected_provenance=_legacy_provenance(),
    )

    assert set(scores["generator_id"]) == {"G1"}
    assert set(scores["score_view"]) == {"primary_sender_resolved_state"}
    assert set(scores["condition"]) == {"CTRL", "IZ"}
    assert scores["out_of_fold"].all()
    assert not scores["outcome_agnostic"].any()
    assert scores["condition_gate_used"].all()
    observed = scores.set_index("sample_id")["score"]
    assert observed.loc["sample-ctrl"] == pytest.approx(0.25)
    assert observed.loc["sample-iz"] == pytest.approx(0.75)


def test_real_inference_matrix_excludes_annotation_only_views() -> None:
    rows = [
        {"generator_id": generator, "score_view": view, "row": index}
        for index, (generator, views) in enumerate(
            real_runner.INFERENCE_SCORE_VIEWS.items()
        )
        for view in views
    ]
    rows.append(
        {
            "generator_id": "G5",
            "score_view": "coupling_prior_annotation",
            "row": 999,
        }
    )
    selected = real_runner._inference_score_views(
        pd.DataFrame.from_records(rows),
        {
            "inference_score_views": {
                key: list(value)
                for key, value in real_runner.INFERENCE_SCORE_VIEWS.items()
            }
        },
    )

    assert len(selected) == 7
    assert "coupling_prior_annotation" not in set(selected["score_view"])


def test_sender_ledger_excludes_g4_and_withholds_formal_inference() -> None:
    ledger = build_sender_resolved_event_ledger(_effects(), _bundle(), _contract())

    assert set(ledger["generator_id"]) == {"G0", "G2", "G3", "G5"}
    assert len(ledger) == 8
    assert not ledger["formal_inference_allowed"].any()
    assert ledger[["p_value", "q_value"]].isna().all(axis=None)
    assert ledger["one_standard_error_stable"].all()
    directions = ledger.set_index(["generator_id", "interaction_id"])["condition"]
    assert directions.loc[("G0", "i1")] == "IZ"
    assert directions.loc[("G0", "i2")] == "CTRL"


def test_sender_ledger_rejects_accidental_formal_fields() -> None:
    effects = _effects()
    effects.loc[effects.index[0], "p_value"] = 0.01

    with pytest.raises(ValueError, match="withheld formal p/q"):
        build_sender_resolved_event_ledger(effects, _bundle(), _contract())


def test_small_des_matrix_covers_continuous_native_and_fixed_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(real_runner, "EVENT_BUDGETS", (2,))
    ledger = build_sender_resolved_event_ledger(_effects(), _bundle(), _contract())

    rankings, scores, coverage = evaluate_v7_real_spatial_des(
        ledger,
        pair_axes=real_runner._pair_axes(("X", "Y", "Z"), _contract()),
        expected=_expected(),
        contract=_contract(),
        resource_id="fixture_resource",
    )

    assert set(scores["generator_id"]) == {"G0", "G2", "G3", "G5"}
    assert set(scores["des_variant"]) == {
        "continuous_weighted_des",
        "diagnostic_one_se_native_count_des",
        "top_k_count_des",
    }
    assert set(scores["status"]) == {"observed"}
    assert len(rankings) == 144
    assert len(scores) == 96
    assert len(coverage) == 96
    top_k = rankings["des_variant"].eq("top_k_count_des")
    assert set(rankings.loc[top_k, "event_budget"]) == {2}


def test_pair_axes_require_two_canonical_cell_types() -> None:
    with pytest.raises(ValueError, match="at least two canonical cell types"):
        real_runner._pair_axes(("X",), _contract())
