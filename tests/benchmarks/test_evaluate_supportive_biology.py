from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]
from benchmarks.metrics.evaluate_supportive_biology import (
    EVIDENCE_CLASSES,
    REPORT_SUPPORT_STATUSES,
    _validated_frozen_edge_ids,
    effect_table_from_external_long,
    effect_table_from_score_table,
    evaluate_supportive_biology,
    load_biology_mappings,
    load_supportive_biology,
    read_external_long_for_effects,
    write_biology_support,
)


def _truth(observations: list[dict[str, object]]) -> dict[str, object]:
    return {
        "truth_set_id": "locked-test-v1",
        "locked_at": "2026-07-12T00:00:00+08:00",
        "datasets": {
            "test_dataset": {
                "primary_contrast": "Tumor_vs_Normal",
                "expected_observations": observations,
            }
        },
        "evaluation_policy": {
            "preregistered_before_method_outputs": True,
            "real_data_is_not_edge_level_ground_truth": True,
            "do_not_compute_real_data_edge_auroc": True,
            "resource_absence_is_not_method_failure": True,
        },
    }


def _observation(
    identifier: str,
    *,
    interactions: list[list[str]] | None = None,
    sender: str = "A",
    receiver: str = "B",
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": identifier,
        "direction": "Tumor_up",
        "sender": sender,
        "receiver": receiver,
        "metric": "locked_rank_and_direction",
    }
    if interactions is not None:
        result["interactions"] = interactions
    return result


def _effect_rows() -> pd.DataFrame:
    identity: dict[str, object] = {
        "method": "method-a",
        "method_version": "1",
        "analysis_track": "lr_stlr",
        "resource": "resource-a",
        "resource_version": "1",
        "resource_mode": "H-common",
        "score_semantics": "rank_strength",
        "universe_id": "universe-a",
    }
    rows = [
        ("A", "B", "i1", "L1", "R1", 10.0, "exploratory"),
        ("A", "B", "i2", "L2", "R2", 1.0, "exploratory"),
        ("A", "B", "i3", "LX", "RX", 0.0, "exploratory"),
        ("A", "B", "i4", "L4", "R4", -5.0, "exploratory"),
        ("A", "B", "i5", "L3", "R3", None, "not_estimable"),
        ("A", "C", "i6", "LC", "RC", 8.0, "exploratory"),
        ("D", "E", "i7", "LD", "RD", -1.0, "exploratory"),
    ]
    return pd.DataFrame(
        [
            identity
            | {
                "sender": sender,
                "receiver": receiver,
                "interaction_id": interaction,
                "ligand": ligand,
                "receptor": receptor,
                "effect": effect,
                "status": status,
            }
            for sender, receiver, interaction, ligand, receptor, effect, status in rows
        ]
    )


def test_locked_lr_evidence_classes_and_report_mapping() -> None:
    observations = [
        _observation("exact", interactions=[["L1", "R1"]]),
        _observation(
            "partial",
            interactions=[["L1", "R1"], ["L2", "R2"]],
        ),
        _observation("opposite", interactions=[["L4", "R4"]]),
        _observation("not_estimable", interactions=[["L3", "R3"]]),
        _observation("not_covered", interactions=[["L9", "R9"]]),
    ]

    result = evaluate_supportive_biology(
        _effect_rows(),
        _truth(observations),
        dataset_id="test_dataset",
        reference="Normal",
        target="Tumor",
        top_fraction=0.25,
    ).set_index("observation_id")

    assert set(result["evidence_class"]).issubset(EVIDENCE_CLASSES)
    assert set(result["support_status"]).issubset(REPORT_SUPPORT_STATUSES)
    assert result.loc["exact", "evidence_class"] == "exact"
    assert result.loc["exact", "support_status"] == "supported"
    assert result.loc["partial", "evidence_class"] == "partial"
    assert result.loc["partial", "n_components_strong"] == 1
    assert result.loc["partial", "n_components_directional"] == 1
    assert result.loc["opposite", "evidence_class"] == "not_supported"
    assert result.loc["opposite", "support_status"] == "discordant"
    assert result.loc["not_estimable", "evidence_class"] == "not_estimable"
    assert result.loc["not_estimable", "support_status"] == "not_estimable"
    assert result.loc["not_covered", "evidence_class"] == "not_estimable"
    assert result.loc["not_covered", "support_status"] == "not_covered"
    assert "AUROC" not in " ".join(result["evidence_note"])


def test_network_observation_uses_locked_dyads() -> None:
    truth = _truth(
        [
            {
                "id": "hub",
                "direction": "Tumor_up",
                "sender": "A",
                "receivers": ["B", "C"],
                "metric": "network_rank",
            }
        ]
    )

    row = evaluate_supportive_biology(
        _effect_rows(),
        truth,
        dataset_id="test_dataset",
        reference="Normal",
        target="Tumor",
        network_top_fraction=0.5,
    ).iloc[0]

    assert row["evidence_class"] == "exact"
    assert row["n_components_expected"] == 2
    evidence = json.loads(row["component_evidence_json"])
    assert {item["component_id"] for item in evidence} == {
        "network:A->B",
        "network:A->C",
    }


def test_explicit_aliases_do_not_add_observations(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        yaml.safe_dump(
            {
                "cell_types": {"Endo": "Endothelial Cell"},
                "genes": {"VEGFR1": "FLT1"},
            }
        ),
        encoding="utf-8",
    )
    table = _effect_rows().iloc[[0]].copy()
    table["receiver"] = "Endo"
    table["receptor"] = "VEGFR1"
    truth = _truth(
        [
            _observation(
                "mapped",
                interactions=[["L1", "FLT1"]],
                receiver="Endothelial Cell",
            )
        ]
    )

    result = evaluate_supportive_biology(
        table,
        truth,
        dataset_id="test_dataset",
        reference="Normal",
        target="Tumor",
        mappings=load_biology_mappings(mapping_path),
    )

    assert result["observation_id"].tolist() == ["mapped"]
    assert result.loc[0, "evidence_class"] == "exact"
    assert str(result.loc[0, "mapping_id"]).startswith("biology_mapping_")


def test_sparse_effect_table_does_not_confuse_cell_pair_with_resource_absence() -> None:
    table = _effect_rows().iloc[[0]].copy()
    truth = _truth(
        [
            _observation(
                "different_pair",
                interactions=[["L1", "R1"]],
                sender="A",
                receiver="C",
            )
        ]
    )

    row = evaluate_supportive_biology(
        table,
        truth,
        dataset_id="test_dataset",
        reference="Normal",
        target="Tumor",
    ).iloc[0]

    assert row["evidence_class"] == "not_estimable"
    assert row["support_status"] == "not_estimable"
    component = json.loads(row["component_evidence_json"])[0]
    assert component["covered"] is True
    assert component["reason_code"] == "locked_component_cell_pair_not_estimable"


def test_mapping_config_rejects_posthoc_observations(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump({"observations": {"new_finding": {"direction": "Tumor_up"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only cell_types and genes"):
        load_biology_mappings(path)


def test_load_truth_requires_real_data_guardrails(tmp_path: Path) -> None:
    truth = _truth([_observation("one", interactions=[["L1", "R1"]])])
    truth["evaluation_policy"]["do_not_compute_real_data_edge_auroc"] = False  # type: ignore[index]
    path = tmp_path / "truth.yaml"
    path.write_text(yaml.safe_dump(truth), encoding="utf-8")

    with pytest.raises(ValueError, match="do_not_compute_real_data_edge_auroc"):
        load_supportive_biology(path)


def _score_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in ("s1", "s2", "s3"):
        for context, score_one, score_two in (
            ("Normal", 0.1, 0.9),
            ("Tumor", 0.9, 0.1),
        ):
            for interaction, ligand, receptor, score in (
                ("i1", "L1", "R1", score_one),
                ("i2", "L2", "R2", score_two),
            ):
                rows.append(
                    {
                        "dataset": "test_dataset",
                        "method": "method-a",
                        "method_version": "1",
                        "analysis_track": "lr_stlr",
                        "resource": "resource-a",
                        "resource_version": "1",
                        "resource_mode": "H-common",
                        "score_semantics": "magnitude",
                        "universe_id": "u1",
                        "sample_id": f"{subject}_{context}",
                        "subject_id": subject,
                        "context": context,
                        "contrast": "Tumor_vs_Normal",
                        "sender": "A",
                        "receiver": "B",
                        "interaction_id": interaction,
                        "ligand": ligand,
                        "receptor": receptor,
                        "score": score,
                        "score_direction": "higher",
                        "status": "observed",
                        "universe_member": True,
                        "universe_size": 2,
                    }
                )
    return pd.DataFrame(rows)


def test_score_table_conversion_preserves_paired_subject_unit() -> None:
    effects = effect_table_from_score_table(
        _score_table(),
        reference="Normal",
        target="Tumor",
        design="paired",
        min_support=3,
        contrast="Tumor_vs_Normal",
    )

    first = effects.loc[effects["interaction_id"].eq("i1")].iloc[0]
    second = effects.loc[effects["interaction_id"].eq("i2")].iloc[0]
    assert first["effect"] > 0
    assert second["effect"] < 0
    assert first["n_pairs"] == 3
    assert first["status"] == "exploratory"


def test_external_track_b_retains_source_agnostic_semantics() -> None:
    score = _score_table()
    rows = pd.DataFrame(
        {
            "run_id": "target-run-1",
            "dataset_id": score["dataset"],
            "method_id": "nichenet_prior_activity",
            "method_version": "1",
            "analysis_track": "ligand_target_program",
            "resource_id": "nichenet",
            "resource_version": "1",
            "resource_mode": "native",
            "universe_id": "target-u1",
            "universe_member": True,
            "universe_size": 2,
            "sample_id": score["sample_id"],
            "subject_id": score["subject_id"],
            "context_json": score["context"].map(
                lambda value: json.dumps({"condition": value})
            ),
            "sender": "__source_agnostic__",
            "receiver": score["receiver"],
            "interaction_id": score["interaction_id"],
            "ligand": score["ligand"],
            "receptor": "__target_program__",
            "score": score["score"],
            "score_name": "ligand_target_program_proxy",
            "score_direction": "higher",
            "status": "ok",
        }
    )

    effects = effect_table_from_external_long(
        rows,
        context_key="condition",
        reference="Normal",
        target="Tumor",
        design="paired",
        min_support=3,
    )

    assert set(effects["analysis_track"]) == {"ligand_target_program"}
    assert set(effects["sender"]) == {"__source_agnostic__"}
    assert effects.loc[effects["interaction_id"].eq("i1"), "effect"].iloc[0] > 0

    biology = evaluate_supportive_biology(
        effects,
        _truth([_observation("reported_lr", interactions=[["L1", "R1"]])]),
        dataset_id="test_dataset",
        reference="Normal",
        target="Tumor",
    ).iloc[0]
    assert biology["evidence_class"] == "not_estimable"
    assert biology["support_status"] == "not_estimable"
    component = json.loads(biology["component_evidence_json"])[0]
    assert component["reason_code"] == (
        "track_b_does_not_estimate_lr_or_sender_identity"
    )


def test_external_long_keeps_unestimable_edges_explicit() -> None:
    score = _score_table()
    rows = pd.DataFrame(
        {
            "run_id": "run-1",
            "dataset_id": score["dataset"],
            "method_id": "method-a",
            "method_version": "1",
            "analysis_track": "lr_stlr",
            "resource_id": "resource-a",
            "resource_version": "1",
            "resource_mode": "H-common",
            "universe_id": "u1",
            "universe_member": True,
            "universe_size": 2,
            "sample_id": score["sample_id"],
            "subject_id": score["subject_id"],
            "context_json": score["context"].map(
                lambda value: json.dumps({"condition": value})
            ),
            "sender": score["sender"],
            "receiver": score["receiver"],
            "interaction_id": score["interaction_id"],
            "ligand": score["ligand"],
            "receptor": score["receptor"],
            "score": score["score"],
            "score_name": "magnitude",
            "score_direction": "higher",
            "status": "ok",
        }
    )
    unavailable = rows["interaction_id"].eq("i2")
    rows.loc[unavailable, "status"] = "insufficient_cells"
    rows.loc[unavailable, "score"] = None

    effects = effect_table_from_external_long(
        rows,
        context_key="condition",
        reference="Normal",
        target="Tumor",
        design="paired",
        min_support=3,
    ).set_index("interaction_id")

    assert effects.loc["i1", "status"] == "exploratory"
    assert effects.loc["i2", "status"] == "not_estimable"
    assert np.isnan(float(str(effects.loc["i2", "effect"])))

    rows.loc[unavailable, "status"] = "unsupported_resource"
    unsupported = effect_table_from_external_long(
        rows,
        context_key="condition",
        reference="Normal",
        target="Tumor",
        design="paired",
        min_support=3,
    ).set_index("interaction_id")
    assert unsupported.loc["i2", "status"] == "not_supported"


def test_external_long_reader_projects_unused_native_columns(tmp_path: Path) -> None:
    score = _score_table()
    table = pd.DataFrame(
        {
            "run_id": "run-1",
            "dataset_id": score["dataset"],
            "method_id": "method-a",
            "method_version": "1",
            "analysis_track": "lr_stlr",
            "resource_mode": "native",
            "resource_id": "native-resource",
            "resource_version": "1",
            "universe_id": "native-u1",
            "universe_member": True,
            "universe_size": 2,
            "sample_id": score["sample_id"],
            "subject_id": score["subject_id"],
            "context_json": score["context"].map(
                lambda value: json.dumps({"condition": value})
            ),
            "sender": score["sender"],
            "receiver": score["receiver"],
            "interaction_id": score["interaction_id"],
            "ligand": score["ligand"],
            "receptor": score["receptor"],
            "target": None,
            "score": score["score"],
            "score_name": "magnitude",
            "score_direction": "higher",
            "rank": score.groupby(["sample_id"])["score"].rank(
                ascending=False, method="average"
            ),
            "status": "ok",
            "large_unused_payload": ["x" * 1000] * len(score),
        }
    )
    path = tmp_path / "native_long.parquet"
    second_run = table.copy()
    second_run["run_id"] = "run-2"
    pd.concat([table, second_run], ignore_index=True).to_parquet(path, index=False)

    projected = read_external_long_for_effects(path, run_id="run-1")

    assert "large_unused_payload" not in projected
    assert len(projected) == len(table)
    assert set(projected["run_id"]) == {"run-1"}
    assert set(projected.columns) == {
        "run_id",
        "dataset_id",
        "method_id",
        "method_version",
        "analysis_track",
        "resource_mode",
        "resource_id",
        "resource_version",
        "universe_id",
        "universe_member",
        "universe_size",
        "sample_id",
        "subject_id",
        "context_json",
        "sender",
        "receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "target",
        "score",
        "score_name",
        "score_direction",
        "rank",
        "status",
    }


def test_frozen_edge_ids_fall_back_for_different_sample_order() -> None:
    working = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "interaction_id": ["i1", "i2", "i2", "i1"],
        }
    )
    sample_codes, labels = pd.factorize(working["sample_id"], sort=False)

    edge_ids, universe = _validated_frozen_edge_ids(
        working,
        edge_columns=["interaction_id"],
        sample_codes=sample_codes,
        sample_count=len(labels),
        declared_size=2,
    )

    by_interaction = dict(
        zip(universe["interaction_id"], universe["_edge_id"], strict=True)
    )
    assert edge_ids.tolist() == [
        by_interaction["i1"],
        by_interaction["i2"],
        by_interaction["i2"],
        by_interaction["i1"],
    ]

    duplicate = working.copy()
    duplicate.loc[3, "interaction_id"] = "i2"
    with pytest.raises(ValueError, match="duplicate sample-edge"):
        _validated_frozen_edge_ids(
            duplicate,
            edge_columns=["interaction_id"],
            sample_codes=sample_codes,
            sample_count=len(labels),
            declared_size=2,
        )


def test_write_report_ready_table(tmp_path: Path) -> None:
    evaluated = evaluate_supportive_biology(
        _effect_rows(),
        _truth([_observation("one", interactions=[["L1", "R1"]])]),
        dataset_id="test_dataset",
        reference="Normal",
        target="Tumor",
        top_fraction=0.25,
    )
    output = write_biology_support(evaluated, tmp_path / "biology_support.tsv")

    persisted = pd.read_csv(output, sep="\t")
    assert persisted["observation_id"].tolist() == ["one"]
    assert persisted["support_status"].tolist() == ["supported"]
