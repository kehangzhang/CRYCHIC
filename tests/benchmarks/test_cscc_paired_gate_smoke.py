from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from benchmarks import run_cscc_paired_gate_smoke as cscc
from crychic.core import canonical_digest
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)


def _definitions() -> list[Mapping[str, object]]:
    config = cscc.load_smoke_config()
    values = cast(Sequence[Mapping[str, object]], config["diagnostic_interactions"])
    return list(values)


def _cell_obs(*, cells_per_stratum: int = 3) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    cell_ids: list[str] = []
    for subject in ("s1", "s2"):
        for condition in ("Normal", "Tumor"):
            for cell_type in ("CD1C", "Epithelial"):
                for index in range(cells_per_stratum):
                    cell_ids.append(f"{subject}_{condition}_{cell_type}_{index}")
                    rows.append(
                        {
                            "sample_id": f"{subject}_{condition}",
                            "subject_id": subject,
                            "condition": condition,
                            "cell_type": cell_type,
                        }
                    )
    return pd.DataFrame(rows, index=cell_ids)


def _expected_capped_ids(obs: pd.DataFrame, *, cap: int) -> set[str]:
    selected: set[str] = set()
    for _, group in obs.groupby(
        ["subject_id", "condition", "cell_type"], observed=True, sort=True
    ):
        ranked = sorted(
            map(str, group.index),
            key=lambda cell_id: (
                hashlib.sha256(cell_id.encode("utf-8")).digest(),
                cell_id,
            ),
        )
        selected.update(ranked[:cap])
    return selected


def test_config_builds_explicit_two_fold_tumor_minus_normal_spec() -> None:
    config = cscc.load_smoke_config()
    crychic_config, spec = cscc.build_crossfit_spec(config)

    assert crychic_config.context_keys == ("condition",)
    assert crychic_config.counts_layer == "counts"
    assert spec.allowed_n_splits == (2,)
    assert spec.outer_fold_partition_seed == 1442362020
    assert spec.outer_fold_partition_seed != crychic_config.random_seed
    assert spec.autonomous_program_use_scope == "biological_analysis"
    assert spec.min_train_subjects_per_context == 4
    assert spec.min_test_subjects_per_context == 4
    assert spec.training_spec.sender_parameters.min_subjects == 4
    assert spec.contrasts[0].name == "tumor_vs_normal"
    assert dict(spec.contrasts[0].weights) == {"Normal": -1.0, "Tumor": 1.0}
    assert spec.penalty_tuning_spec is not None
    assert spec.penalty_tuning_spec.lambda1_fractions == (1.0, 0.1)
    assert spec.penalty_tuning_spec.lambda2_fractions == (0.0,)


def test_sha256_cell_cap_is_order_independent_and_keeps_full_gene_axis(
    tmp_path: Path,
) -> None:
    obs = _cell_obs()
    positions, audit = cscc.deterministic_cell_cap(
        obs,
        cell_types=("CD1C", "Epithelial"),
        conditions=("Normal", "Tumor"),
        minimum_cells=2,
        cap=2,
    )
    selected_ids = set(map(str, obs.index[positions]))
    expected_ids = _expected_capped_ids(obs, cap=2)

    assert selected_ids == expected_ids
    assert audit["n_subjects"] == 2
    assert audit["n_strata"] == 8
    assert audit["n_selected_cells"] == 16
    assert audit["selected_cell_id_digest"] == canonical_digest(sorted(expected_ids))

    reversed_obs = obs.iloc[::-1].copy()
    reversed_positions, reversed_audit = cscc.deterministic_cell_cap(
        reversed_obs,
        cell_types=("CD1C", "Epithelial"),
        conditions=("Normal", "Tumor"),
        minimum_cells=2,
        cap=2,
    )
    assert set(map(str, reversed_obs.index[reversed_positions])) == expected_ids
    assert reversed_audit["selected_cell_id_digest"] == audit["selected_cell_id_digest"]

    counts = sparse.csr_matrix(
        np.tile(np.asarray([[1, 2, 3]], dtype=np.int32), (len(obs), 1))
    )
    adata = ad.AnnData(X=counts.astype(np.float32), obs=obs.copy())
    adata.var_names = ["G1", "G2", "G3"]
    adata.layers["counts"] = counts
    path = tmp_path / "paired.h5ad"
    adata.write_h5ad(path)
    dataset = {
        "expected_h5ad": {
            "bytes": path.stat().st_size,
            "sha256": cscc.sha256_file(path),
            "shape": [len(obs), 3],
        },
        "cell_types": ["CD1C", "Epithelial"],
        "conditions": {"positive": "Tumor", "negative": "Normal"},
        "minimum_cells_per_stratum": 2,
        "cell_cap_per_stratum": 2,
        "expected_subject_count": 2,
        "expected_condition_count": 2,
        "expected_cell_type_count": 2,
        "expected_stratum_count": 8,
        "expected_selected_cell_count": 16,
        "expected_selected_cell_id_digest": canonical_digest(sorted(expected_ids)),
    }

    subset, genes, input_audit = cscc.prepare_input(path, dataset_config=dataset)

    assert subset.shape == (16, 3)
    assert genes == ("G1", "G2", "G3")
    assert input_audit["normalization_denominator_gene_count"] == 3
    assert input_audit["full_gene_axis_retained_for_fold_cpm"] is True
    assert subset.layers["counts"].dtype == np.dtype("int32")


def _harmonized_bundle() -> ResourceBundle:
    config = cscc.load_smoke_config()
    resource = cast(Mapping[str, object], config["harmonized_resource"])
    interactions = []
    for definition in _definitions():
        ligand = str(definition["ligand"])
        receptor = str(definition["receptor"])
        interaction_id = str(definition["interaction_id"])
        interactions.append(
            Interaction(
                interaction_id=interaction_id,
                source_interaction_id=interaction_id,
                ligand_name=ligand,
                receptor_name=receptor,
                ligand_subunits=(ligand,),
                receptor_subunits=(receptor,),
                ligand_is_complex=False,
                receptor_is_complex=False,
                direction="Ligand-Receptor",
                source=str(resource["resource_id"]),
                version=str(resource["version"]),
                species=Species.HUMAN,
                gene_namespace=GeneNamespace.HGNC_SYMBOL,
                evidence=(
                    f"cellchat:{definition['cellchat_source_interaction_id']}",
                    f"cellphonedb:{definition['cellphonedb_source_interaction_id']}",
                ),
            )
        )
    interactions.append(
        Interaction(
            interaction_id="decoy",
            source_interaction_id="decoy",
            ligand_name="DECOY_L",
            receptor_name="DECOY_R",
            ligand_subunits=("DECOY_L",),
            receptor_subunits=("DECOY_R",),
            ligand_is_complex=False,
            receptor_is_complex=False,
            direction="Ligand-Receptor",
            source=str(resource["resource_id"]),
            version=str(resource["version"]),
            species=Species.HUMAN,
            gene_namespace=GeneNamespace.HGNC_SYMBOL,
        )
    )
    return ResourceBundle(
        resource_id=str(resource["resource_id"]),
        version=str(resource["version"]),
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=tuple(interactions),
        mapping_report=MappingReport(
            source_rows=len(interactions),
            loaded_rows=len(interactions),
            mapped_entities=2 * len(interactions),
        ),
        manifest_digest="a" * 64,
        source_files=("harmonized_lr.tsv", "manifest.json"),
        license="test",
        citation="test",
    )


def test_harmonized_anchor_selection_preserves_exact_loader_semantics() -> None:
    selected, manifest = cscc.select_anchor_interactions(
        _harmonized_bundle(),
        _definitions(),
        expected_id_digest=cscc.EXPECTED_INTERACTION_ID_DIGEST,
        expected_fields_digest=cscc.EXPECTED_INTERACTION_FIELDS_DIGEST,
    )

    assert len(selected.interactions) == 14
    assert selected.mapping_report.loaded_rows == 14
    assert selected.mapping_report.mapped_entities == 17
    assert len(manifest) == 14
    assert all(len(item.evidence) == 2 for item in selected.interactions)
    assert all(
        item.source_interaction_id == item.interaction_id
        for item in selected.interactions
    )

    changed = [dict(item) for item in _definitions()]
    changed[0]["cellphonedb_source_interaction_id"] = "changed"
    with pytest.raises(ValueError, match="molecular identity changed"):
        cscc.select_anchor_interactions(
            _harmonized_bundle(),
            changed,
            expected_id_digest=cscc.EXPECTED_INTERACTION_ID_DIGEST,
            expected_fields_digest=cscc.EXPECTED_INTERACTION_FIELDS_DIGEST,
        )


def _target_prior_fixture() -> tuple[TargetPrior, list[dict[str, object]]]:
    drivers = tuple(sorted(str(item["ligand"]) for item in _definitions()))
    first_targets = [f"SHARED_{index:02d}" for index in range(20)]
    targets_by_driver: dict[str, list[str]] = {drivers[0]: first_targets}
    for driver in drivers[1:-1]:
        targets_by_driver[driver] = [
            *first_targets[:8],
            *(f"{driver}_UNIQUE_{index:02d}" for index in range(12)),
        ]
    targets_by_driver[drivers[-1]] = [
        *first_targets[:13],
        *(f"{drivers[-1]}_UNIQUE_{index:02d}" for index in range(7)),
    ]
    target_ids = tuple(
        sorted({gene for genes in targets_by_driver.values() for gene in genes})
    )
    target_index = {target: index for index, target in enumerate(target_ids)}
    indices: list[int] = []
    weights: list[float] = []
    ranks: list[int] = []
    indptr = [0]
    links: list[dict[str, object]] = []
    for driver in drivers:
        for rank, target in enumerate(targets_by_driver[driver], start=1):
            weight = 1.0 / rank
            indices.append(target_index[target])
            weights.append(weight)
            ranks.append(rank)
            links.append(
                {"ligand": driver, "target": target, "rank": rank, "weight": weight}
            )
        indptr.append(len(indices))
    prior = TargetPrior(
        resource_id="nichenet",
        version="test",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=target_ids,
        driver_ids=drivers,
        indptr=tuple(indptr),
        target_indices=tuple(indices),
        weights=tuple(weights),
        ranks=tuple(ranks),
        direction=1,
        evidence="synthetic ranked links",
        mapping_report=MappingReport(
            source_rows=len(indices),
            loaded_rows=len(indices),
            mapped_entities=len(drivers) + len(target_ids),
        ),
        manifest_digest="b" * 64,
    )
    return prior, links


def test_target_prior_freezes_fourteen_by_twenty_links_and_171_targets() -> None:
    source, links = _target_prior_fixture()
    diagnostic, manifest = cscc.build_smoke_target_prior(
        source,
        definitions=_definitions(),
        available_genes=source.target_ids,
        top_n=20,
        expected_driver_count=14,
        expected_link_count=280,
        expected_target_count=171,
        expected_link_digest=canonical_digest(links),
    )

    assert len(diagnostic.driver_ids) == 14
    assert diagnostic.nnz == 280
    assert len(diagnostic.target_ids) == 171
    assert manifest["selected_target_link_digest"] == canonical_digest(links)
    assert diagnostic.manifest_digest == source.manifest_digest

    with pytest.raises(ValueError, match="link identity changed"):
        cscc.build_smoke_target_prior(
            source,
            definitions=_definitions(),
            available_genes=source.target_ids,
            top_n=20,
            expected_driver_count=14,
            expected_link_count=280,
            expected_target_count=171,
            expected_link_digest="0" * 64,
        )


def test_fold_support_summary_deduplicates_exact_rows_and_rejects_conflicts() -> None:
    base = {
        "receiver": "CD1C",
        "diagnostic_id": "CCL3_CCR5",
        "interaction_id": "anchor",
        "status": "supported",
        "reason_code": None,
        "ligand_contrast_gate": True,
    }
    fold_1 = {"fold_id": "fold-1", **base}
    fold_2 = {"fold_id": "fold-2", **base}

    summary = cscc.summarize_fold_supports((fold_1, fold_1.copy(), fold_2))

    assert summary == [
        {
            "receiver": "CD1C",
            "diagnostic_id": "CCL3_CCR5",
            "interaction_id": "anchor",
            "n_unique_folds": 2,
            "support_status_counts": {"supported": 2},
            "support_reason_counts": {"none": 2},
            "ligand_contrast_gate_counts": {"True": 2},
        }
    ]

    conflicting = dict(fold_1)
    conflicting["status"] = "not_estimable"
    with pytest.raises(ValueError, match="conflicting duplicate"):
        cscc.summarize_fold_supports((fold_1, conflicting))


def test_diagnostic_score_summary_is_paired_deidentified_and_noncertifying() -> None:
    interactions = [
        {"interaction_id": "i1", "ligand": "L1", "receptor": "R1"},
        {"interaction_id": "i2", "ligand": "L2", "receptor": "R2"},
    ]
    rows = []
    for subject_index in range(1, 4):
        subject = f"p{subject_index}"
        for context in ("Normal", "Tumor"):
            for mode in ("state", "ecosystem"):
                for interaction_id in ("i1", "i2"):
                    positive = interaction_id == "i1"
                    score = (
                        0.9
                        if positive and context == "Tumor"
                        else 0.1
                        if positive
                        else 0.2
                        if context == "Tumor"
                        else 0.8
                    )
                    rows.append(
                        {
                            "sample_id": f"{subject}_{context}",
                            "subject_id": subject,
                            "context_id": str(
                                cscc.context_id(
                                    {"condition": context}, ("condition",)
                                )
                            ),
                            "sender": "Sender",
                            "receiver": "Receiver",
                            "interaction_id": interaction_id,
                            "mode": mode,
                            "sender_resolved_strength": score,
                            "status": "ok",
                        }
                    )
    application = SimpleNamespace(sender_scores=pd.DataFrame(rows))
    selected = SimpleNamespace(lambda1_fraction=0.1, lambda2_fraction=0.0)
    tuning = SimpleNamespace(selected_candidate=selected)
    model = SimpleNamespace(
        penalty_tuning_artifact=tuning,
        receiver="Receiver",
        diagnostic_status="observed",
        official_incremental_status="not_estimable",
    )
    fold = SimpleNamespace(
        fold_id="fold-1",
        family_common_applications=(application,),
        receiver_incremental_models=(model,),
    )
    artifacts = cast(cscc.CrossFitArtifacts, SimpleNamespace(folds=(fold,)))

    summary = cscc.compact_diagnostic_score_summary(
        artifacts,
        interaction_manifest=interactions,
        resource_id="resource",
        resource_version="1",
    )

    assert summary["scope"] == (
        "exploratory_unadjusted_noncertified_paired_rank_effects"
    )
    assert summary["inferential_fields_available"] == []
    assert summary["raw_sample_and_subject_rows_exported"] is False
    assert len(cast(list[object], summary["paired_effects"])) == 4
    assert {row["n_pairs"] for row in summary["paired_effects"]} == {3}
    assert {row["status"] for row in summary["paired_effects"]} == {"exploratory"}
    assert all("subject_id" not in row for row in summary["paired_effects"])
    assert summary["selected_penalties"] == [
        {
            "fold_id": "fold-1",
            "receiver": "Receiver",
            "diagnostic_status": "observed",
            "official_status": "not_estimable",
            "selected_lambda1_fraction": 0.1,
            "selected_lambda2_fraction": 0.0,
        }
    ]
