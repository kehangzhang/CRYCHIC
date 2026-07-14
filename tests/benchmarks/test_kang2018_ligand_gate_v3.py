from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from benchmarks import run_kang2018_ligand_gate_v3 as kang
from crychic.attribution import ReceptorGatePolicy
from crychic.core import canonical_digest
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)


def _interaction(
    source_id: str,
    ligand: str,
    receptor_name: str,
    receptor_subunits: tuple[str, ...],
) -> Interaction:
    return Interaction(
        interaction_id=f"interaction_{source_id}",
        source_interaction_id=source_id,
        ligand_name=ligand,
        receptor_name=receptor_name,
        ligand_subunits=(ligand,),
        receptor_subunits=receptor_subunits,
        ligand_is_complex=False,
        receptor_is_complex=len(receptor_subunits) > 1,
        direction="Ligand-Receptor",
        source="cellchatdb_v2",
        version="test",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _bundle() -> ResourceBundle:
    interactions = (
        _interaction("CXCL10_CXCR3", "CXCL10", "CXCR3", ("CXCR3",)),
        _interaction(
            "IFNB1_IFNAR1_IFNAR2",
            "IFNB1",
            "IFNAR1_IFNAR2",
            ("IFNAR1", "IFNAR2"),
        ),
        _interaction("CCL5_CCR5", "CCL5", "CCR5", ("CCR5",)),
        _interaction("DECOY_R", "DECOY", "R", ("R",)),
    )
    return ResourceBundle(
        resource_id="cellchatdb_v2",
        version="test",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=interactions,
        mapping_report=MappingReport(
            source_rows=len(interactions),
            loaded_rows=len(interactions),
            mapped_entities=9,
        ),
        manifest_digest="a" * 64,
        source_files=("cellchat/test",),
        license="GPL-3",
        citation="test fixture",
    )


def _definitions() -> list[dict[str, object]]:
    config = kang.load_benchmark_config()
    definitions = cast(
        Sequence[Mapping[str, object]], config["diagnostic_interactions"]
    )
    return [dict(item) for item in definitions]


def _target_prior() -> TargetPrior:
    drivers = ("CCL5", "CXCL10", "IFNB1")
    targets = tuple(
        sorted(f"{driver}_T{rank:02d}" for driver in drivers for rank in range(1, 22))
    )
    target_index = {target: index for index, target in enumerate(targets)}
    indices = tuple(
        target_index[f"{driver}_T{rank:02d}"]
        for driver in drivers
        for rank in range(1, 22)
    )
    return TargetPrior(
        resource_id="nichenet",
        version="test",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=targets,
        driver_ids=drivers,
        indptr=(0, 21, 42, 63),
        target_indices=indices,
        weights=tuple(1.0 / rank for _ in drivers for rank in range(1, 22)),
        ranks=tuple(rank for _ in drivers for rank in range(1, 22)),
        direction=1,
        evidence="synthetic ranked links",
        mapping_report=MappingReport(
            source_rows=63,
            loaded_rows=63,
            mapped_entities=66,
        ),
        manifest_digest="b" * 64,
    )


def test_frozen_interaction_selection_is_exact_and_one_to_one() -> None:
    dataset = kang.load_benchmark_config()["dataset"]
    assert isinstance(dataset, dict)
    assert "subject_ids" not in dataset
    assert dataset["expected_subject_count"] == 8
    selected, manifest = kang.select_diagnostic_interactions(_bundle(), _definitions())

    assert {item.source_interaction_id for item in selected.interactions} == {
        "CXCL10_CXCR3",
        "IFNB1_IFNAR1_IFNAR2",
        "CCL5_CCR5",
    }
    assert [item["diagnostic_id"] for item in manifest] == [
        "CXCL10_CXCR3",
        "IFNB1_IFNAR1_IFNAR2",
        "CCL5_CCR5",
    ]
    assert selected.mapping_report.loaded_rows == 3

    changed = _definitions()
    changed[0]["receptor_subunits"] = ["OTHER"]
    with pytest.raises(ValueError, match="molecular identity changed"):
        kang.select_diagnostic_interactions(_bundle(), changed)


def test_subject_set_manifest_is_nonplaintext_but_not_claimed_anonymous() -> None:
    manifest = kang.subject_set_manifest(("donor-1", "donor-2"))

    assert manifest["n_subjects"] == 2
    assert manifest["privacy_semantics"] == (
        "deterministic_nonplaintext_identifier_for_alignment_not_anonymization"
    )
    assert "donor-1" not in json.dumps(manifest, sort_keys=True)
    assert "donor-2" not in json.dumps(manifest, sort_keys=True)


def test_complete_config_checksum_rejects_any_policy_mutation(
    tmp_path: Path,
) -> None:
    assert kang.sha256_file(kang.DEFAULT_CONFIG) == kang.EXPECTED_CONFIG_SHA256
    frozen = kang.load_benchmark_config()
    _, spec = kang.build_crossfit_spec(
        frozen,
        minimum_effect=0.0,
        receptor_gate_threshold=0.1,
    )
    assert spec.outer_fold_partition_seed == 18021988
    assert spec.autonomous_program_use_scope == "biological_analysis"
    config = json.loads(kang.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["crossfit"]["outer_fold_partition_seed"] += 1
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config checksum mismatch"):
        kang.load_benchmark_config(mutated)


def test_diagnostic_target_prior_selects_three_top20_observable_columns() -> None:
    source = _target_prior()
    definitions = _definitions()
    links = [
        {
            "ligand": ligand,
            "target": f"{ligand}_T{rank:02d}",
            "rank": rank,
            "weight": 1.0 / rank,
        }
        for ligand in source.driver_ids
        for rank in (
            tuple(rank for rank in range(1, 22) if rank != 2)
            if ligand == "CCL5"
            else tuple(range(1, 21))
        )
    ]
    available = {
        *(link["target"] for link in links),
        *(
            str(gene)
            for item in definitions
            for gene in cast(Sequence[object], item["ligand_subunits"])
        ),
        *(
            str(gene)
            for item in definitions
            for gene in cast(Sequence[object], item["receptor_subunits"])
        ),
        "ISG15",
        "HLA-A",
        "ACTB",
    }
    policy = {
        "nichenet_top_h5ad_observable_targets_per_ligand": 20,
        "expected_diagnostic_driver_count": 3,
        "expected_diagnostic_link_count": 60,
        "expected_diagnostic_target_link_digest": canonical_digest(links),
        "model_target_prior_scope": (
            "three_diagnostic_ligands_top20_h5ad_observable_links_each"
        ),
        "pre_specified_gene_sets": {
            "interferon_stimulated": ["ISG15"],
            "antigen_presentation": ["HLA-A"],
            "housekeeping": ["ACTB"],
        },
    }

    diagnostic, manifest = kang.build_diagnostic_target_prior(
        source,
        definitions=definitions,
        gene_subset_config=policy,
        available_genes=sorted(available),
    )

    assert diagnostic.driver_ids == ("CCL5", "CXCL10", "IFNB1")
    assert diagnostic.nnz == 60
    assert "CCL5_T02" not in diagnostic.target_ids
    assert "CCL5_T21" in diagnostic.target_ids
    assert manifest["selected_target_link_digest"] == canonical_digest(links)
    assert manifest["diagnostic_link_count"] == 60
    assert manifest["all_diagnostic_prior_targets_observed_in_h5ad"] is True
    skipped = manifest["unavailable_higher_ranked_nichenet_targets"]
    assert isinstance(skipped, dict)
    assert skipped["CCL5"] == [{"target": "CCL5_T02", "rank": 2, "weight": 0.5}]


def test_input_verification_and_cell_subset_retain_full_gene_axis(
    tmp_path: Path,
) -> None:
    obs = pd.DataFrame(
        {
            "sample_id": ["s:ctrl", "s:ctrl", "s:stim", "s:stim"],
            "subject_id": ["s", "s", "s", "s"],
            "condition": ["ctrl", "ctrl", "stim", "stim"],
            "cell_type": ["A", "B", "A", "B"],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    counts = sparse.csr_matrix(np.arange(1, 13, dtype=np.int32).reshape(4, 3))
    adata = ad.AnnData(X=counts.astype(np.float32), obs=obs)
    adata.var_names = ["G1", "G2", "G3"]
    adata.layers["counts"] = counts
    conversion = {
        "dataset_id": "synthetic",
        "source_archive_sha256": "a" * 64,
        "genes_sha256": "b" * 64,
        "metadata_sha256": "c" * 64,
        "counts_semantics": "raw non-negative integer UMI",
    }
    adata.uns["crychic_conversion"] = conversion
    path = tmp_path / "synthetic.h5ad"
    adata.write_h5ad(path)
    dataset = {
        "expected_h5ad": {
            "bytes": path.stat().st_size,
            "sha256": kang.sha256_file(path),
            "crychic_conversion": conversion,
        },
        "cell_types": ["A", "B"],
        "expected_subject_count": 1,
        "conditions": {"positive": "stim", "negative": "ctrl"},
    }
    gene_scope = {
        "expected_input_gene_count": 3,
        "ann_data_feature_scope": "full_input_transcriptome",
        "library_denominator": "all_input_genes_within_each_training_or_heldout_fold",
    }

    verification, genes = kang.verify_input_artifact(
        path,
        dataset_config=dataset,
        expected_gene_count=3,
    )
    subset, audit = kang._prepare_input(
        path,
        dataset_config=dataset,
        gene_scope_config=gene_scope,
        minimum_cells=1,
    )

    assert verification["checksum_verified"] is True
    assert genes == ("G1", "G2", "G3")
    assert subset.shape == (4, 3)
    assert (
        audit["normalization_scope"][  # type: ignore[index]
            "all_input_genes_retained_before_fold_normalization"
        ]
        is True
    )


def test_compact_paired_summary_keeps_status_but_not_subject_values() -> None:
    table = pd.DataFrame(
        [
            {
                "receiver": "CD8 T cells",
                "interaction_id": "cxcl10",
                "mode": "state",
                "subject_id": subject,
                "context_id": context,
                "score": value,
                "status": status,
                "reason_code": reason,
            }
            for subject, context, value, status, reason in (
                ("s1", "stim-id", 0.8, "observed", None),
                ("s1", "ctrl-id", 0.2, "observed", None),
                ("s2", "stim-id", 0.4, "observed", None),
                ("s2", "ctrl-id", 0.1, "observed", None),
                ("s3", "stim-id", None, "not_estimable", "missing_parent"),
                ("s3", "ctrl-id", None, "not_estimable", "missing_parent"),
            )
        ]
    )

    records = kang.paired_effect_summary(
        table,
        group_columns=("receiver", "interaction_id", "mode"),
        value_column="score",
        positive_context_id="stim-id",
        negative_context_id="ctrl-id",
    )

    assert len(records) == 1
    record = records[0]
    assert record["n_complete_subjects"] == 2
    assert record["mean_paired_effect"] == pytest.approx(0.45)
    assert record["status_counts"] == {"not_estimable": 2, "observed": 4}
    assert record["reason_counts"] == {"missing_parent": 2, "none": 4}
    assert "subject_ids" not in record
    assert "paired_effects" not in record


def test_receptor_gate_records_require_exact_diagnostic_key_coverage() -> None:
    interaction_manifest = [
        {
            "diagnostic_id": item["diagnostic_id"],
            "harmonized_interaction_id": f"interaction_{item['diagnostic_id']}",
        }
        for item in _definitions()
    ]
    driver_by_interaction = tuple(
        sorted(
            (
                str(item["harmonized_interaction_id"]),
                str(item["diagnostic_id"]).split("_", maxsplit=1)[0],
            )
            for item in interaction_manifest
        )
    )
    source_basis = SimpleNamespace(
        driver_ids=("CCL5", "CXCL10", "IFNB1"),
        receptor_eligible=np.asarray([False, True, False]),
        receptor_gate_threshold=0.1,
        gate_policy=ReceptorGatePolicy.HARD_ELIGIBILITY_V2,
    )
    receiver_artifact = SimpleNamespace(
        receiver="CD8 T cells",
        driver_by_interaction=driver_by_interaction,
        receptor_gates=(("CCL5", 0.01), ("CXCL10", 0.2), ("IFNB1", 0.05)),
        source_basis=source_basis,
        receptor_gate_manifest_id="gate-manifest",
        receptor_evidence_digest="evidence-digest",
    )
    model = SimpleNamespace(receiver_family_artifact=receiver_artifact)
    fold = SimpleNamespace(
        fold_id="fold-1",
        training=SimpleNamespace(
            training_subject_ids=("s1", "s2"),
            cell_type_ids=("CD8 T cells",),
        ),
        application=SimpleNamespace(heldout_subject_ids=("s3", "s4")),
        receiver_family_models=(model,),
    )
    artifacts = SimpleNamespace(folds=(fold,))

    records = kang.compact_receptor_gate_records(
        artifacts,
        interaction_manifest=interaction_manifest,
    )

    assert len(records) == 3
    cxcl10 = next(row for row in records if row["diagnostic_id"] == "CXCL10_CXCR3")
    assert cxcl10["receptor_gate"] == pytest.approx(0.2)
    assert cxcl10["receptor_gate_threshold"] == pytest.approx(0.1)
    assert cxcl10["receptor_eligible"] is True
    assert cxcl10["receptor_gate_manifest_id"] == "gate-manifest"
    assert cxcl10["receptor_evidence_digest"] == "evidence-digest"

    fold.receiver_family_models = (model, model)
    with pytest.raises(ValueError, match="exact fold-receiver-interaction"):
        kang.compact_receptor_gate_records(
            artifacts,
            interaction_manifest=interaction_manifest,
        )


def test_cli_writes_compact_json_and_forwards_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run_benchmark(**kwargs: Any) -> dict[str, object]:
        observed.update(kwargs)
        return {"schema_version": kang.SCHEMA_VERSION, "claims": kang.CLAIMS}

    monkeypatch.setattr(kang, "run_benchmark", fake_run_benchmark)
    output = tmp_path / "nested/result.json"
    exit_code = kang.main(
        (
            "--workspace-root",
            str(tmp_path),
            "--output",
            str(output),
            "--minimum-effect",
            "0.02",
            "--receptor-gate-threshold",
            "0.01",
        )
    )

    assert exit_code == 0
    assert observed["workspace_root"] == tmp_path.resolve()
    assert observed["minimum_effect"] == 0.02
    assert observed["receptor_gate_threshold"] == 0.01
    assert json.loads(output.read_text(encoding="utf-8"))["claims"] == kang.CLAIMS
