from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import benchmarks.adapters.crychic.run_kuppe_ctrl_iz as module
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.crychic.des_postprocess import heldout_sample_coverage_audit
from benchmarks.adapters.crychic.postprocess_kuppe_spatial_des import (
    run as postprocess_spatial_des,
)
from scipy import sparse

from crychic import CrychicConfig
from crychic.resources import (
    GeneNamespace,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.workflow import CrossFitSpec


def _write_connectomedb_fixture(tmp_path: Path) -> tuple[Path, Path]:
    records: list[dict[str, object]] = []
    for index, (ligand, receptor) in enumerate((("L1", "R1"), ("L2", "R2"))):
        source_id = f"connectomedb2020_test_{index}"
        record: dict[str, object] = {
            "harmonized_interaction_id": source_id,
            "ligand": ligand,
            "receptor": receptor,
            "interaction_label": f"{ligand} {receptor}",
            "source": "synthetic",
            "pmid_support": "12345",
            "ligand_hgnc_id": f"HGNC:L{index}",
            "ligand_location": "secreted",
            "receptor_hgnc_id": f"HGNC:R{index}",
            "hgnc_pair": f"HGNC:L{index} HGNC:R{index}",
            "secondary_source": "",
        }
        for method in module.PAPER_METHODS:
            record[f"{method}_source_interaction_id"] = source_id
            record[f"{method}_covered"] = True
        records.append(record)
    table = pd.DataFrame.from_records(
        records, columns=list(module.CONNECTOMEDB_ADAPTER_COLUMNS)
    )
    table_path = tmp_path / "connectomedb2020.tsv"
    table.to_csv(table_path, sep="\t", index=False, lineterminator="\n")

    manifest: dict[str, Any] = {
        "schema_version": module.RESOURCE_SCHEMA,
        "resource_id": "ConnectomeDB2020_test_human",
        "version": "test-1",
        "species": "human",
        "gene_namespace": "HGNC symbol",
        "directionality": "ligand_to_receptor",
        "rows": len(table),
        "payload": {
            "filename": table_path.name,
            "bytes": table_path.stat().st_size,
            "sha256": module.sha256_file(table_path),
        },
        "citation": "Synthetic ConnectomeDB2020 fixture",
        "license": {"fixture": "CC0"},
    }
    manifest["manifest_payload_sha256"] = module._payload_sha256(
        manifest, digest_field="manifest_payload_sha256"
    )
    manifest_path = tmp_path / "connectomedb2020.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return table_path, manifest_path


def _write_kuppe_fixture(tmp_path: Path) -> tuple[Path, Path]:
    genes = ("L1", "R1", "L2", "R2", "T1")
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    cell_ids: list[str] = []
    subject_conditions = (
        ("CTRL_1", "CTRL"),
        ("CTRL_2", "CTRL"),
        ("CTRL_3", "CTRL"),
        ("IZ_1", "IZ"),
        ("IZ_2", "IZ"),
        ("IZ_3", "IZ"),
    )
    profiles = {
        "Sender": [8, 0, 5, 0, 1],
        "Receiver": [0, 8, 0, 5, 2],
    }
    for subject_id, condition in subject_conditions:
        sample_id = f"sample_{subject_id}"
        for cell_type, profile in profiles.items():
            for replicate in range(2):
                rows.append(profile)
                metadata.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "condition": condition,
                        "cell_type": cell_type,
                    }
                )
                cell_ids.append(f"{sample_id}_{cell_type}_{replicate}")
    counts = sparse.csr_matrix(np.asarray(rows, dtype=np.int32))
    data = ad.AnnData(
        X=counts.astype(np.float64),
        obs=pd.DataFrame(metadata, index=cell_ids),
        var=pd.DataFrame(index=pd.Index(genes, dtype=object)),
    )
    data.layers["counts"] = counts
    input_path = tmp_path / "kuppe_ctrl_iz.h5ad"
    data.write_h5ad(input_path)

    manifest: dict[str, Any] = {
        "schema_version": module.PREPARATION_SCHEMA,
        "dataset_id": module.DATASET_ID,
        "mode": "counts_ready_h5ad",
        "selection": {"values": ["CTRL", "IZ"]},
        "matrices": {"counts_layer": "counts"},
        "output": {
            "filename": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "sha256": module.sha256_file(input_path),
        },
    }
    manifest["manifest_payload_sha256"] = module._payload_sha256(
        manifest, digest_field="manifest_payload_sha256"
    )
    manifest_path = tmp_path / "kuppe_ctrl_iz.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return input_path, manifest_path


def _target_prior() -> TargetPrior:
    return TargetPrior(
        resource_id="nichenet_test",
        version="test-1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=("T1",),
        driver_ids=("L1", "L2"),
        indptr=(0, 1, 2),
        target_indices=(0, 0),
        weights=(1.0, 0.8),
        ranks=(1, 1),
        direction=1,
        evidence="synthetic NicheNet fixture",
        mapping_report=MappingReport(2, 2, 3),
        manifest_digest="b" * 64,
    )


class _FakeResult:
    def __init__(self, path: Path, scores: pd.DataFrame) -> None:
        self.path = path
        self.sender_queries = 0
        self.lr_queries = 0
        self._scores = scores.assign(
            raw_sender_evidence=scores["global_sender_lr_score"],
            assignment_weight=1.0,
        )
        lr_keys = [
            "crossfit_id",
            "spec_id",
            "repeat_id",
            "fold_id",
            "contrast_id",
            "contrast",
            "sample_id",
            "subject_id",
            "context_id",
            "receiver",
            "family_id",
            "driver_id",
            "interaction_id",
            "mode",
        ]
        self._lr = (
            self._scores.loc[:, [*lr_keys, "global_sender_lr_score"]]
            .drop_duplicates(lr_keys)
            .rename(columns={"global_sender_lr_score": "availability"})
            .assign(receptor_eligible=True, prior_quality=1.0)
        )
        self._manifest: dict[str, object] = {
            "schema_version": "9.0.0",
            "status": "complete",
            "crossfit_result_id": "crossfit-result-test",
        }
        path.mkdir(parents=True)
        (path / "crossfit_manifest.json").write_text(
            json.dumps(self._manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        downstream_keys = [
            "crossfit_id",
            "spec_id",
            "repeat_id",
            "fold_id",
            "contrast_id",
            "contrast",
            "receiver",
            "subject_id",
            "family_id",
        ]
        (
            self._scores.loc[:, downstream_keys]
            .drop_duplicates()
            .assign(differential_effect=0.1, status="observed", reason_code=None)
            .to_parquet(path / "descriptive_differential.parquet", index=False)
        )

    @property
    def manifest(self) -> dict[str, object]:
        return dict(self._manifest)

    def query_contrast_common_sender_lr_scores(
        self,
        *,
        contrast: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        self.sender_queries += 1
        assert contrast == module.CONTRAST
        assert mode == "state"
        assert status is None
        return self._scores.copy(deep=True)

    def query_contrast_common_lr_scores(
        self,
        *,
        contrast: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        self.lr_queries += 1
        assert contrast == module.CONTRAST
        assert mode == "state"
        assert status is None
        return self._lr.copy(deep=True)

    def read_state_semantic_availability(self) -> pd.DataFrame:
        return (
            self._scores.loc[
                :,
                [
                    "crossfit_id",
                    "fold_id",
                    "sample_id",
                    "subject_id",
                    "sender",
                    "receiver",
                    "interaction_id",
                    "mode",
                    "global_sender_lr_score",
                    "status",
                    "reason_code",
                ],
            ]
            .rename(columns={"global_sender_lr_score": "availability_score"})
            .copy(deep=True)
        )


def test_kuppe_ctrl_iz_cli_exports_subject_equal_directional_rankings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, input_manifest = _write_kuppe_fixture(tmp_path)
    resource_path, resource_manifest = _write_connectomedb_fixture(tmp_path)
    captured: dict[str, Any] = {}

    class FakeCrychic:
        def __init__(
            self,
            config: CrychicConfig,
            *,
            resource_bundle: ResourceBundle,
            target_prior: TargetPrior,
        ) -> None:
            captured["config"] = config
            captured["bundle"] = resource_bundle
            captured["prior"] = target_prior

        def fit_descriptive(
            self,
            data: ad.AnnData,
            *,
            spec: CrossFitSpec,
            n_jobs: int,
            output_dir: Path,
        ) -> _FakeResult:
            captured["spec"] = spec
            captured["n_jobs"] = n_jobs
            bundle = captured["bundle"]
            interactions = bundle.interactions
            sample_rows = data.obs.loc[
                :, ["sample_id", "subject_id", "condition"]
            ].drop_duplicates()
            records: list[dict[str, object]] = []
            for sample in sample_rows.itertuples(index=False):
                subject_number = int(str(sample.subject_id).rsplit("_", 1)[1])
                fold_id = f"fold-{1 + (subject_number - 1) % module.OUTER_FOLDS}"
                for interaction in interactions:
                    ligand = str(interaction.ligand_name)
                    values = {
                        ("CTRL", "L1"): 0.2,
                        ("IZ", "L1"): 0.8,
                        ("CTRL", "L2"): 0.7,
                        ("IZ", "L2"): 0.1,
                    }
                    records.append(
                        {
                            "crossfit_id": "crossfit-test",
                            "spec_id": spec.spec_id,
                            "repeat_id": spec.repeat_id,
                            "fold_id": fold_id,
                            "contrast_id": "contrast-test",
                            "contrast": module.CONTRAST,
                            "sample_id": str(sample.sample_id),
                            "subject_id": str(sample.subject_id),
                            "context_id": f"condition={sample.condition}",
                            "sender": "Sender",
                            "receiver": "Receiver",
                            "family_id": f"family-{ligand}",
                            "driver_id": ligand,
                            "interaction_id": interaction.interaction_id,
                            "mode": "state",
                            "global_sender_lr_score": values[
                                (str(sample.condition), ligand)
                            ],
                            "status": "observed",
                            "reason_code": None,
                        }
                    )
            result = _FakeResult(output_dir, pd.DataFrame.from_records(records))
            captured["result"] = result
            return result

    monkeypatch.setattr(module, "Crychic", FakeCrychic)
    monkeypatch.setattr(
        module,
        "load_nichenet_target_prior",
        lambda *args, **kwargs: _target_prior(),
    )
    output_dir = tmp_path / "run"

    module.main(
        [
            str(input_path),
            str(output_dir),
            "--input-manifest",
            str(input_manifest),
            "--connectomedb-resource",
            str(resource_path),
            "--connectomedb-manifest",
            str(resource_manifest),
            "--database-root",
            str(tmp_path / "database"),
            "--seed",
            "17",
            "--threads",
            "2",
            "--fold-jobs",
            "3",
            "--min-cells",
            "1",
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["parameters"]["seed"] == 17
    assert manifest["parameters"]["threads"] == 2
    assert manifest["parameters"]["blas_threads_per_fold"] == 2
    assert manifest["parameters"]["fold_jobs"] == 3
    assert manifest["parameters"]["effective_fold_jobs"] == 2
    assert manifest["parameters"]["outer_folds"] == 2
    assert manifest["resource"]["interactions"] == 2
    assert manifest["resource"]["observation_label_dependency"] is False
    assert manifest["crossfit_result"]["heldout_fold_audit"]["n_folds"] == 2
    assert manifest["crossfit_result"]["heldout_fold_audit"]["n_samples"] == 6
    assert manifest["crossfit_result"]["heldout_fold_audit"]["n_subjects"] == 6
    assert manifest["score_semantics"]["p_value"] == "not_emitted"
    assert manifest["score_semantics"]["q_value"] == "not_emitted"

    config = captured["config"]
    spec = captured["spec"]
    assert captured["n_jobs"] == 3
    assert captured["result"].sender_queries == 1
    assert captured["result"].lr_queries == 1
    assert [mode.value for mode in config.communication_modes] == ["state"]
    assert spec.allowed_n_splits == (2,)
    assert spec.outer_fold_partition_seed == 17
    assert spec.training_spec.max_interactions is None
    assert spec.contrasts[0].name == module.CONTRAST

    scores = pd.read_parquet(output_dir / module.SCORE_FILENAME)
    score_layers = pd.read_parquet(output_dir / module.SCORE_LAYER_FILENAME)
    differences = pd.read_parquet(output_dir / module.DIFFERENCE_FILENAME)
    ranking = pd.read_parquet(output_dir / module.RANKING_FILENAME)
    directed_effects = pd.read_parquet(output_dir / module.DIRECTED_EFFECT_FILENAME)
    mechanistic_effects = pd.read_parquet(
        output_dir / module.MECHANISTIC_DIRECTED_EFFECT_FILENAME
    )
    direct_effects = pd.read_parquet(
        output_dir / module.SENDER_SPECIFIC_DIRECTED_EFFECT_FILENAME
    )
    des_rankings = pd.read_csv(
        output_dir / module.UNORDERED_RANKING_FILENAME,
        sep="\t",
    )
    mechanistic_des_rankings = pd.read_csv(
        output_dir / module.MECHANISTIC_UNORDERED_RANKING_FILENAME,
        sep="\t",
    )
    assert tuple(scores.columns) == module.SCORE_COLUMNS
    assert tuple(score_layers.columns) == module.SCORE_LAYER_COLUMNS
    assert score_layers["selected_score"].equals(
        score_layers["mechanistic_sender_lr_score"]
    )
    assert set(score_layers["selected_score_policy"]) == {"annotate"}
    assert len(scores) == 12
    assert not {"p_value", "q_value", "pval", "qval"}.intersection(scores.columns)
    assert not scores["formal_inference_allowed"].any()
    by_ligand = differences.set_index("ligand")
    assert by_ligand.loc["L1", "strength_difference_IZ_minus_CTRL"] == pytest.approx(
        0.6
    )
    assert by_ligand.loc["L2", "strength_difference_IZ_minus_CTRL"] == pytest.approx(
        -0.6
    )
    assert set(ranking["direction"]) == {"IZ_over_CTRL", "CTRL_over_IZ"}
    assert ranking["n_comparable_lr"].eq(2).all()
    assert ranking["differential_lr_count"].eq(1).all()
    assert ranking["summed_positive_strength_difference"].tolist() == pytest.approx(
        [0.6, 0.6]
    )
    assert not ranking["formal_inference_allowed"].any()
    assert set(des_rankings["condition"]) == {"CTRL", "IZ"}
    assert des_rankings["ranked_strength"].tolist() == pytest.approx([1.0, 1.0])
    assert des_rankings["condition_specific_directed_lr"].eq(1).all()
    assert des_rankings["status"].eq("observed").all()
    assert not des_rankings["formal_inference_allowed"].any()
    directed_by_ligand = directed_effects.set_index("ligand")
    assert directed_by_ligand.loc[
        "L1", "effect_target_minus_reference"
    ] == pytest.approx(0.6)
    assert directed_by_ligand.loc[
        "L2", "effect_target_minus_reference"
    ] == pytest.approx(-0.6)
    assert module.UNORDERED_RANKING_FILENAME in manifest["outputs"]
    assert module.SCORE_LAYER_FILENAME in manifest["outputs"]
    assert module.MECHANISTIC_DIRECTED_EFFECT_FILENAME in manifest["outputs"]
    direct_by_ligand = direct_effects.set_index("ligand")
    assert direct_by_ligand.loc[
        "L1", "effect_target_minus_reference"
    ] == pytest.approx(0.6)
    assert direct_by_ligand.loc[
        "L2", "effect_target_minus_reference"
    ] == pytest.approx(-0.6)
    assert direct_effects["one_standard_error_stable"].all()
    assert mechanistic_des_rankings["ranked_strength"].tolist() == pytest.approx(
        [0.6, 0.6]
    )
    assert module.SENDER_SPECIFIC_DIRECTED_EFFECT_FILENAME in manifest["outputs"]
    assert module.MECHANISTIC_UNORDERED_RANKING_FILENAME in manifest["outputs"]
    assert module.PAIR_OPPORTUNITY_FILENAME in manifest["outputs"]
    assert module.REASON_WATERFALL_FILENAME in manifest["outputs"]
    assert module.EDGE_COMPONENT_DELTA_FILENAME in manifest["outputs"]
    assert mechanistic_effects["effect_semantics"].str.contains("selected_score").all()

    for filename in (
        module.DIRECTED_EFFECT_FILENAME,
        module.UNORDERED_RANKING_FILENAME,
    ):
        (output_dir / filename).unlink()
        manifest["outputs"].pop(filename)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    postprocessed = postprocess_spatial_des(output_dir)
    assert postprocessed["spatial_des_postprocess"]["full_model_refit"] is False
    assert postprocessed["spatial_des_postprocess"]["ranking_source"] == (
        "persisted_sender_specific_direct_effects"
    )
    assert (output_dir / module.DIRECTED_EFFECT_FILENAME).is_file()
    assert (output_dir / module.UNORDERED_RANKING_FILENAME).is_file()

    for filename in (
        module.DIRECTED_EFFECT_FILENAME,
        module.UNORDERED_RANKING_FILENAME,
        module.SENDER_SPECIFIC_DIRECTED_EFFECT_FILENAME,
    ):
        (output_dir / filename).unlink()
        postprocessed["outputs"].pop(filename)
    postprocessed["schema_version"] = "crychic-kuppe-ctrl-iz-crossfit-run-v2"
    (output_dir / "manifest.json").write_text(
        json.dumps(postprocessed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy = postprocess_spatial_des(output_dir)
    assert legacy["spatial_des_postprocess"]["ranking_source"] == (
        "legacy_global_sender_lr_scores"
    )


def test_heldout_coverage_rejects_silently_missing_input_sample() -> None:
    expected = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "subject_id": ["p1", "p2"],
        }
    )
    complete = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s2"],
            "subject_id": ["p1", "p2", "p2"],
        }
    )
    assert heldout_sample_coverage_audit(complete, expected) == {
        "all_input_samples_observed": True,
        "all_input_subjects_observed": True,
        "n_samples": 2,
        "n_subjects": 2,
    }
    with pytest.raises(ValueError, match="held-out sample coverage"):
        heldout_sample_coverage_audit(
            complete.loc[complete["sample_id"].eq("s1")], expected
        )


def test_connectomedb_bundle_fails_closed_on_payload_tamper(tmp_path: Path) -> None:
    table_path, manifest_path = _write_connectomedb_fixture(tmp_path)
    with table_path.open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match=r"payload\.bytes"):
        module.load_connectomedb2020_bundle(table_path, manifest_path)
