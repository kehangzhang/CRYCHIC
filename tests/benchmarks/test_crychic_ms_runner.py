from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import benchmarks.adapters.crychic.run_kuppe_ctrl_iz as kuppe_module
import benchmarks.adapters.crychic.run_ms_ctrl_ca as module
import numpy as np
import pandas as pd
import pytest
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
        source_id = f"connectomedb2020_ms_test_{index}"
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
        for method in kuppe_module.PAPER_METHODS:
            record[f"{method}_source_interaction_id"] = source_id
            record[f"{method}_covered"] = True
        records.append(record)
    table = pd.DataFrame.from_records(
        records, columns=list(kuppe_module.CONNECTOMEDB_ADAPTER_COLUMNS)
    )
    table_path = tmp_path / "connectomedb2020.tsv"
    table.to_csv(table_path, sep="\t", index=False, lineterminator="\n")
    manifest: dict[str, Any] = {
        "schema_version": kuppe_module.RESOURCE_SCHEMA,
        "resource_id": "ConnectomeDB2020_ms_test_human",
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
        "citation": "Synthetic ConnectomeDB2020 MS fixture",
        "license": {"fixture": "CC0"},
    }
    manifest["manifest_payload_sha256"] = kuppe_module._payload_sha256(
        manifest, digest_field="manifest_payload_sha256"
    )
    manifest_path = tmp_path / "connectomedb2020.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return table_path, manifest_path


def _write_ms_fixture(tmp_path: Path) -> tuple[Path, Path]:
    genes = ("L1", "R1", "L2", "R2", "T1")
    samples = (
        ("CTRL_S1", "CTRL_1", "Ctrl", "1"),
        ("CTRL_S2", "CTRL_2", "Ctrl", "2"),
        ("CTRL_S3", "CTRL_3", "Ctrl", "3"),
        ("CA_S1A", "CA_1", "CA", "1"),
        ("CA_S1B", "CA_1", "CA", "1"),
        ("CA_S2", "CA_2", "CA", "2"),
        ("CA_S3", "CA_3", "CA", "3"),
    )
    profiles = {
        "A": [8, 0, 5, 0, 1],
        "B": [0, 8, 0, 5, 2],
    }
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    cell_ids: list[str] = []
    for sample_id, subject_id, lesion_type, batch in samples:
        for cell_type, profile in profiles.items():
            for replicate in range(2):
                rows.append(profile)
                metadata.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "lesion_type": lesion_type,
                        "batch": batch,
                        "cell_type": cell_type,
                    }
                )
                cell_ids.append(f"{sample_id}_{cell_type}_{replicate}")
    counts = sparse.csr_matrix(np.asarray(rows, dtype=np.int32))
    lineage = {
        "schema_version": module.PREPARATION_SCHEMA,
        "dataset_id": module.DATASET_ID,
        "source_filename": "UCSC_Lerma_Martin_MS_snRNA.h5ad",
        "source_sha256": module.EXPECTED_SOURCE_SHA256,
        "selection_field": "lesion_type",
        "selection_values": ["CA", "Ctrl"],
        "preserve_observation_order": True,
        "preserve_variable_axis": True,
        "design": "independent subjects; ~ batch + lesion_type",
        "primary_contrast": module.CONTRAST,
    }
    data = ad.AnnData(
        X=counts.astype(np.float64),
        obs=pd.DataFrame(metadata, index=cell_ids),
        var=pd.DataFrame(index=pd.Index(genes, dtype=object)),
    )
    data.layers["counts"] = counts
    data.uns["crychic_analysis_subset"] = lineage
    input_path = tmp_path / "ms_ctrl_ca.h5ad"
    data.write_h5ad(input_path)
    manifest = {
        "output": input_path.name,
        "output_sha256": module.sha256_file(input_path),
        "shape": list(data.shape),
        "subjects_by_context": {"CA": 3, "Ctrl": 3},
        "samples_by_context": {"CA": 4, "Ctrl": 3},
        "lineage": lineage,
    }
    manifest_path = tmp_path / "ms_ctrl_ca.subset.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return input_path, manifest_path


def _target_prior() -> TargetPrior:
    return TargetPrior(
        resource_id="nichenet_ms_test",
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
        evidence="synthetic NicheNet MS fixture",
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
            "crossfit_result_id": "crossfit-result-ms-test",
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


def test_ms_cli_subject_averages_repeats_and_exports_unordered_des(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "EXPECTED_SHAPE", (28, 5))
    monkeypatch.setattr(module, "EXPECTED_SAMPLES", {"Ctrl": 3, "CA": 4})
    monkeypatch.setattr(module, "EXPECTED_SUBJECTS", {"Ctrl": 3, "CA": 3})
    monkeypatch.setattr(
        module,
        "EXPECTED_SAMPLES_PER_SUBJECT",
        {"Ctrl": (1, 1, 1), "CA": (1, 1, 2)},
    )
    input_path, input_manifest = _write_ms_fixture(tmp_path)
    monkeypatch.setattr(
        module, "EXPECTED_OUTPUT_SHA256", module.sha256_file(input_path)
    )
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
            sample_rows = data.obs.loc[
                :, ["sample_id", "subject_id", "lesion_type"]
            ].drop_duplicates()
            records: list[dict[str, object]] = []
            for sample in sample_rows.itertuples(index=False):
                subject_number = int(str(sample.subject_id).rsplit("_", 1)[1])
                fold_id = f"fold-{1 + (subject_number - 1) % module.OUTER_FOLDS}"
                for interaction in captured["bundle"].interactions:
                    ligand = str(interaction.ligand_name)
                    if sample.lesion_type == "Ctrl":
                        score = 0.1 if ligand == "L1" else 0.2
                    elif ligand == "L1":
                        score = 0.9 if sample.subject_id == "CA_1" else 0.3
                    else:
                        score = 0.8
                    records.append(
                        {
                            "crossfit_id": "crossfit-ms-test",
                            "spec_id": spec.spec_id,
                            "repeat_id": spec.repeat_id,
                            "fold_id": fold_id,
                            "contrast_id": "contrast-ms-test",
                            "contrast": module.CONTRAST,
                            "sample_id": str(sample.sample_id),
                            "subject_id": str(sample.subject_id),
                            "context_id": f"lesion_type={sample.lesion_type}",
                            "sender": "A" if ligand == "L1" else "B",
                            "receiver": "B" if ligand == "L1" else "A",
                            "family_id": f"family-{ligand}",
                            "driver_id": ligand,
                            "interaction_id": interaction.interaction_id,
                            "mode": "state",
                            "global_sender_lr_score": score,
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
            "2",
            "--min-cells",
            "1",
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["parameters"]["threads"] == 2
    assert manifest["parameters"]["blas_threads_per_fold"] == 2
    assert manifest["parameters"]["fold_jobs"] == 2
    assert manifest["parameters"]["effective_fold_jobs"] == 2
    assert manifest["input"]["subject_support"] == {"CA": 3, "Ctrl": 3}
    assert manifest["input"]["repeated_subjects"] == [
        {"lesion_type": "CA", "n_samples": 2, "subject_id": "CA_1"}
    ]
    assert manifest["crossfit_result"]["heldout_fold_audit"] == {
        "all_input_samples_observed": True,
        "all_input_subjects_observed": True,
        "heldout_subjects_by_fold": {
            "fold-1": ["CA_1", "CA_3", "CTRL_1", "CTRL_3"],
            "fold-2": ["CA_2", "CTRL_2"],
        },
        "n_folds": 2,
        "n_samples": 7,
        "n_subjects": 6,
        "subject_heldout_once": True,
        "technical_samples_subject_blocked": True,
    }
    assert manifest["score_semantics"]["p_value"] == "not_emitted"
    assert manifest["score_semantics"]["q_value"] == "not_emitted"

    config = captured["config"]
    spec = captured["spec"]
    assert captured["n_jobs"] == 2
    assert captured["result"].sender_queries == 1
    assert captured["result"].lr_queries == 1
    assert config.context_keys == ("lesion_type",)
    assert config.covariates == ("batch",)
    assert config.categorical_covariates == ("batch",)
    assert config.design == "~ batch + lesion_type"
    assert spec.allowed_n_splits == (2,)
    assert spec.contrasts[0].name == "CA_vs_Ctrl"

    scores = pd.read_parquet(output_dir / module.SCORE_FILENAME)
    score_layers = pd.read_parquet(output_dir / module.SCORE_LAYER_FILENAME)
    effects = pd.read_parquet(output_dir / module.DIRECTED_EFFECT_FILENAME)
    mechanistic_effects = pd.read_parquet(
        output_dir / module.MECHANISTIC_DIRECTED_EFFECT_FILENAME
    )
    rankings = pd.read_csv(output_dir / module.UNORDERED_RANKING_FILENAME, sep="\t")
    assert len(scores) == 14
    assert tuple(score_layers.columns) == module.SCORE_LAYER_COLUMNS
    assert score_layers["selected_score"].equals(
        score_layers["mechanistic_sender_lr_score"]
    )
    assert set(score_layers["selected_score_policy"]) == {"annotate"}
    assert mechanistic_effects["effect_semantics"].str.contains("selected_score").all()
    assert effects["n_samples_target"].eq(4).all()
    assert effects["n_subjects_target"].eq(3).all()
    effect_by_ligand = effects.set_index("ligand")["effect_target_minus_reference"]
    assert effect_by_ligand.loc["L1"] == pytest.approx(0.4)
    assert effect_by_ligand.loc["L2"] == pytest.approx(0.6)
    assert rankings[["sender", "receiver"]].to_dict(orient="records") == [
        {"sender": "A", "receiver": "B"},
        {"sender": "A", "receiver": "B"},
    ]
    by_condition = rankings.set_index("condition")
    assert by_condition.loc["CA", "ranked_strength"] == pytest.approx(1.0)
    assert by_condition.loc["CA", "condition_specific_directed_lr"] == 2
    assert by_condition.loc["Ctrl", "ranked_strength"] == pytest.approx(0.0)
    assert by_condition.loc["Ctrl", "condition_specific_directed_lr"] == 0
    assert by_condition["estimable_directed_lr"].eq(2).all()
    assert not rankings["formal_inference_allowed"].any()
    forbidden = {"p", "p_value", "q", "q_value", "pval", "qval"}
    assert not forbidden.intersection(scores.columns)
    assert not forbidden.intersection(effects.columns)
    assert not forbidden.intersection(rankings.columns)
    assert module.SCORE_LAYER_FILENAME in manifest["outputs"]
    assert module.MECHANISTIC_DIRECTED_EFFECT_FILENAME in manifest["outputs"]


def test_ms_subset_manifest_fails_closed_on_output_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "EXPECTED_SHAPE", (28, 5))
    monkeypatch.setattr(module, "EXPECTED_SAMPLES", {"Ctrl": 3, "CA": 4})
    monkeypatch.setattr(module, "EXPECTED_SUBJECTS", {"Ctrl": 3, "CA": 3})
    input_path, manifest_path = _write_ms_fixture(tmp_path)
    monkeypatch.setattr(
        module, "EXPECTED_OUTPUT_SHA256", module.sha256_file(input_path)
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="output_sha256"):
        module.validate_subset_manifest(input_path, manifest_path)
