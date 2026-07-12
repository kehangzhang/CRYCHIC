from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.common import (
    LONG_TABLE_COLUMNS,
    materialize_fixed_universe,
    validate_long_table,
)
from benchmarks.adapters.crychic import readback
from benchmarks.adapters.crychic.readback import convert_result_to_long
from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.adapters.crychic.run_hcommon import validate_hcommon_workflow

from crychic.core import stable_id
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeResult:
    def __init__(
        self,
        *,
        bundle: ResourceBundle,
        sample_scores: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        self.manifest: Mapping[str, Any] = {
            "run_id": "source_run",
            "result_schema_version": "0.1.0",
            "resource_digests": {
                f"{bundle.resource_id}:{bundle.version}": bundle.manifest_digest
            },
            "workflow_parameters": {
                "method_version": "0.1.0-exploratory",
                "pseudobulk": {"min_cells": 2},
            },
        }
        self.config: Mapping[str, Any] = {
            "sample_key": "sample_id",
            "subject_key": "subject_id",
            "cell_type_key": "cell_type",
            "context_keys": ["condition"],
        }
        self.provenance: Mapping[str, Any] = {"package_version": "0.1.0"}
        self._tables = {
            "sample_scores": sample_scores,
            "interactions": interactions,
        }

    def read_table(
        self,
        name: str,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        result = self._tables[name].copy()
        if filters:
            for field, value in filters.items():
                result = result.loc[result[field].eq(cast(Any, value))]
        if columns is not None:
            result = result.loc[:, list(columns)]
        return result.reset_index(drop=True)


def _interaction(
    interaction_id: str,
    ligand: str,
    receptor: str,
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=f"native_{interaction_id}",
        ligand_name=ligand,
        receptor_name=receptor,
        ligand_subunits=(ligand,),
        receptor_subunits=(receptor,),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="toy",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def test_hcommon_workflow_rejects_data_driven_interaction_cap() -> None:
    accepted = validate_hcommon_workflow(
        {"workflow": {"min_cells": 10, "max_interactions": None}}
    )
    assert accepted["max_interactions"] is None

    with pytest.raises(ValueError, match="requires max_interactions=None"):
        validate_hcommon_workflow(
            {"workflow": {"min_cells": 10, "max_interactions": 100}}
        )


def test_hcommon_configs_are_versioned_without_rewriting_frozen_history() -> None:
    historical_caps = {
        "multicondition_v01_initial.json": 800,
        "multicondition_v01_final.json": 800,
        "synthetic_multimethod_v01.json": 5,
    }
    for name, cap in historical_caps.items():
        config = json.loads(
            (REPO_ROOT / "benchmarks" / "configs" / name).read_text(
                encoding="utf-8"
            )
        )
        assert all(
            dataset["workflow"]["max_interactions"] == cap
            for dataset in config["datasets"].values()
        )

    historical_candidate = json.loads(
        (
            REPO_ROOT
            / "benchmarks/configs/downstream_support_candidate_holdout_v01.json"
        ).read_text(encoding="utf-8")
    )
    assert historical_candidate["workflow"]["max_interactions"] == 5

    for name in (
        "multicondition_hcommon_nocap_v02.json",
        "synthetic_multimethod_nocap_v02.json",
    ):
        config = json.loads(
            (REPO_ROOT / "benchmarks" / "configs" / name).read_text(
                encoding="utf-8"
            )
        )
        assert all(
            dataset["workflow"]["max_interactions"] is None
            for dataset in config["datasets"].values()
        )
        base_path = REPO_ROOT / config["historical_base_config"]
        assert hashlib.sha256(base_path.read_bytes()).hexdigest() == config[
            "historical_base_sha256"
        ]

    candidate = json.loads(
        (
            REPO_ROOT
            / "benchmarks/configs/downstream_support_candidate_holdout_nocap_v02.json"
        ).read_text(encoding="utf-8")
    )
    assert candidate["workflow"]["max_interactions"] is None
    base_path = REPO_ROOT / candidate["historical_base_config"]
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == candidate[
        "historical_base_sha256"
    ]


def _bundle() -> ResourceBundle:
    return ResourceBundle(
        resource_id="toy",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(
            _interaction("i1", "L1", "R1"),
            _interaction("i2", "L2", "R2"),
        ),
        mapping_report=MappingReport(2, 2, 4),
        manifest_digest="d" * 64,
        source_files=("toy.tsv",),
        license="CC0",
        citation="Toy resource",
    )


def _adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2"],
            "subject_id": ["p1"] * 4 + ["p2"] * 3,
            "cell_type": ["A", "A", "B", "B", "A", "A", "B"],
            "condition": ["A"] * 4 + ["B"] * 3,
        },
        index=[f"cell_{index}" for index in range(7)],
    )
    return ad.AnnData(
        X=np.ones((7, 3)),
        obs=obs,
        var=pd.DataFrame(index=["L1", "R1", "L2"]),
    )


def _edge_id() -> str:
    return cast(
        str,
        stable_id(
            "communication_edge",
            {"interaction_id": "i1", "receiver": "B", "sender": "A"},
        ),
    )


def _sample_scores(functional_ids: tuple[str, ...] = ("functional_1",)) -> pd.DataFrame:
    records = []
    for functional_id in functional_ids:
        records.extend(
            [
                {
                    "sample_id": "s1",
                    "subject_id": "p1",
                    "context_id": "context_a",
                    "context_json": '{"condition":"A"}',
                    "edge_id": _edge_id(),
                    "scoring_functional_id": functional_id,
                    "repeat_id": "repeat-0",
                    "fold_id": "in_sample",
                    "mode": "state",
                    "availability": 0.6,
                    "sender_component": 0.7,
                    "prior_quality": 1.0,
                    "comm_strength": 0.8,
                    "status": "ok",
                    "reason_code": "exploratory_not_cross_fitted",
                },
                {
                    "sample_id": "s2",
                    "subject_id": "p2",
                    "context_id": "context_b",
                    "context_json": '{"condition":"B"}',
                    "edge_id": _edge_id(),
                    "scoring_functional_id": functional_id,
                    "repeat_id": "repeat-0",
                    "fold_id": "in_sample",
                    "mode": "state",
                    "availability": 0.4,
                    "sender_component": 0.5,
                    "prior_quality": np.nan,
                    "comm_strength": np.nan,
                    "status": "missing",
                    "reason_code": "missing_core_evidence:prior_quality",
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _interactions(contrasts: tuple[str, ...] = ("global:A",)) -> pd.DataFrame:
    records = []
    for contrast in contrasts:
        records.extend(
            [
                {
                    "context_id": "context_a",
                    "sender": "A",
                    "receiver": "B",
                    "interaction_id": "i1",
                    "mode": "state",
                    "contrast": contrast,
                    "availability": 0.6,
                    "assignment_weight": 0.7,
                    "comm_strength": 0.8,
                    "prior_quality": 1.0,
                },
                {
                    "context_id": "context_b",
                    "sender": "A",
                    "receiver": "B",
                    "interaction_id": "i1",
                    "mode": "state",
                    "contrast": contrast,
                    "availability": 0.4,
                    "assignment_weight": 0.5,
                    "comm_strength": np.nan,
                    "prior_quality": np.nan,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def test_readback_materializes_fixed_universe_and_statuses() -> None:
    bundle = _bundle()
    result = FakeResult(
        bundle=bundle,
        sample_scores=_sample_scores(),
        interactions=_interactions(),
    )
    table, views = convert_result_to_long(
        result,
        _adata(),
        bundle,
        dataset_id="toy_dataset",
        resource_mode="H-common",
    )

    assert tuple(table.columns) == LONG_TABLE_COLUMNS
    assert table.groupby("sample_id").size().to_dict() == {"s1": 8, "s2": 8}
    assert table["universe_size"].unique().tolist() == [8]
    assert table["analysis_track"].unique().tolist() == ["lr_stlr"]
    assert table["score_name"].unique().tolist() == ["comm_strength"]
    assert table["score_direction"].unique().tolist() == ["higher"]
    assert table["interaction_id"].isin(["i1", "i2"]).all()
    assert table["differential_effect"].isna().all()
    assert table["differential_p_value"].isna().all()
    assert table["differential_q_value"].isna().all()
    assert table["within_dataset_p_value"].isna().all()
    assert table["reason_code"].str.startswith("v0_1_inferential_disabled").all()

    observed = table.loc[
        table[["sample_id", "sender", "receiver", "interaction_id"]]
        .eq(["s1", "A", "B", "i1"])
        .all(axis=1)
    ].iloc[0]
    assert observed["status"] == "ok"
    assert observed["score"] == 0.8

    low_support = table.loc[
        table[["sample_id", "sender", "receiver", "interaction_id"]]
        .eq(["s2", "A", "B", "i1"])
        .all(axis=1)
    ].iloc[0]
    assert low_support["status"] == "insufficient_cells"
    assert pd.isna(low_support["score"])

    missing_gene = table.loc[
        table[["sample_id", "sender", "receiver", "interaction_id"]]
        .eq(["s1", "A", "A", "i2"])
        .all(axis=1)
    ].iloc[0]
    assert missing_gene["status"] == "missing"
    assert "missing_required_lr_input" in missing_gene["reason_code"]
    assert views[0]["contrast_candidates"] == ["global:A"]


def test_readback_uses_one_run_per_scoring_functional() -> None:
    bundle = _bundle()
    functionals = ("functional_1", "functional_2")
    result = FakeResult(
        bundle=bundle,
        sample_scores=_sample_scores(functionals),
        interactions=_interactions(("global:A", "global:B")),
    )
    table, views = convert_result_to_long(
        result,
        _adata(),
        bundle,
        dataset_id="toy_dataset",
        resource_mode="native",
    )

    assert table["run_id"].nunique() == 2
    assert table.groupby("run_id").size().unique().tolist() == [16]
    assert {view["scoring_functional_id"] for view in views} == set(functionals)
    assert all(
        view["contrast_candidates"] == ["global:A", "global:B"] for view in views
    )


def test_readback_optimized_validation_preserves_every_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    result = FakeResult(
        bundle=bundle,
        sample_scores=_sample_scores(("functional_1", "functional_2")),
        interactions=_interactions(("global:A", "global:B")),
    )
    optimized, optimized_views = convert_result_to_long(
        result,
        _adata(),
        bundle,
        dataset_id="toy_dataset",
        resource_mode="native",
    )

    materialize = materialize_fixed_universe
    apply_statuses = readback._apply_crychic_statuses

    def materialize_with_validation(
        observed: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        kwargs["validate"] = True
        return materialize(observed, **kwargs)

    def apply_with_validation(
        table: pd.DataFrame,
        raw: pd.DataFrame,
        *,
        validate: bool = True,
    ) -> pd.DataFrame:
        del validate
        return apply_statuses(table, raw, validate=True)

    monkeypatch.setattr(
        readback, "materialize_fixed_universe", materialize_with_validation
    )
    monkeypatch.setattr(readback, "_apply_crychic_statuses", apply_with_validation)
    fully_validated, fully_validated_views = convert_result_to_long(
        result,
        _adata(),
        bundle,
        dataset_id="toy_dataset",
        resource_mode="native",
    )

    pd.testing.assert_frame_equal(optimized, fully_validated, check_exact=True)
    assert optimized_views == fully_validated_views


def test_readback_validates_each_score_view_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated_run_ids: list[tuple[str, ...]] = []
    validate = validate_long_table

    def record_validation(table: pd.DataFrame) -> pd.DataFrame:
        validated_run_ids.append(tuple(table["run_id"].astype(str).unique()))
        return validate(table)

    monkeypatch.setattr(readback, "validate_long_table", record_validation)
    table, _ = convert_result_to_long(
        FakeResult(
            bundle=_bundle(),
            sample_scores=_sample_scores(("functional_1", "functional_2")),
            interactions=_interactions(("global:A", "global:B")),
        ),
        _adata(),
        _bundle(),
        dataset_id="toy_dataset",
        resource_mode="native",
    )

    assert len(validated_run_ids) == 2
    assert all(len(run_ids) == 1 for run_ids in validated_run_ids)
    assert {run_ids[0] for run_ids in validated_run_ids} == set(table["run_id"])


def test_readback_per_view_validation_remains_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apply_statuses = readback._apply_crychic_statuses

    def corrupt_status_after_apply(
        table: pd.DataFrame,
        raw: pd.DataFrame,
        *,
        validate: bool = True,
    ) -> pd.DataFrame:
        result = apply_statuses(table, raw, validate=validate)
        result.loc[result.index[0], "status"] = "invalid_status"
        return result

    monkeypatch.setattr(
        readback, "_apply_crychic_statuses", corrupt_status_after_apply
    )
    with pytest.raises(ValueError, match="invalid statuses"):
        convert_result_to_long(
            FakeResult(
                bundle=_bundle(),
                sample_scores=_sample_scores(),
                interactions=_interactions(),
            ),
            _adata(),
            _bundle(),
            dataset_id="toy_dataset",
            resource_mode="native",
        )


def test_readback_rejects_run_id_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(readback, "canonical_digest", lambda *_args, **_kwargs: "run")
    with pytest.raises(ValueError, match="distinct run IDs"):
        convert_result_to_long(
            FakeResult(
                bundle=_bundle(),
                sample_scores=_sample_scores(("functional_1", "functional_2")),
                interactions=_interactions(("global:A", "global:B")),
            ),
            _adata(),
            _bundle(),
            dataset_id="toy_dataset",
            resource_mode="native",
        )


def test_harmonized_bundle_retains_interaction_id(tmp_path: Path) -> None:
    table_path = tmp_path / "harmonized_lr.tsv"
    table = pd.DataFrame(
        {
            "harmonized_interaction_id": ["harmonized_1"],
            "ligand": ["L1"],
            "receptor": ["R1"],
            "cellchat_source_interaction_id": ["cc_1"],
            "cellphonedb_source_interaction_id": ["cpdb_1"],
        }
    )
    table.to_csv(table_path, sep="\t", index=False)
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "crychic-harmonized-lr-v1",
                "resource_id": "harmonized_toy",
                "version": "1",
                "species": "human",
                "gene_namespace": "HGNC symbol",
                "license": "CC0",
                "citation": ["Toy citation"],
                "payload": {
                    "filename": table_path.name,
                    "sha256": digest,
                    "rows": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = harmonized_resource_bundle(table_path, manifest_path)

    assert bundle.resource_id == "harmonized_toy"
    assert bundle.interactions[0].interaction_id == "harmonized_1"
    assert bundle.interactions[0].source_interaction_id == "harmonized_1"


def test_harmonized_bundle_accepts_synthetic_fixture_schema(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "harmonized_lr.tsv"
    table = pd.DataFrame(
        {
            "harmonized_interaction_id": ["harmonized_1"],
            "ligand": ["CXCL10"],
            "receptor": ["CXCR3"],
            "cellchat_source_interaction_id": ["CXCL10_CXCR3"],
            "cellphonedb_source_interaction_id": ["CPI-1"],
        }
    )
    table.to_csv(table_path, sep="\t", index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "crychic-harmonized-lr-v1-synthetic-fixture",
                "resource_id": "synthetic_fixture",
                "version": "1",
                "species": "human",
                "gene_namespace": "HGNC symbol",
                "license": "intersection-only",
                "citation": ["Synthetic fixture derived from frozen resources"],
                "payload": {
                    "filename": table_path.name,
                    "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
                    "rows": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = harmonized_resource_bundle(table_path, manifest_path)

    assert bundle.resource_id == "synthetic_fixture"
    assert len(bundle.interactions) == 1
