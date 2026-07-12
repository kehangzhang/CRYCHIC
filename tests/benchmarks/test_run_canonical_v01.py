from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.run_canonical_v01 import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    _matching_metrics,
    _summary_completion,
    analysis_lineage,
    load_benchmark_config,
    run_dataset,
    sha256_file,
)
from scipy import sparse


def test_canonical_config_declares_three_pinned_inputs() -> None:
    config = load_benchmark_config(DEFAULT_CONFIG)

    assert list(config["datasets"]) == ["kang2018", "ad_skin", "trophoblast"]
    assert all(len(spec["input_sha256"]) == 64 for spec in config["datasets"].values())
    assert config["datasets"]["ad_skin"]["input_mode"] == "normalized_only"
    assert len(config["datasets"]["trophoblast"]["include_cell_types"]) == 12
    assert config["datasets"]["kang2018"]["lr_resource"] == "cellchat_human"
    assert config["datasets"]["ad_skin"]["lr_resource"] == "cellchat_human"
    assert config["datasets"]["trophoblast"]["lr_resource"] == "cellphonedb_human"


def test_benchmark_config_rejects_unknown_downstream_support_method(
    tmp_path: Path,
) -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    first = next(iter(config["datasets"].values()))
    first["workflow"]["downstream_attribution_support_method"] = "silent-switch"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid downstream attribution support"):
        load_benchmark_config(path)


def test_analysis_lineage_commits_to_transform_spec() -> None:
    source = "a" * 64
    first = analysis_lineage(
        source,
        {"operation": "obs_membership_subset", "include_values": ["A", "B"]},
    )
    repeated = analysis_lineage(
        source,
        {"include_values": ["A", "B"], "operation": "obs_membership_subset"},
    )
    changed = analysis_lineage(
        source,
        {"operation": "obs_membership_subset", "include_values": ["A"]},
    )

    assert first["digest"] == repeated["digest"]
    assert first["digest"] != changed["digest"]


def test_summary_merge_ignores_stale_config_metrics(tmp_path: Path) -> None:
    datasets = {
        "current": {"output_name": "current"},
        "stale": {"output_name": "stale"},
    }
    for name, digest in (("current", "a" * 64), ("stale", "b" * 64)):
        output = tmp_path / name
        output.mkdir()
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "dataset": name,
                    "benchmark_config": {"sha256": digest},
                }
            ),
            encoding="utf-8",
        )

    records = _matching_metrics(datasets, tmp_path, "a" * 64)

    assert [record["dataset"] for record in records] == ["current"]


def test_subset_invocation_status_is_distinct_from_config_completeness() -> None:
    completion = _summary_completion(
        {"requested"},
        selected=("requested",),
        configured=("requested", "not_run"),
    )

    assert completion == {
        "status": "complete",
        "configured_datasets_status": "incomplete",
        "configured_datasets_completed": ["requested"],
        "configured_datasets_missing": ["not_run"],
    }


def test_subset_invocation_fails_when_requested_result_is_missing() -> None:
    completion = _summary_completion(
        {"other"},
        selected=("requested",),
        configured=("requested", "other"),
    )

    assert completion["status"] == "failed"
    assert completion["configured_datasets_status"] == "incomplete"


@pytest.mark.skipif(
    not (REPO_ROOT.parent / "databases" / "cellchat").is_dir(),
    reason="canonical database directory is unavailable",
)
def test_tiny_dataset_reaches_real_resource_dry_run(tmp_path: Path) -> None:
    input_path = tmp_path / "tiny.h5ad"
    obs = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "subject_id": ["d1", "d1", "d2", "d2", "d1", "d1", "d2", "d2"],
            "condition": ["ctrl"] * 4 + ["stim"] * 4,
            "cell_type": ["sender", "receiver"] * 4,
        },
        index=[f"cell-{index}" for index in range(8)],
    )
    counts = sparse.csr_matrix(
        np.array(
            [
                [2, 0, 1],
                [0, 2, 1],
                [3, 0, 1],
                [0, 3, 1],
                [4, 0, 1],
                [0, 4, 1],
                [5, 0, 1],
                [0, 5, 1],
            ],
            dtype=np.int32,
        )
    )
    adata = ad.AnnData(counts.copy(), obs=obs)
    adata.var_names = ["CXCL10", "CXCR3", "ACTB"]
    adata.layers["counts"] = counts
    adata.write_h5ad(input_path)
    digest = sha256_file(input_path)
    spec = {
        "dataset_id": "tiny",
        "input": str(input_path),
        "input_sha256": digest,
        "output_name": "tiny",
        "input_mode": "counts",
        "lr_resource": "cellchat_human",
        "target_prior": None,
        "config": {
            "context_keys": ["condition"],
            "counts_layer": "counts",
            "sample_key": "sample_id",
            "subject_key": "subject_id",
            "cell_type_key": "cell_type",
            "species": "human",
            "gene_namespace": "hgnc_symbol",
            "random_seed": 7,
        },
        "workflow": {
            "min_cells": 1,
            "min_samples_per_context": 2,
            "min_subjects_per_context": 2,
            "min_pooled_availability": 0.0,
            "max_interactions": 10,
        },
    }
    resources = load_benchmark_config(DEFAULT_CONFIG)["resources"]

    metrics = run_dataset(
        "tiny",
        spec,
        repo_root=REPO_ROOT,
        database_root=REPO_ROOT.parent / "databases",
        output_root=tmp_path / "results",
        resources_config=resources,
        benchmark_config_path=DEFAULT_CONFIG,
        benchmark_config_sha256=sha256_file(DEFAULT_CONFIG),
        dry_run_only=True,
    )

    assert metrics["status"] == "dry_run_complete"
    assert metrics["input"]["checksum_verified"] is True
    assert metrics["analysis_lineage"]["source_sha256"] == digest
    assert metrics["memory_measurement"] == {
        "isolated_process": False,
        "peak_rss_scope": "unspecified_process_scope",
    }
    assert metrics["dry_run"]["can_fit"] is True
    assert (tmp_path / "results" / "tiny" / "metrics.json").is_file()
    assert (tmp_path / "results" / "tiny" / "benchmark_manifest.json").is_file()
