from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
from benchmarks.openproblems import run_liana_source_target as runner
from benchmarks.openproblems.common import sha256_file, validate_predictions, write_json


def _input(path: Path) -> Path:
    data = ad.AnnData(
        X=np.array(
            [
                [4, 1],
                [3, 1],
                [1, 4],
                [1, 3],
                [2, 2],
                [2, 2],
            ],
            dtype=np.int64,
        ),
        obs=pd.DataFrame(
            {
                "label": pd.Categorical(
                    ["B", "B", "A", "A", "C", "C"],
                    categories=["B", "A", "C"],
                )
            },
            index=[f"c{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=["L", "R"]),
    )
    data.uns["ccc_target"] = pd.DataFrame(
        {"source": ["B"], "target": ["A"], "response": [1]}
    )
    data.uns["spatial_proxy"] = np.eye(3)
    data.write_h5ad(path)
    return path


def _resource(path: Path) -> Path:
    path.mkdir()
    table_path = path / "liana_consensus_mouse.parquet"
    pd.DataFrame(
        {
            "interaction_id": ["i1"],
            "ligand": ["L"],
            "receptor": ["R"],
            "input_available": [True],
        }
    ).to_parquet(table_path, index=False)
    write_json(
        path / "resource_manifest.json",
        {
            "resource_id": "toy_resource",
            "files": {
                "resource": {
                    "filename": table_path.name,
                    "sha256": sha256_file(table_path),
                }
            },
        },
    )
    return path


def test_liana_runner_removes_truth_before_every_method(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []

    def rank_aggregate(data, **kwargs):
        assert not data.uns
        calls.append("rank")
        return pd.DataFrame(
            {
                "source": ["B", "A"],
                "target": ["A", "B"],
                "cellphone_pvals": [0.01, 0.2],
                "lr_means": [0.8, 0.4],
                "magnitude_rank": [0.1, 0.7],
                "specificity_rank": [0.2, 0.8],
                "scaled_weight": [0.7, 0.3],
                "lr_logfc": [0.6, 0.2],
                "spec_weight": [0.9, 0.1],
                "lrscore": [0.8, 0.2],
            }
        )

    def cellchat(data, **kwargs):
        assert not data.uns
        calls.append("cellchat")
        return pd.DataFrame(
            {
                "source": ["B", "A"],
                "target": ["A", "B"],
                "cellchat_pvals": [0.01, 0.2],
                "lr_probs": [0.7, 0.3],
            }
        )

    fake_liana = SimpleNamespace(
        mt=SimpleNamespace(rank_aggregate=rank_aggregate, cellchat=cellchat)
    )
    fake_scanpy = SimpleNamespace(
        pp=SimpleNamespace(
            normalize_total=lambda *a, **k: None,
            log1p=lambda *a, **k: None,
        )
    )
    monkeypatch.setattr(runner, "_load_runtime", lambda: (fake_liana, fake_scanpy))
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "test")

    output = tmp_path / "output"
    manifest = runner.run(
        _input(tmp_path / "input.h5ad"),
        _resource(tmp_path / "resource"),
        output,
        n_perms=2,
        n_jobs=1,
        overwrite=False,
    )

    assert calls == ["rank", "cellchat"]
    isolation = manifest["input"]["truth_and_proxy_isolation"]
    assert isolation["all_uns_removed_before_method_execution"] is True
    assert set(isolation["removed_key_names"]) == {"ccc_target", "spatial_proxy"}
    assert len(manifest["predictions"]) == 16
    for path in (output / "predictions").glob("*.parquet"):
        validate_predictions(pd.read_parquet(path), labels=("B", "A", "C"))
