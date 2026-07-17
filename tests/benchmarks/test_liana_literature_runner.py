from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.common import sha256_file, write_json
from benchmarks.adapters.liana import run_literature as runner


def _input(path: Path) -> Path:
    adata = ad.AnnData(
        X=np.log1p(np.array([[4, 0, 2], [3, 0, 1], [0, 4, 1], [0, 3, 2]], dtype=float)),
        obs=pd.DataFrame(
            {"cell_type": ["A", "A", "B", "B"]},
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(index=["L1", "R1", "G1"]),
    )
    adata.write_h5ad(path)
    return path


def _rank_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A"],
            "target": ["B"],
            "ligand_complex": ["L1"],
            "receptor_complex": ["R1"],
            "lr_means": [2.0],
            "cellphone_pvals": [0.01],
            "expr_prod": [2.0],
            "scaled_weight": [1.0],
            "lr_logfc": [0.5],
            "spec_weight": [0.8],
            "lrscore": [0.9],
            "specificity_rank": [0.1],
            "magnitude_rank": [0.1],
        }
    )


def _cellchat_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A"],
            "target": ["B"],
            "ligand_complex": ["L1"],
            "receptor_complex": ["R1"],
            "lr_probs": [0.7],
            "cellchat_pvals": [0.02],
        }
    )


def _fake_liana(calls: list[tuple[str, dict[str, object]]]) -> SimpleNamespace:
    def select_resource(name: str) -> pd.DataFrame:
        assert name == "consensus"
        return pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]})

    def rank_aggregate(_adata, **kwargs):
        calls.append(("rank_aggregate", kwargs))
        return _rank_result()

    def cellchat(_adata, **kwargs):
        calls.append(("cellchat", kwargs))
        return _cellchat_result()

    return SimpleNamespace(
        rs=SimpleNamespace(select_resource=select_resource),
        mt=SimpleNamespace(rank_aggregate=rank_aggregate, cellchat=cellchat),
    )


def _harmonized(tmp_path: Path) -> tuple[Path, Path]:
    table = tmp_path / "harmonized_lr.tsv"
    pd.DataFrame(
        {
            "harmonized_interaction_id": ["h1"],
            "ligand": ["L1"],
            "receptor": ["R1"],
            "cellchat_source_interaction_id": ["cc1"],
            "cellphonedb_source_interaction_id": ["cp1"],
        }
    ).to_csv(table, sep="\t", index=False)
    manifest = tmp_path / "resource_manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "crychic-harmonized-lr-v1-synthetic-fixture",
            "resource_id": "toy-common",
            "version": "1",
            "species": "human",
            "gene_namespace": "HGNC symbol",
            "license": "CC0",
            "citation": "Synthetic fixture",
            "payload": {"filename": table.name, "sha256": sha256_file(table)},
        },
    )
    return table, manifest


def test_native_runner_freezes_calls_raw_outputs_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _input(tmp_path / "input.h5ad")
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setitem(sys.modules, "liana", _fake_liana(calls))
    monkeypatch.setattr(
        runner,
        "_distribution_version",
        lambda name: "1.7.3" if name == "liana" else "test",
    )
    output = tmp_path / "native"
    manifest = runner.run_literature_benchmark(
        input_path,
        output,
        dataset_id="cite",
        n_perms=17,
        seed=23,
        min_cells=1,
    )

    assert [name for name, _ in calls] == ["rank_aggregate", "cellchat"]
    for _, kwargs in calls:
        assert kwargs["groupby"] == "cell_type"
        assert kwargs["n_perms"] == 17
        assert kwargs["seed"] == 23
        assert kwargs["n_jobs"] == 1
        assert kwargs["return_all_lrs"] is True
        assert list(kwargs["resource"].columns) == ["ligand", "receptor"]
    assert manifest["status"] == "complete"
    assert manifest["resource"]["mode"] == "native"
    assert manifest["resource"]["interactions"] == 1
    assert manifest["parameters"]["groupby"] == "cell_type"
    assert manifest["method"]["entrypoints"] == [
        "liana.mt.rank_aggregate",
        "liana.mt.cellchat",
    ]
    assert manifest["output"]["rank_aggregate"]["rows"] == 1
    assert manifest["output"]["cellchat"]["rows"] == 1
    assert pd.read_parquet(output / runner.RANK_OUTPUT).equals(_rank_result())
    assert pd.read_parquet(output / runner.CELLCHAT_OUTPUT).equals(_cellchat_result())
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stored["output"]["rank_aggregate"]["sha256"] == sha256_file(
        output / runner.RANK_OUTPUT
    )


def test_hcommon_runner_verifies_resource_and_has_reproducible_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _input(tmp_path / "input-common.h5ad")
    resource, resource_manifest = _harmonized(tmp_path)
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setitem(sys.modules, "liana", _fake_liana(calls))
    monkeypatch.setattr(runner, "_distribution_version", lambda _name: "test")
    first = runner.run_literature_benchmark(
        input_path,
        tmp_path / "first",
        dataset_id="cite",
        resource_mode="H-common",
        harmonized_resource=resource,
        harmonized_manifest=resource_manifest,
        n_perms=11,
        seed=7,
        min_cells=1,
    )
    second = runner.run_literature_benchmark(
        input_path,
        tmp_path / "second",
        dataset_id="cite",
        resource_mode="H-common",
        harmonized_resource=resource,
        harmonized_manifest=resource_manifest,
        n_perms=11,
        seed=7,
        min_cells=1,
    )

    assert first["run_id"] == second["run_id"]
    assert first["resource"]["id"] == "toy-common"
    assert first["resource"]["payload_sha256"] == sha256_file(resource)
    assert first["resource"]["manifest_sha256"] == sha256_file(resource_manifest)


def test_native_custom_runner_uses_checksum_pinned_mouse_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _input(tmp_path / "input-mouse.h5ad")
    resource = tmp_path / "mouse.tsv"
    pd.DataFrame(
        {"interaction_id": ["m1"], "ligand": ["L1"], "receptor": ["R1"]}
    ).to_csv(resource, sep="\t", index=False)
    resource_manifest = tmp_path / "mouse-manifest.json"
    write_json(
        resource_manifest,
        {
            "resource_id": "mouse-consensus",
            "task_version": "v1",
            "species": "mouse",
            "gene_namespace": "MGI symbol",
            "files": {
                "resource_tsv": {
                    "filename": resource.name,
                    "sha256": sha256_file(resource),
                }
            },
        },
    )
    calls: list[tuple[str, dict[str, object]]] = []
    fake = _fake_liana(calls)
    fake.rs.select_resource = lambda _name: pytest.fail(
        "default human consensus must not be selected"
    )
    monkeypatch.setitem(sys.modules, "liana", fake)
    monkeypatch.setattr(runner, "_distribution_version", lambda _name: "test")

    manifest = runner.run_literature_benchmark(
        input_path,
        tmp_path / "mouse-output",
        dataset_id="mouse-cite",
        custom_resource=resource,
        custom_resource_manifest=resource_manifest,
        min_cells=1,
    )

    assert manifest["resource"]["id"] == "mouse-consensus"
    assert manifest["resource"]["species"] == "mouse"
    assert manifest["resource"]["gene_namespace"] == "MGI symbol"
    assert manifest["resource"]["payload_sha256"] == sha256_file(resource)


def test_runner_rejects_invalid_seed_and_missing_hcommon_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _input(tmp_path / "invalid.h5ad")
    monkeypatch.setitem(sys.modules, "liana", _fake_liana([]))
    with pytest.raises(ValueError, match="uint32"):
        runner.run_literature_benchmark(
            input_path,
            tmp_path / "bad-seed",
            dataset_id="cite",
            seed=-1,
        )
    with pytest.raises(ValueError, match="H-common requires"):
        runner.run_literature_benchmark(
            input_path,
            tmp_path / "bad-resource",
            dataset_id="cite",
            resource_mode="H-common",
        )
