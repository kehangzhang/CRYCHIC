from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.scdiffcom import run as module
from scipy import sparse
from scipy.io import mmread


def _resource(rows: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.DataFrame(
        {
            "harmonized_interaction_id": [f"pair-{index}" for index in range(rows)],
            "ligand": [f"L{index}" for index in range(rows)],
            "receptor": [f"R{index}" for index in range(rows)],
            "scdiffcom_covered": ["True"] * rows,
        }
    )
    lri = pd.DataFrame(
        {
            "LRI": table["harmonized_interaction_id"],
            "LIGAND_1": table["ligand"],
            "LIGAND_2": pd.Series(pd.NA, index=table.index, dtype="string"),
            "RECEPTOR_1": table["receptor"],
            "RECEPTOR_2": pd.Series(pd.NA, index=table.index, dtype="string"),
            "RECEPTOR_3": pd.Series(pd.NA, index=table.index, dtype="string"),
        }
    )
    return table, lri


def test_input_manifest_digest_supports_both_prepared_schemas() -> None:
    digest = "a" * 64
    assert (
        module._input_manifest_sha256(
            {"output": {"filename": "input.h5ad", "sha256": digest}},
            "input.h5ad",
        )
        == digest
    )
    assert (
        module._input_manifest_sha256(
            {"output": "input.h5ad", "output_sha256": digest}, "input.h5ad"
        )
        == digest
    )


def test_validate_resource_builds_exact_six_column_simple_lri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource_path = tmp_path / "resource.tsv"
    table, _ = _resource(module.RESOURCE_ROWS)
    table.to_csv(resource_path, sep="\t", index=False)
    digest = hashlib.sha256(resource_path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "RESOURCE_SHA256", digest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "resource_id": module.RESOURCE_ID,
                "rows": module.RESOURCE_ROWS,
                "payload": {"sha256": digest},
            }
        ),
        encoding="utf-8",
    )

    _, lri, _ = module._validate_resource(resource_path, manifest_path)

    assert tuple(lri.columns) == module.LRI_COLUMNS
    assert len(lri) == module.RESOURCE_ROWS
    assert lri["LIGAND_2"].isna().all()
    assert lri["RECEPTOR_2"].isna().all()
    assert lri["RECEPTOR_3"].isna().all()


def test_prepare_exports_only_resource_genes_and_raw_counts(tmp_path: Path) -> None:
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "B", "A", "B"],
            "condition": ["case", "case", "control", "control"],
        },
        index=[f"cell-{index}" for index in range(4)],
    )
    counts = sparse.csr_matrix(
        np.asarray(
            [
                [1, 2, 0, 9, 0],
                [0, 1, 3, 8, 0],
                [2, 0, 1, 7, 4],
                [1, 1, 1, 6, 3],
            ],
            dtype=np.int32,
        )
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts.shape),
        obs=obs,
        var=pd.DataFrame(index=["L0", "R0", "L1", "unrelated", "R1"]),
    )
    adata.layers["counts"] = counts
    input_path = tmp_path / "input.h5ad"
    adata.write_h5ad(input_path)
    resource, lri = _resource()
    output = tmp_path / "export"
    output.mkdir()

    prepared = module._prepare_exports(
        input_path,
        resource,
        lri,
        output,
        cell_type_key="cell_type",
        condition_key="condition",
        target="case",
        reference="control",
        counts_layer="counts",
    )

    genes = prepared.paths["genes"].read_text().splitlines()
    observed = mmread(prepared.paths["counts"], spmatrix=True).tocsr()
    assert genes == ["L0", "R0", "L1", "R1"]
    assert observed.shape == (4, 4)
    np.testing.assert_array_equal(
        observed.toarray(), counts[:, [0, 1, 2, 4]].T.toarray()
    )
    assert prepared.matched_interactions == 2


def test_rankings_keep_full_universe_and_do_not_impute_ineligible_pairs() -> None:
    calls = pd.DataFrame(
        {
            "condition": ["case", "control", "case"],
            "sender": ["B", "A", "A"],
            "receiver": ["A", "B", "A"],
            "LRI": ["lr1", "lr1", "lr2"],
        }
    )

    rankings = module._build_rankings(
        calls,
        source_cell_types=("A", "B", "C"),
        eligible_cell_types=("A", "B"),
        target="case",
        reference="control",
        dataset_id="fixture",
        matched_interactions=2,
    ).set_index(["condition", "sender", "receiver"])

    assert len(rankings) == 12
    assert rankings.loc[("case", "A", "B"), "ranked_strength"] == 1
    assert rankings.loc[("case", "B", "B"), "ranked_strength"] == 0
    assert rankings.loc[("control", "A", "B"), "ranked_strength"] == 1
    assert rankings.loc[("case", "A", "C"), "status"] == "not_estimable"
    assert pd.isna(rankings.loc[("case", "A", "C"), "ranked_strength"])


@pytest.mark.parametrize("cores", [0, 9, True])
def test_run_rejects_out_of_bounds_future_workers(
    tmp_path: Path, cores: object
) -> None:
    with pytest.raises(ValueError, match="cores"):
        module.run(
            tmp_path / "input.h5ad",
            tmp_path / "output",
            input_manifest=tmp_path / "input.json",
            resource_path=tmp_path / "resource.tsv",
            resource_manifest=tmp_path / "resource.json",
            environment_manifest=tmp_path / "environment.json",
            rscript=tmp_path / "Rscript",
            dataset_id="fixture",
            condition_key="condition",
            target="case",
            reference="control",
            cores=cast(int, cores),
        )
