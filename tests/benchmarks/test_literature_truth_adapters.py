from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.literature.citeseq import (
    build_citeseq_receptor_truth,
    label_citeseq_method_scores,
)
from benchmarks.literature.ipf import (
    build_ipf_truth_universe,
    load_ipf_gold_standard,
)


def test_ipf_loader_deduplicates_utf16_release(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "source": ["AT1", "AT1", "AT1"],
            "target": ["AT2", "AT2", "AT1"],
            "ligand": ["tgfb1", "tgfb1", "hbegf"],
            "receptor": ["tgfbr1", "tgfbr1", "egfr"],
        }
    )
    path = tmp_path / "gold.txt"
    source.to_csv(path, sep="\t", index=False, encoding="utf-16")

    result = load_ipf_gold_standard(path, require_published_shape=False)

    assert len(result) == 2
    assert set(result["ligand"]) == {"TGFB1", "HBEGF"}
    assert set(result["is_positive"]) == {1}


def test_ipf_universe_crosses_cell_pairs_with_intact_ligand_receptor_pairs() -> None:
    gold = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "A"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
        }
    )
    predictions = pd.DataFrame(
        {"source": ["C"], "target": ["A"], "ligand": ["LX"], "receptor": ["RX"]}
    )

    result = build_ipf_truth_universe(gold, predicted_edges=predictions)

    assert len(result) == 9
    assert result["is_positive"].sum() == 2
    assert not (result["ligand"].eq("L1") & result["receptor"].eq("R2")).any()
    added = result.loc[result["source"].eq("C")].iloc[0]
    assert added["is_positive"] == 0


def test_citeseq_truth_uses_sample_sd_threshold_and_alias_expansion() -> None:
    means = pd.DataFrame(
        {
            "protein": ["CD3_TotalSeqB"] * 3 + ["CD45RA_TotalSeqB"] * 3,
            "receiver": ["c1", "c2", "c3"] * 2,
            "adt_mean": [0.0, 0.0, 10.0, 0.0, 0.0, 10.0],
        }
    )
    aliases = {"CD3": ("CD3D", "CD3E"), "CD45RA": ("PTPRC",)}

    result = build_citeseq_receptor_truth(
        means,
        dataset="pbmc",
        aliases=aliases,
        z_threshold=1.0,
    )

    assert len(result) == 9
    positives = result.loc[result["is_positive"].eq(1)]
    assert set(positives["receiver"]) == {"c3"}
    assert set(positives["receptor"]) == {"CD3D", "CD3E", "PTPRC"}
    assert result["adt_z"].max() == pytest.approx(2 / 3**0.5)


def test_citeseq_constant_protein_is_not_positive() -> None:
    means = pd.DataFrame(
        {
            "protein": ["CD4", "CD4"],
            "receiver": ["c1", "c2"],
            "adt_mean": [2.0, 2.0],
        }
    )
    result = build_citeseq_receptor_truth(
        means,
        dataset="pbmc",
        aliases={"CD4": ("CD4",)},
    )

    assert set(result["truth_status"]) == {
        "not_estimable_constant_or_single_cluster_protein"
    }
    assert result["is_positive"].sum() == 0


def test_citeseq_labels_only_method_edges_with_protein_truth() -> None:
    truth = pd.DataFrame(
        {
            "dataset": ["pbmc", "pbmc"],
            "receiver": ["c1", "c2"],
            "receptor": ["CD4", "CD4"],
            "is_positive": [1, 0],
            "truth_status": ["observed", "observed"],
        }
    )
    scores = pd.DataFrame(
        {
            "dataset_id": ["pbmc", "pbmc"],
            "target": ["c1", "c2"],
            "receptor_complex": ["cd4", "missing"],
            "score": [0.9, 0.1],
        }
    )

    result = label_citeseq_method_scores(
        scores,
        truth,
        dataset_column="dataset_id",
        receiver_column="target",
        receptor_column="receptor_complex",
    )

    assert len(result) == 1
    assert result.iloc[0]["is_positive"] == 1
