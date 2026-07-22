from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import pandas as pd
from benchmarks.adapters.common import sha256_file
from benchmarks.comprehensive.generate_three_group_fixture import generate


def _resource(path: Path) -> Path:
    table = pd.DataFrame(
        {
            "harmonized_interaction_id": [
                "CXCL10__CXCR3",
                "CCL5__CCR5",
                "VEGFA__FLT1",
                "CXCL12__CXCR4",
                "EGF__EGFR",
            ],
            "ligand": ["CXCL10", "CCL5", "VEGFA", "CXCL12", "EGF"],
            "receptor": ["CXCR3", "CCR5", "FLT1", "CXCR4", "EGFR"],
            "cellchat_source_interaction_id": [f"cc{i}" for i in range(5)],
            "cellphonedb_source_interaction_id": [f"cp{i}" for i in range(5)],
            "liana_source_interaction_id": [f"li{i}" for i in range(5)],
            "liana_covered": [True] * 5,
            "scseqcommdiff_source_interaction_id": [f"sq{i}" for i in range(5)],
            "scseqcommdiff_covered": [True] * 5,
        }
    )
    table.to_csv(path, sep="\t", index=False)
    path.with_name("manifest.json").write_text("{}\n", encoding="utf-8")
    return path


def test_generate_three_group_fixture_is_published_and_checksum_bound(
    tmp_path: Path,
) -> None:
    resource = _resource(tmp_path / "harmonized_lr.tsv")
    output = tmp_path / "fixture"

    manifest = generate(
        output,
        resource,
        seeds=(17,),
        group_subjects={"A": 8, "B": 9, "C": 10},
        mean_cells_per_sample=60,
    )

    assert manifest["status"] == "complete"
    assert len(manifest["records"]) == 2
    assert (output / "manifest.json").is_file()
    assert (
        sha256_file(output / "event_truth.tsv")
        == manifest["outputs"]["truth"]["sha256"]
    )

    config = json.loads((output / "crychic_config.json").read_text(encoding="utf-8"))
    assert len(config["datasets"]) == 2
    assert {value["benchmark_scope"] for value in config["datasets"].values()} == {
        "independent_subject_three_group_hcommon",
    }
    for dataset in config["datasets"].values():
        input_path = Path(dataset["input"])
        assert input_path.is_file()
        assert sha256_file(input_path) == dataset["input_sha256"]

    truth = pd.read_csv(output / "event_truth.tsv", sep="\t")
    active = truth.loc[truth["scenario"].eq("active")]
    positives = active.groupby("contrast", observed=True)["truth_label"].sum()
    assert positives.to_dict() == {"B_vs_A": 2, "C_vs_A": 2, "C_vs_B": 3}
    assert not truth.loc[truth["scenario"].eq("global_null"), "truth_label"].any()

    for record in manifest["records"]:
        full = ad.read_h5ad(output / record["h5ad"], backed="r")
        try:
            assert set(full.obs["condition"].astype(str)) == {"A", "B", "C"}
            assert full.obs["subject_id"].nunique() == 27
            assert (
                full.obs.groupby("condition", observed=True)["subject_id"]
                .nunique()
                .to_dict()
                == {"A": 8, "B": 9, "C": 10}
            )
            assert "simulation_truth" not in full.uns
            assert full.uns["expression_contract"]["x_semantics"] == (
                "log1p_counts_per_10000"
            )
            assert not pd.api.types.is_integer_dtype(full.X.dtype)
            assert pd.api.types.is_integer_dtype(full.layers["counts"].dtype)
            assert full.n_vars == 23
        finally:
            full.file.close()
        for pair_record in record["pairs"]:
            pair = ad.read_h5ad(output / pair_record["h5ad"], backed="r")
            pair_manifest = json.loads(
                (output / pair_record["manifest"]).read_text(encoding="utf-8")
            )
            try:
                assert set(pair.obs["condition"].astype(str)) == {
                    pair_record["target"],
                    pair_record["reference"],
                }
                expected_subjects = {"A": 8, "B": 9, "C": 10}
                assert pair.obs["subject_id"].nunique() == (
                    expected_subjects[pair_record["target"]]
                    + expected_subjects[pair_record["reference"]]
                )
                pair_path = output / pair_record["h5ad"]
                assert sha256_file(pair_path) == pair_manifest["output"]["sha256"]
            finally:
                pair.file.close()
