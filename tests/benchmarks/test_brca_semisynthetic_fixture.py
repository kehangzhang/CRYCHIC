from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.comprehensive.generate_brca_semisynthetic import generate


def _write_source(path: Path) -> tuple[str, ...]:
    ligands = tuple(f"L{index}" for index in range(8))
    receptors = tuple(f"R{index}" for index in range(8))
    targets = tuple(f"T{index}" for index in range(8))
    genes = (*ligands, *receptors, *targets, "BG1", "BG2")
    gene_index = {gene: index for index, gene in enumerate(genes)}
    count_parts: list[np.ndarray] = []
    observations: list[pd.DataFrame] = []
    for expansion in ("E", "NE"):
        for subject_index in range(4):
            subject = f"{expansion}{subject_index}"
            for timepoint in ("Pre", "On"):
                condition = f"{timepoint}{expansion}"
                sample = f"{subject}_{timepoint}"
                for cell_type in ("A", "B"):
                    counts = np.ones((12, len(genes)), dtype=np.int32)
                    counts[:, gene_index["BG1"]] = 8
                    counts[:, gene_index["BG2"]] = 6
                    if cell_type == "A":
                        counts[:, : len(ligands)] = 3
                    else:
                        counts[:, len(ligands) : len(ligands) + len(receptors)] = 3
                        counts[
                            :,
                            len(ligands)
                            + len(receptors) : len(ligands)
                            + len(receptors)
                            + len(targets),
                        ] = 3
                    count_parts.append(counts)
                    observations.append(
                        pd.DataFrame(
                            {
                                "sample_id": sample,
                                "subject_id": subject,
                                "family_id": subject,
                                "timepoint": timepoint,
                                "expansion": expansion,
                                "condition": condition,
                                "cell_type": cell_type,
                            },
                            index=[
                                f"{sample}_{cell_type}_{cell}" for cell in range(12)
                            ],
                        )
                    )
    counts = sparse.csr_matrix(np.vstack(count_parts), dtype=np.int32)
    library = np.asarray(counts.sum(axis=1)).ravel()
    normalized = counts.astype(float).multiply((1.0e4 / library)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    obs = pd.concat(observations)
    obs.index = pd.Index(obs.index, dtype="string")
    for column in obs:
        obs[column] = obs[column].astype("string")
    source = ad.AnnData(
        X=normalized,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene", dtype="string")),
    )
    source.layers["counts"] = counts
    with ad.settings.override(allow_write_nullable_strings=True):
        source.write_h5ad(path)
    return genes


def test_brca_semisynthetic_fixture_is_truth_and_checksum_bound(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.h5ad"
    _write_source(source_path)
    resource = pd.DataFrame(
        {
            "harmonized_interaction_id": [f"I{index}" for index in range(8)],
            "ligand": [f"L{index}" for index in range(8)],
            "receptor": [f"R{index}" for index in range(8)],
            "cellchat_source_interaction_id": [f"C{index}" for index in range(8)],
            "scseqcommdiff_covered": ["true"] * 8,
        }
    )
    resource_path = tmp_path / "harmonized_lr.tsv"
    resource.to_csv(resource_path, sep="\t", index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"license": "fixture"}), encoding="utf-8"
    )
    prior = pd.DataFrame(
        {
            "ligand": [f"L{index}" for index in range(8)],
            "target": [f"T{index}" for index in range(8)],
            "weight": np.linspace(1.0, 0.5, 8),
            "rank": [1] * 8,
        }
    )
    prior_path = tmp_path / "prior.parquet"
    prior.to_parquet(prior_path, index=False)
    output = tmp_path / "fixture"

    manifest = generate(
        source_path,
        resource_path,
        prior_path,
        output,
        seeds=(17,),
        minimum_source_cells=10,
        cap_per_sample_cell_type=10,
        n_interactions=8,
        events_per_class=1,
        targets_per_ligand=1,
        background_genes=2,
        ligand_add=4,
        receptor_add=4,
        target_add=2,
    )

    assert manifest["status"] == "complete"
    assert Path(manifest["database_root"]).is_dir()
    config = json.loads((output / "crychic_config.json").read_text(encoding="utf-8"))
    assert config["database_root"] == manifest["database_root"]
    assert manifest["design"]["subjects_by_expansion"] == {"E": 4, "NE": 4}
    assert manifest["records"][0]["shape"] == [320, 26]
    bundle = harmonized_resource_bundle(
        output / manifest["resource"]["filename"],
        output / manifest["resource"]["manifest"],
    )
    assert len(bundle.interactions) == 8
    truth = pd.read_csv(output / "event_truth.tsv", sep="\t")
    plan = pd.read_csv(output / "injection_plan.tsv", sep="\t")
    assert len(truth) == 32
    assert int(truth["truth_label"].sum()) == 2
    assert plan["event_class"].value_counts().to_dict() == {
        "no_effect": 4,
        "positive_did": 1,
        "negative_did": 1,
        "time_main_only": 1,
        "expansion_main_only": 1,
    }
    assert set(truth.loc[truth["truth_label"].eq(1), "truth_direction"]) == {-1, 1}
    for name, record in manifest["outputs"].items():
        observed = hashlib.sha256(
            (output / record["filename"]).read_bytes()
        ).hexdigest()
        assert observed == record["sha256"], name

    dataset = manifest["records"][0]
    fixture = ad.read_h5ad(output / dataset["h5ad"])
    counts = sparse.csr_matrix(fixture.layers["counts"])
    assert counts.dtype.kind in {"i", "u"}
    assert np.all(counts.data >= 0)
    assert np.any(np.abs(fixture.X.data - np.round(fixture.X.data)) > 1e-6)
    positive = plan.loc[plan["event_class"].eq("positive_did")].iloc[0]
    ligand = str(positive["ligand"])
    sender = str(positive["selected_sender"])
    gene = fixture.var_names.get_loc(ligand)
    mask = fixture.obs["cell_type"].astype(str).eq(sender)
    on_e = mask & fixture.obs["condition"].astype(str).eq("OnE")
    pre_e = mask & fixture.obs["condition"].astype(str).eq("PreE")
    assert float(counts[on_e.to_numpy(), gene].mean()) > float(
        counts[pre_e.to_numpy(), gene].mean()
    )

    expected_conditions = {"E": {"PreE", "OnE"}, "NE": {"PreNE", "OnNE"}}
    for expansion, pair_record in dataset["pairwise_inputs"].items():
        pair_path = output / pair_record["output"]["path"]
        pair_manifest_path = output / pair_record["manifest"]["path"]
        assert hashlib.sha256(pair_path.read_bytes()).hexdigest() == pair_record[
            "output"
        ]["sha256"]
        assert hashlib.sha256(pair_manifest_path.read_bytes()).hexdigest() == (
            pair_record["manifest"]["sha256"]
        )
        pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
        assert pair_manifest["output"]["filename"] == pair_path.name
        assert pair_manifest["output"]["sha256"] == pair_record["output"]["sha256"]
        pair = ad.read_h5ad(pair_path)
        assert pair.shape == (160, 26)
        assert set(pair.obs["condition"].astype(str)) == expected_conditions[expansion]
        pair_design = pair.obs[["sample_id", "condition"]].astype(str).drop_duplicates()
        assert not pair_design.duplicated("sample_id", keep=False).any()
        assert pair_record["dimensions"]["units_by_condition"] == {
            condition: 4 for condition in expected_conditions[expansion]
        }

    null_output = tmp_path / "null_fixture"
    null_manifest = generate(
        source_path,
        resource_path,
        prior_path,
        null_output,
        seeds=(23,),
        fixture_mode="randomized_global_null",
        minimum_source_cells=10,
        cap_per_sample_cell_type=10,
        n_interactions=8,
        events_per_class=1,
        targets_per_ligand=1,
        background_genes=2,
        ligand_add=4,
        receptor_add=4,
        target_add=2,
    )
    assert null_manifest["fixture_mode"] == "randomized_global_null"
    assert not null_manifest["injection"]["enabled"]
    assert null_manifest["records"][0]["expansion_assignment_digest"].startswith(
        "expansion_assignment_"
    )
    null_truth = pd.read_csv(null_output / "event_truth.tsv", sep="\t")
    null_plan = pd.read_csv(null_output / "injection_plan.tsv", sep="\t")
    assert not null_truth["truth_label"].any()
    assert set(null_plan["event_class"]) == {"no_effect"}
    null_fixture = ad.read_h5ad(
        null_output / null_manifest["records"][0]["h5ad"]
    )
    assert not null_fixture.uns["semisynthetic_contract"][
        "count_injection_enabled"
    ]
    assert (
        null_fixture.obs["timepoint"].astype(str)
        + null_fixture.obs["expansion"].astype(str)
        == null_fixture.obs["condition"].astype(str)
    ).all()
