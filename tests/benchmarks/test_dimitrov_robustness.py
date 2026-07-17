from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from benchmarks.literature.robustness import (
    build_design,
    replace_resource_interactions,
    reshuffle_labels,
    select_top_edges,
    subsample_cells,
    top_edge_overlap,
)


def _data() -> ad.AnnData:
    labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    return ad.AnnData(
        X=np.ones((30, 4)),
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(labels)},
            index=[f"cell-{index}" for index in range(30)],
        ),
        var=pd.DataFrame(index=["L1", "R1", "G1", "G2"]),
    )


def test_paper_design_has_four_series_eight_levels_and_five_replicates() -> None:
    first = build_design()
    second = build_design()

    assert first == second
    assert len(first) == 4 * 8 * 5
    assert {item.proportion for item in first} == {
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
    }


def test_cell_perturbations_are_deterministic_and_match_requested_scope() -> None:
    data = _data()
    subset, subset_audit = subsample_cells(
        data, label_key="cell_type", proportion=0.2, seed=7
    )
    shuffled, shuffle_audit = reshuffle_labels(
        data, label_key="cell_type", proportion=0.2, seed=7
    )

    assert subset.n_obs == 24
    assert set(subset.obs["cell_type"].value_counts()) == {8}
    assert np.isclose(subset_audit["actual_proportion"], 0.2)
    original = data.obs["cell_type"].astype(str).to_numpy()
    changed = shuffled.obs["cell_type"].astype(str).to_numpy() != original
    assert changed.sum() == 6
    assert np.isclose(shuffle_audit["actual_proportion"], 0.2)


def test_resource_replacement_preserves_requested_lr_and_uses_new_gene_pools() -> None:
    resource = pd.DataFrame(
        {"ligand": ["L1", "L2", "L3", "L4"], "receptor": ["R1", "R2", "R3", "R4"]}
    )
    replaced, audit = replace_resource_interactions(
        resource,
        genes=tuple(f"G{index}" for index in range(20)),
        proportion=0.5,
        seed=11,
        preserved_pairs=frozenset({("L1", "R1")}),
    )

    assert tuple(replaced.iloc[0]) == ("L1", "R1")
    assert audit["replaced_rows"] == 2
    assert len(replaced) == len(resource)
    assert not replaced.duplicated().any()
    changed = (replaced != resource).any(axis=1)
    assert changed.sum() == 2
    assert set(replaced.loc[changed, "ligand"]).isdisjoint({"L1", "L2", "L3", "L4"})
    assert set(replaced.loc[changed, "receptor"]).isdisjoint({"R1", "R2", "R3", "R4"})
    assert set(replaced.loc[changed, "ligand"]).isdisjoint(
        set(replaced.loc[changed, "receptor"])
    )


def test_top_selection_and_overlap_use_direction_and_full_edge_identity() -> None:
    scores = pd.DataFrame(
        {
            "method": ["higher"] * 3 + ["lower"] * 3,
            "source": ["A"] * 6,
            "target": ["B"] * 6,
            "ligand": ["L1", "L2", "L3"] * 2,
            "receptor": ["R1", "R2", "R3"] * 2,
            "score": [3.0, 2.0, 1.0, 0.03, 0.02, 0.01],
            "score_direction": ["higher"] * 3 + ["lower"] * 3,
        }
    )
    top = select_top_edges(scores, top_n=2)

    assert list(top.loc[top["method"].eq("higher"), "ligand"]) == ["L1", "L2"]
    assert list(top.loc[top["method"].eq("lower"), "ligand"]) == ["L3", "L2"]
    overlap = top_edge_overlap(
        top.loc[top["method"].eq("higher")],
        top.loc[top["method"].eq("lower")],
    )
    assert overlap["intersection_edges"] == 1
    assert overlap["baseline_recovery"] == 0.5
    assert np.isclose(overlap["jaccard"], 1 / 3)
