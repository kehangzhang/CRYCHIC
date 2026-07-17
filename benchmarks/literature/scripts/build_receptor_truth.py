#!/usr/bin/env python3
"""Build Dimitrov-style receptor-protein truth tables from prepared ADT means."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

THRESHOLD = 1.645

# Canonical receptor-oriented interpretation of the paper's org.Hs.eg.db query
# plus its manual_list. Spurious alias collisions returned by org.Hs.eg.db
# (e.g. CD20 -> KRT20 and PD-1 -> RPL17) are intentionally excluded, matching
# the downstream receptor-universe filter in get_adt_means().
ADT_TO_RECEPTOR: dict[str, tuple[str, ...]] = {
    "CD3": ("CD3D", "CD3E", "CD3G", "CD247"),
    "CD4": ("CD4",),
    "CD8": ("CD8A",),
    "CD8a": ("CD8A",),
    "CD11b": ("ITGAM",),
    "CD11c": ("ITGAX",),
    "CD14": ("CD14",),
    "CD15": ("FUT4",),
    "CD16": ("FCGR3A", "FCGR3B"),
    "CD19": ("CD19",),
    "CD20": ("MS4A1",),
    "CD25": ("IL2RA",),
    "CD27": ("CD27",),
    "CD28": ("CD28",),
    "CD34": ("CD34",),
    "CD45RA": ("PTPRC",),
    "CD45RO": ("PTPRC",),
    "CD56": ("NCAM1",),
    "CD62L": ("SELL",),
    "CD69": ("CD69",),
    "CD80": ("CD80",),
    "CD86": ("CD86",),
    "CD127": ("IL7R",),
    "CD137": ("TNFRSF9",),
    "CD197": ("CCR7",),
    "CD274": ("CD274",),
    "CD278": ("ICOS",),
    "CD335": ("NCR1",),
    "PD-1": ("PDCD1",),
    "HLA-DR": ("CD74",),
    "TIGIT": ("TIGIT",),
}

MANUAL_FEATURES = {"CD3", "CD45RA", "CD45RO", "HLA-DR"}


def alias_table(
    dataset_id: str,
    observed_features: list[str],
    *,
    dataset_dir: Path | None = None,
) -> pd.DataFrame:
    frozen_path = None if dataset_dir is None else dataset_dir / "adt_aliases_input.tsv"
    if frozen_path is not None and frozen_path.is_file():
        aliases = pd.read_csv(frozen_path, sep="\t", dtype=str)
        required = {"adt_feature", "receptor_gene", "mapping_source"}
        missing_columns = required.difference(aliases.columns)
        if missing_columns or aliases.empty:
            raise ValueError(
                f"{dataset_id}: frozen ADT aliases are invalid: "
                f"{sorted(missing_columns)}"
            )
        aliases = aliases.loc[:, sorted(required)].copy()
        aliases["adt_feature"] = aliases["adt_feature"].astype(str)
        aliases["receptor_gene"] = aliases["receptor_gene"].astype(str)
        aliases["mapping_source"] = aliases["mapping_source"].astype(str)
        unknown = sorted(set(aliases["adt_feature"]) - set(observed_features))
        if unknown:
            raise ValueError(
                f"{dataset_id}: frozen aliases contain unobserved ADTs: {unknown}"
            )
        if aliases.duplicated(["adt_feature", "receptor_gene"]).any():
            raise ValueError(
                f"{dataset_id}: frozen ADT aliases contain duplicate pairs"
            )
        aliases.insert(0, "dataset_id", dataset_id)
        return aliases.loc[
            :, ["dataset_id", "adt_feature", "receptor_gene", "mapping_source"]
        ].sort_values(["adt_feature", "receptor_gene"], ignore_index=True)

    missing = sorted(set(observed_features) - set(ADT_TO_RECEPTOR))
    if missing:
        raise ValueError(f"{dataset_id}: unmapped ADT features: {missing}")
    rows = []
    for feature in observed_features:
        for receptor in ADT_TO_RECEPTOR[feature]:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "adt_feature": feature,
                    "receptor_gene": receptor,
                    "mapping_source": (
                        "paper_manual_list_canonicalized"
                        if feature in MANUAL_FEATURES
                        else "org.Hs.eg.db_alias_receptor_filtered"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_one(dataset_dir: Path, dataset_id: str) -> dict[str, object]:
    means_path = dataset_dir / "adt_cluster_means.tsv"
    cells_path = dataset_dir / "cell_clusters.tsv"
    means = pd.read_csv(means_path, sep="\t")
    cells = pd.read_csv(cells_path, sep="\t", dtype=str)
    required = {
        "dataset_id",
        "cluster_id",
        "adt_feature",
        "mean_clr",
        "n_cells",
        "z_across_clusters",
    }
    if not required.issubset(means.columns):
        raise ValueError(
            f"{dataset_id}: ADT means lack {required - set(means.columns)}"
        )
    if means["z_across_clusters"].isna().any():
        raise ValueError(f"{dataset_id}: undefined protein z scores")

    observed_features = sorted(means["adt_feature"].astype(str).unique())
    aliases = alias_table(
        dataset_id,
        observed_features,
        dataset_dir=dataset_dir,
    )
    aliases.to_csv(dataset_dir / "adt_aliases.tsv", sep="\t", index=False)

    expanded = means.merge(
        aliases[["adt_feature", "receptor_gene", "mapping_source"]],
        on="adt_feature",
        how="inner",
        validate="many_to_many",
    )
    if expanded.empty:
        raise ValueError(f"{dataset_id}: no ADT features map to receptor truth")
    expanded["dimitrov_threshold"] = THRESHOLD
    expanded["is_positive"] = expanded["z_across_clusters"] >= THRESHOLD
    expanded["label"] = expanded["is_positive"].astype(int)
    expanded = expanded[
        [
            "dataset_id",
            "cluster_id",
            "receptor_gene",
            "adt_feature",
            "mapping_source",
            "mean_clr",
            "z_across_clusters",
            "dimitrov_threshold",
            "is_positive",
            "label",
            "n_cells",
        ]
    ].sort_values(["cluster_id", "receptor_gene", "adt_feature"])
    expanded.to_csv(
        dataset_dir / "receptor_truth_author_rows.tsv", sep="\t", index=False
    )

    collapsed_rows = []
    for (cluster_id, receptor_gene), group in expanded.groupby(
        ["cluster_id", "receptor_gene"], sort=True, observed=True
    ):
        best = group.loc[group["z_across_clusters"].idxmax()]
        collapsed_rows.append(
            {
                "dataset_id": dataset_id,
                "target_cell_type": cluster_id,
                "receptor_gene": receptor_gene,
                "protein_z_max": float(best["z_across_clusters"]),
                "protein_mean_clr_at_max_z": float(best["mean_clr"]),
                "matched_adt_features": ",".join(sorted(group["adt_feature"].unique())),
                "n_matched_adt_features": int(group["adt_feature"].nunique()),
                "dimitrov_threshold": THRESHOLD,
                "is_positive": bool(group["is_positive"].any()),
                "label": int(group["is_positive"].any()),
            }
        )
    truth = pd.DataFrame(collapsed_rows).sort_values(
        ["target_cell_type", "receptor_gene"]
    )
    if truth.duplicated(["target_cell_type", "receptor_gene"]).any():
        raise AssertionError(f"{dataset_id}: collapsed truth keys are not unique")
    truth.to_csv(dataset_dir / "receptor_truth.tsv", sep="\t", index=False)

    return {
        "dataset_id": dataset_id,
        "n_cells": len(cells),
        "n_clusters": int(cells["cluster_id"].nunique()),
        "n_adt_features": len(observed_features),
        "n_mapped_adt_features": int(aliases["adt_feature"].nunique()),
        "n_unmapped_adt_features": len(
            set(observed_features) - set(aliases["adt_feature"])
        ),
        "n_mapped_receptor_genes": int(aliases["receptor_gene"].nunique()),
        "n_author_rows": len(expanded),
        "n_author_positive_rows": int(expanded["is_positive"].sum()),
        "n_unique_truth_rows": len(truth),
        "n_unique_positive_labels": int(truth["is_positive"].sum()),
        "truth_key_unique": True,
        "threshold": THRESHOLD,
        "truth_file": str(Path(dataset_id) / "receptor_truth.tsv"),
        "author_rows_file": str(
            Path(dataset_id) / "receptor_truth_author_rows.tsv"
        ),
        "aliases_file": str(Path(dataset_id) / "adt_aliases.tsv"),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=root / "prepared")
    parser.add_argument("--dataset", action="append")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=root / "evidence" / "receptor_truth_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_root = args.prepared_root.expanduser().resolve()
    datasets = args.dataset or sorted(
        path.name
        for path in prepared_root.iterdir()
        if path.is_dir() and (path / "adt_cluster_means.tsv").is_file()
    )
    records = [
        build_one(prepared_root / dataset_id, dataset_id)
        for dataset_id in datasets
    ]
    manifest = {
        "schema_version": "liana-citeseq-receptor-truth-manifest-v1",
        "generated_on": date.today().isoformat(),
        "threshold_rule": (
            "ADT CLR mean z-scored across clusters; positive iff z >= 1.645"
        ),
        "collapse_rule": (
            "unique target_cell_type x receptor_gene; if multiple ADTs map to one "
            "gene, "
            "use maximum z and positive if any ADT is positive"
        ),
        "author_row_semantics": (
            "receptor_truth_author_rows.tsv preserves ADT-to-gene expansion before "
            "collapse"
        ),
        "records": records,
    }
    output = args.manifest_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
