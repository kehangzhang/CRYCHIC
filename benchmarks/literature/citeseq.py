"""CITE-seq receptor-protein truth from Dimitrov et al.

Antibody-derived tag (ADT) abundance is z-scored across cell clusters for each
surface protein. A protein-cluster pair is positive when z >= 1.645. This is a
protein-specificity proxy, not direct evidence that any ligand-receptor event
occurred.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

RECEPTOR_SPECIFICITY_Z_THRESHOLD = 1.645

# The paper used org.Hs.eg.db aliases and manually resolved non-standard ADT
# labels. This compact map covers the four public 10x human datasets used in
# the small-scale reproduction. Values deliberately expand ambiguous proteins.
DEFAULT_HUMAN_ADT_RECEPTOR_ALIASES: dict[str, tuple[str, ...]] = {
    "CD3": ("CD3D", "CD3E", "CD3G", "CD247"),
    "CD4": ("CD4",),
    "CD8A": ("CD8A",),
    "CD11B": ("ITGAM",),
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


def _canonical_adt_name(value: object) -> str:
    name = str(value).strip().split("_", maxsplit=1)[0]
    if not name:
        raise ValueError("ADT names must be non-empty")
    if name.upper().endswith("CONTROL") or "_CONTROL" in str(value).upper():
        return ""
    upper = name.upper()
    if upper == "CD8A":
        return "CD8A"
    return upper if upper not in {"PD-1", "HLA-DR"} else name.upper()


def _normalized_aliases(
    aliases: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for raw_adt, raw_genes in aliases.items():
        adt = _canonical_adt_name(raw_adt)
        genes = tuple(
            sorted(
                {str(gene).strip().upper() for gene in raw_genes if str(gene).strip()}
            )
        )
        if not adt or not genes:
            raise ValueError("ADT alias mappings require non-empty names and genes")
        result[adt] = genes
    return result


def build_citeseq_receptor_truth(
    adt_cluster_means: pd.DataFrame,
    *,
    dataset: str,
    protein_column: str = "protein",
    cluster_column: str = "receiver",
    abundance_column: str = "adt_mean",
    aliases: Mapping[str, Sequence[str]] = DEFAULT_HUMAN_ADT_RECEPTOR_ALIASES,
    z_threshold: float = RECEPTOR_SPECIFICITY_Z_THRESHOLD,
) -> pd.DataFrame:
    """Create alias-expanded receptor labels from long-form ADT cluster means."""

    required = {protein_column, cluster_column, abundance_column}
    missing = required.difference(adt_cluster_means.columns)
    if missing:
        raise ValueError(f"ADT cluster means are missing columns: {sorted(missing)}")
    if not dataset.strip():
        raise ValueError("dataset must be non-empty")
    if not np.isfinite(z_threshold):
        raise ValueError("z_threshold must be finite")
    source = adt_cluster_means.loc[:, list(required)].rename(
        columns={
            protein_column: "protein",
            cluster_column: "receiver",
            abundance_column: "adt_mean",
        }
    )
    if source[["protein", "receiver"]].isna().any().any():
        raise ValueError("ADT protein and receiver identifiers must not be missing")
    source["protein"] = source["protein"].map(_canonical_adt_name)
    source = source.loc[source["protein"].ne("")].copy()
    source["receiver"] = source["receiver"].astype(str).str.strip()
    source["adt_mean"] = pd.to_numeric(source["adt_mean"], errors="coerce")
    if source["receiver"].eq("").any() or source["adt_mean"].isna().any():
        raise ValueError("ADT receivers must be non-empty and means must be numeric")
    if np.isinf(source["adt_mean"]).any():
        raise ValueError("ADT means must be finite")
    if source.duplicated(["protein", "receiver"]).any():
        raise ValueError("ADT cluster means contain duplicate protein-receiver rows")

    grouped = source.groupby("protein", sort=False, observed=True)["adt_mean"]
    mean = grouped.transform("mean")
    standard_deviation = grouped.transform("std")
    source["adt_z"] = (source["adt_mean"] - mean) / standard_deviation
    source["truth_status"] = np.where(
        standard_deviation.gt(0) & standard_deviation.notna(),
        "observed",
        "not_estimable_constant_or_single_cluster_protein",
    )
    source["is_positive"] = (
        source["adt_z"].ge(z_threshold) & source["truth_status"].eq("observed")
    ).astype(int)

    alias_map = _normalized_aliases(aliases)
    records: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        for receptor in alias_map.get(str(row.protein), ()):
            records.append(
                {
                    "dataset": dataset,
                    "protein": row.protein,
                    "receiver": row.receiver,
                    "receptor": receptor,
                    "adt_mean": float(str(row.adt_mean)),
                    "adt_z": float(str(row.adt_z)),
                    "z_threshold": float(z_threshold),
                    "is_positive": int(str(row.is_positive)),
                    "truth_status": row.truth_status,
                }
            )
    if not records:
        raise ValueError("no ADT proteins matched the supplied receptor aliases")
    result = pd.DataFrame.from_records(records)
    if result.duplicated(["dataset", "receiver", "receptor"]).any():
        # CD45RA and CD45RO both map to PTPRC. Preserve the strongest protein
        # evidence for the receptor-cluster label without double counting it.
        result = (
            result.sort_values(
                ["dataset", "receiver", "receptor", "adt_z"],
                ascending=[True, True, True, False],
                kind="stable",
            )
            .drop_duplicates(["dataset", "receiver", "receptor"], keep="first")
            .reset_index(drop=True)
        )
    return result.sort_values(
        ["dataset", "receiver", "receptor"], kind="stable", ignore_index=True
    )


def label_citeseq_method_scores(
    scores: pd.DataFrame,
    receptor_truth: pd.DataFrame,
    *,
    dataset_column: str = "dataset",
    receiver_column: str = "receiver",
    receptor_column: str = "receptor",
) -> pd.DataFrame:
    """Attach ADT receptor-specificity labels to communication score rows."""

    score_required = {dataset_column, receiver_column, receptor_column}
    truth_required = {"dataset", "receiver", "receptor", "is_positive", "truth_status"}
    missing_score = score_required.difference(scores.columns)
    missing_truth = truth_required.difference(receptor_truth.columns)
    if missing_score:
        raise ValueError(f"method scores are missing columns: {sorted(missing_score)}")
    if missing_truth:
        raise ValueError(f"receptor truth is missing columns: {sorted(missing_truth)}")
    source = scores.rename(
        columns={
            dataset_column: "dataset",
            receiver_column: "receiver",
            receptor_column: "receptor",
        }
    ).copy(deep=True)
    for column in ("dataset", "receiver", "receptor"):
        source[column] = source[column].astype(str).str.strip()
    source["receptor"] = source["receptor"].str.upper()
    truth = receptor_truth.loc[:, list(truth_required)].copy(deep=True)
    return source.merge(
        truth,
        on=["dataset", "receiver", "receptor"],
        how="inner",
        validate="many_to_one",
    )


__all__ = [
    "DEFAULT_HUMAN_ADT_RECEPTOR_ALIASES",
    "RECEPTOR_SPECIFICITY_Z_THRESHOLD",
    "build_citeseq_receptor_truth",
    "label_citeseq_method_scores",
]
