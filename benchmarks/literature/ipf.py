"""Truth construction for the Xie et al. IPF CCI benchmark.

The paper evaluates source-target-ligand-receptor tetrads. Its negative universe
is not an experimentally verified negative set: it crosses all ordered cell
pairs with each intact ligand-receptor pair in the manually curated positives,
then augments that base with tetrads returned by the compared methods.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

IPF_EDGE_COLUMNS = ("source", "target", "ligand", "receptor")
IPF_CANONICAL_CELL_TYPES = (
    "AT1",
    "AT2",
    "Endothelial",
    "Fibroblast",
    "Macrophage",
    "Mast",
    "Monocyte",
    "Tcell",
)


def _canonical_edges(table: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    missing = set(IPF_EDGE_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"{table_name} is missing columns: {sorted(missing)}")
    selected = table.loc[:, list(IPF_EDGE_COLUMNS)].copy(deep=True)
    if selected.isna().any().any():
        raise ValueError(f"{table_name} edge identifiers must not be missing")
    for column in IPF_EDGE_COLUMNS:
        selected[column] = selected[column].astype(str).str.strip()
        if selected[column].eq("").any():
            raise ValueError(f"{table_name}.{column} contains empty identifiers")
    selected["ligand"] = selected["ligand"].str.upper()
    selected["receptor"] = selected["receptor"].str.upper()
    return selected.drop_duplicates(ignore_index=True)


def load_ipf_gold_standard(
    path: str | Path,
    *,
    require_published_shape: bool = True,
) -> pd.DataFrame:
    """Load and canonicalize the UTF-16 gold-standard table published by Xie.

    The upstream file contains 253 rows but three exact duplicates. The paper's
    reported gold standard and the canonical result therefore contain 250
    unique tetrads.
    """

    source = pd.read_csv(path, sep="\t", encoding="utf-16")
    source.columns = [str(column).lstrip("\ufeff") for column in source.columns]
    result = _canonical_edges(source, table_name="IPF gold standard")
    if require_published_shape:
        observed_cells = set(result["source"]).union(result["target"])
        expected_cells = set(IPF_CANONICAL_CELL_TYPES)
        if len(result) != 250 or observed_cells != expected_cells:
            raise ValueError(
                "IPF gold standard does not match the published 250-tetrad, "
                "eight-cell-type release"
            )
    result["is_positive"] = 1
    return result


def build_ipf_truth_universe(
    gold: pd.DataFrame,
    *,
    predicted_edges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the paper's labeled IPF tetrad universe.

    Additional method predictions are included even when they use a cell or
    molecule absent from the manually curated table. Such rows are labeled
    negative, matching the paper's operational definition rather than claiming
    that they are experimentally disproven interactions.
    """

    positives = _canonical_edges(gold, table_name="IPF gold standard")
    cells = tuple(sorted(set(positives["source"]).union(positives["target"])))
    ligand_receptors = tuple(
        positives.loc[:, ["ligand", "receptor"]]
        .drop_duplicates()
        .sort_values(["ligand", "receptor"], kind="stable")
        .itertuples(index=False, name=None)
    )
    cartesian = pd.DataFrame.from_records(
        (
            (source, target, ligand, receptor)
            for source, target in product(cells, cells)
            for ligand, receptor in ligand_receptors
        ),
        columns=IPF_EDGE_COLUMNS,
    )
    if predicted_edges is not None:
        predictions = _canonical_edges(
            predicted_edges,
            table_name="IPF method predictions",
        )
        universe = pd.concat((cartesian, predictions), ignore_index=True)
    else:
        universe = cartesian
    universe = universe.drop_duplicates(ignore_index=True)
    positive_keys = pd.MultiIndex.from_frame(positives.loc[:, list(IPF_EDGE_COLUMNS)])
    universe_keys = pd.MultiIndex.from_frame(universe.loc[:, list(IPF_EDGE_COLUMNS)])
    universe["is_positive"] = universe_keys.isin(positive_keys).astype(int)
    return universe.sort_values(
        list(IPF_EDGE_COLUMNS), kind="stable", ignore_index=True
    )


__all__ = [
    "IPF_CANONICAL_CELL_TYPES",
    "IPF_EDGE_COLUMNS",
    "build_ipf_truth_universe",
    "load_ipf_gold_standard",
]
