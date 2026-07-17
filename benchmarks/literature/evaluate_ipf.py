"""Evaluate the Xie IPF gold standard on a frozen operational universe."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from benchmarks.literature.evaluate_citeseq import (
    component_scores_from_cellchat,
    component_scores_from_liana,
    scores_from_crychic_availability,
)
from benchmarks.literature.ipf import IPF_EDGE_COLUMNS, build_ipf_truth_universe
from benchmarks.metrics import (
    LiteratureGoldStandardResult,
    LiteratureGoldStandardSpec,
    evaluate_literature_gold_standard,
)


def _canonical_scores(scores: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "resource_mode",
        "method",
        "sender",
        "receiver",
        "ligand",
        "receptor",
        "score",
        "score_direction",
        "status",
    }
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"IPF component scores are missing: {sorted(missing)}")
    result = scores.rename(columns={"sender": "source", "receiver": "target"})
    result = result.loc[
        :,
        [
            "dataset",
            "resource_mode",
            "method",
            *IPF_EDGE_COLUMNS,
            "score",
            "score_direction",
            "status",
        ],
    ].copy()
    for column in IPF_EDGE_COLUMNS:
        result[column] = result[column].astype(str).str.strip()
    result["ligand"] = result["ligand"].str.upper().str.split(r"[_&]")
    result["receptor"] = result["receptor"].str.upper().str.split(r"[_&]")
    result = result.explode("ligand", ignore_index=True).explode(
        "receptor", ignore_index=True
    )
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    result = result.loc[
        result["status"].eq("observed") & np.isfinite(result["score"])
    ].copy()
    result["score"] = np.where(
        result["score_direction"].eq("higher"),
        result["score"],
        -result["score"],
    )
    keys = ["dataset", "resource_mode", "method", *IPF_EDGE_COLUMNS]
    result = result.groupby(keys, sort=True, observed=True, as_index=False).agg(
        score=("score", "max")
    )
    result["score_direction"] = "higher"
    result["status"] = "observed"
    return cast(pd.DataFrame, result)


def restrict_ipf_universe_to_resource(
    universe: pd.DataFrame, resource: pd.DataFrame
) -> pd.DataFrame:
    """Restrict an IPF universe to intact LR pairs in a predeclared resource."""

    required = {"ligand", "receptor"}
    missing = required.difference(resource.columns)
    if missing:
        raise ValueError(f"IPF resource is missing: {sorted(missing)}")
    pairs = resource.loc[:, ["ligand", "receptor"]].copy()
    pairs["ligand"] = pairs["ligand"].astype(str).str.strip().str.upper()
    pairs["receptor"] = pairs["receptor"].astype(str).str.strip().str.upper()
    pairs = pairs.drop_duplicates(ignore_index=True)
    selected = universe.merge(
        pairs.assign(_resource_covered=True),
        on=["ligand", "receptor"],
        how="inner",
        validate="many_to_one",
    ).drop(columns="_resource_covered")
    if selected.empty or not selected["is_positive"].any():
        raise ValueError("restricted IPF universe contains no positive gold edges")
    return selected.sort_values(
        list(IPF_EDGE_COLUMNS), kind="stable", ignore_index=True
    )


def _worst_rank_sentinel(scores: pd.Series) -> float:
    minimum = float(scores.min())
    maximum = float(scores.max())
    margin = max(1.0, abs(maximum - minimum) * 0.01)
    return minimum - margin


def evaluate_ipf_components(
    scores: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    universe_mode: str,
    resource: pd.DataFrame | None = None,
    n_bootstrap: int = 200,
    n_negative_samples: int = 100,
    random_seed: int = 20260715,
) -> tuple[pd.DataFrame, LiteratureGoldStandardResult]:
    """Evaluate continuous method scores with paper-defined absent predictions."""

    canonical = _canonical_scores(scores)
    universe = build_ipf_truth_universe(gold)
    if resource is not None:
        universe = restrict_ipf_universe_to_resource(universe, resource)
    universe = universe.assign(universe_mode=universe_mode)
    n_positive = int(universe["is_positive"].sum())
    materialized: list[pd.DataFrame] = []
    results: list[LiteratureGoldStandardResult] = []
    for (dataset, resource_mode, method_name), method in canonical.groupby(
        ["dataset", "resource_mode", "method"], sort=True, observed=True
    ):
        sentinel = _worst_rank_sentinel(method["score"])
        selected = universe.merge(
            method.loc[:, [*IPF_EDGE_COLUMNS, "score"]],
            on=list(IPF_EDGE_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        selected["raw_returned"] = selected["score"].notna()
        selected["score"] = selected["score"].fillna(sentinel)
        selected["dataset"] = dataset
        selected["resource_mode"] = resource_mode
        selected["method"] = method_name
        selected["score_direction"] = "higher"
        selected["status"] = "observed"
        truth = selected.loc[
            :,
            [
                "dataset",
                "resource_mode",
                "universe_mode",
                *IPF_EDGE_COLUMNS,
                "is_positive",
            ],
        ].drop_duplicates(ignore_index=True)
        spec = LiteratureGoldStandardSpec(
            key_columns=IPF_EDGE_COLUMNS,
            method_columns=("method",),
            stratum_columns=("dataset", "resource_mode", "universe_mode"),
            top_k=n_positive,
            n_bootstrap=n_bootstrap,
            n_negative_samples=n_negative_samples,
            negative_to_positive_ratio=1.0,
            random_seed=random_seed,
        )
        result = evaluate_literature_gold_standard(selected, truth, spec)
        raw_returned = selected["raw_returned"]
        raw_positive = selected["is_positive"].eq(1)
        audit = {
            "raw_returned_edges": int(raw_returned.sum()),
            "raw_absent_edges_filled_worst": int((~raw_returned).sum()),
            "raw_returned_positive_edges": int((raw_returned & raw_positive).sum()),
            "raw_returned_negative_edges": int((raw_returned & ~raw_positive).sum()),
            "raw_positive_return_fraction": float(
                (raw_returned & raw_positive).sum() / n_positive
            ),
            "missing_score_policy_ipf": "paper_absent_prediction_ranked_after_returned",
        }
        point = result.point_estimates.assign(**audit)
        results.append(
            LiteratureGoldStandardResult(
                point_estimates=point,
                bootstrap_summary=result.bootstrap_summary,
                negative_sampling_summary=result.negative_sampling_summary,
            )
        )
        materialized.append(selected)
    if not results:
        raise ValueError("no IPF methods were available for evaluation")
    return pd.concat(materialized, ignore_index=True), LiteratureGoldStandardResult(
        point_estimates=pd.concat(
            [result.point_estimates for result in results], ignore_index=True
        ),
        bootstrap_summary=pd.concat(
            [result.bootstrap_summary for result in results], ignore_index=True
        ),
        negative_sampling_summary=pd.concat(
            [result.negative_sampling_summary for result in results], ignore_index=True
        ),
    )


def _write_result(result: LiteratureGoldStandardResult, output: Path) -> None:
    result.point_estimates.to_csv(output / "point_estimates.tsv", sep="\t", index=False)
    result.bootstrap_summary.to_csv(
        output / "bootstrap_summary.tsv", sep="\t", index=False
    )
    result.negative_sampling_summary.to_csv(
        output / "negative_sampling_summary.tsv", sep="\t", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("liana_raw", type=Path)
    parser.add_argument("gold", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--resource-mode", required=True)
    parser.add_argument("--universe-mode", required=True)
    parser.add_argument("--resource", type=Path)
    parser.add_argument("--crychic", type=Path)
    parser.add_argument("--cellchat", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--n-negative-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    scores = [
        component_scores_from_liana(
            pd.read_parquet(args.liana_raw),
            dataset=args.dataset,
            resource_mode=args.resource_mode,
        )
    ]
    if args.crychic is not None:
        scores.append(
            scores_from_crychic_availability(
                pd.read_parquet(args.crychic),
                dataset=args.dataset,
                resource_mode=args.resource_mode,
            )
        )
    if args.cellchat is not None:
        scores.append(
            component_scores_from_cellchat(
                pd.read_parquet(args.cellchat),
                dataset=args.dataset,
                resource_mode=args.resource_mode,
            )
        )
    gold = pd.read_csv(args.gold, sep="\t")
    resource = None if args.resource is None else pd.read_csv(args.resource, sep="\t")
    materialized, result = evaluate_ipf_components(
        pd.concat(scores, ignore_index=True),
        gold,
        universe_mode=args.universe_mode,
        resource=resource,
        n_bootstrap=args.n_bootstrap,
        n_negative_samples=args.n_negative_samples,
        random_seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    materialized.to_parquet(
        args.output_dir / "materialized_scores.parquet", index=False
    )
    _write_result(result, args.output_dir)


if __name__ == "__main__":
    main()


__all__ = [
    "evaluate_ipf_components",
    "restrict_ipf_universe_to_resource",
]
