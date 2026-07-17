"""Evaluate CytoSig activity agreement for CRYCHIC and LIANA methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from benchmarks.openproblems.common import sha256_file, write_json

RANDOM_AUPRC = 0.24117674611812784
RANDOM_ODDS_RATIO = 0.3269435569755058
RANK_CUTOFFS = (100, 250, 500, 1000, 2500, 5000, 10000)


def _truth(input_h5ad: Path) -> pd.DataFrame:
    data = ad.read_h5ad(input_h5ad, backed="r")
    try:
        truth = data.uns["ccc_target"].loc[
            :, ["ligand", "target", "response"]
        ].copy()
    finally:
        data.file.close()
    truth["ligand"] = truth["ligand"].astype(str)
    truth["target"] = truth["target"].astype(str)
    truth["response"] = pd.to_numeric(truth["response"], errors="raise").astype(int)
    if truth.duplicated(["ligand", "target"]).any() or not set(
        truth["response"]
    ).issubset({0, 1}):
        raise ValueError("CytoSig truth has an invalid schema")
    return truth.reset_index(drop=True)


def _crychic_scores(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    definitions = (
        (
            "crychic_availability_state",
            "CRYCHIC availability-state",
            "availability_state",
            "state_status",
        ),
        (
            "crychic_availability_ecosystem",
            "CRYCHIC availability-ecosystem",
            "availability_ecosystem",
            "ecosystem_status",
        ),
    )
    rows: list[pd.DataFrame] = []
    for method_id, method_name, score_column, status_column in definitions:
        selected = raw.loc[raw[status_column].astype(str).eq("observed")]
        rows.append(
            pd.DataFrame(
                {
                    "method_id": method_id,
                    "method_name": method_name,
                    "source": selected["source"].astype(str),
                    "target": selected["target"].astype(str),
                    "ligand": selected["ligand"].astype(str),
                    "receptor": selected["receptor"].astype(str),
                    "score": pd.to_numeric(selected[score_column], errors="coerce"),
                    "score_direction": "higher",
                    "status": "observed",
                    "implementation": "CRYCHIC current source",
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    return result.loc[result["score"].notna()].reset_index(drop=True)


def _liana_scores(path: Path) -> pd.DataFrame:
    result = pd.read_parquet(path).copy()
    required = {
        "method_id",
        "method_name",
        "source",
        "target",
        "ligand",
        "receptor",
        "score",
        "score_direction",
        "status",
    }
    if required.difference(result.columns):
        raise ValueError("standardized LIANA cytokine scores have an invalid schema")
    result["implementation"] = "LIANA 1.7.3 current Python"
    return result.loc[:, [*required, "implementation"]].copy()


def _bh(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if not valid.any():
        return result
    ordered = numeric.loc[valid].sort_values(kind="stable")
    count = len(ordered)
    adjusted = (ordered.to_numpy() * count / np.arange(1, count + 1))[::-1]
    adjusted = np.minimum.accumulate(adjusted)[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def fisher_rank_curves(scores: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    joined = scores.merge(
        truth,
        on=["ligand", "target"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (method_id, method_name), group in joined.groupby(
        ["method_id", "method_name"], sort=True, observed=True
    ):
        group = group.copy()
        group["rank"] = group["score"].rank(
            method="min", ascending=False, na_option="bottom"
        )
        total_positive = int(group["response"].eq(1).sum())
        total_negative = int(group["response"].eq(0).sum())
        max_rank = int(group["rank"].max())
        for cutoff in RANK_CUTOFFS:
            if cutoff > max_rank:
                rows.append(
                    {
                        "method_id": method_id,
                        "method_name": method_name,
                        "rank_cutoff": cutoff,
                        "max_rank": max_rank,
                        "status": "not_estimable",
                        "reason_code": "cutoff_exceeds_returned_rank_range",
                    }
                )
                continue
            top = group["rank"].le(cutoff)
            tp = int((top & group["response"].eq(1)).sum())
            fp = int((top & group["response"].eq(0)).sum())
            fn = total_positive - tp
            tn = total_negative - fp
            odds, p_value = fisher_exact([[tp, fp], [fn, tn]])
            rows.append(
                {
                    "method_id": method_id,
                    "method_name": method_name,
                    "rank_cutoff": cutoff,
                    "max_rank": max_rank,
                    "odds_ratio": float(odds),
                    "p_value": float(p_value),
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_negative": tn,
                    "total_positive": total_positive,
                    "total_negative": total_negative,
                    "status": "observed",
                    "reason_code": None,
                }
            )
    result = pd.DataFrame.from_records(rows)
    observed = result["status"].eq("observed")
    result["q_value_bh_within_cutoff"] = np.nan
    if observed.any():
        result.loc[observed, "q_value_bh_within_cutoff"] = result.loc[
            observed
        ].groupby("rank_cutoff", observed=True)["p_value"].transform(_bh)
    return result


def _unique_metrics(scores: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method_id, method_name), method in scores.groupby(
        ["method_id", "method_name"], sort=True, observed=True
    ):
        for aggregation in ("max", "sum"):
            grouped = (
                method.groupby(["ligand", "target"], observed=True)["score"]
                .agg(aggregation)
                .rename("score")
                .reset_index()
            )
            joined = truth.merge(
                grouped,
                on=["ligand", "target"],
                how="left",
                validate="one_to_one",
            )
            if joined["score"].notna().sum() == 0:
                continue
            matched_rows = int(joined["score"].notna().sum())
            minimum = float(joined["score"].dropna().min())
            joined["score"] = joined["score"].fillna(
                minimum - np.finfo(float).eps
            )
            response = joined["response"].to_numpy(int)
            score = joined["score"].to_numpy(float)
            precision, recall, _ = precision_recall_curve(response, score)
            pr_auc = float(auc(recall, precision))
            ranked = joined.sort_values("score", ascending=False)
            top_n = int(len(ranked) * 0.05)
            top = ranked.iloc[:top_n]
            rest = ranked.iloc[top_n:]
            tp = int(top["response"].eq(1).sum())
            fp = int(top["response"].eq(0).sum())
            fn = int(rest["response"].eq(1).sum())
            tn = int(rest["response"].eq(0).sum())
            numerator = tp * tn
            denominator = fp * fn
            raw_or = (
                np.nan
                if numerator == 0 and denominator == 0
                else np.inf
                if denominator == 0
                else numerator / denominator
            )
            transformed_or = (
                np.nan
                if np.isnan(raw_or)
                else 1.0
                if np.isinf(raw_or)
                else 1.0 - 1.0 / (1.0 + raw_or / 2.0)
            )
            scaled_pr = (pr_auc - RANDOM_AUPRC) / (1.0 - RANDOM_AUPRC)
            scaled_or = (transformed_or - RANDOM_ODDS_RATIO) / (
                1.0 - RANDOM_ODDS_RATIO
            )
            rows.append(
                {
                    "method_id": method_id,
                    "method_name": method_name,
                    "aggregation": aggregation,
                    "openproblems_score": float(np.nanmean([scaled_pr, scaled_or])),
                    "precision_recall_auc": float(scaled_pr),
                    "odds_ratio": float(scaled_or),
                    "precision_recall_auc_raw": pr_auc,
                    "odds_ratio_raw": transformed_or,
                    "auroc": float(roc_auc_score(response, score)),
                    "average_precision": float(
                        average_precision_score(response, score)
                    ),
                    "truth_rows": len(joined),
                    "truth_positive": int(response.sum()),
                    "matched_rows": matched_rows,
                    "top_n": top_n,
                    "top_true_positive": tp,
                }
            )
    return pd.DataFrame.from_records(rows)


def _official_reference(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame.from_records(
        {
            "method_id": item["method_id"],
            "openproblems_score": item["mean_score"],
            "precision_recall_auc": item["scaled_scores"]["auprc"],
            "odds_ratio": item["scaled_scores"]["odds_ratio"],
            "precision_recall_auc_raw": item["metric_values"]["auprc"],
            "odds_ratio_raw": item["metric_values"]["odds_ratio"],
            "code_version": item["code_version"],
            "commit_sha": item["commit_sha"],
        }
        for item in payload
        if item.get("dataset_id") == "tnbc_data"
    )


def _json_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    safe = table.astype("object").where(pd.notna(table), None)
    records = safe.to_dict(orient="records")
    for record in records:
        for key, value in tuple(record.items()):
            if isinstance(value, float) and not np.isfinite(value):
                record[key] = None
    return records


def run(
    input_h5ad: Path,
    crychic_scores: Path,
    liana_scores: Path,
    official_results: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"cytokine evaluation exists: {manifest_path}")
    truth = _truth(input_h5ad)
    scores = pd.concat(
        (_crychic_scores(crychic_scores), _liana_scores(liana_scores)),
        ignore_index=True,
    )
    curves = fisher_rank_curves(scores, truth)
    unique = _unique_metrics(scores, truth)
    reference = _official_reference(official_results)
    score_path = output_dir / "standardized_lr_scores.parquet"
    curves_path = output_dir / "paper_fisher_or_curves.tsv"
    unique_path = output_dir / "controlled_unique_ligand_target_metrics.tsv"
    reference_path = output_dir / "official_v1_reference.tsv"
    scores.to_parquet(score_path, index=False)
    curves.to_csv(curves_path, sep="\t", index=False, na_rep="")
    unique.to_csv(unique_path, sep="\t", index=False, na_rep="")
    reference.to_csv(reference_path, sep="\t", index=False, na_rep="")
    manifest: dict[str, object] = {
        "schema_version": "crychic-cytokine-activity-evaluation-v1",
        "status": "complete",
        "input": {
            "filename": input_h5ad.name,
            "sha256": sha256_file(input_h5ad),
            "truth_rows": len(truth),
            "truth_positive": int(truth["response"].sum()),
        },
        "paper_protocol": {
            "grain": "returned LR rows joined on ligand and receiver target",
            "rank_cutoffs": list(RANK_CUTOFFS),
            "test": "two-sided Fisher exact test on top-vs-remainder",
            "multiplicity": "BH across methods separately at each cutoff",
        },
        "controlled_protocol": {
            "grain": "unique ligand x target",
            "aggregations": ["max", "sum"],
            "metrics": [
                "OpenProblems v1 scaled PR-AUC",
                "OpenProblems v1 scaled odds ratio",
                "AUROC",
                "average precision",
            ],
            "official_random_raw": {
                "precision_recall_auc": RANDOM_AUPRC,
                "odds_ratio": RANDOM_ODDS_RATIO,
            },
        },
        "inputs": {
            "crychic_scores": {
                "filename": crychic_scores.name,
                "sha256": sha256_file(crychic_scores),
            },
            "liana_scores": {
                "filename": liana_scores.name,
                "sha256": sha256_file(liana_scores),
            },
            "official_results": {
                "filename": official_results.name,
                "sha256": sha256_file(official_results),
            },
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (score_path, curves_path, unique_path, reference_path)
        },
        "controlled_results": _json_records(unique),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("crychic_scores", type=Path)
    parser.add_argument("liana_scores", type=Path)
    parser.add_argument("official_results", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.input_h5ad,
        args.crychic_scores,
        args.liana_scores,
        args.official_results,
        args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
