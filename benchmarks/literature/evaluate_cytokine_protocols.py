"""Audit cytokine-activity rankings under explicit Dimitrov protocol variants."""

from __future__ import annotations

import argparse
import importlib.metadata
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

from benchmarks.literature.evaluate_cytokine_activity import (
    RANK_CUTOFFS,
    _crychic_scores,
    _liana_scores,
)
from benchmarks.openproblems.common import sha256_file, write_json

KEY_COLUMNS = ("source", "target", "ligand", "receptor")
TRUTH_COLUMNS = ("ligand", "target", "response")
PRIMARY_METHOD_IDS = (
    "crychic_availability_state",
    "cellchat_composite",
    "cellphonedb_composite",
    "connectome_specificity",
    "logfc_specificity",
    "natmi_specificity",
    "singlecellsignalr_lrscore",
    "liana_specificity_consensus",
)


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


def _read_truth(path: Path) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".h5ad":
        data = ad.read_h5ad(path, backed="r")
        try:
            table = data.uns["ccc_target"].loc[:, list(TRUTH_COLUMNS)].copy()
        finally:
            data.file.close()
        source_kind = "openproblems_h5ad_binary_truth"
    else:
        table = pd.read_csv(path, sep="\t")
        source_kind = "reconstructed_activity_truth_table"
    missing = set(TRUTH_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"cytokine truth is missing columns: {sorted(missing)}")
    table = table.loc[:, list(TRUTH_COLUMNS)].copy()
    table["ligand"] = table["ligand"].astype(str)
    table["target"] = table["target"].astype(str)
    table["response"] = pd.to_numeric(table["response"], errors="raise").astype(int)
    if table.duplicated(["ligand", "target"]).any() or not set(
        table["response"]
    ).issubset({0, 1}):
        raise ValueError("cytokine truth must be unique binary ligand-target rows")
    return table.reset_index(drop=True), source_kind


def _load_manifest(path: Path, schema: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != schema or manifest.get("status") != "complete":
        raise ValueError(f"invalid run manifest: {path}")
    return manifest


def _load_and_validate_scores(
    crychic_scores: Path,
    liana_scores: Path,
    crychic_manifest_path: Path,
    liana_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    crychic_manifest = _load_manifest(
        crychic_manifest_path, "crychic-cytokine-availability-run-v1"
    )
    liana_manifest = _load_manifest(
        liana_manifest_path, "crychic-cytokine-liana-run-v1"
    )
    if sha256_file(crychic_scores) != crychic_manifest["output"]["sha256"]:
        raise ValueError("CRYCHIC score checksum does not match its manifest")
    liana_record = liana_manifest["outputs"].get(liana_scores.name)
    if liana_record is None or sha256_file(liana_scores) != liana_record["sha256"]:
        raise ValueError("LIANA score checksum does not match its manifest")
    if crychic_manifest["input"]["sha256"] != liana_manifest["input"]["sha256"]:
        raise ValueError(
            "CRYCHIC and LIANA scores were generated from different inputs"
        )
    if crychic_manifest["resource"]["sha256"] != liana_manifest["resource"]["sha256"]:
        raise ValueError("CRYCHIC and LIANA scores used different resources")

    crychic = _crychic_scores(crychic_scores)
    crychic_version = crychic_manifest.get("versions", {}).get("CRYCHIC", "unknown")
    crychic["implementation"] = f"CRYCHIC {crychic_version}"
    liana = _liana_scores(liana_scores)
    liana_version = liana_manifest.get("versions", {}).get("liana", "unknown")
    liana["implementation"] = f"LIANA {liana_version} current Python"
    scores = pd.concat((crychic, liana), ignore_index=True)
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.loc[
        scores["status"].astype(str).eq("observed") & scores["score"].notna()
    ].copy()
    return scores, crychic_manifest, liana_manifest


def _collapse_scores(scores: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    truth_ligands = set(truth["ligand"])
    truth_targets = set(truth["target"])
    matched = scores.loc[
        scores["ligand"].isin(truth_ligands) & scores["target"].isin(truth_targets)
    ].copy()
    group_columns = ["method_id", "method_name", "implementation", *KEY_COLUMNS]
    return (
        matched.groupby(group_columns, observed=True, sort=False)["score"]
        .max()
        .reset_index()
    )


def _fisher_counts(
    response: pd.Series,
    top: pd.Series,
    definition: str,
) -> tuple[float, float, dict[str, int]]:
    positive = response.eq(1)
    negative = response.eq(0)
    tp = int((top & positive).sum())
    fp = int((top & negative).sum())
    fn = int((~top & positive).sum())
    tn = int((~top & negative).sum())
    if definition == "paper_text_top_vs_remainder":
        table = [[tp, fp], [fn, tn]]
    elif definition == "author_released_code_top_vs_total":
        table = [[fp + tn, tp + fn], [fp, tp]]
    else:
        raise ValueError(f"unknown Fisher definition: {definition}")
    odds_ratio, p_value = fisher_exact(table)
    return (
        float(odds_ratio),
        float(p_value),
        {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
    )


def _method_frames(
    scores: pd.DataFrame,
    truth: pd.DataFrame,
    protocol: str,
) -> list[tuple[tuple[str, str, str], pd.DataFrame, int, int]]:
    joined_scores = scores.merge(
        truth, on=["ligand", "target"], how="inner", validate="many_to_one"
    )
    methods = list(
        joined_scores.loc[:, ["method_id", "method_name", "implementation"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if protocol == "current_returned_universe":
        frames = []
        for method in methods:
            group = joined_scores.loc[joined_scores["method_id"].eq(method[0])].copy()
            group["_returned"] = True
            frames.append((method, group, len(group), 0))
        return frames
    if protocol != "paper_union_max_imputed":
        raise ValueError(f"unknown protocol: {protocol}")

    union = (
        joined_scores.loc[:, [*KEY_COLUMNS, "response"]]
        .drop_duplicates(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    frames = []
    for method in methods:
        returned = joined_scores.loc[
            joined_scores["method_id"].eq(method[0]), [*KEY_COLUMNS, "score"]
        ]
        frame = union.merge(
            returned, on=list(KEY_COLUMNS), how="left", validate="one_to_one"
        )
        returned_rows = int(frame["score"].notna().sum())
        if returned_rows == 0:
            continue
        frame["_returned"] = frame["score"].notna()
        frame["score"] = frame["score"].fillna(float(frame["score"].min()))
        frames.append((method, frame, returned_rows, len(frame) - returned_rows))
    return frames


def fisher_rank_curves(
    scores: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    protocols = ("current_returned_universe", "paper_union_max_imputed")
    definitions = (
        "paper_text_top_vs_remainder",
        "author_released_code_top_vs_total",
    )
    for protocol in protocols:
        for method, frame, returned_rows, missing_rows in _method_frames(
            scores, truth, protocol
        ):
            frame = frame.copy()
            frame["rank"] = frame["score"].rank(
                method="min", ascending=False, na_option="bottom"
            )
            returned_max_rank = int(frame.loc[frame["_returned"], "rank"].max())
            for cutoff in RANK_CUTOFFS:
                estimable = cutoff <= returned_max_rank
                for definition in definitions:
                    record: dict[str, Any] = {
                        "protocol": protocol,
                        "fisher_definition": definition,
                        "method_id": method[0],
                        "method_name": method[1],
                        "implementation": method[2],
                        "rank_cutoff": cutoff,
                        "returned_rows": returned_rows,
                        "missing_rows": missing_rows,
                        "evaluation_rows": len(frame),
                        "returned_max_rank": returned_max_rank,
                    }
                    if not estimable:
                        record.update(
                            status="not_estimable",
                            reason_code="cutoff_exceeds_returned_rank_range",
                        )
                    else:
                        odds, p_value, counts = _fisher_counts(
                            frame["response"], frame["rank"].le(cutoff), definition
                        )
                        record.update(
                            odds_ratio=odds,
                            p_value=p_value,
                            status="observed",
                            reason_code=None,
                            **counts,
                        )
                    rows.append(record)
    result = pd.DataFrame.from_records(rows)
    for column in ("p_value", "odds_ratio"):
        if column not in result:
            result[column] = np.nan
    result["primary_8arm_substitute"] = result["method_id"].isin(PRIMARY_METHOD_IDS)
    result["q_value_bh_all_arms"] = np.nan
    result["q_value_bh_primary_8arm"] = np.nan
    observed = result["status"].eq("observed")
    family = ["protocol", "fisher_definition", "rank_cutoff"]
    if observed.any():
        result.loc[observed, "q_value_bh_all_arms"] = (
            result.loc[observed]
            .groupby(family, observed=True)["p_value"]
            .transform(_bh)
        )
        primary = observed & result["primary_8arm_substitute"]
        result.loc[primary, "q_value_bh_primary_8arm"] = (
            result.loc[primary].groupby(family, observed=True)["p_value"].transform(_bh)
        )
    return result


def _curve_metrics(response: np.ndarray, score: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(response, score)
    return {
        "auroc": (
            float(roc_auc_score(response, score))
            if len(np.unique(response)) == 2
            else np.nan
        ),
        "average_precision": float(average_precision_score(response, score)),
        "pr_auc_trapezoid": float(auc(recall, precision)),
    }


def _balanced_auprc(
    response: np.ndarray,
    score: np.ndarray,
    *,
    seed: int = 1234,
    iterations: int = 100,
) -> dict[str, float]:
    positive = np.flatnonzero(response == 1)
    negative = np.flatnonzero(response == 0)
    if len(positive) == 0 or len(negative) == 0:
        return {
            "balanced_average_precision_mean": np.nan,
            "balanced_average_precision_sd": np.nan,
            "balanced_pr_auc_trapezoid_mean": np.nan,
            "balanced_pr_auc_trapezoid_sd": np.nan,
        }
    rng = np.random.RandomState(seed)
    average_precision: list[float] = []
    trapezoid: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(negative, size=len(positive), replace=True)
        selected = np.concatenate((positive, sampled))
        metrics = _curve_metrics(response[selected], score[selected])
        average_precision.append(metrics["average_precision"])
        trapezoid.append(metrics["pr_auc_trapezoid"])
    return {
        "balanced_average_precision_mean": float(np.mean(average_precision)),
        "balanced_average_precision_sd": float(np.std(average_precision, ddof=1)),
        "balanced_pr_auc_trapezoid_mean": float(np.mean(trapezoid)),
        "balanced_pr_auc_trapezoid_sd": float(np.std(trapezoid, ddof=1)),
    }


def ranking_metrics(scores: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol in ("current_returned_universe", "paper_union_max_imputed"):
        for method, frame, returned_rows, missing_rows in _method_frames(
            scores, truth, protocol
        ):
            response = frame["response"].to_numpy(dtype=int)
            score = frame["score"].to_numpy(dtype=float)
            rows.append(
                {
                    "protocol": protocol,
                    "aggregation": "stlr_rows",
                    "method_id": method[0],
                    "method_name": method[1],
                    "implementation": method[2],
                    "evaluation_rows": len(frame),
                    "returned_rows": returned_rows,
                    "missing_rows": missing_rows,
                    "positive_rows": int(response.sum()),
                    **_curve_metrics(response, score),
                    **_balanced_auprc(response, score),
                }
            )

    for (method_id, method_name, implementation), method in scores.groupby(
        ["method_id", "method_name", "implementation"],
        observed=True,
        sort=True,
    ):
        for aggregation in ("max", "sum"):
            grouped = (
                method.groupby(["ligand", "target"], observed=True)["score"]
                .agg(aggregation)
                .rename("score")
                .reset_index()
            )
            frame = truth.merge(
                grouped,
                on=["ligand", "target"],
                how="left",
                validate="one_to_one",
            )
            returned_rows = int(frame["score"].notna().sum())
            if returned_rows == 0:
                continue
            minimum = float(frame["score"].dropna().min())
            frame["score"] = frame["score"].fillna(np.nextafter(minimum, -np.inf))
            response = frame["response"].to_numpy(dtype=int)
            score = frame["score"].to_numpy(dtype=float)
            rows.append(
                {
                    "protocol": "controlled_unique_ligand_target",
                    "aggregation": aggregation,
                    "method_id": method_id,
                    "method_name": method_name,
                    "implementation": implementation,
                    "evaluation_rows": len(frame),
                    "returned_rows": returned_rows,
                    "missing_rows": len(frame) - returned_rows,
                    "positive_rows": int(response.sum()),
                    **_curve_metrics(response, score),
                    **_balanced_auprc(response, score),
                }
            )
    return pd.DataFrame.from_records(rows)


def score_inventory(scores: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in scores.groupby(
        ["method_id", "method_name", "implementation"],
        observed=True,
        sort=True,
    ):
        joined = group.merge(
            truth, on=["ligand", "target"], how="inner", validate="many_to_one"
        )
        rows.append(
            {
                "method_id": method[0],
                "method_name": method[1],
                "implementation": method[2],
                "raw_score_rows": len(group),
                "cytokine_matched_rows": len(joined),
                "unique_stlr_rows": len(joined.drop_duplicates(list(KEY_COLUMNS))),
                "unique_ligand_target_rows": len(
                    joined.drop_duplicates(["ligand", "target"])
                ),
                "matched_positive_rows": int(joined["response"].eq(1).sum()),
            }
        )
    return pd.DataFrame.from_records(rows)


def run(
    truth_path: Path,
    crychic_scores: Path,
    liana_scores: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    truth_status: str,
    crychic_manifest: Path | None = None,
    liana_manifest: Path | None = None,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"cytokine protocol audit exists: {manifest_path}")
    crychic_manifest = crychic_manifest or crychic_scores.parent / "manifest.json"
    liana_manifest = liana_manifest or liana_scores.parent / "manifest.json"
    truth, truth_source_kind = _read_truth(truth_path)
    scores, crychic_run, liana_run = _load_and_validate_scores(
        crychic_scores,
        liana_scores,
        crychic_manifest,
        liana_manifest,
    )
    if sha256_file(truth_path) != crychic_run["input"]["sha256"]:
        if truth_path.suffix.lower() == ".h5ad":
            raise ValueError("truth H5AD is not the method input recorded in manifests")
    collapsed = _collapse_scores(scores, truth)
    curves = fisher_rank_curves(collapsed, truth)
    metrics = ranking_metrics(collapsed, truth)
    inventory = score_inventory(collapsed, truth)
    paths = {
        "fisher": output_dir / "fisher_rank_curves.tsv",
        "ranking": output_dir / "ranking_metrics.tsv",
        "inventory": output_dir / "score_inventory.tsv",
    }
    curves.to_csv(paths["fisher"], sep="\t", index=False, na_rep="")
    metrics.to_csv(paths["ranking"], sep="\t", index=False, na_rep="")
    inventory.to_csv(paths["inventory"], sep="\t", index=False, na_rep="")
    method_family = sorted(collapsed["method_id"].unique())
    evaluator_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "schema_version": "crychic-cytokine-protocol-audit-v1",
        "status": "complete",
        "dataset_id": dataset_id,
        "truth": {
            "status": truth_status,
            "source_kind": truth_source_kind,
            "filename": truth_path.name,
            "sha256": sha256_file(truth_path),
            "rows": len(truth),
            "positive_rows": int(truth["response"].sum()),
        },
        "method_families": {
            "all_score_arms": method_family,
            "primary_8arm_substitute": list(PRIMARY_METHOD_IDS),
            "primary_family_note": (
                "Current-method report family: CRYCHIC plus available composite/"
                "component methods and LIANA specificity consensus; Crosstalk is "
                "absent."
            ),
        },
        "protocols": {
            "current_returned_universe": (
                "Existing current-method substitute; each method uses only returned "
                "STLR rows."
            ),
            "paper_union_max_imputed": (
                "Common returned STLR union with missing scores tied at each method's "
                "worst returned score; both paper-text and released-author-code Fisher "
                "tables emitted."
            ),
            "controlled_unique_ligand_target": (
                "Unique ligand-target truth with max/sum aggregation and missing "
                "scores strictly after returned scores."
            ),
        },
        "balanced_auprc": {
            "iterations": 100,
            "negative_sampling": "with replacement to match all positives",
            "seed": 1234,
            "random_expected_value": 0.5,
        },
        "run_linkage": {
            "input_sha256": crychic_run["input"]["sha256"],
            "resource_sha256": crychic_run["resource"]["sha256"],
            "crychic_manifest_sha256": sha256_file(crychic_manifest),
            "liana_manifest_sha256": sha256_file(liana_manifest),
        },
        "evaluator": {
            "module": "benchmarks.literature.evaluate_cytokine_protocols",
            "source_sha256": sha256_file(evaluator_path),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("anndata", "numpy", "pandas", "scikit-learn", "scipy")
            },
        },
        "limitations": [
            (
                "The current union includes CRYCHIC and is dominated by its broader "
                "returned STLR coverage; it is a protocol extension, not the "
                "historical paper union."
            ),
            (
                "Current LIANA 1.7.3 consensus is not the paper's frozen LIANA "
                "0.0.5 OmniPath snapshot."
            ),
            (
                "Current LIANA run used "
                f"{liana_run['parameters']['n_perms']} permutations; the paper "
                "used 1000."
            ),
            "Crosstalk is unavailable in the current Python LIANA score table.",
            (
                "Balanced AUPRC uses a NumPy RNG and a clean mean of 100 scalar "
                "metrics, not the released R threshold-row weighting quirk."
            ),
        ],
        "outputs": {
            path.name: {"sha256": sha256_file(path)} for path in paths.values()
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth_path", type=Path)
    parser.add_argument("crychic_scores", type=Path)
    parser.add_argument("liana_scores", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--truth-status", required=True)
    parser.add_argument("--crychic-manifest", type=Path)
    parser.add_argument("--liana-manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.truth_path,
        args.crychic_scores,
        args.liana_scores,
        args.output_dir,
        dataset_id=args.dataset_id,
        truth_status=args.truth_status,
        crychic_manifest=args.crychic_manifest,
        liana_manifest=args.liana_manifest,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
