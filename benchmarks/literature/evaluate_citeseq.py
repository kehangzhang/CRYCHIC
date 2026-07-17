"""Evaluate CITE-seq receptor specificity using the Dimitrov protocol."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.metrics import (
    LiteratureGoldStandardResult,
    LiteratureGoldStandardSpec,
    evaluate_literature_gold_standard,
)

EDGE_COLUMNS = ("sender", "receiver", "ligand", "receptor")


@dataclass(frozen=True, slots=True)
class ComponentScoreSpec:
    """One literature method score and its optional published filter."""

    method: str
    score_column: str
    direction: str
    filter_column: str | None = None
    filter_operator: str | None = None
    filter_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("component score direction must be higher or lower")
        filters = (
            self.filter_column,
            self.filter_operator,
            self.filter_threshold,
        )
        if any(value is None for value in filters) != all(
            value is None for value in filters
        ):
            raise ValueError("component filter fields must be all set or all absent")
        if self.filter_operator not in {None, "le", "ge"}:
            raise ValueError("component filter operator must be le or ge")


LIANA_COMPONENT_SPECS = (
    ComponentScoreSpec("LIANA magnitude consensus", "magnitude_rank", "lower"),
    ComponentScoreSpec("LIANA specificity consensus", "specificity_rank", "lower"),
    ComponentScoreSpec(
        "CellPhoneDB composite",
        "lr_means",
        "higher",
        "cellphone_pvals",
        "le",
        0.05,
    ),
    ComponentScoreSpec("CellPhoneDB p-value", "cellphone_pvals", "lower"),
    ComponentScoreSpec("Connectome specificity", "scaled_weight", "higher"),
    ComponentScoreSpec("logFC specificity", "lr_logfc", "higher"),
    ComponentScoreSpec("NATMI specificity", "spec_weight", "higher"),
    ComponentScoreSpec(
        "SingleCellSignalR LRscore",
        "lrscore",
        "higher",
        "lrscore",
        "ge",
        0.5,
    ),
)

CELLCHAT_COMPONENT_SPECS = (
    ComponentScoreSpec(
        "CellChat composite",
        "lr_probs",
        "higher",
        "cellchat_pvals",
        "le",
        0.05,
    ),
    ComponentScoreSpec("CellChat p-value", "cellchat_pvals", "lower"),
)


def _apply_filter(table: pd.DataFrame, spec: ComponentScoreSpec) -> pd.DataFrame:
    if spec.filter_column is None:
        return table
    if spec.filter_threshold is None:
        raise RuntimeError("validated component filter threshold is unavailable")
    values = pd.to_numeric(table[spec.filter_column], errors="coerce")
    if spec.filter_operator == "le":
        return table.loc[values.le(spec.filter_threshold)]
    if spec.filter_operator == "ge":
        return table.loc[values.ge(spec.filter_threshold)]
    raise RuntimeError("validated component filter operator is unavailable")


def component_scores_from_liana(
    raw: pd.DataFrame,
    *,
    dataset: str,
    resource_mode: str,
    specs: tuple[ComponentScoreSpec, ...] = LIANA_COMPONENT_SPECS,
) -> pd.DataFrame:
    """Convert LIANA component columns to one oriented long score table."""

    required = {"source", "target", "ligand_complex", "receptor_complex"}
    required.update(spec.score_column for spec in specs)
    required.update(
        spec.filter_column for spec in specs if spec.filter_column is not None
    )
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"LIANA component output is missing: {sorted(missing)}")
    frames: list[pd.DataFrame] = []
    for spec in specs:
        selected = _apply_filter(raw, spec)
        frame = selected.loc[
            :, ["source", "target", "ligand_complex", "receptor_complex"]
        ].rename(
            columns={
                "source": "sender",
                "target": "receiver",
                "ligand_complex": "ligand",
                "receptor_complex": "receptor",
            }
        )
        frame["score"] = pd.to_numeric(
            selected[spec.score_column], errors="coerce"
        ).to_numpy()
        frame = frame.loc[np.isfinite(frame["score"])].copy()
        frame.insert(0, "method", spec.method)
        frame.insert(0, "resource_mode", resource_mode)
        frame.insert(0, "dataset", dataset)
        frame["status"] = "observed"
        frame["score_direction"] = spec.direction
        frames.append(frame)
    return _collapse_scores(pd.concat(frames, ignore_index=True))


def component_scores_from_cellchat(
    raw: pd.DataFrame,
    *,
    dataset: str,
    resource_mode: str,
) -> pd.DataFrame:
    """Convert a LIANA CellChat run to composite and p-value scores."""

    return component_scores_from_liana(
        raw,
        dataset=dataset,
        resource_mode=resource_mode,
        specs=CELLCHAT_COMPONENT_SPECS,
    )


def scores_from_crychic_availability(
    raw: pd.DataFrame,
    *,
    dataset: str,
    resource_mode: str,
) -> pd.DataFrame:
    """Convert the dedicated static CRYCHIC availability adapter output."""

    required = {*EDGE_COLUMNS, "availability_state", "status"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"CRYCHIC availability output is missing: {sorted(missing)}")
    result = raw.loc[:, [*EDGE_COLUMNS, "availability_state", "status"]].rename(
        columns={"availability_state": "score"}
    )
    result.insert(0, "method", "CRYCHIC availability-state")
    result.insert(0, "resource_mode", resource_mode)
    result.insert(0, "dataset", dataset)
    result["score_direction"] = "higher"
    return _collapse_scores(result)


def _collapse_scores(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy(deep=True)
    for column in ("ligand", "receptor"):
        scores[column] = scores[column].astype(str).str.strip().str.upper()
    keys = ["dataset", "resource_mode", "method", *EDGE_COLUMNS]
    if scores.empty:
        raise ValueError("component score table must not be empty")
    if scores.loc[:, keys].isna().any().any():
        raise ValueError("component score identifiers must not be missing")
    records: list[pd.DataFrame] = []
    for (_, _, _), method in scores.groupby(
        ["dataset", "resource_mode", "method"], sort=True, observed=True
    ):
        directions = method["score_direction"].astype(str).unique()
        if len(directions) != 1 or directions[0] not in {"higher", "lower"}:
            raise ValueError("score direction must be constant per method")
        aggregation = "max" if directions[0] == "higher" else "min"
        records.append(
            method.groupby(keys, sort=True, observed=True, as_index=False).agg(
                score=("score", aggregation),
                status=("status", "first"),
                score_direction=("score_direction", "first"),
            )
        )
    return pd.concat(records, ignore_index=True)


def _annotated_scores(
    scores: pd.DataFrame, receptor_truth: pd.DataFrame
) -> pd.DataFrame:
    required = {"dataset", "receiver", "receptor", "is_positive", "truth_status"}
    missing = required.difference(receptor_truth.columns)
    if missing:
        raise ValueError(f"CITE-seq receptor truth is missing: {sorted(missing)}")
    truth = receptor_truth.loc[:, list(required)].copy()
    truth = truth.loc[truth["truth_status"].eq("observed")]
    return scores.merge(
        truth,
        on=["dataset", "receiver", "receptor"],
        how="inner",
        validate="many_to_one",
    )


def _combine_results(
    results: list[LiteratureGoldStandardResult],
) -> LiteratureGoldStandardResult:
    if not results:
        raise ValueError("at least one CITE-seq evaluation result is required")
    return LiteratureGoldStandardResult(
        point_estimates=pd.concat(
            [result.point_estimates for result in results], ignore_index=True
        ),
        bootstrap_summary=pd.concat(
            [result.bootstrap_summary for result in results], ignore_index=True
        ),
        negative_sampling_summary=pd.concat(
            [result.negative_sampling_summary for result in results],
            ignore_index=True,
        ),
    )


def _resource_fixed_universe(
    receptor_truth: pd.DataFrame,
    *,
    dataset: str,
    resource: pd.DataFrame,
) -> pd.DataFrame:
    required = {"dataset", "receiver", "receptor", "is_positive", "truth_status"}
    missing_truth = required.difference(receptor_truth.columns)
    missing_resource = {"ligand", "receptor"}.difference(resource.columns)
    if missing_truth:
        raise ValueError(f"CITE-seq receptor truth is missing: {sorted(missing_truth)}")
    if missing_resource:
        raise ValueError(
            f"CITE-seq fixed resource is missing: {sorted(missing_resource)}"
        )
    truth = receptor_truth.loc[
        receptor_truth["dataset"].eq(dataset)
        & receptor_truth["truth_status"].eq("observed"),
        ["receiver", "receptor", "is_positive"],
    ].copy()
    if truth.empty:
        raise ValueError(f"CITE-seq truth has no observed labels for {dataset}")
    truth["receptor"] = truth["receptor"].astype(str).str.strip().str.upper()
    pairs = resource.loc[:, ["ligand", "receptor"]].copy()
    pairs["ligand"] = pairs["ligand"].astype(str).str.strip().str.upper()
    pairs["receptor"] = pairs["receptor"].astype(str).str.strip().str.upper()
    pairs = pairs.drop_duplicates(ignore_index=True)
    pairs = pairs.loc[pairs["receptor"].isin(truth["receptor"])].copy()
    if pairs.empty:
        raise ValueError("CITE-seq fixed resource has no receptor truth coverage")
    cells = pd.DataFrame({"sender": sorted(truth["receiver"].astype(str).unique())})
    receivers = pd.DataFrame(
        {"receiver": sorted(truth["receiver"].astype(str).unique())}
    )
    universe = cells.merge(receivers, how="cross").merge(pairs, how="cross")
    universe = universe.merge(
        truth,
        on=["receiver", "receptor"],
        how="left",
        validate="many_to_one",
    )
    if universe["is_positive"].isna().any():
        raise ValueError("CITE-seq fixed universe contains unlabeled receptor targets")
    universe["is_positive"] = universe["is_positive"].astype(int)
    return universe.sort_values(list(EDGE_COLUMNS), kind="stable", ignore_index=True)


def _worst_score_sentinel(scores: pd.Series) -> float:
    minimum = float(scores.min())
    maximum = float(scores.max())
    return minimum - max(1.0, abs(maximum - minimum) * 0.01)


def _evaluate_resource_fixed(
    scores: pd.DataFrame,
    receptor_truth: pd.DataFrame,
    *,
    resource: pd.DataFrame,
    n_bootstrap: int,
    n_negative_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, LiteratureGoldStandardResult]:
    results: list[LiteratureGoldStandardResult] = []
    retained: list[pd.DataFrame] = []
    for (dataset, resource_mode), arm in scores.groupby(
        ["dataset", "resource_mode"], sort=True, observed=True
    ):
        universe = _resource_fixed_universe(
            receptor_truth, dataset=str(dataset), resource=resource
        ).assign(
            dataset=dataset,
            resource_mode=resource_mode,
            universe_mode="resource_fixed",
        )
        n_positive = int(universe["is_positive"].sum())
        for method_name, method in arm.groupby("method", sort=True, observed=True):
            directions = method["score_direction"].astype(str).unique()
            if len(directions) != 1:
                raise ValueError("CITE-seq fixed method has inconsistent direction")
            oriented = method.loc[:, [*EDGE_COLUMNS, "score", "status"]].copy()
            oriented = oriented.loc[oriented["status"].eq("observed")].copy()
            if directions[0] == "lower":
                oriented["score"] = -oriented["score"]
            elif directions[0] != "higher":
                raise ValueError("CITE-seq fixed method direction is invalid")
            oriented = oriented.groupby(
                list(EDGE_COLUMNS), sort=True, observed=True, as_index=False
            ).agg(score=("score", "max"))
            sentinel = _worst_score_sentinel(oriented["score"])
            selected = universe.merge(
                oriented,
                on=list(EDGE_COLUMNS),
                how="left",
                validate="one_to_one",
            )
            selected["raw_returned"] = selected["score"].notna()
            selected["score"] = selected["score"].fillna(sentinel)
            selected["method"] = method_name
            selected["score_direction"] = "higher"
            selected["status"] = "observed"
            truth = selected.loc[
                :,
                [
                    "dataset",
                    "resource_mode",
                    "universe_mode",
                    *EDGE_COLUMNS,
                    "is_positive",
                ],
            ]
            spec = LiteratureGoldStandardSpec(
                key_columns=EDGE_COLUMNS,
                method_columns=("method",),
                stratum_columns=("dataset", "resource_mode", "universe_mode"),
                top_k=n_positive,
                n_bootstrap=n_bootstrap,
                n_negative_samples=n_negative_samples,
                negative_to_positive_ratio=1.0,
                random_seed=random_seed,
            )
            result = evaluate_literature_gold_standard(selected, truth, spec)
            returned = selected["raw_returned"]
            positive = selected["is_positive"].eq(1)
            point = result.point_estimates.assign(
                raw_returned_edges=int(returned.sum()),
                raw_absent_edges_filled_worst=int((~returned).sum()),
                raw_returned_positive_edges=int((returned & positive).sum()),
                raw_positive_return_fraction=float(
                    (returned & positive).sum() / n_positive
                ),
                fixed_resource_lr_pairs=int(
                    universe.loc[:, ["ligand", "receptor"]].drop_duplicates().shape[0]
                ),
                missing_score_policy_citeseq=(
                    "fixed_resource_absent_prediction_ranked_after_returned"
                ),
            )
            results.append(
                LiteratureGoldStandardResult(
                    point_estimates=point,
                    bootstrap_summary=result.bootstrap_summary,
                    negative_sampling_summary=result.negative_sampling_summary,
                )
            )
            retained.append(selected)
    return pd.concat(retained, ignore_index=True), _combine_results(results)


def evaluate_citeseq_components(
    scores: pd.DataFrame,
    receptor_truth: pd.DataFrame,
    *,
    universe_mode: str,
    resource: pd.DataFrame | None = None,
    n_bootstrap: int = 200,
    n_negative_samples: int = 100,
    random_seed: int = 20260715,
) -> tuple[pd.DataFrame, LiteratureGoldStandardResult]:
    """Evaluate method-independent or strict intersection score universes."""

    if universe_mode not in {"independent", "intersection", "resource_fixed"}:
        raise ValueError(
            "universe_mode must be independent, intersection, or resource_fixed"
        )
    collapsed = _collapse_scores(scores)
    if universe_mode == "resource_fixed":
        if resource is None:
            raise ValueError("resource_fixed requires a frozen resource table")
        return _evaluate_resource_fixed(
            collapsed,
            receptor_truth,
            resource=resource,
            n_bootstrap=n_bootstrap,
            n_negative_samples=n_negative_samples,
            random_seed=random_seed,
        )
    if resource is not None:
        raise ValueError("resource is only valid for resource_fixed evaluation")
    annotated = _annotated_scores(collapsed, receptor_truth)
    if annotated.empty:
        raise ValueError("no method scores match CITE-seq receptor truth")
    annotated["universe_mode"] = universe_mode
    group_columns = ["dataset", "resource_mode"]
    results: list[LiteratureGoldStandardResult] = []
    retained: list[pd.DataFrame] = []
    for (dataset, resource_mode), arm in annotated.groupby(
        group_columns, sort=True, observed=True
    ):
        if universe_mode == "intersection":
            method_count = arm["method"].nunique()
            support = arm.groupby(list(EDGE_COLUMNS), observed=True)["method"].nunique()
            common = support.loc[support.eq(method_count)].index
            keys = pd.MultiIndex.from_frame(arm.loc[:, list(EDGE_COLUMNS)])
            arm = arm.loc[keys.isin(common)].copy()
            if arm.empty:
                raise ValueError("CITE-seq method intersection is empty")
            truth = (
                arm.loc[:, [*EDGE_COLUMNS, "is_positive"]]
                .drop_duplicates(ignore_index=True)
                .assign(
                    dataset=dataset,
                    resource_mode=resource_mode,
                    universe_mode=universe_mode,
                )
            )
            n_positive = int(truth["is_positive"].sum())
            spec = LiteratureGoldStandardSpec(
                key_columns=EDGE_COLUMNS,
                method_columns=("method",),
                stratum_columns=("dataset", "resource_mode", "universe_mode"),
                top_k=max(1, n_positive),
                n_bootstrap=n_bootstrap,
                n_negative_samples=n_negative_samples,
                negative_to_positive_ratio=1.0,
                random_seed=random_seed,
            )
            results.append(evaluate_literature_gold_standard(arm, truth, spec))
            retained.append(arm)
            continue

        for method_name, method in arm.groupby("method", sort=True, observed=True):
            truth = method.loc[:, [*EDGE_COLUMNS, "is_positive"]].copy()
            truth = truth.assign(
                dataset=dataset,
                resource_mode=resource_mode,
                universe_mode=universe_mode,
            )
            n_positive = int(truth["is_positive"].sum())
            spec = LiteratureGoldStandardSpec(
                key_columns=EDGE_COLUMNS,
                method_columns=("method",),
                stratum_columns=("dataset", "resource_mode", "universe_mode"),
                top_k=max(1, n_positive),
                n_bootstrap=n_bootstrap,
                n_negative_samples=n_negative_samples,
                negative_to_positive_ratio=1.0,
                random_seed=random_seed,
            )
            results.append(
                evaluate_literature_gold_standard(
                    method.assign(method=method_name), truth, spec
                )
            )
            retained.append(method)
    return pd.concat(retained, ignore_index=True), _combine_results(results)


def _write_result(result: LiteratureGoldStandardResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("receptor_truth", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--resource-mode", required=True)
    parser.add_argument("--crychic", type=Path)
    parser.add_argument("--cellchat", type=Path)
    parser.add_argument("--resource", type=Path)
    parser.add_argument(
        "--universe-mode",
        choices=("independent", "intersection", "resource_fixed"),
        default="independent",
    )
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--n-negative-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--exclude-method", action="append", default=[])
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
    combined = pd.concat(scores, ignore_index=True)
    if args.exclude_method:
        combined = combined.loc[
            ~combined["method"].isin(set(args.exclude_method))
        ].copy()
        if combined.empty:
            raise ValueError("method exclusion removed every CITE-seq score")
    retained, result = evaluate_citeseq_components(
        combined,
        pd.read_csv(args.receptor_truth, sep="\t"),
        universe_mode=args.universe_mode,
        resource=(
            None if args.resource is None else pd.read_csv(args.resource, sep="\t")
        ),
        n_bootstrap=args.n_bootstrap,
        n_negative_samples=args.n_negative_samples,
        random_seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retained.to_parquet(args.output_dir / "evaluated_scores.parquet", index=False)
    _write_result(result, args.output_dir)


if __name__ == "__main__":
    main()


__all__ = [
    "CELLCHAT_COMPONENT_SPECS",
    "LIANA_COMPONENT_SPECS",
    "ComponentScoreSpec",
    "component_scores_from_cellchat",
    "component_scores_from_liana",
    "evaluate_citeseq_components",
    "scores_from_crychic_availability",
]
