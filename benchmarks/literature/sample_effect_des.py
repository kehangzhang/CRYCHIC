"""Build descriptive spatial-DES rankings from sample-level CCC scores.

This is a common downstream sensitivity arm, not the native significance arm
from Cesaro et al.  Each method's score is oriented so that larger values mean
stronger communication, technical samples are averaged within
``subject x condition``, and the target-minus-reference subject mean is
computed per directed LR edge.  Positive and negative effects are then summed
separately after collapsing both communication directions into an unordered
cell-type pair, matching the paper's spatial comparison grain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

SCHEMA_VERSION = "crychic-sample-effect-des-ranking-v1"
ELIGIBLE_STATUSES = frozenset({"ok", "not_returned"})
MISSING_STATUSES = frozenset(
    {
        "cell_type_missing",
        "failed",
        "insufficient_cells",
        "method_failed",
        "resource_unavailable",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SampleEffectDESSpec:
    """Frozen descriptive contrast and support policy."""

    context_key: str
    reference: str
    target: str
    min_subjects_per_context: int = 3

    def __post_init__(self) -> None:
        for field in ("context_key", "reference", "target"):
            value = getattr(self, field)
            if not value or value != value.strip():
                raise ValueError(f"{field} must be a canonical non-empty string")
        if self.reference == self.target:
            raise ValueError("reference and target must differ")
        if (
            isinstance(self.min_subjects_per_context, bool)
            or self.min_subjects_per_context < 2
        ):
            raise ValueError("min_subjects_per_context must be an integer >= 2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_value(table: pd.DataFrame, column: str) -> str:
    if column not in table:
        raise ValueError(f"sample score table is missing {column!r}")
    values = table[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"sample score table requires exactly one {column}")
    return str(values[0])


def _sample_design(scores: pd.DataFrame, context_key: str) -> pd.DataFrame:
    required = {"sample_id", "subject_id", "context_json"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"sample score table is missing columns: {sorted(missing)}")
    design = scores.loc[:, sorted(required)].drop_duplicates(ignore_index=True)
    if design.duplicated("sample_id").any():
        raise ValueError("sample IDs map to multiple subjects or contexts")

    def context(value: object) -> str:
        try:
            payload = json.loads(str(value))
        except json.JSONDecodeError as error:
            raise ValueError("context_json contains invalid JSON") from error
        if not isinstance(payload, dict) or context_key not in payload:
            raise ValueError(f"context_json is missing {context_key!r}")
        result = payload[context_key]
        if not isinstance(result, str) or not result or result != result.strip():
            raise ValueError(f"context_json {context_key!r} must be canonical")
        return result

    design["condition"] = design["context_json"].map(context)
    return design.loc[:, ["sample_id", "subject_id", "condition"]]


def _oriented_strength(scores: pd.DataFrame) -> pd.Series:
    directions = scores["score_direction"].dropna().astype(str).unique()
    if len(directions) != 1 or directions[0] not in {"higher", "lower"}:
        raise ValueError("score_direction must be one constant 'higher' or 'lower'")
    statuses = scores["status"].astype(str)
    invalid = set(statuses).difference(ELIGIBLE_STATUSES | MISSING_STATUSES)
    if invalid:
        raise ValueError(f"sample score table contains unsupported statuses: {invalid}")
    numeric = pd.to_numeric(scores["score"], errors="coerce")
    if numeric.loc[statuses.eq("ok")].isna().any():
        raise ValueError("ok sample score rows require finite scores")
    if np.isinf(numeric.fillna(0.0)).any():
        raise ValueError("sample scores must be finite")
    if numeric.loc[~statuses.eq("ok")].notna().any():
        raise ValueError("non-ok sample score rows require missing scores")

    strength = pd.Series(np.nan, index=scores.index, dtype=float)
    observed = statuses.eq("ok")
    strength.loc[observed] = numeric.loc[observed]
    if directions[0] == "lower":
        if ((strength.loc[observed] < 0) | (strength.loc[observed] > 1)).any():
            raise ValueError("lower-is-better rank scores must lie in [0, 1]")
        strength.loc[observed] = 1.0 - strength.loc[observed]
    else:
        # Native probability scales differ across methods. Ranking within each
        # sample preserves order while making the common effect arm comparable.
        strength.loc[observed] = scores.loc[observed].groupby(
            "sample_id", observed=True, sort=False
        )["score"].rank(method="average", pct=True)
    strength.loc[statuses.eq("not_returned")] = 0.0
    return strength


def build_sample_effect_rankings(
    scores: pd.DataFrame,
    specification: SampleEffectDESSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unordered condition rankings and directed LR effect diagnostics."""
    required = {
        "dataset_id",
        "method_id",
        "method_version",
        "resource_id",
        "sample_id",
        "subject_id",
        "context_json",
        "sender",
        "receiver",
        "interaction_id",
        "score",
        "score_direction",
        "status",
    }
    missing = required.difference(scores.columns)
    if missing or scores.empty:
        raise ValueError(f"sample score table is empty or missing: {sorted(missing)}")
    identity = ["sample_id", "sender", "receiver", "interaction_id"]
    if scores.loc[:, identity].isna().any().any():
        raise ValueError("sample/edge identifiers must be complete")
    if scores.duplicated(identity).any():
        raise ValueError("sample score table contains duplicate sample-edge rows")

    spec = specification
    design = _sample_design(scores, spec.context_key)
    observed_conditions = set(design["condition"])
    requested = {spec.reference, spec.target}
    if observed_conditions != requested:
        raise ValueError(
            f"observed conditions {sorted(observed_conditions)} != {sorted(requested)}"
        )
    table = scores.loc[:, list(required)].copy(deep=True)
    table["strength"] = _oriented_strength(table)
    table = table.merge(
        design,
        on=["sample_id", "subject_id"],
        how="left",
        validate="many_to_one",
    )

    edge_keys = ["sender", "receiver", "interaction_id"]
    subject = (
        table.groupby(
            ["subject_id", "condition", *edge_keys],
            observed=True,
            sort=False,
            dropna=False,
        )["strength"]
        .mean()
        .reset_index()
    )
    summary = (
        subject.groupby(["condition", *edge_keys], observed=True, sort=False)[
            "strength"
        ]
        .agg(["mean", "count"])
        .reset_index()
        .pivot(index=edge_keys, columns="condition", values=["mean", "count"])
    )
    summary_columns = cast(pd.MultiIndex, summary.columns)
    summary.columns = [
        f"{stat}_{condition}" for stat, condition in summary_columns.to_list()
    ]
    effects = summary.reset_index()
    reference_mean = f"mean_{spec.reference}"
    target_mean = f"mean_{spec.target}"
    reference_count = f"count_{spec.reference}"
    target_count = f"count_{spec.target}"
    for column in (reference_mean, target_mean, reference_count, target_count):
        if column not in effects:
            effects[column] = np.nan
    estimable = (
        effects[reference_count].fillna(0).ge(spec.min_subjects_per_context)
        & effects[target_count].fillna(0).ge(spec.min_subjects_per_context)
    )
    effects["effect"] = effects[target_mean] - effects[reference_mean]
    effects.loc[~estimable, "effect"] = np.nan
    effects["status"] = np.where(estimable, "observed", "not_estimable")
    effects["reason_code"] = np.where(
        estimable, None, "insufficient_subjects_per_context"
    )
    effects["sender_unordered"] = effects[["sender", "receiver"]].min(axis=1)
    effects["receiver_unordered"] = effects[["sender", "receiver"]].max(axis=1)

    eligible = effects.loc[effects["status"].eq("observed")].copy()
    eligible["target_contribution"] = eligible["effect"].clip(lower=0.0)
    eligible["reference_contribution"] = (-eligible["effect"]).clip(lower=0.0)
    grouped = eligible.groupby(
        ["sender_unordered", "receiver_unordered"], observed=True, sort=True
    )
    pair_summary = grouped.agg(
        target_strength=("target_contribution", "sum"),
        reference_strength=("reference_contribution", "sum"),
        estimable_directed_lr=("effect", "size"),
        target_specific_directed_lr=(
            "target_contribution",
            lambda values: int(np.count_nonzero(values)),
        ),
        reference_specific_directed_lr=(
            "reference_contribution",
            lambda values: int(np.count_nonzero(values)),
        ),
    ).reset_index()
    pair_universe = effects.loc[
        :, ["sender_unordered", "receiver_unordered"]
    ].drop_duplicates(ignore_index=True)
    pair = pair_universe.merge(
        pair_summary,
        on=["sender_unordered", "receiver_unordered"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "estimable_directed_lr",
        "target_specific_directed_lr",
        "reference_specific_directed_lr",
    ):
        pair[column] = pair[column].fillna(0).astype(int)
    dataset = _single_value(scores, "dataset_id")
    method = _single_value(scores, "method_id")
    common = {
        "dataset": dataset,
        "method": method,
        "method_version": _single_value(scores, "method_version"),
        "resource": _single_value(scores, "resource_id"),
        "ranking_semantics": (
            "sum_condition_specific_subject_mean_rank_effects_after_collapsing_"
            "communication_directions;descriptive_common_sensitivity_arm"
        ),
    }
    records: list[dict[str, Any]] = []
    for row in pair.to_dict(orient="records"):
        pair_observed = int(row["estimable_directed_lr"]) > 0
        for condition, strength_field, count_field in (
            (spec.target, "target_strength", "target_specific_directed_lr"),
            (
                spec.reference,
                "reference_strength",
                "reference_specific_directed_lr",
            ),
        ):
            records.append(
                {
                    **common,
                    "condition": condition,
                    "sender": row["sender_unordered"],
                    "receiver": row["receiver_unordered"],
                    "ranked_strength": (
                        row[strength_field] if pair_observed else np.nan
                    ),
                    "condition_specific_directed_lr": row[count_field],
                    "estimable_directed_lr": row["estimable_directed_lr"],
                    "status": "observed" if pair_observed else "not_estimable",
                }
            )
    rankings = pd.DataFrame.from_records(records).sort_values(
        ["dataset", "method", "condition", "sender", "receiver"],
        kind="stable",
        ignore_index=True,
    )
    effects.insert(0, "dataset", dataset)
    effects.insert(1, "method", method)
    return rankings, effects


def run(
    input_parquet: Path,
    output_dir: Path,
    *,
    specification: SampleEffectDESSpec,
    overwrite: bool,
) -> dict[str, Any]:
    if not input_parquet.is_file():
        raise FileNotFoundError(input_parquet)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset_id",
        "method_id",
        "method_version",
        "resource_id",
        "sample_id",
        "subject_id",
        "context_json",
        "sender",
        "receiver",
        "interaction_id",
        "score",
        "score_direction",
        "status",
    ]
    scores = pd.read_parquet(input_parquet, columns=columns)
    rankings, effects = build_sample_effect_rankings(scores, specification)
    ranking_path = output_dir / "condition_cell_pair_rankings.tsv"
    effects_path = output_dir / "directed_lr_effects.parquet"
    rankings.to_csv(ranking_path, sep="\t", index=False)
    effects.to_parquet(effects_path, index=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "input": {
            "filename": input_parquet.name,
            "bytes": input_parquet.stat().st_size,
            "sha256": _sha256(input_parquet),
            "rows": len(scores),
        },
        "specification": {
            "context_key": specification.context_key,
            "reference": specification.reference,
            "target": specification.target,
            "min_subjects_per_context": specification.min_subjects_per_context,
            "statistical_unit": "subject_id",
            "technical_sample_policy": "mean within subject x condition",
            "formal_p_or_q_emitted": False,
        },
        "outputs": {
            "rankings": {
                "filename": ranking_path.name,
                "rows": len(rankings),
                "sha256": _sha256(ranking_path),
            },
            "effects": {
                "filename": effects_path.name,
                "rows": len(effects),
                "sha256": _sha256(effects_path),
            },
        },
        "deviation_from_cesaro_native_arm": (
            "continuous common subject-mean score effects replace each method's "
            "native significance/filtering rule; report as sensitivity only"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_parquet", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--context-key", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--min-subjects-per-context", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.input_parquet,
        args.output_dir,
        specification=SampleEffectDESSpec(
            context_key=args.context_key,
            reference=args.reference,
            target=args.target,
            min_subjects_per_context=args.min_subjects_per_context,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
