"""Paired score-formula ablations for mechanistic candidate development."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from crychic.scoring import mechanistic_strength, weighted_geometric_strength

FORMULAS = (
    "legacy_geometric_sender_total",
    "mechanistic_sender_unresolved",
    "mechanistic_sender_conserved_total",
)
REQUIRED_COLUMNS = {
    "seed",
    "scenario",
    "known_edge_id",
    "sender_id",
    "availability",
    "legacy_downstream",
    "incremental_downstream",
    "sender_weight",
    "prior_quality",
    "status",
    "reason_code",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreAblationResult:
    """Sender-grain components, edge scores, and paired formula diagnostics."""

    sender_scores: pd.DataFrame
    edge_scores: pd.DataFrame
    summary: pd.DataFrame

    def __post_init__(self) -> None:
        for field_name in ("sender_scores", "edge_scores", "summary"):
            table = getattr(self, field_name)
            if not isinstance(table, pd.DataFrame) or table.empty:
                raise ValueError(f"{field_name} must be a non-empty DataFrame")
            object.__setattr__(self, field_name, table.copy(deep=True))


def _prepare(records: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(records.columns)
    if missing:
        raise ValueError(f"score ablation records are missing: {sorted(missing)}")
    if records.empty:
        raise ValueError("score ablation records must not be empty")
    table = records.loc[:, sorted(REQUIRED_COLUMNS)].copy(deep=True)
    identifiers = ("scenario", "known_edge_id", "sender_id")
    if table.loc[:, list(identifiers)].isna().any().any():
        raise ValueError("score ablation identifiers must be complete")
    for column in identifiers:
        table[column] = table[column].astype(str)
    key = ["seed", "scenario", "known_edge_id", "sender_id"]
    if table.duplicated(key).any():
        raise ValueError("score ablation sender-grain keys must be unique")
    invalid_status = set(table["status"].astype(str)).difference(
        {"observed", "not_estimable"}
    )
    if invalid_status:
        raise ValueError(f"score ablation has invalid statuses: {invalid_status}")
    observed = table["status"].eq("observed")
    numeric_columns = (
        "availability",
        "legacy_downstream",
        "incremental_downstream",
        "sender_weight",
        "prior_quality",
    )
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
        values = table.loc[observed, column]
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"observed {column} must lie in [0, 1]")
        if table.loc[~observed, column].notna().any():
            raise ValueError(f"not-estimable {column} must be NA")
    if (observed & table["reason_code"].notna()).any():
        raise ValueError("observed score ablation rows must not carry reasons")
    if ((~observed) & table["reason_code"].isna()).any():
        raise ValueError("not-estimable score ablation rows require reasons")
    edge_key = ["seed", "scenario", "known_edge_id"]
    for key_values, group in table.groupby(edge_key, sort=False, observed=True):
        if not group["status"].eq("observed").all():
            continue
        for column in (
            "availability",
            "legacy_downstream",
            "incremental_downstream",
            "prior_quality",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"edge components must be sender-invariant for {key_values!r}"
                )
        if not np.isclose(group["sender_weight"].sum(), 1.0, atol=1e-12, rtol=0):
            raise ValueError(f"sender weights must sum to one for {key_values!r}")
    return table


def _sender_scores(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_record in table.to_dict(orient="records"):
        record = cast(dict[str, object], raw_record)
        if record["status"] != "observed":
            legacy = None
            core = None
            adjusted = None
            resolved = None
        else:
            availability = float(cast(float, record["availability"]))
            legacy_downstream = float(cast(float, record["legacy_downstream"]))
            incremental = float(cast(float, record["incremental_downstream"]))
            sender = float(cast(float, record["sender_weight"]))
            quality = float(cast(float, record["prior_quality"]))
            legacy = weighted_geometric_strength(
                {
                    "availability": availability,
                    "downstream": legacy_downstream,
                    "sender": sender,
                    "prior_quality": quality,
                },
                weights={
                    "availability": 1.0,
                    "downstream": 1.0,
                    "sender": 1.0,
                    "prior_quality": 1.0,
                },
                scales={
                    "availability": 1.0,
                    "downstream": 1.0,
                    "sender": 1.0,
                    "prior_quality": 1.0,
                },
            )
            core, adjusted, resolved = mechanistic_strength(
                availability=availability,
                incremental_downstream=incremental,
                prior_quality=quality,
                sender_weight=sender,
            )
        rows.append(
            {
                **record,
                "legacy_sender_strength": legacy,
                "mechanistic_lr_core": core,
                "mechanistic_prior_adjusted": adjusted,
                "mechanistic_sender_strength": resolved,
            }
        )
    return pd.DataFrame(rows)


def _edge_scores(sender_scores: pd.DataFrame) -> pd.DataFrame:
    edge_key = ["seed", "scenario", "known_edge_id"]
    rows: list[dict[str, object]] = []
    for key, group in sender_scores.groupby(edge_key, sort=False, observed=True):
        seed, scenario, edge_id = key
        observed = group["status"].eq("observed").all()
        if observed:
            core_values = group["mechanistic_prior_adjusted"].drop_duplicates()
            if len(core_values) != 1:
                raise ValueError(
                    "sender rows changed the sender-unresolved LR strength"
                )
            unresolved = float(core_values.iloc[0])
            legacy_total = float(group["legacy_sender_strength"].sum())
            conserved_total = float(group["mechanistic_sender_strength"].sum())
            conservation_error = conserved_total - unresolved
            reason = None
            status = "observed"
        else:
            unresolved = None
            legacy_total = None
            conserved_total = None
            conservation_error = None
            reasons = sorted(set(group["reason_code"].dropna().astype(str)))
            reason = ",".join(reasons) or "edge_component_not_estimable"
            status = "not_estimable"
        rows.append(
            {
                "seed": seed,
                "scenario": str(scenario),
                "known_edge_id": str(edge_id),
                "legacy_geometric_sender_total": legacy_total,
                "mechanistic_sender_unresolved": unresolved,
                "mechanistic_sender_conserved_total": conserved_total,
                "sender_conservation_error": conservation_error,
                "n_senders": len(group),
                "status": status,
                "reason_code": reason,
            }
        )
    return pd.DataFrame(rows)


def _paired_margin(
    edge_scores: pd.DataFrame, formula: str, *, negative_scenario: str
) -> tuple[float | None, int, str | None]:
    active = edge_scores.loc[
        edge_scores["scenario"].eq("active") & edge_scores["status"].eq("observed"),
        ["seed", "known_edge_id", formula],
    ]
    negative = edge_scores.loc[
        edge_scores["scenario"].eq(negative_scenario)
        & edge_scores["status"].eq("observed"),
        ["seed", "known_edge_id", formula],
    ]
    paired = active.merge(
        negative,
        on=["seed", "known_edge_id"],
        suffixes=("_active", "_negative"),
        validate="one_to_one",
    )
    expected = max(len(active), len(negative))
    if len(active) != len(negative) or len(paired) != expected or expected == 0:
        return None, len(paired), "insufficient_paired_edge_seed_support"
    margin = paired[f"{formula}_active"] - paired[f"{formula}_negative"]
    return float(margin.mean()), len(paired), None


def _summary(
    edge_scores: pd.DataFrame, negative_scenarios: tuple[str, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for formula in FORMULAS:
        for negative in negative_scenarios:
            margin, n_pairs, reason = _paired_margin(
                edge_scores, formula, negative_scenario=negative
            )
            rows.append(
                {
                    "formula": formula,
                    "metric": f"active_minus_{negative}_paired_margin",
                    "scenario": negative,
                    "estimate": margin,
                    "status": "observed" if reason is None else "not_estimable",
                    "reason_code": reason,
                    "n_eligible": n_pairs,
                }
            )
        for scenario in negative_scenarios:
            selected = edge_scores.loc[edge_scores["scenario"].eq(scenario)]
            observed = selected.loc[selected["status"].eq("observed"), formula]
            coverage = len(observed) / len(selected) if len(selected) else 0.0
            false_activation = float((observed > 0).mean()) if len(observed) else None
            rows.append(
                {
                    "formula": formula,
                    "metric": "negative_false_activation_rate",
                    "scenario": scenario,
                    "estimate": false_activation,
                    "status": "observed" if len(observed) else "not_estimable",
                    "reason_code": (
                        None if len(observed) else "no_observed_negative_edges"
                    ),
                    "n_eligible": len(observed),
                    "comparison_coverage": coverage,
                }
            )
    conservation = edge_scores.loc[edge_scores["status"].eq("observed")]
    rows.append(
        {
            "formula": "mechanistic_sender_conserved_total",
            "metric": "maximum_absolute_sender_conservation_error",
            "scenario": "all",
            "estimate": float(conservation["sender_conservation_error"].abs().max()),
            "status": "observed",
            "reason_code": None,
            "n_eligible": len(conservation),
        }
    )
    return pd.DataFrame(rows)


def evaluate_score_ablation(
    records: pd.DataFrame,
    *,
    negative_scenarios: Sequence[str] = (
        "global_null",
        "ligand_only",
        "target_only",
        "receiver_autonomous",
        "receptor_knockout",
    ),
) -> ScoreAblationResult:
    """Evaluate legacy and mechanistic formulas on paired component evidence."""

    negatives = tuple(map(str, negative_scenarios))
    if not negatives or "active" in negatives or len(set(negatives)) != len(negatives):
        raise ValueError("negative_scenarios must be unique and exclude active")
    table = _prepare(records)
    sender_scores = _sender_scores(table)
    edge_scores = _edge_scores(sender_scores)
    return ScoreAblationResult(
        sender_scores=sender_scores,
        edge_scores=edge_scores,
        summary=_summary(edge_scores, negatives),
    )


__all__ = [
    "FORMULAS",
    "ScoreAblationResult",
    "evaluate_score_ablation",
]
