"""Subject-equal directed effects and unordered spatial-DES rankings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DIRECTED_EFFECT_COLUMNS = (
    "reference_condition",
    "target_condition",
    "mean_strength_reference",
    "mean_strength_target",
    "n_samples_reference",
    "n_samples_target",
    "n_subjects_reference",
    "n_subjects_target",
    "effect_target_minus_reference",
    "status",
    "reason_code",
    "effect_semantics",
    "formal_inference_allowed",
)

UNORDERED_DES_RANKING_COLUMNS = (
    "dataset",
    "method",
    "method_version",
    "resource",
    "ranking_semantics",
    "condition",
    "sender",
    "receiver",
    "ranked_strength",
    "condition_specific_directed_lr",
    "estimable_directed_lr",
    "minimum_reference_subjects",
    "minimum_target_subjects",
    "rank_by_ranked_strength",
    "rank_by_condition_specific_directed_lr",
    "status",
    "reason_code",
    "formal_inference_allowed",
)


def _canonical_text(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def heldout_sample_coverage_audit(
    scores: pd.DataFrame,
    expected_sample_metadata: pd.DataFrame,
) -> dict[str, object]:
    """Fail closed unless every prepared sample and subject has held-out rows."""

    required = {"sample_id", "subject_id"}
    missing_scores = required.difference(scores.columns)
    missing_expected = required.difference(expected_sample_metadata.columns)
    if missing_scores or scores.empty:
        raise ValueError(
            f"held-out scores are empty or missing: {sorted(missing_scores)}"
        )
    if missing_expected or expected_sample_metadata.empty:
        raise ValueError(
            f"expected sample metadata are empty or missing: {sorted(missing_expected)}"
        )
    expected = (
        expected_sample_metadata.loc[:, ["sample_id", "subject_id"]]
        .astype(str)
        .drop_duplicates(ignore_index=True)
    )
    observed = (
        scores.loc[:, ["sample_id", "subject_id"]]
        .astype(str)
        .drop_duplicates(ignore_index=True)
    )
    if expected["sample_id"].duplicated().any():
        raise ValueError("expected sample_id maps to multiple subjects")
    if observed["sample_id"].duplicated().any():
        raise ValueError("held-out sample_id maps to multiple subjects")
    expected_pairs = set(expected.itertuples(index=False, name=None))
    observed_pairs = set(observed.itertuples(index=False, name=None))
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs.difference(observed_pairs))
        unexpected = sorted(observed_pairs.difference(expected_pairs))
        raise ValueError(
            "held-out sample coverage disagrees with prepared input; "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected_subjects = set(expected["subject_id"])
    observed_subjects = set(observed["subject_id"])
    if observed_subjects != expected_subjects:
        raise ValueError("held-out subject coverage disagrees with prepared input")
    return {
        "all_input_samples_observed": True,
        "all_input_subjects_observed": True,
        "n_samples": len(expected_pairs),
        "n_subjects": len(expected_subjects),
    }


def subject_equal_directed_lr_effects(
    scores: pd.DataFrame,
    *,
    reference: str,
    target: str,
    condition_column: str,
    edge_columns: Sequence[str],
    score_column: str = "global_sender_lr_score",
    min_subjects_per_condition: int = 3,
) -> pd.DataFrame:
    """Average technical samples within subject before target-reference effects."""

    reference = _canonical_text(reference, field="reference")
    target = _canonical_text(target, field="target")
    condition_column = _canonical_text(condition_column, field="condition_column")
    if reference == target:
        raise ValueError("reference and target must differ")
    if (
        isinstance(min_subjects_per_condition, bool)
        or not isinstance(min_subjects_per_condition, int)
        or min_subjects_per_condition < 1
    ):
        raise ValueError("min_subjects_per_condition must be a positive integer")
    edges = tuple(edge_columns)
    if (
        not edges
        or len(set(edges)) != len(edges)
        or any(not column or column != column.strip() for column in edges)
    ):
        raise ValueError("edge_columns must contain unique canonical names")
    identity = {"sender", "receiver", "interaction_id"}
    if not identity.issubset(edges):
        raise ValueError(
            "edge_columns must include sender, receiver, and interaction_id"
        )
    required = {
        "sample_id",
        "subject_id",
        condition_column,
        score_column,
        "status",
        *edges,
    }
    missing = required.difference(scores.columns)
    if missing or scores.empty:
        raise ValueError(
            f"sample sender-LR scores are empty or missing: {sorted(missing)}"
        )
    observed_conditions = set(scores[condition_column].astype(str))
    if observed_conditions != {reference, target}:
        raise ValueError(
            "sample sender-LR conditions disagree with the declared contrast: "
            f"{sorted(observed_conditions)}"
        )
    sample_design = scores.loc[
        :, ["sample_id", "subject_id", condition_column]
    ].drop_duplicates(ignore_index=True)
    if (
        sample_design["sample_id"].isna().any()
        or sample_design.duplicated("sample_id").any()
    ):
        raise ValueError("sample_id maps to multiple subject or condition values")
    base = scores.loc[:, list(edges)].drop_duplicates(ignore_index=True)
    if base.isna().any().any():
        raise ValueError("directed sender-receiver-LR identifiers must be complete")
    if base.duplicated(["sender", "receiver", "interaction_id"]).any():
        raise ValueError("directed sender-receiver-LR resource mapping is ambiguous")

    statuses = scores["status"].astype(str)
    supported_statuses = {"observed", "structural_zero", "not_estimable"}
    invalid_statuses = set(statuses).difference(supported_statuses)
    if invalid_statuses:
        raise ValueError(
            f"sample sender-LR scores contain invalid statuses: {invalid_statuses}"
        )
    all_strengths = pd.to_numeric(scores[score_column], errors="coerce")
    if all_strengths.loc[statuses.eq("not_estimable")].notna().any():
        raise ValueError("not_estimable sender-LR rows must not carry strengths")
    if not all_strengths.loc[statuses.eq("structural_zero")].eq(0.0).all():
        raise ValueError("structural_zero sender-LR rows require exact zero")
    usable = scores.loc[statuses.isin(("observed", "structural_zero"))].copy()
    usable["_strength"] = all_strengths.loc[usable.index]
    if usable["_strength"].isna().any() or np.isinf(usable["_strength"]).any():
        raise ValueError("eligible sender-LR rows require finite strengths")

    subject = (
        usable.groupby(
            [condition_column, "subject_id", *edges],
            observed=True,
            sort=False,
        )
        .agg(
            subject_strength=("_strength", "mean"),
            n_samples=("sample_id", "nunique"),
        )
        .reset_index()
    )
    summary = (
        subject.groupby([condition_column, *edges], observed=True, sort=False)
        .agg(
            mean_strength=("subject_strength", "mean"),
            n_samples=("n_samples", "sum"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )

    def condition_summary(condition: str, suffix: str) -> pd.DataFrame:
        selected = summary.loc[
            summary[condition_column].astype(str).eq(condition)
        ].drop(columns=condition_column)
        return selected.rename(
            columns={
                "mean_strength": f"mean_strength_{suffix}",
                "n_samples": f"n_samples_{suffix}",
                "n_subjects": f"n_subjects_{suffix}",
            }
        )

    result = base.merge(
        condition_summary(reference, "reference"),
        on=list(edges),
        how="left",
        validate="one_to_one",
    ).merge(
        condition_summary(target, "target"),
        on=list(edges),
        how="left",
        validate="one_to_one",
    )
    for column in (
        "n_samples_reference",
        "n_samples_target",
        "n_subjects_reference",
        "n_subjects_target",
    ):
        result[column] = result[column].fillna(0).astype(np.int64)
    estimable = (
        result["mean_strength_reference"].notna()
        & result["mean_strength_target"].notna()
        & result["n_subjects_reference"].ge(min_subjects_per_condition)
        & result["n_subjects_target"].ge(min_subjects_per_condition)
    )
    result["effect_target_minus_reference"] = np.nan
    result.loc[estimable, "effect_target_minus_reference"] = (
        result.loc[estimable, "mean_strength_target"]
        - result.loc[estimable, "mean_strength_reference"]
    )
    result["reference_condition"] = reference
    result["target_condition"] = target
    result["status"] = np.where(estimable, "observed", "not_estimable")
    result["reason_code"] = np.where(
        estimable, None, "insufficient_subject_score_coverage"
    )
    result["effect_semantics"] = (
        f"target_minus_reference_subject_equal_mean_{score_column}_descriptive"
    )
    result["formal_inference_allowed"] = False
    return result.loc[:, [*edges, *DIRECTED_EFFECT_COLUMNS]].sort_values(
        ["sender", "receiver", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )


def unordered_cell_pair_des_rankings(
    effects: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    method_version: str,
    resource: str,
) -> pd.DataFrame:
    """Collapse both directed communication axes into condition-specific pairs."""

    identity_values = {
        "dataset": dataset,
        "method": method,
        "method_version": method_version,
        "resource": resource,
    }
    identity_values = {
        field: _canonical_text(value, field=field)
        for field, value in identity_values.items()
    }
    required = {
        "sender",
        "receiver",
        "interaction_id",
        "reference_condition",
        "target_condition",
        "n_subjects_reference",
        "n_subjects_target",
        "effect_target_minus_reference",
        "status",
    }
    missing = required.difference(effects.columns)
    if missing or effects.empty:
        raise ValueError(f"directed LR effects are empty or missing: {sorted(missing)}")
    if effects.duplicated(["sender", "receiver", "interaction_id"]).any():
        raise ValueError("directed LR effects contain duplicate directed edges")
    identifiers = effects.loc[:, ["sender", "receiver", "interaction_id"]]
    if identifiers.isna().any().any():
        raise ValueError("directed LR effect identifiers must be complete")
    canonical = identifiers.map(
        lambda value: isinstance(value, str) and bool(value) and value == value.strip()
    )
    if not canonical.all().all():
        raise ValueError("directed LR effect identifiers must be canonical strings")
    references = effects["reference_condition"].astype(str).unique()
    targets = effects["target_condition"].astype(str).unique()
    if len(references) != 1 or len(targets) != 1 or references[0] == targets[0]:
        raise ValueError("directed LR effects require one valid reference/target pair")
    reference = str(references[0])
    target = str(targets[0])

    statuses = effects["status"].astype(str)
    invalid_statuses = set(statuses).difference({"observed", "not_estimable"})
    if invalid_statuses:
        raise ValueError(
            f"directed LR effects contain invalid statuses: {invalid_statuses}"
        )
    numeric = pd.to_numeric(effects["effect_target_minus_reference"], errors="coerce")
    observed = statuses.eq("observed")
    if numeric.loc[observed].isna().any() or np.isinf(numeric.loc[observed]).any():
        raise ValueError("observed directed LR effects must be finite")
    if numeric.loc[~observed].notna().any():
        raise ValueError("non-estimable directed LR effects must not carry an effect")

    working = effects.copy(deep=True)
    working["_effect"] = numeric
    working["_sender_unordered"] = [
        min(str(sender), str(receiver))
        for sender, receiver in working[["sender", "receiver"]].itertuples(
            index=False, name=None
        )
    ]
    working["_receiver_unordered"] = [
        max(str(sender), str(receiver))
        for sender, receiver in working[["sender", "receiver"]].itertuples(
            index=False, name=None
        )
    ]
    pair_keys = ["_sender_unordered", "_receiver_unordered"]
    pair_universe = working.loc[:, pair_keys].drop_duplicates(ignore_index=True)
    eligible = working.loc[observed].copy()
    eligible["_target_contribution"] = eligible["_effect"].clip(lower=0.0)
    eligible["_reference_contribution"] = (-eligible["_effect"]).clip(lower=0.0)
    eligible["_target_specific"] = eligible["_target_contribution"].gt(0.0)
    eligible["_reference_specific"] = eligible["_reference_contribution"].gt(0.0)
    summary = (
        eligible.groupby(pair_keys, observed=True, sort=True)
        .agg(
            target_strength=("_target_contribution", "sum"),
            reference_strength=("_reference_contribution", "sum"),
            target_specific_directed_lr=("_target_specific", "sum"),
            reference_specific_directed_lr=("_reference_specific", "sum"),
            estimable_directed_lr=("_effect", "size"),
            minimum_reference_subjects=("n_subjects_reference", "min"),
            minimum_target_subjects=("n_subjects_target", "min"),
        )
        .reset_index()
    )
    pairs = pair_universe.merge(
        summary, on=pair_keys, how="left", validate="one_to_one"
    )
    integer_columns = (
        "target_specific_directed_lr",
        "reference_specific_directed_lr",
        "estimable_directed_lr",
        "minimum_reference_subjects",
        "minimum_target_subjects",
    )
    for column in integer_columns:
        pairs[column] = pairs[column].fillna(0).astype(np.int64)

    records: list[dict[str, object]] = []
    semantics = (
        "sum_positive_subject_equal_directed_lr_effects_after_unordered_"
        "cell_pair_collapse;descriptive_spatial_des_sensitivity_arm"
    )
    for row in pairs.to_dict(orient="records"):
        estimable = int(row["estimable_directed_lr"]) > 0
        for condition, strength_field, count_field in (
            (target, "target_strength", "target_specific_directed_lr"),
            (
                reference,
                "reference_strength",
                "reference_specific_directed_lr",
            ),
        ):
            records.append(
                {
                    **identity_values,
                    "ranking_semantics": semantics,
                    "condition": condition,
                    "sender": row["_sender_unordered"],
                    "receiver": row["_receiver_unordered"],
                    "ranked_strength": row[strength_field] if estimable else np.nan,
                    "condition_specific_directed_lr": row[count_field],
                    "estimable_directed_lr": row["estimable_directed_lr"],
                    "minimum_reference_subjects": row["minimum_reference_subjects"],
                    "minimum_target_subjects": row["minimum_target_subjects"],
                    "status": "observed" if estimable else "not_estimable",
                    "reason_code": (
                        None if estimable else "no_subject_supported_directed_lr"
                    ),
                    "formal_inference_allowed": False,
                }
            )
    result = pd.DataFrame.from_records(records)
    result["rank_by_ranked_strength"] = np.nan
    result["rank_by_condition_specific_directed_lr"] = np.nan
    condition_groups = result.groupby(
        "condition", observed=True, sort=False
    ).groups.items()
    for _, indices in condition_groups:
        selected = result.index.isin(indices) & result["status"].eq("observed")
        result.loc[selected, "rank_by_ranked_strength"] = result.loc[
            selected, "ranked_strength"
        ].rank(method="min", ascending=False)
        result.loc[selected, "rank_by_condition_specific_directed_lr"] = result.loc[
            selected, "condition_specific_directed_lr"
        ].rank(method="min", ascending=False)
    return result.loc[:, list(UNORDERED_DES_RANKING_COLUMNS)].sort_values(
        ["condition", "sender", "receiver"],
        kind="stable",
        ignore_index=True,
    )


__all__ = [
    "DIRECTED_EFFECT_COLUMNS",
    "UNORDERED_DES_RANKING_COLUMNS",
    "heldout_sample_coverage_audit",
    "subject_equal_directed_lr_effects",
    "unordered_cell_pair_des_rankings",
]
