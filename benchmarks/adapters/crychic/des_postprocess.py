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

ADJUSTED_DIRECTED_EFFECT_COLUMNS = (
    *DIRECTED_EFFECT_COLUMNS,
    "effect_standard_error_hc2",
    "effect_signal_to_noise",
    "one_standard_error_stable",
    "n_model_subjects",
    "n_contrast_informative_subjects",
    "model_rank",
    "residual_degrees_of_freedom",
    "covariate_adjustment",
    "response_transform",
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

PAIR_OPPORTUNITY_COLUMNS = (
    "sender",
    "receiver",
    "n_resource_directed_lr",
    "n_estimable_directed_lr",
    "n_target_up_directed_lr",
    "n_reference_up_directed_lr",
    "n_zero_effect_directed_lr",
    "n_one_se_stable_target_up_directed_lr",
    "n_one_se_stable_reference_up_directed_lr",
    "target_up_evidence",
    "reference_up_evidence",
    "target_opportunity_normalized_magnitude",
    "reference_opportunity_normalized_magnitude",
    "absolute_strength_target",
    "absolute_strength_reference",
    "minimum_reference_subjects",
    "minimum_target_subjects",
    "status",
    "reason_code",
)

REASON_WATERFALL_COLUMNS = ("score_layer", "status", "reason_code", "rows")


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


def sender_specific_direct_scores(
    score_layers: pd.DataFrame,
    semantic_availability: pd.DataFrame,
    *,
    condition_column: str,
    edge_columns: Sequence[str],
    covariate_columns: Sequence[str] = (),
    response_transform: str = "identity",
) -> pd.DataFrame:
    """Bind exact held-out sender-specific availability to the score universe.

    A missing semantic row is never imputed from the mechanistic layer because
    its structural-zero state may come from an eligibility gate rather than an
    exact sender-specific availability zero.
    """

    condition_column = _canonical_text(condition_column, field="condition_column")
    edges = tuple(edge_columns)
    covariates = tuple(covariate_columns)
    if not edges or len(set(edges)) != len(edges):
        raise ValueError("edge_columns must contain unique canonical names")
    if len(set(covariates)) != len(covariates):
        raise ValueError("covariate_columns must contain unique canonical names")
    if set(edges).intersection(covariates):
        raise ValueError("edge and covariate columns must not overlap")
    if response_transform not in {"identity", "sqrt"}:
        raise ValueError("response_transform must be 'identity' or 'sqrt'")

    join_keys = (
        "fold_id",
        "sample_id",
        "subject_id",
        "sender",
        "receiver",
        "interaction_id",
    )
    score_columns = tuple(
        dict.fromkeys(
            (
                *join_keys,
                condition_column,
                *covariates,
                *edges,
                "mechanistic_status",
                "mechanistic_reason_code",
            )
        )
    )
    missing_scores = set(score_columns).difference(score_layers.columns)
    availability_columns = (
        *join_keys,
        "mode",
        "availability_score",
        "status",
        "reason_code",
    )
    missing_availability = set(availability_columns).difference(
        semantic_availability.columns
    )
    if missing_scores or score_layers.empty:
        raise ValueError(
            f"score layers are empty or missing: {sorted(missing_scores)}"
        )
    if missing_availability or semantic_availability.empty:
        raise ValueError(
            "semantic availability is empty or missing: "
            f"{sorted(missing_availability)}"
        )

    base = score_layers.loc[:, list(score_columns)]
    if base.duplicated(list(join_keys)).any():
        raise ValueError("score-layer sample sender-LR keys must be unique")
    availability = semantic_availability.loc[
        semantic_availability["mode"].astype(str).eq("state"),
        list(availability_columns),
    ].copy()
    if availability.empty or availability.duplicated(list(join_keys)).any():
        raise ValueError(
            "state semantic availability keys must be non-empty and unique"
        )
    availability_status = availability["status"].astype(str)
    invalid_status = set(availability_status).difference(
        {"observed", "not_estimable"}
    )
    if invalid_status:
        raise ValueError(
            "state semantic availability contains invalid statuses: "
            f"{sorted(invalid_status)}"
        )
    availability_score = pd.to_numeric(
        availability["availability_score"], errors="coerce"
    )
    observed_availability = availability_status.eq("observed")
    if (
        availability_score.loc[observed_availability].isna().any()
        or np.isinf(availability_score.loc[observed_availability]).any()
        or not availability_score.loc[observed_availability]
        .between(0.0, 1.0)
        .all()
        or availability_score.loc[~observed_availability].notna().any()
    ):
        raise ValueError(
            "state semantic availability score/status semantics are invalid"
        )
    availability["availability_score"] = availability_score
    availability = availability.rename(
        columns={
            "status": "_availability_status",
            "reason_code": "_availability_reason_code",
        }
    ).drop(columns="mode")

    result = base.merge(
        availability,
        on=list(join_keys),
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    matched = result["_merge"].eq("both")
    matched_observed = matched & result["_availability_status"].astype(str).eq(
        "observed"
    )
    matched_not_estimable = matched & result["_availability_status"].astype(
        str
    ).eq("not_estimable")
    mechanism_status = result["mechanistic_status"].astype(str)
    missing_observed_mechanism = ~matched & mechanism_status.eq("observed")
    if missing_observed_mechanism.any():
        raise ValueError(
            "observed mechanistic rows are missing sender-specific availability"
        )
    raw = pd.Series(np.nan, index=result.index, dtype=float)
    raw.loc[matched_observed] = result.loc[
        matched_observed, "availability_score"
    ].to_numpy(dtype=float, copy=False)
    direct_status = pd.Series("not_estimable", index=result.index, dtype=object)
    direct_status.loc[matched_observed] = "observed"
    direct_reason = pd.Series(
        "sender_specific_availability_not_estimable",
        index=result.index,
        dtype=object,
    )
    direct_reason.loc[matched_observed] = None
    direct_reason.loc[matched_not_estimable] = result.loc[
        matched_not_estimable, "_availability_reason_code"
    ].where(
        result.loc[matched_not_estimable, "_availability_reason_code"].notna(),
        "sender_specific_availability_not_estimable",
    )
    transformed = raw if response_transform == "identity" else np.sqrt(raw)

    output_columns = tuple(
        dict.fromkeys(
            (
                "fold_id",
                "sample_id",
                "subject_id",
                condition_column,
                *covariates,
                *edges,
            )
        )
    )
    output = result.loc[:, list(output_columns)].copy()
    output["sender_specific_availability_state"] = raw.to_numpy(copy=False)
    output["direct_response_score"] = transformed.to_numpy(copy=False)
    output["direct_response_status"] = direct_status.to_numpy(copy=False)
    output["direct_response_reason_code"] = direct_reason.to_numpy(copy=False)
    output["direct_response_transform"] = response_transform
    not_estimable = output["direct_response_status"].eq("not_estimable")
    if output.loc[not_estimable, "direct_response_score"].notna().any():
        raise RuntimeError("not-estimable direct responses must remain missing")
    return output


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


def adjusted_subject_directed_lr_effects(
    direct_scores: pd.DataFrame,
    *,
    reference: str,
    target: str,
    condition_column: str,
    edge_columns: Sequence[str],
    categorical_covariates: Sequence[str] = (),
    score_column: str = "direct_response_score",
    status_column: str = "direct_response_status",
    min_subjects_per_condition: int = 3,
) -> pd.DataFrame:
    """Fit vectorized subject-level contrasts with HC2 robust uncertainty.

    Technical samples are averaged within subject.  Edges sharing the same
    subject-support mask are solved in one matrix operation, which keeps the
    cost close to the preceding groupby rather than the number of LR edges.
    """

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
    covariates = tuple(categorical_covariates)
    if not edges or len(set(edges)) != len(edges):
        raise ValueError("edge_columns must contain unique canonical names")
    if not {"sender", "receiver", "interaction_id"}.issubset(edges):
        raise ValueError(
            "edge_columns must include sender, receiver, and interaction_id"
        )
    if len(set(covariates)) != len(covariates):
        raise ValueError("categorical_covariates must contain unique names")
    required = {
        "sample_id",
        "subject_id",
        condition_column,
        score_column,
        status_column,
        *covariates,
        *edges,
    }
    missing = required.difference(direct_scores.columns)
    if missing or direct_scores.empty:
        raise ValueError(f"direct scores are empty or missing: {sorted(missing)}")

    design_columns = ["sample_id", "subject_id", condition_column, *covariates]
    sample_design = direct_scores.loc[:, design_columns].drop_duplicates(
        ignore_index=True
    )
    if sample_design["sample_id"].isna().any() or sample_design.duplicated(
        "sample_id"
    ).any():
        raise ValueError("sample_id maps to multiple subject/design rows")
    for column in ("sample_id", "subject_id", condition_column, *covariates):
        sample_design[column] = sample_design[column].astype(str)
    observed_conditions = set(sample_design[condition_column])
    if observed_conditions != {reference, target}:
        raise ValueError(
            "direct-score conditions disagree with the declared contrast: "
            f"{sorted(observed_conditions)}"
        )
    subject_design = sample_design.drop(columns="sample_id").drop_duplicates(
        ignore_index=True
    )
    if subject_design.duplicated("subject_id").any():
        raise ValueError(
            "adjusted independent-subject contrast requires one design row per subject"
        )
    subject_design = subject_design.sort_values(
        "subject_id", kind="stable", ignore_index=True
    )

    base = direct_scores.loc[:, list(edges)].drop_duplicates(ignore_index=True)
    if base.isna().any().any():
        raise ValueError("directed sender-receiver-LR identifiers must be complete")
    if base.duplicated(["sender", "receiver", "interaction_id"]).any():
        raise ValueError("directed sender-receiver-LR resource mapping is ambiguous")
    statuses = direct_scores[status_column].astype(str)
    invalid_statuses = set(statuses).difference(
        {"observed", "structural_zero", "not_estimable"}
    )
    if invalid_statuses:
        raise ValueError(
            f"direct scores contain invalid statuses: {sorted(invalid_statuses)}"
        )
    numeric = pd.to_numeric(direct_scores[score_column], errors="coerce")
    if numeric.loc[statuses.eq("not_estimable")].notna().any():
        raise ValueError("not-estimable direct rows must not carry scores")
    if not numeric.loc[statuses.eq("structural_zero")].eq(0.0).all():
        raise ValueError("structural-zero direct rows require exact zero")
    usable = direct_scores.loc[
        statuses.isin(("observed", "structural_zero")),
        ["sample_id", "subject_id", condition_column, *covariates, *edges],
    ].copy()
    usable["_strength"] = numeric.loc[usable.index].to_numpy(copy=False)
    if usable["_strength"].isna().any() or np.isinf(usable["_strength"]).any():
        raise ValueError("eligible direct-score rows require finite strengths")
    usable["sample_id"] = usable["sample_id"].astype(str)
    usable["subject_id"] = usable["subject_id"].astype(str)
    usable[condition_column] = usable[condition_column].astype(str)
    for covariate in covariates:
        usable[covariate] = usable[covariate].astype(str)

    subject = (
        usable.groupby(
            [condition_column, *covariates, "subject_id", *edges],
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

    subjects = subject_design["subject_id"].tolist()
    pivot = subject.pivot(
        index=list(edges), columns="subject_id", values="subject_strength"
    ).reindex(columns=subjects)
    values = pivot.to_numpy(dtype=float)
    present = ~np.isnan(values)
    conditions = subject_design[condition_column].to_numpy(dtype=str)
    reference_mask = conditions == reference
    target_mask = conditions == target
    support = (
        present[:, reference_mask].sum(axis=1) >= min_subjects_per_condition
    ) & (present[:, target_mask].sum(axis=1) >= min_subjects_per_condition)
    supported_rows = np.flatnonzero(support)
    supported_values = values[support]
    supported_present = present[support]
    supported_edges = pivot.index.to_frame(index=False).iloc[
        supported_rows
    ].reset_index(drop=True)

    effect = np.full(len(supported_edges), np.nan, dtype=float)
    standard_error = np.full(len(supported_edges), np.nan, dtype=float)
    n_model_subjects = np.zeros(len(supported_edges), dtype=np.int64)
    n_contrast_informative_subjects = np.zeros(
        len(supported_edges), dtype=np.int64
    )
    model_rank = np.zeros(len(supported_edges), dtype=np.int64)
    residual_df = np.zeros(len(supported_edges), dtype=np.int64)
    if len(supported_edges):
        packed_support = np.packbits(supported_present, axis=1, bitorder="little")
        _, support_codes = np.unique(
            packed_support, axis=0, return_inverse=True
        )
        canonical_reference, canonical_target = sorted((reference, target))
        orientation = 1.0 if target == canonical_target else -1.0
        canonical_indicator = (conditions == canonical_target).astype(float)
        covariate_values = {
            covariate: subject_design[covariate].to_numpy(dtype=str)
            for covariate in covariates
        }
        for support_code in range(int(support_codes.max()) + 1):
            row_indices = np.flatnonzero(support_codes == support_code)
            row_mask = supported_present[row_indices[0]]
            n_observed = int(row_mask.sum())
            design_parts = [
                np.ones(n_observed, dtype=float),
                canonical_indicator[row_mask],
            ]
            for covariate in covariates:
                local = covariate_values[covariate][row_mask]
                levels = sorted(set(local))
                design_parts.extend(
                    (local == level).astype(float) for level in levels[1:]
                )
            design = np.column_stack(design_parts)
            rank = int(np.linalg.matrix_rank(design))
            if rank != design.shape[1] or rank >= n_observed:
                continue
            cross_product_inverse = np.linalg.inv(design.T @ design)
            coefficient_operator = cross_product_inverse @ design.T
            response = supported_values[row_indices][:, row_mask].T
            coefficients = coefficient_operator @ response
            residual = response - design @ coefficients
            leverage = np.sum(
                design * (design @ cross_product_inverse), axis=1
            )
            contrast_weight = coefficient_operator[1]
            numerical_scale = max(
                1.0,
                float(np.max(np.abs(coefficient_operator))),
                float(np.max(np.abs(leverage))),
            )
            tolerance = 256.0 * np.finfo(np.float64).eps * numerical_scale
            contrast_informative = np.abs(contrast_weight) > tolerance
            leverage_saturated = (1.0 - leverage) <= tolerance
            if (contrast_informative & leverage_saturated).any():
                continue
            hc2_weight = np.zeros(n_observed, dtype=float)
            hc2_weight[contrast_informative] = contrast_weight[
                contrast_informative
            ] / np.sqrt(1.0 - leverage[contrast_informative])
            canonical_effect = coefficients[1]
            effect[row_indices] = orientation * canonical_effect
            standard_error[row_indices] = np.sqrt(
                np.sum((hc2_weight[:, None] * residual) ** 2, axis=0)
            )
            n_model_subjects[row_indices] = n_observed
            n_contrast_informative_subjects[row_indices] = int(
                contrast_informative.sum()
            )
            model_rank[row_indices] = rank
            residual_df[row_indices] = n_observed - rank
        del canonical_reference

    model = supported_edges.copy()
    model["effect_target_minus_reference"] = effect
    model["effect_standard_error_hc2"] = standard_error
    model["n_model_subjects"] = n_model_subjects
    model["n_contrast_informative_subjects"] = (
        n_contrast_informative_subjects
    )
    model["model_rank"] = model_rank
    model["residual_degrees_of_freedom"] = residual_df
    result = result.merge(
        model,
        on=list(edges),
        how="left",
        validate="one_to_one",
    )
    for column in (
        "n_samples_reference",
        "n_samples_target",
        "n_subjects_reference",
        "n_subjects_target",
        "n_model_subjects",
        "n_contrast_informative_subjects",
        "model_rank",
        "residual_degrees_of_freedom",
    ):
        result[column] = result[column].fillna(0).astype(np.int64)
    estimable = (
        result["effect_target_minus_reference"].notna()
        & result["effect_standard_error_hc2"].notna()
        & result["mean_strength_reference"].notna()
        & result["mean_strength_target"].notna()
    )
    signal_to_noise = pd.Series(np.nan, index=result.index, dtype=float)
    positive_se = estimable & result["effect_standard_error_hc2"].gt(0.0)
    signal_to_noise.loc[positive_se] = (
        result.loc[positive_se, "effect_target_minus_reference"]
        / result.loc[positive_se, "effect_standard_error_hc2"]
    )
    zero_se = estimable & result["effect_standard_error_hc2"].eq(0.0)
    zero_effect = zero_se & result["effect_target_minus_reference"].eq(0.0)
    signal_to_noise.loc[zero_effect] = 0.0
    separated = zero_se & ~zero_effect
    signal_to_noise.loc[separated] = (
        np.sign(result.loc[separated, "effect_target_minus_reference"])
        * np.finfo(np.float64).max
    )
    stable = (
        estimable
        & result["effect_target_minus_reference"].ne(0.0)
        & result["effect_target_minus_reference"].abs().ge(
            result["effect_standard_error_hc2"]
        )
    )
    result["reference_condition"] = reference
    result["target_condition"] = target
    result["status"] = np.where(estimable, "observed", "not_estimable")
    result["reason_code"] = np.where(
        estimable, None, "insufficient_subject_or_covariate_model_support"
    )
    covariate_semantics = (
        "none"
        if not covariates
        else "categorical_fixed_effects:" + ",".join(covariates)
    )
    result["effect_semantics"] = (
        "target_minus_reference_subject_level_linear_contrast_specific_hc2_"
        f"{score_column}_exploratory_v3"
    )
    result["formal_inference_allowed"] = False
    result["effect_signal_to_noise"] = signal_to_noise.to_numpy(copy=False)
    result["one_standard_error_stable"] = stable.to_numpy(copy=False)
    result["covariate_adjustment"] = covariate_semantics
    response_transform = (
        direct_scores["direct_response_transform"].astype(str).unique()
        if "direct_response_transform" in direct_scores.columns
        else np.asarray(["unspecified"])
    )
    if len(response_transform) != 1:
        raise ValueError("direct responses must use one response transform")
    result["response_transform"] = str(response_transform[0])
    return result.loc[:, [*edges, *ADJUSTED_DIRECTED_EFFECT_COLUMNS]].sort_values(
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


def stable_breadth_unordered_cell_pair_rankings(
    effects: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    method_version: str,
    resource: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build condition rankings from stable sender-specific differential breadth."""

    identity_values = {
        field: _canonical_text(value, field=field)
        for field, value in {
            "dataset": dataset,
            "method": method,
            "method_version": method_version,
            "resource": resource,
        }.items()
    }
    required = {
        "sender",
        "receiver",
        "interaction_id",
        "reference_condition",
        "target_condition",
        "mean_strength_reference",
        "mean_strength_target",
        "n_subjects_reference",
        "n_subjects_target",
        "effect_target_minus_reference",
        "one_standard_error_stable",
        "status",
    }
    missing = required.difference(effects.columns)
    if missing or effects.empty:
        raise ValueError(
            f"adjusted directed LR effects are empty or missing: {sorted(missing)}"
        )
    if effects.duplicated(["sender", "receiver", "interaction_id"]).any():
        raise ValueError("adjusted directed LR effects contain duplicate edges")
    references = effects["reference_condition"].astype(str).unique()
    targets = effects["target_condition"].astype(str).unique()
    if len(references) != 1 or len(targets) != 1 or references[0] == targets[0]:
        raise ValueError("adjusted directed effects require one reference/target pair")
    reference = str(references[0])
    target = str(targets[0])
    statuses = effects["status"].astype(str)
    invalid_statuses = set(statuses).difference({"observed", "not_estimable"})
    if invalid_statuses:
        raise ValueError(
            f"adjusted directed effects contain invalid statuses: {invalid_statuses}"
        )
    numeric_columns = (
        "mean_strength_reference",
        "mean_strength_target",
        "effect_target_minus_reference",
    )
    numeric = {
        column: pd.to_numeric(effects[column], errors="coerce")
        for column in numeric_columns
    }
    observed = statuses.eq("observed")
    for column, values in numeric.items():
        if values.loc[observed].isna().any() or np.isinf(values.loc[observed]).any():
            raise ValueError(f"observed adjusted effects require finite {column}")
        if column == "effect_target_minus_reference" and values.loc[
            ~observed
        ].notna().any():
            raise ValueError(f"non-estimable adjusted effects must not carry {column}")
        if column != "effect_target_minus_reference" and np.isinf(
            values.loc[~observed].dropna()
        ).any():
            raise ValueError(
                f"non-estimable adjusted effects contain infinite {column}"
            )
    stable = effects["one_standard_error_stable"]
    if stable.loc[observed].isna().any() or not stable.loc[observed].isin(
        (True, False)
    ).all():
        raise ValueError("observed adjusted effects require boolean stability")

    working = effects.copy(deep=False)
    sender = working["sender"].astype(str).to_numpy()
    receiver = working["receiver"].astype(str).to_numpy()
    working = working.assign(
        _sender_unordered=np.minimum(sender, receiver),
        _receiver_unordered=np.maximum(sender, receiver),
        _effect=numeric["effect_target_minus_reference"].to_numpy(copy=False),
        _mean_reference=numeric["mean_strength_reference"].to_numpy(copy=False),
        _mean_target=numeric["mean_strength_target"].to_numpy(copy=False),
    )
    pair_keys = ["_sender_unordered", "_receiver_unordered"]
    resource_summary = (
        working.groupby(pair_keys, observed=True, sort=True)
        .agg(n_resource_directed_lr=("interaction_id", "size"))
        .reset_index()
    )
    eligible = working.loc[observed].copy()
    eligible["_target_up"] = eligible["_effect"].gt(0.0)
    eligible["_reference_up"] = eligible["_effect"].lt(0.0)
    eligible["_zero_effect"] = eligible["_effect"].eq(0.0)
    eligible["_stable_target_up"] = (
        eligible["_target_up"] & eligible["one_standard_error_stable"].astype(bool)
    )
    eligible["_stable_reference_up"] = (
        eligible["_reference_up"]
        & eligible["one_standard_error_stable"].astype(bool)
    )
    eligible["_target_up_evidence"] = eligible["_effect"].clip(lower=0.0)
    eligible["_reference_up_evidence"] = (-eligible["_effect"]).clip(lower=0.0)
    summary = (
        eligible.groupby(pair_keys, observed=True, sort=True)
        .agg(
            n_estimable_directed_lr=("_effect", "size"),
            n_target_up_directed_lr=("_target_up", "sum"),
            n_reference_up_directed_lr=("_reference_up", "sum"),
            n_zero_effect_directed_lr=("_zero_effect", "sum"),
            n_one_se_stable_target_up_directed_lr=("_stable_target_up", "sum"),
            n_one_se_stable_reference_up_directed_lr=(
                "_stable_reference_up",
                "sum",
            ),
            target_up_evidence=("_target_up_evidence", "sum"),
            reference_up_evidence=("_reference_up_evidence", "sum"),
            absolute_strength_target=("_mean_target", "sum"),
            absolute_strength_reference=("_mean_reference", "sum"),
            minimum_reference_subjects=("n_subjects_reference", "min"),
            minimum_target_subjects=("n_subjects_target", "min"),
        )
        .reset_index()
    )
    pairs = resource_summary.merge(
        summary, on=pair_keys, how="left", validate="one_to_one"
    )
    count_columns = (
        "n_estimable_directed_lr",
        "n_target_up_directed_lr",
        "n_reference_up_directed_lr",
        "n_zero_effect_directed_lr",
        "n_one_se_stable_target_up_directed_lr",
        "n_one_se_stable_reference_up_directed_lr",
        "minimum_reference_subjects",
        "minimum_target_subjects",
    )
    for column in count_columns:
        pairs[column] = pairs[column].fillna(0).astype(np.int64)
    estimable_pair = pairs["n_estimable_directed_lr"].gt(0)
    denominator = np.sqrt(pairs["n_estimable_directed_lr"].where(estimable_pair))
    pairs["target_opportunity_normalized_magnitude"] = (
        pairs["target_up_evidence"] / denominator
    )
    pairs["reference_opportunity_normalized_magnitude"] = (
        pairs["reference_up_evidence"] / denominator
    )
    pairs["status"] = np.where(estimable_pair, "observed", "not_estimable")
    pairs["reason_code"] = np.where(
        estimable_pair, None, "no_subject_supported_directed_lr"
    )
    opportunity = pairs.rename(
        columns={
            "_sender_unordered": "sender",
            "_receiver_unordered": "receiver",
        }
    ).loc[:, list(PAIR_OPPORTUNITY_COLUMNS)]

    common = pairs.loc[
        :,
        [
            *pair_keys,
            "n_estimable_directed_lr",
            "minimum_reference_subjects",
            "minimum_target_subjects",
            "status",
            "reason_code",
        ],
    ]
    target_rows = common.assign(
        condition=target,
        ranked_strength=pairs[
            "n_one_se_stable_target_up_directed_lr"
        ].where(estimable_pair),
        condition_specific_directed_lr=pairs[
            "n_one_se_stable_target_up_directed_lr"
        ],
    )
    reference_rows = common.assign(
        condition=reference,
        ranked_strength=pairs[
            "n_one_se_stable_reference_up_directed_lr"
        ].where(estimable_pair),
        condition_specific_directed_lr=pairs[
            "n_one_se_stable_reference_up_directed_lr"
        ],
    )
    result = pd.concat([target_rows, reference_rows], ignore_index=True)
    result = result.rename(
        columns={
            "_sender_unordered": "sender",
            "_receiver_unordered": "receiver",
            "n_estimable_directed_lr": "estimable_directed_lr",
        }
    )
    for field, value in identity_values.items():
        result[field] = value
    result["ranking_semantics"] = (
        "count_one_standard_error_stable_sender_specific_hc2_adjusted_"
        "directed_lr_effects_after_unordered_cell_pair_collapse;exploratory_v3"
    )
    result["formal_inference_allowed"] = False
    result["rank_by_ranked_strength"] = np.nan
    result["rank_by_condition_specific_directed_lr"] = np.nan
    eligible_ranking = result["status"].eq("observed")
    result.loc[eligible_ranking, "rank_by_ranked_strength"] = (
        result.loc[eligible_ranking]
        .groupby("condition", observed=True, sort=False)["ranked_strength"]
        .rank(method="min", ascending=False)
    )
    result.loc[eligible_ranking, "rank_by_condition_specific_directed_lr"] = (
        result.loc[eligible_ranking]
        .groupby("condition", observed=True, sort=False)[
            "condition_specific_directed_lr"
        ]
        .rank(method="min", ascending=False)
    )
    rankings = result.loc[:, list(UNORDERED_DES_RANKING_COLUMNS)].sort_values(
        ["condition", "sender", "receiver"],
        kind="stable",
        ignore_index=True,
    )
    opportunity = opportunity.sort_values(
        ["sender", "receiver"], kind="stable", ignore_index=True
    )
    return rankings, opportunity


def condition_ranking_diagnostics(rankings: pd.DataFrame) -> dict[str, object]:
    """Report per-condition score degeneracy independently of sample scores."""

    required = {"condition", "ranked_strength", "status"}
    missing = required.difference(rankings.columns)
    if missing or rankings.empty:
        raise ValueError(f"condition rankings are empty or missing: {sorted(missing)}")
    conditions: dict[str, dict[str, object]] = {}
    for condition, group in rankings.groupby("condition", observed=True, sort=True):
        observed = group.loc[group["status"].astype(str).eq("observed")]
        score = pd.to_numeric(observed["ranked_strength"], errors="coerce")
        if score.isna().any() or np.isinf(score).any():
            raise ValueError("observed condition rankings require finite strengths")
        counts = score.value_counts(dropna=False)
        tied_rows = int(counts.loc[counts.gt(1)].sum()) if not counts.empty else 0
        unique = int(score.nunique(dropna=True))
        n_observed = len(score)
        opportunity_correlation: float | None = None
        if "estimable_directed_lr" in observed.columns and n_observed > 1:
            opportunity = pd.to_numeric(
                observed["estimable_directed_lr"], errors="coerce"
            )
            if (
                opportunity.notna().all()
                and score.nunique(dropna=True) > 1
                and opportunity.nunique(dropna=True) > 1
            ):
                correlation = score.rank(method="average").corr(
                    opportunity.rank(method="average")
                )
                if pd.notna(correlation):
                    opportunity_correlation = float(correlation)
        conditions[str(condition)] = {
            "rows": len(group),
            "observed_rows": n_observed,
            "unique_score_count": unique,
            "nonzero_rows": int(score.ne(0.0).sum()),
            "all_zero": bool(n_observed and score.eq(0.0).all()),
            "tie_fraction": tied_rows / n_observed if n_observed else 1.0,
            "degenerate_ranking": bool(n_observed > 1 and unique <= 1),
            "spearman_ranked_strength_vs_estimable_directed_lr": (
                opportunity_correlation
            ),
            "high_opportunity_correlation": bool(
                opportunity_correlation is not None
                and abs(opportunity_correlation) >= 0.7
            ),
        }
    return {
        "conditions": conditions,
        "any_degenerate_condition": any(
            bool(value["degenerate_ranking"]) for value in conditions.values()
        ),
        "any_high_opportunity_correlation": any(
            bool(value["high_opportunity_correlation"])
            for value in conditions.values()
        ),
    }


def score_reason_waterfall(
    score_layers: pd.DataFrame,
    direct_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Count fail-closed statuses and reasons without reading component ledgers."""

    layer_columns = (
        ("base", "status", "reason_code"),
        ("mechanistic", "mechanistic_status", "mechanistic_reason_code"),
        ("downstream", "downstream_status", "downstream_reason_code"),
        ("selected", "selected_score_status", "selected_score_reason_code"),
    )
    records: list[pd.DataFrame] = []
    for layer, status_column, reason_column in layer_columns:
        missing = {status_column, reason_column}.difference(score_layers.columns)
        if missing:
            raise ValueError(f"score layers are missing waterfall columns: {missing}")
        summary = (
            score_layers.groupby(
                [status_column, reason_column],
                observed=True,
                sort=True,
                dropna=False,
            )
            .size()
            .reset_index(name="rows")
            .rename(
                columns={status_column: "status", reason_column: "reason_code"}
            )
        )
        summary.insert(0, "score_layer", layer)
        records.append(summary)
    direct_required = {
        "direct_response_status",
        "direct_response_reason_code",
    }
    missing_direct = direct_required.difference(direct_scores.columns)
    if missing_direct:
        raise ValueError(
            f"direct scores are missing waterfall columns: {missing_direct}"
        )
    direct = (
        direct_scores.groupby(
            ["direct_response_status", "direct_response_reason_code"],
            observed=True,
            sort=True,
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .rename(
            columns={
                "direct_response_status": "status",
                "direct_response_reason_code": "reason_code",
            }
        )
    )
    direct.insert(0, "score_layer", "sender_specific_direct")
    records.append(direct)
    return pd.concat(records, ignore_index=True).loc[
        :, list(REASON_WATERFALL_COLUMNS)
    ]


__all__ = [
    "ADJUSTED_DIRECTED_EFFECT_COLUMNS",
    "DIRECTED_EFFECT_COLUMNS",
    "PAIR_OPPORTUNITY_COLUMNS",
    "REASON_WATERFALL_COLUMNS",
    "UNORDERED_DES_RANKING_COLUMNS",
    "adjusted_subject_directed_lr_effects",
    "condition_ranking_diagnostics",
    "heldout_sample_coverage_audit",
    "score_reason_waterfall",
    "sender_specific_direct_scores",
    "stable_breadth_unordered_cell_pair_rankings",
    "subject_equal_directed_lr_effects",
    "unordered_cell_pair_des_rankings",
]
