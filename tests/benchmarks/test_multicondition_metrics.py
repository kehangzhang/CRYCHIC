from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from benchmarks.metrics import multicondition as multicondition_metrics
from benchmarks.metrics.multicondition import (
    EDGE_KEYS,
    MOLECULAR_LR_EQUIVALENCE_COLUMN,
    aggregate_loso_primary_endpoint,
    cross_method_concordance,
    external_long_to_score_table,
    paired_differential_loso_reproducibility,
    paired_edge_effects,
    score_coverage_summary,
    summarize_loso_primary_endpoint,
    summarize_run_performance,
    synthetic_edge_truth_metrics,
    unpaired_differential_split_half_reproducibility,
    unpaired_edge_effects,
    unpaired_leave_one_subject_influence,
    validate_score_table,
    within_context_reproducibility,
)


def _scores(
    *,
    method: str = "m1",
    method_version: str = "1",
    score_direction: str = "higher",
    resource: str = "common",
    resource_version: str = "2026-01",
    score_semantics: str = "strength_not_probability",
    contrast: str = "Tumor-vs-Normal",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_index, subject in enumerate(("p1", "p2", "p3")):
        for context, offset in (("Normal", 0.0), ("Tumor", 2.0)):
            for index in range(4):
                rows.append(
                    {
                        "dataset": "cscc",
                        "method": method,
                        "method_version": method_version,
                        "analysis_track": "lr_stlr",
                        "resource": resource,
                        "resource_version": resource_version,
                        "resource_mode": "H-common",
                        "sample_id": f"{subject}_{context}",
                        "subject_id": subject,
                        "context": context,
                        "contrast": contrast,
                        "sender": "S",
                        "receiver": "R",
                        "interaction_id": f"I{index}",
                        "ligand": f"L{index}",
                        "receptor": f"R{index}",
                        "score": index + offset + subject_index / 10,
                        "score_direction": score_direction,
                        "score_semantics": score_semantics,
                        "status": "observed",
                        "universe_id": "cscc-h-common-v1",
                        "universe_member": True,
                        "universe_size": 4,
                    }
                )
    return pd.DataFrame(rows)


def _multi_family_scores() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    edges = [("small", "R-small", f"LS{index}", f"RS{index}") for index in range(3)] + [
        ("large", "R-large", f"LL{index}", f"RL{index}") for index in range(9)
    ]
    for subject in ("p1", "p2", "p3", "p4"):
        for context in ("Normal", "Tumor"):
            for edge_index, (sender, receiver, ligand, receptor) in enumerate(edges):
                if context == "Normal":
                    score = 0.0
                elif sender == "small":
                    score = float(edge_index + 1)
                elif subject == "p1":
                    score = float(100 - edge_index)
                else:
                    score = float(edge_index)
                rows.append(
                    {
                        "dataset": "cscc",
                        "method": "m1",
                        "method_version": "1",
                        "analysis_track": "lr_stlr",
                        "resource": "common",
                        "resource_version": "2026-01",
                        "resource_mode": "H-common",
                        "score_semantics": "strength_not_probability",
                        "universe_id": "cscc-two-families-v1",
                        "sample_id": f"{subject}_{context}",
                        "subject_id": subject,
                        "context": context,
                        "contrast": "Tumor-vs-Normal",
                        "sender": sender,
                        "receiver": receiver,
                        "interaction_id": f"two-family-{edge_index}",
                        "ligand": ligand,
                        "receptor": receptor,
                        "score": score,
                        "score_direction": "higher",
                        "status": "observed",
                        "universe_member": True,
                        "universe_size": len(edges),
                    }
                )
    return pd.DataFrame(rows)


def _unpaired_scores(*, n_reference: int = 6, n_target: int = 6) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(1701)
    edges: list[tuple[str, str, str, str]] = [
        (
            "S-small" if index < 4 else "S-large",
            "R-small" if index < 4 else "R-large",
            f"L{index}",
            f"R{index}",
        )
        for index in range(8)
    ]
    base: np.ndarray = np.arange(len(edges), dtype=float)
    differential: np.ndarray = np.asarray([2.5, -1.5, 1.0, -2.0, 1.5, -1.0, 2.0, -2.5])
    groups: tuple[tuple[str, str, int, np.ndarray], ...] = (
        ("Ctrl", "c", n_reference, np.zeros(len(edges), dtype=float)),
        ("Case", "t", n_target, differential),
    )
    for context, prefix, n_subjects, context_effect in groups:
        for subject_index in range(n_subjects):
            subject = f"{prefix}{subject_index + 1}"
            noise = rng.normal(0.0, 0.8, len(edges))
            scores = base + context_effect + noise
            for edge_index, (sender, receiver, ligand, receptor) in enumerate(edges):
                rows.append(
                    {
                        "dataset": "ms",
                        "method": "m1",
                        "method_version": "1",
                        "analysis_track": "lr_stlr",
                        "resource": "common",
                        "resource_version": "2026-01",
                        "resource_mode": "H-common",
                        "sample_id": f"{subject}_{context}",
                        "subject_id": subject,
                        "context": context,
                        "contrast": "Case-vs-Ctrl",
                        "sender": sender,
                        "receiver": receiver,
                        "interaction_id": f"I{edge_index}",
                        "ligand": ligand,
                        "receptor": receptor,
                        "score": float(scores[edge_index]),
                        "score_direction": "higher",
                        "score_semantics": "strength_not_probability",
                        "status": "observed",
                        "universe_id": "ms-h-common-v1",
                        "universe_member": True,
                        "universe_size": len(edges),
                    }
                )
    return pd.DataFrame(rows)


def _rich_effect_scores(
    *, paired: bool, random_seed: int | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    if paired:
        samples = [
            (subject, context, f"{subject}_{context}")
            for subject in ("p1", "p2", "p3")
            for context in ("Ctrl", "Case")
        ]
        samples.append(("p1", "Case", "p1_Case_library2"))
    else:
        samples = [
            *((f"c{index}", "Ctrl", f"c{index}_Ctrl") for index in range(1, 4)),
            *((f"t{index}", "Case", f"t{index}_Case") for index in range(1, 4)),
            ("t1", "Case", "t1_Case_library2"),
        ]
    samples = [
        (subject, context, f"{len(samples) - index:02d}_{sample_id}")
        for index, (subject, context, sample_id) in enumerate(samples)
    ]
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(("m1", "m2"), start=1):
        for sample_index, (subject, context, sample_id) in enumerate(samples):
            identity_subject = subject if method_index == 1 else f"m2_{subject}"
            identity_sample = sample_id if method_index == 1 else f"m2_{sample_id}"
            for edge_index in range(15):
                status = "observed"
                if edge_index == 1:
                    status = "not_predicted"
                elif edge_index == 2:
                    status = "missing"
                elif edge_index == 3:
                    status = "cell_type_missing"
                elif edge_index == 4:
                    status = "resource_unavailable"
                elif edge_index == 5:
                    status = "not_supported"
                elif edge_index == 6:
                    status = "failed"
                elif edge_index == 7 and (
                    (paired and subject == "p3" and context == "Case")
                    or (not paired and subject == "t3")
                ):
                    status = "missing"
                elif random_seed is not None and edge_index >= 7:
                    draw = rng.random()
                    if draw < 0.10:
                        status = "missing"
                    elif draw < 0.18:
                        status = "cell_type_missing"
                    elif draw < 0.28:
                        status = "not_predicted"
                score = (
                    float(rng.normal())
                    if random_seed is not None and status == "observed"
                    else float(
                        (edge_index * 7 + sample_index * 3 + method_index) % 17
                        + edge_index / 100
                        + (0.25 if context == "Case" else 0)
                    )
                    if status == "observed"
                    else np.nan
                )
                rows.append(
                    {
                        "dataset": "rich",
                        "method": method,
                        "method_version": str(method_index),
                        "analysis_track": "lr_stlr",
                        "resource": "common",
                        "resource_version": "2026-01",
                        "resource_mode": "H-common",
                        "sample_id": identity_sample,
                        "subject_id": identity_subject,
                        "context": context,
                        "contrast": "Case-vs-Ctrl",
                        "sender": f"S{edge_index % 2}",
                        "receiver": f"R{edge_index % 3}",
                        "interaction_id": f"I{edge_index}",
                        "ligand": f"L{edge_index}",
                        "receptor": f"Q{edge_index}",
                        "score": score,
                        "score_direction": (
                            "higher" if method_index == 1 else "lower"
                        ),
                        "score_semantics": "rank_strength",
                        "status": status,
                        "universe_id": f"rich-v{method_index}",
                        "universe_member": True,
                        "universe_size": 15,
                    }
                )
    return validate_score_table(pd.DataFrame(rows))


def _paired_effects_scalar_reference(
    scores: pd.DataFrame, *, reference: str, target: str, min_pairs: int
) -> pd.DataFrame:
    subset = scores.loc[scores["context"].isin((reference, target))].copy()
    grouping = [
        *multicondition_metrics.METHOD_IDENTITY_KEYS,
        "contrast",
        *EDGE_KEYS,
    ]
    rows: list[dict[str, object]] = []
    for keys, group in subset.groupby(grouping, observed=True, sort=False):
        usable = group.loc[group["comparison_eligible"]]
        subject_scores = (
            usable.groupby(["subject_id", "context"], observed=True, sort=False)[
                "comparison_strength"
            ]
            .mean()
            .unstack("context")
        )
        if reference in subject_scores and target in subject_scores:
            complete = subject_scores[[reference, target]].dropna()
            differences = (complete[target] - complete[reference]).to_numpy(
                dtype=float
            )
        else:
            differences = np.array([], dtype=float)
        n_pairs = len(differences)
        effect = float(differences.mean()) if n_pairs else np.nan
        standard_error = (
            float(differences.std(ddof=1) / math.sqrt(n_pairs))
            if n_pairs >= 2
            else np.nan
        )
        scale = float(differences.std(ddof=1)) if n_pairs >= 2 else np.nan
        nonzero_differences = differences[differences != 0]
        direction_comparable_pairs = len(nonzero_differences)
        if effect > 0 and direction_comparable_pairs:
            direction_consistency = float(np.mean(nonzero_differences > 0))
        elif effect < 0 and direction_comparable_pairs:
            direction_consistency = float(np.mean(nonzero_differences < 0))
        else:
            direction_consistency = np.nan
        row = dict(zip(grouping, keys, strict=True))
        estimable = n_pairs >= min_pairs
        row.update(
            {
                "reference": reference,
                "target": target,
                "effect": effect,
                "effect_semantics": "target_minus_reference_comparison_strength",
                "diagnostic_standard_error": standard_error,
                "standardized_effect": (
                    effect / scale if np.isfinite(scale) and scale > 0 else np.nan
                ),
                "median_effect": (
                    float(np.median(differences)) if n_pairs else np.nan
                ),
                "direction_consistency": direction_consistency,
                "direction_comparable_pairs": direction_comparable_pairs,
                "n_pairs": n_pairs,
                "n_not_predicted_rows": int(
                    group["status"].eq("not_predicted").sum()
                ),
                "n_missing_rows": int(group["status"].eq("missing").sum()),
                "n_resource_unavailable_rows": int(
                    group["status"].eq("resource_unavailable").sum()
                ),
                "n_cell_type_missing_rows": int(
                    group["status"].eq("cell_type_missing").sum()
                ),
                "status": "exploratory" if estimable else "not_estimable",
                "reason_code": (
                    None
                    if estimable
                    else multicondition_metrics._reason_for_unestimable(group, n_pairs)
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _unpaired_effects_scalar_reference(
    scores: pd.DataFrame, *, reference: str, target: str, min_subjects: int
) -> pd.DataFrame:
    scores = multicondition_metrics._normalize_subject_context(scores)
    scores = scores.loc[scores["context"].isin((reference, target))]
    identity_keys = [*multicondition_metrics.METHOD_IDENTITY_KEYS, "contrast"]
    rows: list[dict[str, object]] = []
    for identity_values, identity_group in scores.groupby(
        identity_keys, observed=True, sort=False
    ):
        reference_subjects, target_subjects = (
            multicondition_metrics._validate_unpaired_contexts(
                identity_group,
                reference=reference,
                target=target,
            )
        )
        identity = dict(zip(identity_keys, identity_values, strict=True))
        for edge_values, edge_group in identity_group.groupby(
            list(EDGE_KEYS), observed=True, sort=False
        ):
            usable = edge_group.loc[edge_group["comparison_eligible"]]
            subject_scores = (
                usable.groupby(["subject_id", "context"], observed=True, sort=False)[
                    "comparison_strength"
                ]
                .mean()
                .reset_index()
            )
            reference_values = subject_scores.loc[
                subject_scores["context"].eq(reference), "comparison_strength"
            ].to_numpy(dtype=float)
            target_values = subject_scores.loc[
                subject_scores["context"].eq(target), "comparison_strength"
            ].to_numpy(dtype=float)
            n_reference = len(reference_values)
            n_target = len(target_values)
            effect = (
                float(target_values.mean() - reference_values.mean())
                if n_reference and n_target
                else math.nan
            )
            diagnostic_standard_error = (
                float(
                    math.sqrt(
                        target_values.var(ddof=1) / n_target
                        + reference_values.var(ddof=1) / n_reference
                    )
                )
                if n_reference >= 2 and n_target >= 2
                else math.nan
            )
            pooled_denominator = n_reference + n_target - 2
            pooled_variance = (
                (
                    (n_reference - 1) * reference_values.var(ddof=1)
                    + (n_target - 1) * target_values.var(ddof=1)
                )
                / pooled_denominator
                if n_reference >= 2 and n_target >= 2 and pooled_denominator > 0
                else math.nan
            )
            estimable = n_reference >= min_subjects and n_target >= min_subjects
            row = identity | dict(zip(EDGE_KEYS, edge_values, strict=True))
            row.update(
                {
                    "reference": reference,
                    "target": target,
                    "effect": effect,
                    "effect_semantics": (
                        "target_minus_reference_subject_mean_comparison_strength"
                    ),
                    "diagnostic_standard_error": diagnostic_standard_error,
                    "standardized_effect": (
                        effect / math.sqrt(pooled_variance)
                        if np.isfinite(effect)
                        and np.isfinite(pooled_variance)
                        and pooled_variance > 0
                        else math.nan
                    ),
                    "reference_mean": (
                        float(reference_values.mean()) if n_reference else math.nan
                    ),
                    "target_mean": (
                        float(target_values.mean()) if n_target else math.nan
                    ),
                    "n_reference_subjects": n_reference,
                    "n_target_subjects": n_target,
                    "n_reference_subjects_total": len(reference_subjects),
                    "n_target_subjects_total": len(target_subjects),
                    "n_reference_samples": int(
                        usable.loc[
                            usable["context"].eq(reference), "sample_id"
                        ].nunique()
                    ),
                    "n_target_samples": int(
                        usable.loc[
                            usable["context"].eq(target), "sample_id"
                        ].nunique()
                    ),
                    "minimum_subjects_per_context": min_subjects,
                    "aggregation": (
                        "sample_mean_within_subject_context_then_equal_subject_mean"
                    ),
                    "design": "independent_subject_groups",
                    "n_not_predicted_rows": int(
                        edge_group["status"].eq("not_predicted").sum()
                    ),
                    "n_missing_rows": int(edge_group["status"].eq("missing").sum()),
                    "n_resource_unavailable_rows": int(
                        edge_group["status"].eq("resource_unavailable").sum()
                    ),
                    "n_cell_type_missing_rows": int(
                        edge_group["status"].eq("cell_type_missing").sum()
                    ),
                    "status": "exploratory" if estimable else "not_estimable",
                    "reason_code": (
                        None
                        if estimable
                        else multicondition_metrics._unpaired_support_reason(
                            edge_group,
                            n_reference_subjects=n_reference,
                            n_target_subjects=n_target,
                            min_subjects=min_subjects,
                        )
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _effect_rows(
    *,
    method: str,
    effects: list[float],
    statuses: list[str] | None = None,
    method_version: str = "1",
    resource: str = "common",
    resource_version: str = "2026-01",
) -> pd.DataFrame:
    statuses = statuses or ["exploratory"] * len(effects)
    return pd.DataFrame(
        [
            {
                "dataset": "cscc",
                "method": method,
                "method_version": method_version,
                "analysis_track": "lr_stlr",
                "resource": resource,
                "resource_version": resource_version,
                "resource_mode": "H-common",
                "score_semantics": "rank_strength",
                "contrast": "Tumor-vs-Normal",
                "sender": "S",
                "receiver": "R",
                "interaction_id": f"I{index}",
                "ligand": f"L{index}",
                "receptor": f"R{index}",
                "effect": effect,
                "status": status,
                "universe_id": "cscc-h-common-v1",
            }
            for index, (effect, status) in enumerate(
                zip(effects, statuses, strict=True)
            )
        ]
    )


def test_validation_preserves_raw_direction_and_uses_average_tied_ranks() -> None:
    table = _scores(score_direction="lower")
    sample = table["sample_id"].eq("p1_Normal")
    table.loc[sample, "score"] = [1.0, 1.0, 3.0, np.nan]
    table.loc[sample & table["ligand"].eq("L3"), "status"] = "not_predicted"

    validated = validate_score_table(table)
    observed = validated[validated["sample_id"].eq("p1_Normal")].set_index("ligand")

    assert observed.loc["L0", "score"] == 1.0
    assert observed.loc["L1", "score"] == 1.0
    assert observed.loc["L0", "comparison_rank"] == pytest.approx(1.5)
    assert observed.loc["L1", "comparison_rank"] == pytest.approx(1.5)
    assert observed.loc["L2", "comparison_rank"] == pytest.approx(3.0)
    assert observed.loc["L3", "comparison_rank"] == pytest.approx(4.0)
    assert observed.loc["L3", "comparison_strength"] == 0.0
    strength = pd.to_numeric(observed["comparison_strength"])
    assert strength.loc["L0"] > strength.loc["L2"]


def test_metrics_can_reuse_an_explicitly_validated_score_table() -> None:
    table = _scores()
    prepared = validate_score_table(table)

    expected = score_coverage_summary(table)
    reused = score_coverage_summary(prepared, validated=True)

    pd.testing.assert_frame_equal(reused, expected)
    with pytest.raises(ValueError, match="prepared columns"):
        score_coverage_summary(table, validated=True)


def test_validation_rejects_duplicates_invalid_scores_and_sparse_universe() -> None:
    table = _scores()
    with pytest.raises(ValueError, match="duplicate"):
        validate_score_table(pd.concat([table, table.iloc[[0]]], ignore_index=True))

    invalid = table.copy()
    invalid.loc[0, "score"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_score_table(invalid)

    invalid = table.copy()
    invalid.loc[0, "status"] = "missing"
    with pytest.raises(ValueError, match="non-observed"):
        validate_score_table(invalid)

    with pytest.raises(ValueError, match="same fixed edge universe"):
        validate_score_table(table.drop(index=0))

    shifted = table.copy()
    sample = shifted["sample_id"].eq("p1_Normal")
    shifted.loc[sample & shifted["ligand"].eq("L3"), "interaction_id"] = "I4"
    shifted.loc[sample & shifted["ligand"].eq("L3"), "ligand"] = "L4"
    shifted.loc[sample & shifted["receptor"].eq("R3"), "receptor"] = "R4"
    with pytest.raises(ValueError, match="universe_size"):
        validate_score_table(shifted)


@pytest.mark.parametrize("invalid_id", [None, "", " molecular-a ", 7])
def test_optional_molecular_lr_axis_requires_canonical_nonempty_strings(
    invalid_id: object,
) -> None:
    table = _scores()
    table[MOLECULAR_LR_EQUIVALENCE_COLUMN] = (
        table["interaction_id"]
        .map({f"I{index}": f"molecular-{index}" for index in range(4)})
        .astype(object)
    )
    table.loc[table.index[0], MOLECULAR_LR_EQUIVALENCE_COLUMN] = invalid_id

    with pytest.raises(ValueError, match="canonical non-empty strings"):
        validate_score_table(table)


def test_molecular_lr_source_mapping_is_unique_across_contrasts() -> None:
    table = _scores()
    table[MOLECULAR_LR_EQUIVALENCE_COLUMN] = table["interaction_id"].map(
        {f"I{index}": f"molecular-{index}" for index in range(4)}
    )
    second_contrast = table.copy()
    second_contrast["contrast"] = "Relapse-vs-Normal"
    second_contrast.loc[
        second_contrast["interaction_id"].eq("I0"),
        MOLECULAR_LR_EQUIVALENCE_COLUMN,
    ] = "conflicting-molecular-id"

    with pytest.raises(ValueError, match="exactly one molecular_lr_equivalence_id"):
        validate_score_table(pd.concat([table, second_contrast], ignore_index=True))

    validated = validate_score_table(table)
    assert validated[MOLECULAR_LR_EQUIVALENCE_COLUMN].notna().all()


def test_resource_unavailable_is_a_frozen_edge_state() -> None:
    table = _scores()
    edge = table["interaction_id"].eq("I0")
    table.loc[edge, "score"] = np.nan
    table.loc[edge, "status"] = "resource_unavailable"
    validated = validate_score_table(table)
    assert validated.loc[validated["interaction_id"].eq("I0"), "status"].eq(
        "resource_unavailable"
    ).all()

    invalid = table.copy()
    one_sample = invalid["sample_id"].eq("p1_Normal") & invalid[
        "interaction_id"
    ].eq("I0")
    invalid.loc[one_sample, "score"] = 1.0
    invalid.loc[one_sample, "status"] = "observed"
    with pytest.raises(ValueError, match="frozen method-resource edge state"):
        validate_score_table(invalid)


def test_interaction_id_preserves_distinct_native_rows_with_same_lr_text() -> None:
    table = _scores()
    duplicate_text = table["ligand"].isin(["L0", "L1"])
    table.loc[duplicate_text, "ligand"] = "L-shared"
    table.loc[duplicate_text, "receptor"] = "R-shared"

    validated = validate_score_table(table)
    sample = validated.loc[validated["sample_id"].eq("p1_Normal")]

    assert len(sample) == 4
    assert sample["interaction_id"].nunique() == 4

    invalid = table.copy()
    invalid.loc[invalid["interaction_id"].eq("I1"), "interaction_id"] = "I0"
    with pytest.raises(ValueError, match="duplicate"):
        validate_score_table(invalid)


def test_unavailable_and_missing_states_are_not_zero_scored() -> None:
    table = _scores()
    edge = table["ligand"].eq("L3")
    table.loc[edge, "score"] = np.nan
    table.loc[edge, "status"] = "resource_unavailable"
    one_cell_type = table["sample_id"].eq("p1_Tumor") & table["ligand"].eq("L2")
    table.loc[one_cell_type, "score"] = np.nan
    table.loc[one_cell_type, "status"] = "cell_type_missing"
    one_missing = table["sample_id"].eq("p2_Tumor") & table["ligand"].eq("L1")
    table.loc[one_missing, "score"] = np.nan
    table.loc[one_missing, "status"] = "missing"

    validated = validate_score_table(table)

    unavailable = validated["status"].eq("resource_unavailable")
    missing = validated["status"].isin(["missing", "cell_type_missing"])
    assert validated.loc[unavailable | missing, "comparison_rank"].isna().all()
    assert validated.loc[unavailable | missing, "comparison_strength"].isna().all()
    assert not validated.loc[unavailable | missing, "comparison_eligible"].any()
    assert list(validated.columns[-5:]) == [
        "comparison_eligible",
        "comparison_rank",
        "comparison_strength",
        "eligible_universe_size",
        "comparison_eligible_size",
    ]

    inconsistent = table.copy()
    inconsistent.loc[inconsistent.index[0], "score"] = np.nan
    inconsistent.loc[inconsistent.index[0], "status"] = "resource_unavailable"
    with pytest.raises(ValueError, match="cannot vary by sample"):
        validate_score_table(inconsistent)

    no_scores = table.copy()
    no_scores["score"] = np.nan
    no_scores["status"] = "resource_unavailable"
    stability = within_context_reproducibility(no_scores)
    assert set(stability["status"]) == {"not_estimable"}
    assert set(stability["reason_code"]) == {"no_pairwise_comparable_scores"}


def test_top_k_includes_all_edges_tied_at_the_cutoff() -> None:
    table = _scores()
    for sample_id in table["sample_id"].unique():
        sample = table["sample_id"].eq(sample_id)
        table.loc[sample, "score"] = [4.0, 3.0, 3.0, 1.0]
    # One sample swaps one member of the cutoff tie. Correct average-rank top-2
    # includes both tied edges in both samples, yielding Jaccard one.
    table.loc[table["sample_id"].eq("p1_Normal"), "score"] = [4.0, 3.0, 3.0, 1.0]

    stability = within_context_reproducibility(table, top_k=2)

    assert set(stability["median_top_k_jaccard"]) == {1.0}


def test_paired_effects_use_subject_pairs_and_full_identity_key() -> None:
    table = _scores()
    effects = paired_edge_effects(
        table,
        reference="Normal",
        target="Tumor",
        min_pairs=3,
        contrast="Tumor-vs-Normal",
    )

    assert len(effects) == 4
    assert set(effects["n_pairs"]) == {3}
    # A constant native-score shift cannot change a within-sample rank effect.
    assert set(effects["effect"]) == {0.0}
    assert effects["direction_consistency"].isna().all()
    assert set(effects["direction_comparable_pairs"]) == {0}
    assert set(effects["method_version"]) == {"1"}
    assert set(effects["resource_version"]) == {"2026-01"}
    assert set(effects["score_semantics"]) == {"strength_not_probability"}
    assert set(effects["contrast"]) == {"Tumor-vs-Normal"}


def test_vectorized_paired_effects_exactly_match_scalar_status_rich_reference() -> None:
    table = _rich_effect_scores(paired=True)
    expected = _paired_effects_scalar_reference(
        table,
        reference="Ctrl",
        target="Case",
        min_pairs=3,
    )
    observed = paired_edge_effects(
        table,
        reference="Ctrl",
        target="Case",
        min_pairs=3,
        validated=True,
    )

    pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    assert set(observed["method"]) == {"m1", "m2"}
    assert set(observed["reason_code"].dropna()) == {
        "missing_score",
        "cell_type_missing",
        "resource_unavailable",
        "method_or_resource_not_supported",
        "method_run_failed",
        "insufficient_paired_subjects",
    }
    subjects_by_method = table.groupby("method")["subject_id"].agg(set)
    assert subjects_by_method["m1"].isdisjoint(subjects_by_method["m2"])


def test_paired_effects_bound_subject_matrix_to_each_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _rich_effect_scores(paired=True)
    shapes: list[tuple[int, ...]] = []
    original = multicondition_metrics._rowwise_distribution

    def recording(
        values: np.ndarray, *, order: np.ndarray | None = None
    ) -> tuple[np.ndarray, ...]:
        shapes.append(values.shape)
        return original(values, order=order)

    monkeypatch.setattr(
        multicondition_metrics,
        "_rowwise_distribution",
        recording,
    )
    paired_edge_effects(
        table,
        reference="Ctrl",
        target="Case",
        min_pairs=3,
        validated=True,
    )

    assert shapes == [(15, 3), (15, 3)]


def test_paired_effects_normalize_numeric_subjects_and_contexts() -> None:
    table = _scores()
    table["subject_id"] = table["subject_id"].str.removeprefix("p").astype(int)
    table["context"] = table["context"].replace({"Normal": 0, "Tumor": 1})

    effects = paired_edge_effects(
        table,
        reference="0",
        target="1",
        min_pairs=3,
    )

    assert set(effects["n_pairs"]) == {3}
    assert set(effects["reference"]) == {"0"}
    assert set(effects["target"]) == {"1"}


def test_unpaired_effects_average_libraries_before_equal_subject_means() -> None:
    table = _unpaired_scores(n_reference=4, n_target=4)
    baseline = unpaired_edge_effects(
        table,
        reference="Ctrl",
        target="Case",
        min_subjects=3,
    ).sort_values(list(EDGE_KEYS), ignore_index=True)

    duplicate_library = table.loc[table["subject_id"].eq("t1")].copy()
    duplicate_library["sample_id"] = duplicate_library["sample_id"] + "_library2"
    augmented = pd.concat([table, duplicate_library], ignore_index=True)
    repeated = unpaired_edge_effects(
        augmented,
        reference="Ctrl",
        target="Case",
        min_subjects=3,
    ).sort_values(list(EDGE_KEYS), ignore_index=True)

    np.testing.assert_allclose(baseline["effect"], repeated["effect"])
    assert set(repeated["n_target_subjects"]) == {4}
    assert set(repeated["n_target_samples"]) == {5}
    assert set(repeated["status"]) == {"exploratory"}
    assert set(repeated["aggregation"]) == {
        "sample_mean_within_subject_context_then_equal_subject_mean"
    }


def test_vectorized_unpaired_effects_exactly_match_scalar_reference() -> None:
    table = _rich_effect_scores(paired=False)
    expected = _unpaired_effects_scalar_reference(
        table,
        reference="Ctrl",
        target="Case",
        min_subjects=3,
    )
    observed = unpaired_edge_effects(
        table,
        reference="Ctrl",
        target="Case",
        min_subjects=3,
        validated=True,
    )

    pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    assert set(observed["method"]) == {"m1", "m2"}
    assert set(observed["reason_code"].dropna()) == {
        "insufficient_subjects_both_contexts",
        "resource_unavailable",
        "method_or_resource_not_supported",
        "method_run_failed",
        "insufficient_target_subjects",
    }


def test_vectorized_effects_match_scalar_references_for_50_random_tables() -> None:
    for seed in range(25):
        paired_table = _rich_effect_scores(paired=True, random_seed=seed)
        paired_expected = _paired_effects_scalar_reference(
            paired_table,
            reference="Ctrl",
            target="Case",
            min_pairs=3,
        )
        paired_observed = paired_edge_effects(
            paired_table,
            reference="Ctrl",
            target="Case",
            min_pairs=3,
            validated=True,
        )
        pd.testing.assert_frame_equal(
            paired_observed,
            paired_expected,
            check_exact=True,
            obj=f"paired seed {seed}",
        )

        unpaired_table = _rich_effect_scores(paired=False, random_seed=seed)
        unpaired_expected = _unpaired_effects_scalar_reference(
            unpaired_table,
            reference="Ctrl",
            target="Case",
            min_subjects=3,
        )
        unpaired_observed = unpaired_edge_effects(
            unpaired_table,
            reference="Ctrl",
            target="Case",
            min_subjects=3,
            validated=True,
        )
        pd.testing.assert_frame_equal(
            unpaired_observed,
            unpaired_expected,
            check_exact=True,
            obj=f"unpaired seed {seed}",
        )


def test_unpaired_effects_reject_group_overlap_and_report_support() -> None:
    overlapping = _unpaired_scores(n_reference=3, n_target=3)
    overlapping.loc[overlapping["subject_id"].eq("t1"), "subject_id"] = "c1"
    with pytest.raises(ValueError, match="disjoint subject groups"):
        unpaired_edge_effects(
            overlapping,
            reference="Ctrl",
            target="Case",
            min_subjects=3,
        )

    insufficient = _unpaired_scores(n_reference=3, n_target=2)
    effects = unpaired_edge_effects(
        insufficient,
        reference="Ctrl",
        target="Case",
        min_subjects=3,
    )

    assert set(effects["status"]) == {"not_estimable"}
    assert set(effects["reason_code"]) == {"insufficient_target_subjects"}
    assert set(effects["n_reference_subjects"]) == {3}
    assert set(effects["n_target_subjects"]) == {2}


def test_unpaired_split_half_is_seeded_stratified_and_reports_valid_repeats() -> None:
    table = _unpaired_scores(n_reference=8, n_target=8)
    first = unpaired_differential_split_half_reproducibility(
        table,
        reference="Ctrl",
        target="Case",
        n_repeats=50,
        min_subjects_per_half=2,
        top_k=2,
        random_seed=37,
    )
    shuffled = table.sample(frac=1.0, random_state=11).reset_index(drop=True)
    repeated = unpaired_differential_split_half_reproducibility(
        shuffled,
        reference="Ctrl",
        target="Case",
        n_repeats=50,
        min_subjects_per_half=2,
        top_k=2,
        random_seed=37,
    )
    different_seed = unpaired_differential_split_half_reproducibility(
        table,
        reference="Ctrl",
        target="Case",
        n_repeats=50,
        min_subjects_per_half=2,
        top_k=2,
        random_seed=38,
    )

    pd.testing.assert_frame_equal(first, repeated)
    row = first.iloc[0]
    assert row["status"] == "observed"
    assert 2 <= row["n_repeats_valid"] <= 50
    assert row["ci_lower"] <= row["estimate"] <= row["ci_upper"]
    assert row["n_reference_subjects"] == 8
    assert row["n_target_subjects"] == 8
    assert row["reference_half_size_minimum"] == 4
    assert row["target_half_size_minimum"] == 4
    assert row["split_unit"] == "subject_stratified_within_context"
    assert (
        row["split_assignment_sha256"]
        != different_seed.iloc[0]["split_assignment_sha256"]
    )

    incomplete = table.copy()
    missing = incomplete["subject_id"].eq("t1") & incomplete["interaction_id"].eq("I7")
    incomplete.loc[missing, "score"] = np.nan
    incomplete.loc[missing, "status"] = "missing"
    fixed_complete_case = unpaired_differential_split_half_reproducibility(
        incomplete,
        reference="Ctrl",
        target="Case",
        n_repeats=10,
        min_subjects_per_half=2,
        random_seed=37,
    ).iloc[0]
    assert fixed_complete_case["frozen_eligible_edges"] == 8
    assert fixed_complete_case["complete_subject_edges"] == 7
    assert fixed_complete_case["complete_subject_edge_coverage"] == pytest.approx(7 / 8)

    insufficient = unpaired_differential_split_half_reproducibility(
        _unpaired_scores(n_reference=3, n_target=4),
        reference="Ctrl",
        target="Case",
        n_repeats=10,
        min_subjects_per_half=2,
    ).iloc[0]
    assert insufficient["status"] == "not_estimable"
    assert insufficient["reason_code"] == (
        "insufficient_subjects_for_stratified_halves"
    )


def test_unpaired_leave_one_subject_is_labelled_as_influence_only() -> None:
    table = _unpaired_scores(n_reference=5, n_target=5)
    # This edge is always last-ranked, so the full and leave-one-out effects are
    # both zero and it must not enter direction-agreement denominators.
    table.loc[table["interaction_id"].eq("I0"), "score"] = -1_000_000.0
    influence = unpaired_leave_one_subject_influence(
        table,
        reference="Ctrl",
        target="Case",
        min_remaining_subjects=3,
        top_k=2,
    )

    assert len(influence) == 10
    assert set(influence["diagnostic_name"]) == {"leave_one_subject_influence"}
    assert set(influence["interpretation"]) == {
        "influence_diagnostic_not_independent_replication"
    }
    assert set(influence["status"]) == {"descriptive"}
    assert set(influence["n_reference_subjects_remaining"]) == {4, 5}
    assert set(influence["n_target_subjects_remaining"]) == {4, 5}
    assert (
        influence["direction_comparable_edges"] < influence["complete_subject_edges"]
    ).all()


def test_loso_reproducibility_blocks_multiple_samples_by_subject() -> None:
    table = _scores()
    duplicate_region = table[table["subject_id"].eq("p1")].copy()
    duplicate_region["sample_id"] = duplicate_region["sample_id"] + "_region2"
    # Create subject-specific, nonconstant differential ranks.
    tumor = table["context"].eq("Tumor")
    table.loc[tumor, "score"] = table.loc[tumor, "score"] + table.loc[
        tumor, "ligand"
    ].str.removeprefix("L").astype(int) * table.loc[tumor, "subject_id"].map(
        {"p1": 1, "p2": -1, "p3": 2}
    )
    duplicate_tumor = duplicate_region["context"].eq("Tumor")
    duplicate_region.loc[duplicate_tumor, "score"] = duplicate_region.loc[
        duplicate_tumor, "score"
    ] + duplicate_region.loc[duplicate_tumor, "ligand"].str.removeprefix("L").astype(
        int
    )
    augmented = pd.concat([table, duplicate_region], ignore_index=True)

    loso = paired_differential_loso_reproducibility(
        augmented,
        reference="Normal",
        target="Tumor",
        min_subjects=3,
        top_k=2,
    )

    assert len(loso) == 3
    assert set(loso["held_out_subject"]) == {"p1", "p2", "p3"}
    assert set(loso["n_paired_subjects"]) == {3}
    assert set(loso["n_training_subjects"]) == {2}
    assert set(loso["minimum_training_edge_support"]) == {2}


def test_paired_loso_normalizes_numeric_subjects_and_contexts() -> None:
    table = _scores()
    table["subject_id"] = table["subject_id"].str.removeprefix("p").astype(int)
    table["context"] = table["context"].replace({"Normal": 0, "Tumor": 1})

    loso = paired_differential_loso_reproducibility(
        table,
        reference="0",
        target="1",
        min_subjects=3,
        top_k=2,
    )

    assert len(loso) == 3
    assert set(loso["held_out_subject"]) == {"1", "2", "3"}
    assert set(loso["n_paired_subjects"]) == {3}


def test_cross_method_filters_not_estimable_and_excludes_zero_zero_direction() -> None:
    first = _effect_rows(
        method="m1",
        effects=[0.0, 1.0, -1.0, 4.0, 99.0],
        statuses=["exploratory"] * 4 + ["not_estimable"],
    )
    second = _effect_rows(
        method="m2",
        effects=[0.0, 2.0, 3.0, 8.0, -99.0],
        statuses=["exploratory"] * 4 + ["not_estimable"],
        method_version="2",
    )

    comparison = cross_method_concordance(
        pd.concat([first, second], ignore_index=True), minimum_shared_edges=3
    )

    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["shared_edges"] == 4
    assert row["direction_comparable_edges"] == 3
    assert row["direction_agreement"] == pytest.approx(2 / 3)
    assert row["method_version_left"] == "1"
    assert row["method_version_right"] == "2"


def test_cross_method_does_not_merge_resource_or_semantic_variants() -> None:
    base = _effect_rows(method="m1", effects=[1.0, 2.0, 3.0])
    same_method_new_resource = _effect_rows(
        method="m1",
        effects=[3.0, 2.0, 1.0],
        resource="other",
        resource_version="2026-02",
    )
    competitor = _effect_rows(method="m2", effects=[1.0, 2.0, 3.0])

    comparison = cross_method_concordance(
        pd.concat([base, same_method_new_resource, competitor], ignore_index=True),
        minimum_shared_edges=3,
    )

    assert len(comparison) == 3
    assert {comparison.loc[index, "resource_left"] for index in comparison.index} <= {
        "common",
        "other",
    }
    assert set(comparison["contrast"]) == {"Tumor-vs-Normal"}

    incompatible = pd.concat([base, competitor.iloc[:-1]], ignore_index=True)
    with pytest.raises(ValueError, match="same edge universe"):
        cross_method_concordance(incompatible, minimum_shared_edges=3)


def test_external_long_table_conversion_is_explicit_about_sparse_rows() -> None:
    rows = _scores().iloc[[0, 1, 4]].copy()
    external = pd.DataFrame(
        {
            "method_id": rows["method"],
            "method_version": rows["method_version"],
            "analysis_track": rows["analysis_track"],
            "resource_id": rows["resource"],
            "resource_version": rows["resource_version"],
            "resource_mode": rows["resource_mode"],
            "dataset_id": rows["dataset"],
            "universe_id": rows["universe_id"],
            "universe_member": rows["universe_member"],
            "universe_size": rows["universe_size"],
            "sample_id": rows["sample_id"],
            "subject_id": rows["subject_id"],
            "context_json": [
                json.dumps({"condition": value}) for value in rows["context"]
            ],
            "sender": rows["sender"],
            "receiver": rows["receiver"],
            "interaction_id": rows["interaction_id"],
            "ligand": rows["ligand"],
            "receptor": rows["receptor"],
            "score": rows["score"],
            "score_name": "magnitude_rank",
            "score_direction": "lower",
            "status": "ok",
            MOLECULAR_LR_EQUIVALENCE_COLUMN: rows["interaction_id"].map(
                {f"I{index}": f"molecular-{index}" for index in range(4)}
            ),
        }
    )

    mapped = external_long_to_score_table(
        external,
        dataset="cscc",
        context_key="condition",
        contrast="Tumor-vs-Normal",
        validate=False,
    )

    assert tuple(mapped.loc[:, EDGE_KEYS].columns) == EDGE_KEYS
    assert set(mapped["method"]) == {"m1"}
    assert set(mapped["score_direction"]) == {"lower"}
    assert set(mapped["status"]) == {"observed"}
    assert set(mapped[MOLECULAR_LR_EQUIVALENCE_COLUMN]) == {
        "molecular-0",
        "molecular-1",
    }
    wrong_track = external.copy()
    wrong_track["analysis_track"] = "ligand_target_program"
    with pytest.raises(ValueError, match="Track B"):
        external_long_to_score_table(
            wrong_track,
            dataset="cscc",
            context_key="condition",
            contrast="Tumor-vs-Normal",
            validate=False,
        )
    with pytest.raises(ValueError, match="materialized frozen edge universe"):
        external_long_to_score_table(
            external,
            dataset="cscc",
            context_key="condition",
            contrast="Tumor-vs-Normal",
        )


def test_external_sparse_statuses_have_explicit_metric_semantics() -> None:
    row = _scores().iloc[[0]].copy()
    external = pd.DataFrame(
        {
            "dataset_id": row["dataset"],
            "method_id": row["method"],
            "method_version": row["method_version"],
            "analysis_track": row["analysis_track"],
            "resource_id": row["resource"],
            "resource_version": row["resource_version"],
            "resource_mode": row["resource_mode"],
            "universe_id": row["universe_id"],
            "universe_member": row["universe_member"],
            "universe_size": row["universe_size"],
            "sample_id": row["sample_id"],
            "subject_id": row["subject_id"],
            "context_json": [json.dumps({"condition": "Normal"})],
            "sender": row["sender"],
            "receiver": row["receiver"],
            "interaction_id": row["interaction_id"],
            "ligand": row["ligand"],
            "receptor": row["receptor"],
            "score": np.nan,
            "score_name": "magnitude_rank",
            "score_direction": "lower",
            "status": "not_returned",
        }
    )
    status_cases = (
        ("not_returned", "not_predicted"),
        ("unsupported_resource", "not_supported"),
        ("insufficient_cells", "cell_type_missing"),
        ("method_failed", "failed"),
    )
    for adapter_status, metric_status in status_cases:
        external["status"] = adapter_status
        mapped = external_long_to_score_table(
            external,
            context_key="condition",
            contrast="Tumor-vs-Normal",
            validate=False,
        )

        assert mapped.loc[0, "status"] == metric_status
        assert pd.isna(mapped.loc[0, "score"])


def test_universe_contract_and_coverage_make_missingness_visible() -> None:
    table = _scores()
    unavailable = table["ligand"].eq("L3")
    table.loc[unavailable, "score"] = np.nan
    table.loc[unavailable, "status"] = "resource_unavailable"
    missing = table["sample_id"].eq("p1_Tumor") & table["ligand"].eq("L2")
    table.loc[missing, "score"] = np.nan
    table.loc[missing, "status"] = "not_estimable"

    coverage = score_coverage_summary(table).iloc[0]

    assert coverage["frozen_universe_edges"] == 4
    assert coverage["resource_covered_edges"] == 3
    assert coverage["resource_coverage_fraction"] == pytest.approx(0.75)
    assert coverage["comparison_coverage_fraction"] == pytest.approx(17 / 18)
    assert coverage["not_estimable_rows"] == 1

    invalid = table.copy()
    invalid["universe_size"] = 5
    with pytest.raises(ValueError, match="does not match"):
        validate_score_table(invalid)


def test_loso_is_family_macro_and_requires_every_training_subject() -> None:
    family_table = _multi_family_scores()
    loso = paired_differential_loso_reproducibility(
        family_table,
        reference="Normal",
        target="Tumor",
        min_subjects=3,
        top_k=2,
    )
    held_p1 = loso.set_index("held_out_subject").loc["p1"]

    assert held_p1["n_eligible_families"] == 2
    assert held_p1["n_estimable_families"] == 2
    # One family is perfectly concordant and the larger family is perfectly
    # discordant; equal family weighting gives zero rather than a size-weighted loss.
    assert held_p1["effect_spearman"] == pytest.approx(0.0, abs=1e-12)

    incomplete = _scores()
    missing = incomplete["sample_id"].eq("p2_Tumor") & incomplete["ligand"].eq("L3")
    incomplete.loc[missing, "score"] = np.nan
    incomplete.loc[missing, "status"] = "missing"
    strict_loso = paired_differential_loso_reproducibility(
        incomplete,
        reference="Normal",
        target="Tumor",
        min_subjects=3,
    ).set_index("held_out_subject")

    assert strict_loso.loc["p1", "training_complete_edges"] == 3
    assert strict_loso.loc["p1", "shared_edge_coverage"] == pytest.approx(0.75)
    assert strict_loso.loc["p1", "minimum_training_edge_support"] == 2


def test_loso_primary_endpoint_bootstraps_subjects_then_datasets() -> None:
    rows: list[dict[str, object]] = []
    for dataset, values in (("d1", (0.2, 0.4, 0.6)), ("d2", (0.6, 0.8, 1.0))):
        for index, value in enumerate(values):
            rows.append(
                {
                    "dataset": dataset,
                    "method": "m1",
                    "method_version": "1",
                    "analysis_track": "lr_stlr",
                    "resource": "common",
                    "resource_version": "2026-01",
                    "resource_mode": "H-common",
                    "score_semantics": "rank_strength",
                    "universe_id": f"{dataset}-common-v1",
                    "contrast": f"target-vs-reference-{dataset}",
                    "held_out_subject": f"{dataset}-p{index}",
                    "effect_spearman": value,
                    "status": "observed",
                }
            )
    loso = pd.DataFrame(rows)

    dataset_summary = summarize_loso_primary_endpoint(
        loso, n_bootstrap=200, random_seed=7
    )
    overall = aggregate_loso_primary_endpoint(
        loso, n_bootstrap=200, random_seed=7
    ).iloc[0]

    assert sorted(dataset_summary["estimate"]) == pytest.approx([0.4, 0.8])
    assert set(dataset_summary["bootstrap_unit"]) == {"held_out_subject"}
    assert overall["estimate"] == pytest.approx(0.6)
    assert overall["n_datasets"] == 2
    assert overall["bootstrap_unit"] == "dataset_then_subject"


def test_truth_metrics_are_synthetic_only_and_tie_aware() -> None:
    table = _scores()
    truth_rows = table.loc[
        table["sample_id"].eq("p1_Normal"),
        ["dataset", "contrast", "universe_id", *EDGE_KEYS],
    ].copy()
    truth_rows["is_positive"] = truth_rows["ligand"].eq("L3").astype(int)
    truth_rows["truth_scope"] = "real_data"

    with pytest.raises(ValueError, match="forbidden"):
        synthetic_edge_truth_metrics(table, truth_rows, top_k=1)

    invalid_truth = truth_rows.copy()
    invalid_truth["truth_scope"] = "simulation"
    invalid_truth["is_positive"] = invalid_truth["is_positive"].astype(float)
    invalid_truth.loc[invalid_truth.index[0], "is_positive"] = 0.5
    with pytest.raises(ValueError, match="binary"):
        synthetic_edge_truth_metrics(table, invalid_truth, top_k=1)

    truth_rows["truth_scope"] = "simulation"
    metrics = synthetic_edge_truth_metrics(table, truth_rows, top_k=1)

    assert len(metrics) == table["sample_id"].nunique()
    assert set(metrics["auroc"]) == {1.0}
    assert set(metrics["average_precision"]) == {1.0}
    assert set(metrics["top_k_realized"]) == {1}
    assert set(metrics["top_k_precision"]) == {1.0}


def test_performance_summary_keeps_failures_and_determinism_scope() -> None:
    runs = pd.DataFrame(
        {
            "dataset": ["cscc"] * 3,
            "method": ["m1"] * 3,
            "method_version": ["1"] * 3,
            "analysis_track": ["lr_stlr"] * 3,
            "resource": ["common"] * 3,
            "resource_version": ["2026-01"] * 3,
            "resource_mode": ["H-common"] * 3,
            "run_id": ["r1", "r2", "r3"],
            "status": ["complete", "complete", "failed"],
            "wall_time_seconds": [10.0, 14.0, np.nan],
            "peak_rss_mb": [100.0, 120.0, np.nan],
            "output_bytes": [1000, 1000, np.nan],
            "threads": [2, 2, 2],
            "determinism_key": ["same", "same", "same"],
            "output_sha256": ["abc", "abc", np.nan],
        }
    )

    summary = summarize_run_performance(runs).iloc[0]

    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["median_wall_time_seconds"] == 12.0
    assert summary["median_peak_rss_mb"] == 110.0
    assert bool(summary["deterministic_output"])
    assert summary["repeated_determinism_groups"] == 1

    invalid = runs.copy()
    invalid["threads"] = 0
    with pytest.raises(ValueError, match="threads"):
        summarize_run_performance(invalid)
