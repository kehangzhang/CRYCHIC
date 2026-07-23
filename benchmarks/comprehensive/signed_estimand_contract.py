"""Run deterministic signed-estimand and invariance release gates.

This benchmark deliberately exercises the public design and OOF inference APIs.
It does not implement a second contrast engine and never manufactures analytic
P values from the diagnostic CR2 covariance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from crychic.design import (
    ContextGraph,
    ContrastSpec,
    balanced_contrast,
    factorial_interaction_contrast,
    global_one_vs_rest,
    local_neighbor_contrast,
)
from crychic.inference import (
    OOFContextEffectResult,
    OOFContextOmnibusSpec,
    OOFEffectSpec,
    fit_oof_context_effect,
    fit_oof_context_omnibus,
)

SCHEMA_VERSION = "crychic-signed-estimand-contract-v1"
ABSOLUTE_TOLERANCE = 1.0e-10


def deterministic_score_table(
    means: Mapping[str, float],
    *,
    scale: float = 1.0,
    sender: str = "Sender",
    receiver: str = "Receiver",
) -> pd.DataFrame:
    """Build a paired OOF table with exact context means and nondegenerate CR2."""

    if len(means) < 2:
        raise ValueError("deterministic fixture requires at least two contexts")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    contexts = tuple(means)
    if len(set(contexts)) != len(contexts):
        raise ValueError("fixture contexts must be unique")
    if any(not isinstance(context, str) or not context for context in contexts):
        raise ValueError("fixture contexts must be non-empty strings")
    numeric_means = {context: float(value) for context, value in means.items()}
    if not all(math.isfinite(value) for value in numeric_means.values()):
        raise ValueError("fixture means must be finite")

    # Both sequences sum to zero inside each fold and are linearly independent.
    first = np.asarray((-7, -5, -3, -1, 1, 3, 5, 7), dtype=float) * 0.01
    second = np.asarray((-3, 5, -7, 1, 7, -1, 3, -5), dtype=float) * 0.01
    profiles = (first, second, -first - second, first - second)
    rows: list[dict[str, object]] = []
    for fold_index in range(2):
        fold_id = f"fold-{fold_index}"
        for within_fold in range(8):
            subject_index = fold_index * 8 + within_fold
            subject_id = f"subject-{subject_index:02d}"
            for context_index, context in enumerate(contexts):
                noise = profiles[context_index % len(profiles)][within_fold]
                rows.append(
                    {
                        "subject_id": subject_id,
                        "sample_id": f"{subject_id}:{context}",
                        "fold_id": fold_id,
                        "context_id": context,
                        "score": scale * (numeric_means[context] + noise),
                        "score_status": "observed",
                        "scoring_function_id": f"frozen-functional:{fold_id}",
                        "sender": sender,
                        "receiver": receiver,
                    }
                )
    return pd.DataFrame.from_records(rows)


def _effect_spec(
    contrast: ContrastSpec,
    contexts: Sequence[str],
    *,
    hypothesis_id: str,
) -> OOFEffectSpec:
    """Adapt a design contrast while retaining zero-weight context support."""

    unknown = set(contrast.weights).difference(contexts)
    if unknown:
        raise ValueError(f"contrast contains unknown contexts: {sorted(unknown)}")
    return OOFEffectSpec(
        hypothesis_id=hypothesis_id,
        contrast_name=contrast.name,
        contrast_weights=tuple(
            (context, float(contrast.weights.get(context, 0.0)))
            for context in contexts
        ),
    )


def _fit(
    table: pd.DataFrame,
    contrast: ContrastSpec,
    contexts: Sequence[str],
    *,
    hypothesis_id: str,
) -> OOFContextEffectResult:
    return fit_oof_context_effect(
        table,
        _effect_spec(contrast, contexts, hypothesis_id=hypothesis_id),
    )


def _evidence(result: OOFContextEffectResult) -> float | None:
    if (
        result.effect is None
        or result.standard_error is None
        or result.standard_error <= 0.0
    ):
        return None
    return abs(result.effect / result.standard_error)


def _close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1.0e-10, abs_tol=ABSOLUTE_TOLERANCE)


def _case_record(
    *,
    case_id: str,
    estimand: str,
    expected_effect: float | None,
    result: OOFContextEffectResult,
    notes: str,
) -> dict[str, object]:
    passed = result.effect_status == "observed" and _close(
        result.effect, expected_effect
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "estimand": estimand,
        "expected_effect": expected_effect,
        "observed_effect": result.effect,
        "standard_error": result.standard_error,
        "absolute_standardized_evidence": _evidence(result),
        "effect_status": result.effect_status,
        "reason_code": result.reason_code,
        "formal_p_value_status": "NE_requires_full_pipeline_resampling",
        "passed": passed,
        "notes": notes,
    }


def _invariance_record(
    *,
    invariant_id: str,
    expected_relation: str,
    passed: bool,
    maximum_absolute_error: float | None = None,
    observed_status: str = "observed",
    notes: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "invariant_id": invariant_id,
        "expected_relation": expected_relation,
        "observed_status": observed_status,
        "maximum_absolute_error": maximum_absolute_error,
        "passed": bool(passed),
        "notes": notes,
    }


def evaluate_contract() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all deterministic cases and end-to-end invariance checks."""

    cases: list[dict[str, object]] = []
    invariants: list[dict[str, object]] = []

    two_contexts = ("A", "B")
    two_table = deterministic_score_table({"A": 2.0, "B": 1.0})
    forward_contrast = balanced_contrast(("A",), ("B",), name="A-minus-B")
    reverse_contrast = balanced_contrast(("B",), ("A",), name="B-minus-A")
    forward = _fit(
        two_table,
        forward_contrast,
        two_contexts,
        hypothesis_id="two-group-forward",
    )
    reverse = _fit(
        two_table,
        reverse_contrast,
        two_contexts,
        hypothesis_id="two-group-reverse",
    )
    cases.extend(
        (
            _case_record(
                case_id="two_group_A_minus_B",
                estimand="pairwise",
                expected_effect=1.0,
                result=forward,
                notes="mu_A=2, mu_B=1",
            ),
            _case_record(
                case_id="two_group_B_minus_A",
                estimand="pairwise_reverse",
                expected_effect=-1.0,
                result=reverse,
                notes="exact reverse of A-minus-B",
            ),
        )
    )
    reverse_errors = (
        abs(float(forward.effect) + float(reverse.effect)),
        abs(float(forward.standard_error) - float(reverse.standard_error)),
        abs(float(_evidence(forward)) - float(_evidence(reverse))),
    )
    invariants.append(
        _invariance_record(
            invariant_id="reverse_contrast",
            expected_relation="effect_reverse=-effect_forward; SE and |z| unchanged",
            maximum_absolute_error=max(reverse_errors),
            passed=max(reverse_errors) <= ABSOLUTE_TOLERANCE,
            notes="Formal P values remain NE on both sides of the contrast.",
        )
    )

    shuffled = two_table.sample(frac=1.0, random_state=20260723).reset_index(
        drop=True
    )
    shuffled_result = _fit(
        shuffled,
        forward_contrast,
        two_contexts,
        hypothesis_id="two-group-forward",
    )
    row_errors = (
        abs(float(forward.effect) - float(shuffled_result.effect)),
        abs(float(forward.standard_error) - float(shuffled_result.standard_error)),
    )
    invariants.append(
        _invariance_record(
            invariant_id="context_row_reordering",
            expected_relation=(
                "effect, SE, source digest, and result identity unchanged"
            ),
            maximum_absolute_error=max(row_errors),
            passed=(
                max(row_errors) <= ABSOLUTE_TOLERANCE
                and forward.source_table_digest == shuffled_result.source_table_digest
                and forward.result_id == shuffled_result.result_id
            ),
            notes="Rows are canonicalized before source hashing and fitting.",
        )
    )

    renamed = two_table.copy()
    renamed["context_id"] = renamed["context_id"].replace(
        {"A": "Alpha", "B": "Beta"}
    )
    renamed["sample_id"] = renamed["sample_id"].str.replace(
        r":A$", ":Alpha", regex=True
    ).str.replace(r":B$", ":Beta", regex=True)
    renamed_result = _fit(
        renamed,
        balanced_contrast(("Alpha",), ("Beta",), name="Alpha-minus-Beta"),
        ("Alpha", "Beta"),
        hypothesis_id="two-group-relabeled",
    )
    relabel_errors = (
        abs(float(forward.effect) - float(renamed_result.effect)),
        abs(float(forward.standard_error) - float(renamed_result.standard_error)),
    )
    invariants.append(
        _invariance_record(
            invariant_id="context_relabeling",
            expected_relation="effect and SE unchanged after bijective relabeling",
            maximum_absolute_error=max(relabel_errors),
            passed=max(relabel_errors) <= ABSOLUTE_TOLERANCE,
            notes="Identity changes because the declared estimand names change.",
        )
    )

    scaled_table = two_table.copy()
    scaled_table["score"] *= 7.5
    scaled_result = _fit(
        scaled_table,
        forward_contrast,
        two_contexts,
        hypothesis_id="two-group-forward",
    )
    scale_errors = (
        abs(float(scaled_result.effect) - 7.5 * float(forward.effect)),
        abs(
            float(scaled_result.standard_error)
            - 7.5 * float(forward.standard_error)
        ),
        abs(float(_evidence(scaled_result)) - float(_evidence(forward))),
    )
    invariants.append(
        _invariance_record(
            invariant_id="positive_score_scaling",
            expected_relation="effect and SE scale; |z| and ranks remain unchanged",
            maximum_absolute_error=max(scale_errors),
            passed=max(scale_errors) <= ABSOLUTE_TOLERANCE,
            notes="Rank invariance is additionally checked across three contrasts.",
        )
    )

    relabeled_parties = two_table.copy()
    relabeled_parties["sender"] = "z_sender"
    relabeled_parties["receiver"] = "a_receiver"
    relabeled_parties = relabeled_parties.iloc[::-1].reset_index(drop=True)
    party_result = _fit(
        relabeled_parties,
        forward_contrast,
        two_contexts,
        hypothesis_id="two-group-forward",
    )
    party_errors = (
        abs(float(forward.effect) - float(party_result.effect)),
        abs(float(forward.standard_error) - float(party_result.standard_error)),
    )
    invariants.append(
        _invariance_record(
            invariant_id="sender_receiver_label_reordering",
            expected_relation="condition effect and SE unchanged",
            maximum_absolute_error=max(party_errors),
            passed=(
                max(party_errors) <= ABSOLUTE_TOLERANCE
                and forward.result_id == party_result.result_id
            ),
            notes="Sender/receiver labels cannot alter the condition contrast.",
        )
    )

    three_contexts = ("A", "B", "C")
    three_table = deterministic_score_table({"A": 3.0, "B": 2.0, "C": 1.0})
    three_specs = (
        (
            "three_group_A_minus_B",
            balanced_contrast(("A",), ("B",), name="A-minus-B"),
            1.0,
            "pairwise",
        ),
        (
            "three_group_A_minus_C",
            balanced_contrast(("A",), ("C",), name="A-minus-C"),
            2.0,
            "pairwise",
        ),
        (
            "three_group_B_minus_C",
            balanced_contrast(("B",), ("C",), name="B-minus-C"),
            1.0,
            "pairwise",
        ),
        (
            "three_group_A_one_vs_rest",
            global_one_vs_rest(three_contexts, "A", name="A-one-vs-rest"),
            1.5,
            "balanced_one_vs_rest",
        ),
    )
    three_results: dict[str, OOFContextEffectResult] = {}
    for case_id, contrast, expected, estimand in three_specs:
        result = _fit(
            three_table,
            contrast,
            three_contexts,
            hypothesis_id=case_id,
        )
        three_results[case_id] = result
        cases.append(
            _case_record(
                case_id=case_id,
                estimand=estimand,
                expected_effect=expected,
                result=result,
                notes="Pairwise and one-vs-rest labels are intentionally distinct.",
            )
        )

    original_ranks = pd.Series(
        [
            abs(float(three_results[case_id].effect))
            for case_id, *_ in three_specs[:3]
        ]
    ).rank(method="average")
    scaled_three = three_table.copy()
    scaled_three["score"] *= 7.5
    scaled_effects = []
    for case_id, contrast, _, _ in three_specs[:3]:
        result = _fit(
            scaled_three,
            contrast,
            three_contexts,
            hypothesis_id=case_id,
        )
        scaled_effects.append(abs(float(result.effect)))
    scaled_ranks = pd.Series(scaled_effects).rank(method="average")
    invariants.append(
        _invariance_record(
            invariant_id="positive_scaling_rank",
            expected_relation="absolute-effect average ranks unchanged",
            maximum_absolute_error=float(
                np.max(np.abs(original_ranks.to_numpy() - scaled_ranks.to_numpy()))
            ),
            passed=original_ranks.equals(scaled_ranks),
            notes="Ties are retained with average ranks.",
        )
    )

    omnibus_spec = OOFContextOmnibusSpec(
        hypothesis_id="three-group-omnibus",
        omnibus_name="A-B-C-equality",
        context_ids=three_contexts,
    )
    omnibus = fit_oof_context_omnibus(three_table, omnibus_spec)
    omnibus_passed = bool(
        omnibus.status == "observed"
        and omnibus.wald_statistic is not None
        and omnibus.wald_statistic > 0.0
    )
    cases.append(
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": "three_group_omnibus_active",
            "estimand": "omnibus_equality",
            "expected_effect": None,
            "observed_effect": omnibus.wald_statistic,
            "standard_error": None,
            "absolute_standardized_evidence": omnibus.wald_statistic,
            "effect_status": omnibus.status,
            "reason_code": omnibus.reason_code,
            "formal_p_value_status": "NE_requires_full_pipeline_resampling",
            "passed": omnibus_passed,
            "notes": "Omnibus detection is separate from post-hoc direction.",
        }
    )
    scaled_omnibus = fit_oof_context_omnibus(scaled_three, omnibus_spec)
    omnibus_error = (
        None
        if omnibus.wald_statistic is None or scaled_omnibus.wald_statistic is None
        else abs(omnibus.wald_statistic - scaled_omnibus.wald_statistic)
    )
    invariants.append(
        _invariance_record(
            invariant_id="positive_scaling_omnibus",
            expected_relation="Wald omnibus statistic unchanged",
            maximum_absolute_error=omnibus_error,
            passed=bool(
                omnibus.status == scaled_omnibus.status == "observed"
                and omnibus_error is not None
                and omnibus_error <= ABSOLUTE_TOLERANCE
            ),
            notes="No analytic omnibus P value is released.",
        )
    )

    factorial_graph = ContextGraph.product(
        {
            "exposure": ContextGraph.chain(("NE", "E")),
            "time": ContextGraph.chain(("Pre", "On")),
        }
    )
    node_labels = {
        node: f"{dict(node)['time']}|{dict(node)['exposure']}"
        for node in factorial_graph.nodes
    }
    did_contrast = factorial_interaction_contrast(
        factorial_graph,
        "time",
        "On",
        "Pre",
        "exposure",
        "E",
        "NE",
        name="time-by-exposure-DID",
    )
    reversed_factor_contrast = factorial_interaction_contrast(
        factorial_graph,
        "exposure",
        "E",
        "NE",
        "time",
        "On",
        "Pre",
        name="exposure-by-time-DID",
    )
    did_weights = {
        node_labels[node]: weight for node, weight in did_contrast.weights.items()
    }
    reverse_factor_weights = {
        node_labels[node]: weight
        for node, weight in reversed_factor_contrast.weights.items()
    }
    did_contexts = tuple(sorted(node_labels.values()))
    did_design = ContrastSpec(
        name=did_contrast.name,
        weights=did_weights,
        family=did_contrast.family,
        mode=did_contrast.mode,
    )
    reverse_factor_design = ContrastSpec(
        name=reversed_factor_contrast.name,
        weights=reverse_factor_weights,
        family=reversed_factor_contrast.family,
        mode=reversed_factor_contrast.mode,
    )
    positive_did_table = deterministic_score_table(
        {"On|E": 3.0, "Pre|E": 1.0, "On|NE": 1.0, "Pre|NE": 1.0}
    )
    negative_did_table = deterministic_score_table(
        {"On|E": 0.0, "Pre|E": 2.0, "On|NE": 1.0, "Pre|NE": 1.0}
    )
    positive_did = _fit(
        positive_did_table,
        did_design,
        did_contexts,
        hypothesis_id="positive-DID",
    )
    negative_did = _fit(
        negative_did_table,
        did_design,
        did_contexts,
        hypothesis_id="negative-DID",
    )
    cases.extend(
        (
            _case_record(
                case_id="factorial_DID_positive",
                estimand="difference_in_differences",
                expected_effect=2.0,
                result=positive_did,
                notes="(OnE-PreE)-(OnNE-PreNE)",
            ),
            _case_record(
                case_id="factorial_DID_negative",
                estimand="difference_in_differences",
                expected_effect=-2.0,
                result=negative_did,
                notes="Negative planted interaction",
            ),
        )
    )
    factor_reordered = _fit(
        positive_did_table,
        reverse_factor_design,
        did_contexts,
        hypothesis_id="positive-DID-factor-reordered",
    )
    factor_error = abs(float(positive_did.effect) - float(factor_reordered.effect))
    invariants.append(
        _invariance_record(
            invariant_id="factor_argument_order",
            expected_relation="DID weights and effect unchanged",
            maximum_absolute_error=factor_error,
            passed=bool(
                did_weights == reverse_factor_weights
                and factor_error <= ABSOLUTE_TOLERANCE
            ),
            notes="Positive and negative levels remain explicitly declared.",
        )
    )

    chain = ContextGraph.chain(("C1", "C2", "C3", "C4"))
    chain_contexts = ("C1", "C2", "C3", "C4")
    chain_table = deterministic_score_table(
        {"C1": 1.0, "C2": 1.0, "C3": 3.0, "C4": 3.0}
    )
    edge_results: dict[str, OOFContextEffectResult] = {}
    for left, right, expected in (
        ("C2", "C1", 0.0),
        ("C3", "C2", 2.0),
        ("C4", "C3", 0.0),
    ):
        if right not in {neighbor for neighbor, _ in chain.neighbors(left)}:
            raise RuntimeError("declared change-point contrast is not a graph edge")
        case_id = f"chain_edge_{left}_minus_{right}"
        result = _fit(
            chain_table,
            balanced_contrast((left,), (right,), name=case_id),
            chain_contexts,
            hypothesis_id=case_id,
        )
        edge_results[case_id] = result
        cases.append(
            _case_record(
                case_id=case_id,
                estimand="local_graph_edge",
                expected_effect=expected,
                result=result,
                notes="Adjacent graph-edge contrast",
            )
        )
    for focal, expected in (("C2", -1.0), ("C3", 1.0)):
        case_id = f"chain_local_{focal}"
        result = _fit(
            chain_table,
            local_neighbor_contrast(chain, focal, name=case_id),
            chain_contexts,
            hypothesis_id=case_id,
        )
        cases.append(
            _case_record(
                case_id=case_id,
                estimand="local_neighbor",
                expected_effect=expected,
                result=result,
                notes="Focal context versus its graph-neighbor mean",
            )
        )
    global_c3 = _fit(
        chain_table,
        global_one_vs_rest(chain, "C3", name="chain-global-C3"),
        chain_contexts,
        hypothesis_id="chain-global-C3",
    )
    cases.append(
        _case_record(
            case_id="chain_global_C3",
            estimand="global_one_vs_rest",
            expected_effect=4.0 / 3.0,
            result=global_c3,
            notes="Global and local graph estimands remain separately labelled.",
        )
    )
    strongest_edge = max(
        edge_results,
        key=lambda case_id: abs(float(edge_results[case_id].effect)),
    )
    invariants.append(
        _invariance_record(
            invariant_id="chain_change_point_localization",
            expected_relation="unique largest adjacent effect is C3-minus-C2",
            maximum_absolute_error=0.0,
            passed=strongest_edge == "chain_edge_C3_minus_C2",
            notes=f"Observed strongest edge: {strongest_edge}",
        )
    )

    missing_score = two_table.copy()
    missing_score.loc[0, "score_status"] = "missing"
    missing_result = _fit(
        missing_score,
        forward_contrast,
        two_contexts,
        hypothesis_id="missing-score",
    )
    invariants.append(
        _invariance_record(
            invariant_id="missing_score_not_zero",
            expected_relation="not_estimable with incomplete coverage reason",
            passed=bool(
                missing_result.effect is None
                and missing_result.effect_status == "not_estimable"
                and missing_result.reason_code == "incomplete_oof_score_coverage"
            ),
            observed_status=missing_result.effect_status,
            notes=str(missing_result.reason_code),
        )
    )
    missing_context = two_table.loc[two_table["context_id"].ne("B")].copy()
    missing_context_result = _fit(
        missing_context,
        forward_contrast,
        two_contexts,
        hypothesis_id="missing-context-cell-type",
    )
    invariants.append(
        _invariance_record(
            invariant_id="missing_context_cell_type_not_zero",
            expected_relation="not_estimable rather than zero effect",
            passed=bool(
                missing_context_result.effect is None
                and missing_context_result.effect_status == "not_estimable"
                and missing_context_result.reason_code
                == "oof_effect_context_universe_mismatch"
            ),
            observed_status=missing_context_result.effect_status,
            notes=str(missing_context_result.reason_code),
        )
    )

    case_table = pd.DataFrame.from_records(cases).sort_values(
        "case_id", kind="stable"
    )
    invariant_table = pd.DataFrame.from_records(invariants).sort_values(
        "invariant_id", kind="stable"
    )
    return case_table.reset_index(drop=True), invariant_table.reset_index(drop=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _report(case_table: pd.DataFrame, invariant_table: pd.DataFrame) -> str:
    case_passed = int(case_table["passed"].sum())
    invariant_passed = int(invariant_table["passed"].sum())
    gate = bool(case_table["passed"].all() and invariant_table["passed"].all())
    lines = [
        "# Signed estimand contract benchmark",
        "",
        f"Gate A: **{'PASS' if gate else 'FAIL'}**",
        "",
        f"- Hand-computable cases: {case_passed}/{len(case_table)} passed",
        f"- End-to-end invariants: {invariant_passed}/{len(invariant_table)} passed",
        "- Formal P values: NE (full-pipeline resampling is required)",
        "",
        "Pairwise, one-vs-rest, omnibus, DID, local-edge, local-neighbor, and "
        "global graph estimands are reported as distinct quantities.",
        "",
        "## Cases",
        "",
        "| case | estimand | expected | observed | status | pass |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in case_table.itertuples(index=False):
        expected = (
            "NE" if pd.isna(row.expected_effect) else f"{row.expected_effect:.8g}"
        )
        observed = (
            "NE" if pd.isna(row.observed_effect) else f"{row.observed_effect:.8g}"
        )
        lines.append(
            f"| {row.case_id} | {row.estimand} | {expected} | {observed} | "
            f"{row.effect_status} | {'yes' if row.passed else 'no'} |"
        )
    lines.extend(
        (
            "",
            "## Invariants",
            "",
            "| invariant | expected relation | status | pass |",
            "|---|---|---|---|",
        )
    )
    for row in invariant_table.itertuples(index=False):
        lines.append(
            f"| {row.invariant_id} | {row.expected_relation} | "
            f"{row.observed_status} | {'yes' if row.passed else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def run_contract(output_dir: Path, *, repo_root: Path) -> dict[str, Any]:
    """Run Gate A and atomically publish checksum-bound evidence."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        case_table, invariant_table = evaluate_contract()
        case_path = temporary / "case_results.tsv"
        invariant_path = temporary / "invariance_results.tsv"
        report_path = temporary / "REPORT.md"
        case_table.to_csv(case_path, sep="\t", index=False)
        invariant_table.to_csv(invariant_path, sep="\t", index=False)
        report_path.write_text(
            _report(case_table, invariant_table), encoding="utf-8"
        )
        passed = bool(
            case_table["passed"].all() and invariant_table["passed"].all()
        )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete" if passed else "failed",
            "gate_a": "PASS" if passed else "FAIL",
            "contract": {
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "formal_p_value_policy": (
                    "not_released_without_full_pipeline_resampling"
                ),
                "n_cases": len(case_table),
                "n_cases_passed": int(case_table["passed"].sum()),
                "n_invariants": len(invariant_table),
                "n_invariants_passed": int(invariant_table["passed"].sum()),
            },
            "code": git_metadata(repo_root),
            "outputs": {
                "case_results": {
                    "path": case_path.name,
                    "sha256": sha256_file(case_path),
                },
                "invariance_results": {
                    "path": invariant_path.name,
                    "sha256": sha256_file(invariant_path),
                },
                "report": {
                    "path": report_path.name,
                    "sha256": sha256_file(report_path),
                },
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_contract(args.output, repo_root=args.repo_root.resolve())
    print(json.dumps(json_safe(manifest), sort_keys=True, allow_nan=False))
    return 0 if manifest["gate_a"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
