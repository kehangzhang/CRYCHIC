from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from crychic.core import ContractError, SeedLineage
from crychic.resampling import (
    ContextPermutationOperation,
    ExchangeabilityDesign,
    apply_context_permutation,
    build_exchangeability_map,
    plan_context_permutations,
    plan_subject_bootstraps,
)


def _independent_metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    assignments = (
        ("u1", "A", "x"),
        ("u2", "B", "x"),
        ("u3", "A", "x"),
        ("u4", "B", "x"),
        ("u5", "A", "y"),
        ("u6", "B", "y"),
        ("u7", "A", "y"),
        ("u8", "B", "y"),
    )
    for subject, condition, batch in assignments:
        rows.append(
            {
                "sample": f"{subject}-1",
                "subject": subject,
                "condition": condition,
                "batch": batch,
            }
        )
    rows.append(
        {"sample": "u1-2", "subject": "u1", "condition": "A", "batch": "x"}
    )
    return pd.DataFrame(reversed(rows))


def _paired_metadata(contexts: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample": f"{subject}-{context}",
                "subject": subject,
                "condition": context,
                "batch": "shared",
            }
            for subject in ("u1", "u2", "u3", "u4")
            for context in contexts
        ]
    )


def _build(metadata: pd.DataFrame):
    return build_exchangeability_map(
        metadata,
        sample_key="sample",
        subject_key="subject",
        context_keys=("condition",),
        strata_keys=("batch",),
        immutable_covariates=("batch",),
    )


def _context_value(context) -> object:
    return dict(context)["condition"]


def test_independent_permutation_preserves_subject_blocks_and_stratum_counts() -> None:
    exchangeability = _build(_independent_metadata())
    plans = plan_context_permutations(
        exchangeability,
        n_permutations=12,
        seed_lineage=SeedLineage(91),
    )

    assert exchangeability.design is ExchangeabilityDesign.INDEPENDENT
    assert exchangeability.operation is (
        ContextPermutationOperation.BETWEEN_SUBJECT_WITHIN_STRATUM
    )
    subject_by_sample = dict(
        zip(
            exchangeability.sample_ids,
            exchangeability.sample_subject_ids,
            strict=True,
        )
    )
    stratum_by_subject = dict(
        zip(
            exchangeability.subject_ids,
            exchangeability.subject_strata,
            strict=True,
        )
    )
    original_by_subject = {
        subject: _context_value(exchangeability.sample_contexts[index])
        for index, subject in enumerate(exchangeability.sample_subject_ids)
    }
    original_counts: dict[tuple[object, ...], list[object]] = {}
    for subject in exchangeability.subject_ids:
        original_counts.setdefault(stratum_by_subject[subject], []).append(
            original_by_subject[subject]
        )

    for plan in plans:
        assigned: dict[str, set[object]] = {}
        for sample, context in zip(
            plan.sample_ids, plan.permuted_contexts, strict=True
        ):
            assigned.setdefault(subject_by_sample[sample], set()).add(
                _context_value(context)
            )
        assert all(len(values) == 1 for values in assigned.values())
        for stratum, original in original_counts.items():
            observed = [
                next(iter(assigned[subject]))
                for subject in exchangeability.subject_ids
                if stratum_by_subject[subject] == stratum
            ]
            assert sorted(observed) == sorted(original)


def test_permutation_plans_are_deterministic_and_call_order_independent() -> None:
    exchangeability = _build(_independent_metadata())
    lineage = SeedLineage(7).derive("formal-null")

    first = plan_context_permutations(
        exchangeability, n_permutations=4, seed_lineage=lineage
    )
    second = plan_context_permutations(
        exchangeability, n_permutations=4, seed_lineage=lineage
    )

    assert [plan.permutation_id for plan in first] == [
        plan.permutation_id for plan in second
    ]
    assert [plan.permuted_contexts for plan in first] == [
        plan.permuted_contexts for plan in second
    ]


def test_paired_binary_plan_only_swaps_complete_subject_blocks() -> None:
    exchangeability = _build(_paired_metadata())
    plans = plan_context_permutations(
        exchangeability,
        n_permutations=8,
        seed_lineage=SeedLineage(17),
    )

    assert exchangeability.design is ExchangeabilityDesign.PAIRED_BINARY
    assert exchangeability.operation is (
        ContextPermutationOperation.WITHIN_SUBJECT_BINARY_SWAP
    )
    rows_by_subject: dict[str, list[int]] = {}
    for index, subject in enumerate(exchangeability.sample_subject_ids):
        rows_by_subject.setdefault(subject, []).append(index)
    for plan in plans:
        for subject, indexes in rows_by_subject.items():
            original = [exchangeability.sample_contexts[index] for index in indexes]
            observed = [plan.permuted_contexts[index] for index in indexes]
            if subject in plan.changed_subject_ids:
                assert observed == list(reversed(original))
            else:
                assert observed == original


def test_apply_permutation_updates_all_sample_cell_rows_without_mutation() -> None:
    metadata = _independent_metadata()
    exchangeability = _build(metadata)
    plan = plan_context_permutations(
        exchangeability,
        n_permutations=1,
        seed_lineage=SeedLineage(3),
    )[0]
    cell_metadata = pd.concat([metadata, metadata], ignore_index=True)
    original = cell_metadata.copy(deep=True)

    applied = apply_context_permutation(cell_metadata, exchangeability, plan)

    pd.testing.assert_frame_equal(cell_metadata, original)
    assert applied.groupby("sample")["condition"].nunique().eq(1).all()
    expected = {
        sample: _context_value(context)
        for sample, context in zip(
            plan.sample_ids, plan.permuted_contexts, strict=True
        )
    }
    assert applied.groupby("sample")["condition"].first().to_dict() == expected


def test_mixed_paired_unpaired_design_fails_closed() -> None:
    metadata = _paired_metadata()
    metadata = metadata.loc[
        ~((metadata["subject"] == "u4") & (metadata["condition"] == "B"))
    ]

    with pytest.raises(ContractError) as error:
        _build(metadata)

    assert error.value.details.code == (
        "mixed_paired_unpaired_exchangeability_unsupported"
    )


def test_multi_context_paired_design_requires_explicit_operation() -> None:
    with pytest.raises(ContractError) as error:
        _build(_paired_metadata(("A", "B", "C")))

    assert error.value.details.code == "multi_context_exchangeability_not_declared"


def test_immutable_covariate_must_be_constant_within_subject() -> None:
    metadata = _independent_metadata()
    metadata.loc[metadata["sample"].eq("u1-2"), "batch"] = "y"

    with pytest.raises(ContractError) as error:
        _build(metadata)

    assert error.value.details.code == (
        "exchangeability_immutable_covariate_varies"
    )


def test_no_multicontext_stratum_is_not_permutable() -> None:
    metadata = _independent_metadata()
    metadata["batch"] = metadata["condition"]

    with pytest.raises(ContractError) as error:
        _build(metadata)

    assert error.value.details.code == "exchangeability_no_permutable_stratum"


def test_subject_bootstrap_preserves_stratum_sizes_and_uses_unique_draw_ids() -> None:
    exchangeability = _build(_independent_metadata())
    plans = plan_subject_bootstraps(
        exchangeability,
        n_bootstraps=5,
        seed_lineage=SeedLineage(23),
    )
    expected_sizes: dict[tuple[object, ...], int] = {}
    for stratum in exchangeability.subject_strata:
        expected_sizes[stratum] = expected_sizes.get(stratum, 0) + 1

    for plan in plans:
        observed_sizes: dict[tuple[object, ...], int] = {}
        for draw in plan.draws:
            observed_sizes[draw.stratum] = observed_sizes.get(draw.stratum, 0) + 1
        assert observed_sizes == expected_sizes
        assert len(plan.draws) == len(exchangeability.subject_ids)
        assert len({draw.bootstrap_subject_id for draw in plan.draws}) == len(
            plan.draws
        )


def test_tampered_permutation_identity_and_wrong_sample_universe_are_rejected() -> None:
    metadata = _independent_metadata()
    exchangeability = _build(metadata)
    plan = plan_context_permutations(
        exchangeability,
        n_permutations=1,
        seed_lineage=SeedLineage(29),
    )[0]

    with pytest.raises(ContractError) as error:
        replace(plan, changed_subject_ids=())
    assert error.value.details.code == "context_permutation_integrity_violation"

    missing = metadata.loc[~metadata["sample"].eq("u1-1")]
    with pytest.raises(ContractError) as error:
        apply_context_permutation(missing, exchangeability, plan)
    assert error.value.details.code == (
        "context_permutation_sample_universe_mismatch"
    )
