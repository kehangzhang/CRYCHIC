from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from crychic.core import ContractError
from crychic.inference.bootstrap_support import (
    SpecificitySupportResult,
    SpecificitySupportSpec,
    SpecificitySupportStatus,
    summarize_specificity_support,
)
from crychic.inference.effects import (
    OOFContextEffectResult,
    OOFEffectSpec,
    fit_oof_context_effect,
)
from crychic.inference.full_pipeline import (
    FullPipelineEffectDistribution,
    FullPipelineEffectDistributionSpec,
    FullPipelineEffectResamplingKind,
    FullPipelineResampleEffectRecord,
    FullPipelineResampleEffectStatus,
    SpecificityDirection,
    summarize_full_pipeline_effect_distribution,
)

_SupportCase = tuple[
    FullPipelineEffectDistribution,
    SpecificitySupportSpec,
    tuple[FullPipelineResampleEffectRecord, ...],
]


def _effect_spec() -> OOFEffectSpec:
    return OOFEffectSpec(
        hypothesis_id="receiver-R-family-F",
        contrast_name="B-vs-A",
        contrast_weights=(("B", 1.0), ("A", -1.0)),
    )


def _point_effect(*, observed: bool = True) -> OOFContextEffectResult:
    rows = [
        {
            "subject_id": f"s{subject:02d}",
            "sample_id": f"s{subject:02d}-{context}",
            "fold_id": f"fold-{subject % 2}",
            "context_id": context,
            "score": float(subject) / 10 + (2.0 if context == "B" else 0.0),
            "score_status": "observed",
            "scoring_function_id": f"functional-{subject % 2}",
        }
        for subject in range(12)
        for context in ("A", "B")
    ]
    if not observed:
        rows[0]["score_status"] = "not_estimable"
    return fit_oof_context_effect(pd.DataFrame(rows), _effect_spec())


def _record(
    kind: FullPipelineEffectResamplingKind,
    index: int,
    effect: float | None,
    *,
    status: FullPipelineResampleEffectStatus = (
        FullPipelineResampleEffectStatus.OBSERVED
    ),
    plan_id: str | None = None,
) -> FullPipelineResampleEffectRecord:
    observed = status is FullPipelineResampleEffectStatus.OBSERVED
    return FullPipelineResampleEffectRecord(
        full_pipeline_record_id=f"pipeline-{kind.value}-{index}",
        plan_id=plan_id or f"plan-{kind.value}-{index:04d}",
        crossfit_id=f"crossfit-{kind.value}-{index}",
        resampling_kind=kind,
        resample_index=index,
        effect_spec_id=_effect_spec().spec_id,
        hypothesis_id=_effect_spec().hypothesis_id,
        effect_result_id=f"effect-{kind.value}-{index}",
        effect=effect if observed else None,
        status=status,
        reason_code=None if observed else f"{status.value}_resample_effect",
    )


def _records(
    effects: tuple[float, ...],
    *,
    statuses: dict[int, FullPipelineResampleEffectStatus] | None = None,
    include_permutations: bool = False,
) -> tuple[FullPipelineResampleEffectRecord, ...]:
    statuses = {} if statuses is None else statuses
    bootstrap = tuple(
        _record(
            FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP,
            index,
            effect,
            status=statuses.get(
                index,
                FullPipelineResampleEffectStatus.OBSERVED,
            ),
        )
        for index, effect in enumerate(effects)
    )
    if not include_permutations:
        return bootstrap
    permutation = (
        _record(
            FullPipelineEffectResamplingKind.CONTEXT_PERMUTATION,
            0,
            100_000.0,
        ),
        _record(
            FullPipelineEffectResamplingKind.CONTEXT_PERMUTATION,
            1,
            None,
            status=FullPipelineResampleEffectStatus.FAILED,
        ),
    )
    return (*bootstrap, *permutation)


def _case(
    effects: tuple[float, ...],
    *,
    direction: SpecificityDirection = SpecificityDirection.GREATER,
    statuses: dict[int, FullPipelineResampleEffectStatus] | None = None,
    point_observed: bool = True,
    include_permutations: bool = False,
) -> _SupportCase:
    distribution_spec = FullPipelineEffectDistributionSpec(
        effect_spec=_effect_spec(),
        minimum_effect=2.0,
        specificity_direction=direction,
    )
    records = _records(
        effects,
        statuses=statuses,
        include_permutations=include_permutations,
    )
    distribution = summarize_full_pipeline_effect_distribution(
        _point_effect(observed=point_observed),
        distribution_spec,
        records,
    )
    plan_ids = tuple(
        item.plan_id
        for item in records
        if item.resampling_kind
        is FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP
    )
    support_spec = SpecificitySupportSpec(
        effect_distribution_spec=distribution_spec,
        hypothesis_universe_id="frozen-hypothesis-universe",
        effect_scale_id="oof-common-functional-unpenalized-effect-v1",
        subject_bootstrap_plan_ids=plan_ids,
    )
    return distribution, support_spec, records


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def observed_case() -> _SupportCase:
    effects = tuple(3.0 if index < 500 else 2.0 for index in range(1_000))
    return _case(effects)


def test_complete_bootstraps_produce_strict_frequentist_support(
    observed_case: _SupportCase,
) -> None:
    distribution, spec, records = observed_case

    result = summarize_specificity_support(distribution, spec, tuple(reversed(records)))
    repeated_spec = SpecificitySupportSpec(
        effect_distribution_spec=spec.effect_distribution_spec,
        hypothesis_universe_id=spec.hypothesis_universe_id,
        effect_scale_id=spec.effect_scale_id,
        subject_bootstrap_plan_ids=tuple(reversed(spec.subject_bootstrap_plan_ids)),
    )
    repeated = summarize_specificity_support(distribution, repeated_spec, records)

    assert result.status is SpecificitySupportStatus.OBSERVED
    assert result.specificity_support == 0.5
    assert result.n_bootstrap_total == 1_000
    assert result.n_bootstrap_observed == 1_000
    assert result.n_bootstrap_not_estimable == 0
    assert result.n_bootstrap_failed == 0
    assert result.bootstrap_plan_set_id == repeated.bootstrap_plan_set_id
    assert result.result_id == repeated.result_id
    assert result.is_posterior_probability is False
    assert result.is_comm_probability is False
    assert result.public_release_allowed is False
    payload = result.to_dict()
    assert payload["specificity_semantics"] == (
        "frequentist_full_pipeline_subject_bootstrap_exceedance_frequency_"
        "not_posterior_v1"
    )
    assert payload["source_binding_status"] == (
        "unbound_numeric_primitive_requires_frozen_workflow_adapter"
    )
    assert payload["public_release_allowed"] is False
    assert "probability" not in payload


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("direction", "expected"),
    [
        (SpecificityDirection.GREATER, 0.25),
        (SpecificityDirection.LESS, 0.25),
        (SpecificityDirection.TWO_SIDED, 0.5),
    ],
)
def test_directional_support_uses_strict_delta_comparison(
    direction: SpecificityDirection,
    expected: float,
) -> None:
    effects = tuple((3.0, 2.0, -3.0, -2.0)[index % 4] for index in range(1_000))
    distribution, spec, records = _case(effects, direction=direction)

    result = summarize_specificity_support(distribution, spec, records)

    assert result.status is SpecificitySupportStatus.OBSERVED
    assert result.specificity_support == expected


def test_fewer_than_1000_bootstraps_is_not_estimable() -> None:
    distribution, spec, records = _case((3.0,) * 999)

    result = summarize_specificity_support(distribution, spec, records)

    assert result.status is SpecificitySupportStatus.NOT_ESTIMABLE
    assert result.specificity_support is None
    assert result.reason_code == "specificity_bootstrap_count_below_1000"


def test_explicit_not_estimable_suppresses_observed_subset_frequency() -> None:
    distribution, spec, records = _case(
        (3.0,) * 1_000,
        statuses={17: FullPipelineResampleEffectStatus.NOT_ESTIMABLE},
    )
    assert distribution.diagnostic_specificity_frequency == 1.0

    result = summarize_specificity_support(distribution, spec, records)

    assert result.status is SpecificitySupportStatus.NOT_ESTIMABLE
    assert result.specificity_support is None
    assert result.n_bootstrap_not_estimable == 1
    assert result.reason_code == "specificity_bootstrap_record_not_estimable"


def test_failed_bootstrap_takes_precedence_over_not_estimable() -> None:
    distribution, spec, records = _case(
        (3.0,) * 1_000,
        statuses={
            1: FullPipelineResampleEffectStatus.NOT_ESTIMABLE,
            2: FullPipelineResampleEffectStatus.FAILED,
        },
    )

    result = summarize_specificity_support(distribution, spec, records)

    assert result.status is SpecificitySupportStatus.FAILED
    assert result.specificity_support is None
    assert result.n_bootstrap_failed == 1
    assert result.n_bootstrap_not_estimable == 1
    assert result.reason_code == "specificity_bootstrap_record_failed"


def test_unobserved_point_effect_is_not_estimable() -> None:
    distribution, spec, records = _case((3.0,) * 1_000, point_observed=False)

    result = summarize_specificity_support(distribution, spec, records)

    assert result.status is SpecificitySupportStatus.NOT_ESTIMABLE
    assert result.specificity_support is None
    assert result.reason_code == "specificity_point_effect_not_observed"


def test_context_permutations_are_rejected_from_primitive() -> None:
    distribution, spec, records = _case(
        (3.0,) * 1_000,
        include_permutations=True,
    )

    with pytest.raises(ContractError, match="bootstrap-only"):
        summarize_specificity_support(distribution, spec, records)


def test_context_permutation_record_is_rejected_even_with_bootstrap_distribution(
    observed_case: _SupportCase,
) -> None:
    distribution, spec, records = observed_case
    permutation = _record(
        FullPipelineEffectResamplingKind.CONTEXT_PERMUTATION,
        0,
        0.0,
    )

    with pytest.raises(ContractError, match="subject-bootstrap records only"):
        summarize_specificity_support(distribution, spec, (*records, permutation))


def test_missing_extra_and_duplicate_source_provenance_are_contract_errors(
    observed_case: _SupportCase,
) -> None:
    distribution, spec, records = observed_case
    extra = _record(
        FullPipelineEffectResamplingKind.SUBJECT_BOOTSTRAP,
        1_001,
        0.0,
    )

    with pytest.raises(ContractError, match="exactly reproduce"):
        summarize_specificity_support(distribution, spec, records[:-1])
    with pytest.raises(ContractError, match="exactly reproduce"):
        summarize_specificity_support(distribution, spec, (*records, extra))
    with pytest.raises(ContractError, match="duplicate provenance"):
        summarize_specificity_support(distribution, spec, (*records, records[0]))


def test_mismatched_bootstrap_plan_set_is_a_contract_error(
    observed_case: _SupportCase,
) -> None:
    distribution, spec, records = observed_case
    mismatched = SpecificitySupportSpec(
        effect_distribution_spec=spec.effect_distribution_spec,
        hypothesis_universe_id=spec.hypothesis_universe_id,
        effect_scale_id=spec.effect_scale_id,
        subject_bootstrap_plan_ids=(
            *spec.subject_bootstrap_plan_ids[:-1],
            "plan-subject_bootstrap-unrun",
        ),
    )

    with pytest.raises(ContractError, match="frozen plan set"):
        summarize_specificity_support(distribution, mismatched, records)


def test_spec_and_result_are_tamper_evident() -> None:
    distribution, spec, records = _case((3.0,) * 1_000)
    result = summarize_specificity_support(distribution, spec, records)
    object.__setattr__(spec, "effect_scale_id", "tampered-scale")
    with pytest.raises(ContractError, match="spec failed integrity"):
        spec.to_dict()
    object.__setattr__(result, "specificity_support", 0.99)
    with pytest.raises(ContractError, match="result failed integrity"):
        result.to_dict()


def test_result_detects_tampered_source_record() -> None:
    distribution, spec, records = _case((3.0,) * 1_000)
    result = summarize_specificity_support(distribution, spec, records)
    object.__setattr__(records[0], "plan_id", "tampered-plan")

    with pytest.raises(ContractError, match="result failed integrity"):
        result.to_dict()


def test_source_distribution_tampering_is_rejected() -> None:
    distribution, spec, records = _case((3.0,) * 1_000)
    object.__setattr__(distribution, "distribution_id", "tampered-distribution")

    with pytest.raises(ContractError, match="distribution failed integrity"):
        summarize_specificity_support(distribution, spec, records)


def test_fixed_minimum_and_producer_owned_result(
    observed_case: _SupportCase,
) -> None:
    _, spec, _ = observed_case
    with pytest.raises(ValueError, match="fixed at 1000"):
        replace(spec, minimum_bootstraps=999)
    with pytest.raises(TypeError, match="producer-owned"):
        SpecificitySupportResult()


def test_free_form_source_ids_do_not_authorize_public_release(
    observed_case: _SupportCase,
) -> None:
    distribution, original_spec, records = observed_case
    forged_spec = SpecificitySupportSpec(
        effect_distribution_spec=original_spec.effect_distribution_spec,
        hypothesis_universe_id="caller-forged-universe-id",
        effect_scale_id="caller-forged-effect-scale-id",
        subject_bootstrap_plan_ids=original_spec.subject_bootstrap_plan_ids,
    )

    result = summarize_specificity_support(distribution, forged_spec, records)

    assert result.status is SpecificitySupportStatus.OBSERVED
    assert result.hypothesis_universe_id == "caller-forged-universe-id"
    assert result.effect_scale_id == "caller-forged-effect-scale-id"
    assert forged_spec.public_release_allowed is False
    assert result.public_release_allowed is False
    assert forged_spec.to_dict()["source_binding_status"] == (
        "unbound_numeric_primitive_requires_frozen_workflow_adapter"
    )
    assert result.to_dict()["source_binding_status"] == (
        "unbound_numeric_primitive_requires_frozen_workflow_adapter"
    )
