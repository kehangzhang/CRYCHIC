from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from time import perf_counter

import pytest

from crychic.core import ContractError, SeedLineage
from crychic.resampling.active_null import (
    ActiveNullPlanStatus,
    ActiveNullReasonCode,
    ActiveNullSpec,
    materialize_active_null_target_prior,
    plan_active_null_target_prior,
    target_prior_content_id,
)
from crychic.resources import GeneNamespace, MappingReport, Species, TargetPrior


def _prior(
    links_by_driver: dict[str, tuple[tuple[str, float, int | None], ...]],
) -> TargetPrior:
    driver_ids = tuple(sorted(links_by_driver))
    target_ids = tuple(
        sorted({target for links in links_by_driver.values() for target, _, _ in links})
    )
    target_index = {target: index for index, target in enumerate(target_ids)}
    has_ranks = {
        rank is not None for links in links_by_driver.values() for _, _, rank in links
    }
    if len(has_ranks) != 1:
        raise ValueError("test prior ranks must be uniformly present or absent")
    ranks_present = has_ranks == {True}
    indptr = [0]
    target_indices: list[int] = []
    weights: list[float] = []
    ranks: list[int] = []
    for driver in driver_ids:
        for target, weight, rank in links_by_driver[driver]:
            target_indices.append(target_index[target])
            weights.append(weight)
            if rank is not None:
                ranks.append(rank)
        indptr.append(len(target_indices))
    edge_count = len(weights)
    return TargetPrior(
        resource_id="active_null_test_prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=target_ids,
        driver_ids=driver_ids,
        indptr=tuple(indptr),
        target_indices=tuple(target_indices),
        weights=tuple(weights),
        ranks=tuple(ranks) if ranks_present else None,
        direction=1,
        evidence="synthetic ranked evidence",
        mapping_report=MappingReport(
            source_rows=edge_count,
            loaded_rows=edge_count,
            mapped_entities=len(driver_ids) + len(target_ids),
        ),
        manifest_digest="active-null-test-manifest",
    )


def _matching_prior(*, n_drivers: int = 80, ranked: bool = True) -> TargetPrior:
    return _prior(
        {
            f"L{index:03d}": (
                (f"T_A_{index:03d}", 1.0, 1 if ranked else None),
                (f"T_B_{index:03d}", 0.5, 11 if ranked else None),
            )
            for index in range(n_drivers)
        }
    )


def _large_sparse_prior() -> TargetPrior:
    return _prior(
        {
            f"L{driver:03d}": tuple(
                (
                    f"T{(driver * 137 + rank * 17) % 1000:04d}",
                    1.0 / rank,
                    rank,
                )
                for rank in range(1, 101)
            )
            for driver in range(30)
        }
    )


def _links(
    prior: TargetPrior,
) -> tuple[tuple[str, str, float, int | None], ...]:
    rows: list[tuple[str, str, float, int | None]] = []
    for driver_index, driver in enumerate(prior.driver_ids):
        for position in range(
            prior.indptr[driver_index], prior.indptr[driver_index + 1]
        ):
            rows.append(
                (
                    driver,
                    prior.target_ids[prior.target_indices[position]],
                    prior.weights[position],
                    None if prior.ranks is None else prior.ranks[position],
                )
            )
    return tuple(rows)


def _driver_stub_multiset(
    prior: TargetPrior,
) -> Counter[tuple[str, float, int | None]]:
    return Counter((driver, weight, rank) for driver, _, weight, rank in _links(prior))


def _target_tier_degree(prior: TargetPrior) -> Counter[tuple[str, int]]:
    if prior.ranks is None:
        raise ValueError("ranked prior required")
    bounds = ActiveNullSpec().rank_tier_upper_bounds

    def tier(rank: int) -> int:
        return next(
            (index for index, upper in enumerate(bounds) if rank <= upper),
            len(bounds),
        )

    return Counter((target, tier(rank)) for _, target, _, rank in _links(prior))


def test_plan_and_materialize_preserve_exact_degree_and_evidence_contract() -> None:
    prior = _matching_prior()
    source_snapshot = asdict(prior)

    plan = plan_active_null_target_prior(
        prior,
        seed_lineage=SeedLineage(901).derive("fold-1", "null-plan-1"),
    )

    assert plan.status is ActiveNullPlanStatus.OBSERVED
    assert plan.reason_code is None
    assert plan.source_edge_count == 160
    assert plan.accepted_switch_count == 20 * plan.source_edge_count
    assert all(
        diagnostic.accepted_switch_count == 20 * diagnostic.edge_count
        for diagnostic in plan.tier_diagnostics
    )
    assert plan.source_overlap_fraction is not None
    assert plan.source_overlap_fraction <= 0.05
    assert len(plan.reassigned_links) == plan.source_edge_count

    materialized = materialize_active_null_target_prior(prior, plan)

    assert asdict(prior) == source_snapshot
    assert materialized.driver_ids == prior.driver_ids
    assert materialized.target_ids == prior.target_ids
    assert Counter(driver for driver, _, _, _ in _links(materialized)) == Counter(
        driver for driver, _, _, _ in _links(prior)
    )
    assert _target_tier_degree(materialized) == _target_tier_degree(prior)
    assert _driver_stub_multiset(materialized) == _driver_stub_multiset(prior)
    assert target_prior_content_id(materialized) != target_prior_content_id(prior)
    assert materialized.manifest_digest != prior.manifest_digest
    assert f"active_null_plan_id={plan.active_null_plan_id}" in (
        materialized.mapping_report.notes
    )


def test_plans_are_seed_deterministic_and_call_order_independent() -> None:
    prior = _matching_prior()
    lineage = SeedLineage(19).derive("fold-2", "null-plan-4")

    first = plan_active_null_target_prior(prior, seed_lineage=lineage)
    second = plan_active_null_target_prior(prior, seed_lineage=lineage)
    other = plan_active_null_target_prior(
        prior,
        seed_lineage=SeedLineage(19).derive("fold-2", "null-plan-5"),
    )

    assert first.to_dict() == second.to_dict()
    assert first.active_null_plan_id == second.active_null_plan_id
    assert other.status is ActiveNullPlanStatus.OBSERVED
    assert other.active_null_plan_id != first.active_null_plan_id
    assert other.reassigned_links != first.reassigned_links


def test_missing_ranks_use_deterministic_weight_order_without_materializing_ranks() -> (
    None
):
    prior = _matching_prior(ranked=False)

    plan = plan_active_null_target_prior(
        prior,
        seed_lineage=SeedLineage(77).derive("rankless"),
    )
    materialized = materialize_active_null_target_prior(prior, plan)

    assert plan.status is ActiveNullPlanStatus.OBSERVED
    assert {link.evidence_rank for link in plan.reassigned_links} == {1, 2}
    assert {link.evidence_tier for link in plan.reassigned_links} == {0}
    assert materialized.ranks is None
    assert _driver_stub_multiset(materialized) == _driver_stub_multiset(prior)
    assert Counter(target for _, target, _, _ in _links(materialized)) == Counter(
        target for _, target, _, _ in _links(prior)
    )


def test_complete_bipartite_tier_is_typed_reassignment_infeasible() -> None:
    prior = _prior(
        {
            "L1": (("T1", 1.0, 1), ("T2", 0.5, 2)),
            "L2": (("T1", 1.0, 1), ("T2", 0.5, 2)),
        }
    )

    plan = plan_active_null_target_prior(
        prior,
        seed_lineage=SeedLineage(5).derive("complete-bipartite"),
    )

    assert plan.status is ActiveNullPlanStatus.NOT_ESTIMABLE
    assert plan.reason_code is ActiveNullReasonCode.REASSIGNMENT_INFEASIBLE
    assert plan.accepted_switch_count == 0
    assert plan.reassigned_links == ()
    with pytest.raises(ContractError) as error:
        materialize_active_null_target_prior(prior, plan)
    assert error.value.details.code == "active_null_plan_not_estimable"


def test_twenty_switches_do_not_override_failed_overlap_gate() -> None:
    prior = _prior(
        {
            "L1": (("T1", 1.0, 1),),
            "L2": (("T2", 1.0, 1),),
        }
    )

    plan = plan_active_null_target_prior(
        prior,
        seed_lineage=SeedLineage(11).derive("two-edge-cycle"),
    )

    assert plan.status is ActiveNullPlanStatus.NOT_ESTIMABLE
    assert plan.reason_code is ActiveNullReasonCode.INSUFFICIENT_REWIRING
    assert plan.accepted_switch_count == 40
    assert plan.source_overlap_count == 2
    assert plan.source_overlap_fraction == 1.0


def test_planner_matches_pre_optimization_frozen_outputs() -> None:
    # Each ID binds every reassigned link, diagnostic, seed, and source identity.
    # These values were captured from the original object-based implementation.
    cases = (
        (
            _matching_prior(),
            SeedLineage(901).derive("fold-1", "null-plan-1"),
            "active_null_plan_83f24687cf3e00a7071001246a035d3b",
        ),
        (
            _matching_prior(ranked=False),
            SeedLineage(77).derive("rankless"),
            "active_null_plan_535345e41ad29c35669ff55703c1ac9e",
        ),
        (
            _prior(
                {
                    "L1": (("T1", 1.0, 1), ("T2", 0.5, 2)),
                    "L2": (("T1", 1.0, 1), ("T2", 0.5, 2)),
                }
            ),
            SeedLineage(5).derive("complete-bipartite"),
            "active_null_plan_4b8d7ca186f6d1991eb7ff6c62fc4a69",
        ),
        (
            _prior(
                {
                    "L1": (("T1", 1.0, 1),),
                    "L2": (("T2", 1.0, 1),),
                }
            ),
            SeedLineage(11).derive("two-edge-cycle"),
            "active_null_plan_7b727201a47591e6cfc722e56862ffed",
        ),
    )

    for prior, lineage, expected_plan_id in cases:
        plan = plan_active_null_target_prior(prior, seed_lineage=lineage)

        assert plan.active_null_plan_id == expected_plan_id


def test_large_sparse_planner_performance_guard() -> None:
    prior = _large_sparse_prior()

    started = perf_counter()
    plan = plan_active_null_target_prior(
        prior,
        seed_lineage=SeedLineage(20260716).derive("synthetic-performance-regression"),
    )
    elapsed_seconds = perf_counter() - started

    assert plan.accepted_switch_count == 20 * prior.nnz == 60_000
    # This is intentionally a broad regression guard, not a microbenchmark.
    # The optimized implementation is below 0.2 s on the reference host.
    assert elapsed_seconds < 3.0, f"planner took {elapsed_seconds:.3f} seconds"


def test_materializer_rejects_a_different_source_prior() -> None:
    prior = _matching_prior()
    plan = plan_active_null_target_prior(
        prior,
        seed_lineage=SeedLineage(31).derive("source-binding"),
    )
    altered = replace(
        prior,
        weights=(prior.weights[0] + 0.01, *prior.weights[1:]),
    )

    with pytest.raises(ContractError) as error:
        materialize_active_null_target_prior(altered, plan)

    assert error.value.details.code == "active_null_source_mismatch"


def test_materializer_rejects_post_construction_plan_mutation() -> None:
    prior = _matching_prior()
    plan = plan_active_null_target_prior(
        prior,
        seed_lineage=SeedLineage(41).derive("integrity"),
    )
    object.__setattr__(
        plan,
        "accepted_switch_count",
        plan.accepted_switch_count + 1,
    )

    with pytest.raises(ContractError) as error:
        materialize_active_null_target_prior(prior, plan)

    assert error.value.details.code == "active_null_plan_integrity_violation"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("rank_tier_upper_bounds", (10, 50, 250)),
        ("accepted_switches_per_edge", 19),
        ("max_overlap_fraction", 0.1),
        ("max_attempts_per_required_switch", 99),
    ),
)
def test_primary_null_contract_cannot_be_runtime_switched(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        ActiveNullSpec(**{field: value})  # type: ignore[arg-type]
