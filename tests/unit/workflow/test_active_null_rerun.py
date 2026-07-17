from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import crychic.workflow.active_null_rerun as rerun_module
from crychic.core import ContractError, CrychicConfig, SeedLineage, stable_id
from crychic.design import balanced_contrast
from crychic.inference import ActiveEdgeScoreStatus, PointActiveEdgeScoreRecord
from crychic.resampling import (
    ActiveNullPlanStatus,
    ActiveNullReasonCode,
    plan_active_null_target_prior,
)
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.scoring import (
    ActiveEdgeCandidate,
    FrozenActiveEdgeUniverse,
    freeze_active_edge_universe,
)
from crychic.workflow.active_edge_scores import CrossFitActiveEdgePointRecords
from crychic.workflow.active_null_rerun import (
    ActiveNullRerunResultStatus,
    ActiveNullRerunStatus,
    run_active_null_reruns,
)
from crychic.workflow.crossfit import CrossFitArtifacts, CrossFitSpec
from crychic.workflow.training import FoldTrainingSpec

_SCORE_VERSION = "receiver_gain_percentile_mechanistic_conserved_sender_v4"
_SCORE_SPEC_ID = "test-global-score-spec"


def _adata() -> AnnData:
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    names: list[str] = []
    for subject_index, subject in enumerate(("s1", "s2", "s3", "s4")):
        for condition in ("control", "stim"):
            for cell_type, values in (
                ("Sender", [10 + subject_index, 1, 0]),
                ("Receiver", [0, 1, 10 + subject_index]),
            ):
                rows.append(values)
                metadata.append(
                    {
                        "sample_id": f"{subject}-{condition}",
                        "subject_id": subject,
                        "cell_type": cell_type,
                        "condition": condition,
                    }
                )
                names.append(f"{subject}-{condition}-{cell_type}")
    counts = sparse.csr_matrix(np.asarray(rows, dtype=np.int64))
    adata = AnnData(
        X=sparse.csr_matrix(counts.shape, dtype=float),
        obs=pd.DataFrame(metadata, index=names),
        var=pd.DataFrame(index=("L", "R", "T")),
    )
    adata.layers["counts"] = counts
    return adata


def _config() -> CrychicConfig:
    return CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=3107,
    )


def _bundle(*, manifest_digest: str = "resource-manifest") -> ResourceBundle:
    interaction = Interaction(
        interaction_id="lr1",
        source_interaction_id="lr1",
        ligand_name="L",
        receptor_name="R",
        ligand_subunits=("L",),
        receptor_subunits=("R",),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="test",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    return ResourceBundle(
        resource_id="active-null-rerun-test",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(interaction,),
        mapping_report=MappingReport(1, 1, 2),
        manifest_digest=manifest_digest,
        source_files=("test.tsv",),
        license="CC0",
        citation="Synthetic test",
    )


def _prior(*, n_drivers: int = 80) -> TargetPrior:
    driver_ids = tuple(f"L{index:03d}" for index in range(n_drivers))
    target_ids = tuple(
        sorted(
            target
            for index in range(n_drivers)
            for target in (f"TA{index:03d}", f"TB{index:03d}")
        )
    )
    target_index = {target: index for index, target in enumerate(target_ids)}
    indptr = tuple(2 * index for index in range(n_drivers + 1))
    return TargetPrior(
        resource_id="active-null-rerun-prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=target_ids,
        driver_ids=driver_ids,
        indptr=indptr,
        target_indices=tuple(
            target_index[target]
            for index in range(n_drivers)
            for target in (f"TA{index:03d}", f"TB{index:03d}")
        ),
        weights=tuple(value for _ in driver_ids for value in (1.0, 0.5)),
        ranks=tuple(value for _ in driver_ids for value in (1, 11)),
        direction=1,
        evidence="synthetic ranked evidence",
        mapping_report=MappingReport(
            source_rows=2 * n_drivers,
            loaded_rows=2 * n_drivers,
            mapped_entities=3 * n_drivers,
        ),
        manifest_digest="source-prior-manifest",
    )


def _spec() -> CrossFitSpec:
    return CrossFitSpec(
        contrasts=(
            balanced_contrast(
                ("stim",),
                ("control",),
                name="stim_vs_control",
            ),
        ),
        training_spec=FoldTrainingSpec(min_cells=1, max_interactions=1),
        allowed_n_splits=(2,),
    )


@dataclass(frozen=True)
class _FakeFoldPlan:
    repeat_id: str
    seed_lineage: SeedLineage
    plan_id: str = "frozen-point-fold-plan"
    partition_seed_lineage: SeedLineage | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "repeat_id": self.repeat_id,
            "seed_lineage": self.seed_lineage.to_dict(),
            "partition_seed_lineage": (
                None
                if self.partition_seed_lineage is None
                else self.partition_seed_lineage.to_dict()
            ),
        }


def _universe(spec: CrossFitSpec) -> FrozenActiveEdgeUniverse:
    contrast_id = stable_id("contrast_manifest", spec.contrasts[0].to_dict())
    candidates = (
        ActiveEdgeCandidate(
            contrast_id=contrast_id,
            context_id="control",
            sender="Sender",
            receiver="Receiver",
            interaction_id="lr1",
            driver_id="L000",
            mode="state",
        ),
        ActiveEdgeCandidate(
            contrast_id=contrast_id,
            context_id="stim",
            sender="Sender",
            receiver="Receiver",
            interaction_id="lr1",
            driver_id="L000",
            mode="state",
        ),
    )
    return freeze_active_edge_universe(
        candidates,
        universe_name="active-null-rerun-test-universe",
        contrast_id=contrast_id,
        score_version=_SCORE_VERSION,
    )


def _score_collection(
    crossfit_id: str,
    universe: FrozenActiveEdgeUniverse,
    *,
    source_collection_id: str,
    repeat_id: str,
    second_not_estimable: bool = False,
) -> CrossFitActiveEdgePointRecords:
    records: list[PointActiveEdgeScoreRecord] = []
    for index, candidate in enumerate(universe.candidates):
        not_estimable = second_not_estimable and index == 1
        records.append(
            PointActiveEdgeScoreRecord(
                candidate_edge_id=candidate.candidate_edge_id,
                stratum_id=candidate.stratum_id(score_version=_SCORE_VERSION),
                score_version=_SCORE_VERSION,
                source_score_collection_id=source_collection_id,
                n_subjects=4,
                score=None if not_estimable else 0.8 - 0.1 * index,
                status=(
                    ActiveEdgeScoreStatus.NOT_ESTIMABLE
                    if not_estimable
                    else ActiveEdgeScoreStatus.OBSERVED
                ),
                reason_code=(
                    "active_edge_candidate_not_in_training_fold"
                    if not_estimable
                    else None
                ),
            )
        )
    collection = object.__new__(CrossFitActiveEdgePointRecords)
    values: dict[str, object] = {
        "source_crossfit_id": crossfit_id,
        "source_oof_audit_id": f"oof-audit-{crossfit_id}",
        "source_registry_id": f"registry-{crossfit_id}",
        "active_edge_universe_id": universe.universe_id,
        "contrast_id": universe.contrast_id,
        "contrast_name": "stim_vs_control",
        "repeat_id": repeat_id,
        "score_version": _SCORE_VERSION,
        "score_spec_id": _SCORE_SPEC_ID,
        "source_score_collection_id": source_collection_id,
        "source_functional_ids": (f"functional-{crossfit_id}",),
        "source_application_ids": (f"application-{crossfit_id}",),
        "source_sender_score_digests": (f"digest-{crossfit_id}",),
        "records": tuple(sorted(records, key=lambda item: item.candidate_edge_id)),
        "_producer_marker": "crychic.workflow.crossfit_active_edge_point_records.v2",
    }
    for name, value in values.items():
        object.__setattr__(collection, name, value)
    object.__setattr__(
        collection,
        "collection_id",
        stable_id(
            "crossfit_active_edge_point_records",
            collection._identity_payload(),
            schema_version="2.0.0",
        ),
    )
    collection._require_intact()
    return collection


def _artifacts(
    *,
    crossfit_id: str,
    root_identity: Any,
    spec: CrossFitSpec,
    fold_plan: _FakeFoldPlan,
    config: CrychicConfig,
    resource_content_id: str,
    prior_content_id: str,
) -> CrossFitArtifacts:
    artifacts = object.__new__(CrossFitArtifacts)
    object.__setattr__(artifacts, "crossfit_id", crossfit_id)
    object.__setattr__(artifacts, "root_input_identity", root_identity)
    object.__setattr__(artifacts, "spec", spec)
    object.__setattr__(artifacts, "fold_plan", fold_plan)
    object.__setattr__(
        artifacts,
        "receiver_universe",
        SimpleNamespace(
            universe_id="active-null-point-receiver-universe",
            receiver_axis_id="active-null-point-receiver-axis",
            receiver_ids=("Receiver", "Sender"),
        ),
    )
    object.__setattr__(
        artifacts,
        "folds",
        (
            SimpleNamespace(
                training=SimpleNamespace(
                    config=config,
                    resource_bundle_content_id=resource_content_id,
                    target_prior_content_id=prior_content_id,
                )
            ),
        ),
    )
    object.__setattr__(
        artifacts,
        "receiver_scoring_registry_id",
        f"registry-{crossfit_id}",
    )
    return artifacts


@dataclass(frozen=True)
class _Fixture:
    adata: AnnData
    config: CrychicConfig
    bundle: ResourceBundle
    prior: TargetPrior
    spec: CrossFitSpec
    universe: FrozenActiveEdgeUniverse
    fold_plan: _FakeFoldPlan
    point_artifacts: CrossFitArtifacts
    point_scores: CrossFitActiveEdgePointRecords


def _fixture(monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    monkeypatch.setattr(CrossFitArtifacts, "_require_intact", lambda self: None)
    adata = _adata()
    config = _config()
    bundle = _bundle()
    prior = _prior()
    spec = _spec()
    universe = _universe(spec)
    snapshot = rerun_module._sanitized_raw_input_snapshot(adata, config)
    fold_plan = _FakeFoldPlan(
        repeat_id=spec.repeat_id,
        seed_lineage=SeedLineage(config.random_seed).derive(
            "subject_crossfit",
            spec.spec_id,
        ),
    )
    resource_id = rerun_module._resource_bundle_content_id(bundle)
    prior_id = rerun_module._target_prior_content_id(prior)
    point_artifacts = _artifacts(
        crossfit_id="point-crossfit",
        root_identity=snapshot.identity,
        spec=spec,
        fold_plan=fold_plan,
        config=config,
        resource_content_id=resource_id,
        prior_content_id=prior_id,
    )
    point_scores = _score_collection(
        point_artifacts.crossfit_id,
        universe,
        source_collection_id="point-score-source",
        repeat_id=spec.repeat_id,
    )
    monkeypatch.setattr(
        rerun_module,
        "adapt_crossfit_active_edge_point_records",
        lambda artifacts, frozen: point_scores,
    )
    return _Fixture(
        adata=adata,
        config=config,
        bundle=bundle,
        prior=prior,
        spec=spec,
        universe=universe,
        fold_plan=fold_plan,
        point_artifacts=point_artifacts,
        point_scores=point_scores,
    )


def _child_for_prior(
    fixture: _Fixture,
    prior: TargetPrior,
    *,
    crossfit_id: str,
    fold_plan: _FakeFoldPlan | None = None,
) -> CrossFitArtifacts:
    return _artifacts(
        crossfit_id=crossfit_id,
        root_identity=fixture.point_artifacts.root_input_identity,
        spec=fixture.spec,
        fold_plan=fixture.fold_plan if fold_plan is None else fold_plan,
        config=fixture.config,
        resource_content_id=rerun_module._resource_bundle_content_id(fixture.bundle),
        prior_content_id=rerun_module._target_prior_content_id(prior),
    )


def test_one_plan_reruns_same_seed_tree_and_keeps_candidate_ne(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(monkeypatch)
    calls: list[
        tuple[Any, CrychicConfig, ResourceBundle, TargetPrior, CrossFitSpec]
    ] = []

    def runner(
        snapshot: Any,
        config: CrychicConfig,
        resource_bundle: ResourceBundle,
        target_prior: TargetPrior,
        *,
        spec: CrossFitSpec,
    ) -> CrossFitArtifacts:
        calls.append((snapshot, config, resource_bundle, target_prior, spec))
        return _child_for_prior(
            fixture,
            target_prior,
            crossfit_id="null-child-0",
        )

    monkeypatch.setattr(rerun_module, "_run_subject_crossfit", runner)
    monkeypatch.setattr(
        rerun_module,
        "adapt_crossfit_active_edge_records_against_universe",
        lambda child, frozen: _score_collection(
            child.crossfit_id,
            frozen,
            source_collection_id="null-score-source-0",
            repeat_id=fixture.spec.repeat_id,
            second_not_estimable=True,
        ),
    )

    result = run_active_null_reruns(
        fixture.adata,
        fixture.config,
        fixture.bundle,
        fixture.prior,
        fixture.point_artifacts,
        fixture.universe,
        n_plans=1,
        retain_children=True,
    )

    assert result.status is ActiveNullRerunResultStatus.SUCCEEDED
    assert len(calls) == 1
    snapshot, config, resource, null_prior, spec = calls[0]
    assert snapshot.identity.identity_id == (
        fixture.point_artifacts.root_input_identity.identity_id
    )
    assert config is fixture.config
    assert config.random_seed == fixture.config.random_seed
    assert resource is fixture.bundle
    assert spec is fixture.spec
    assert rerun_module._target_prior_content_id(null_prior) != (
        rerun_module._target_prior_content_id(fixture.prior)
    )
    assert result.records[0].child is result.children[0]
    assert result.records[0].child_fold_plan_id == fixture.fold_plan.plan_id
    assert result.distribution.complete_rectangle
    assert len(result.distribution.null_records) == 2
    assert {record.status for record in result.distribution.null_records} == {
        ActiveEdgeScoreStatus.OBSERVED,
        ActiveEdgeScoreStatus.NOT_ESTIMABLE,
    }
    ne_record = next(
        record
        for record in result.distribution.null_records
        if record.status is ActiveEdgeScoreStatus.NOT_ESTIMABLE
    )
    assert ne_record.source_score_collection_id == "null-score-source-0"
    assert ne_record.n_subjects == 4
    assert result.to_manifest()["n_plans"] == 1
    assert result.to_manifest()["only_target_prior_content_changes"] is True


def test_planner_ne_planner_error_and_child_failure_emit_complete_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(monkeypatch)
    original_planner = plan_active_null_target_prior
    runner_calls = 0

    def planner(
        prior: TargetPrior,
        *,
        seed_lineage: SeedLineage,
        spec: Any,
    ) -> Any:
        index = int(seed_lineage.path[-1].split("=")[1])
        if index == 1:
            raise ContractError(
                "synthetic planner failure",
                code="synthetic_planner_failure",
                field="planner",
            )
        observed = original_planner(prior, seed_lineage=seed_lineage, spec=spec)
        if index == 0:
            return replace(
                observed,
                status=ActiveNullPlanStatus.NOT_ESTIMABLE,
                reason_code=ActiveNullReasonCode.REASSIGNMENT_INFEASIBLE,
                source_overlap_count=None,
                source_overlap_fraction=None,
                reassigned_links=(),
            )
        return observed

    def failed_runner(*args: Any, **kwargs: Any) -> CrossFitArtifacts:
        nonlocal runner_calls
        runner_calls += 1
        raise ContractError(
            "synthetic child failure",
            code="synthetic_child_failure",
            field="runner",
        )

    monkeypatch.setattr(rerun_module, "plan_active_null_target_prior", planner)
    monkeypatch.setattr(rerun_module, "_run_subject_crossfit", failed_runner)

    result = run_active_null_reruns(
        fixture.adata,
        fixture.config,
        fixture.bundle,
        fixture.prior,
        fixture.point_artifacts,
        fixture.universe,
        n_plans=3,
    )

    assert result.status is ActiveNullRerunResultStatus.PARTIAL
    assert runner_calls == 1
    assert tuple(record.status for record in result.records) == (
        ActiveNullRerunStatus.NOT_ESTIMABLE,
        ActiveNullRerunStatus.FAILED,
        ActiveNullRerunStatus.FAILED,
    )
    assert result.records[0].reason_code == (
        ActiveNullReasonCode.REASSIGNMENT_INFEASIBLE.value
    )
    assert result.records[1].reason_code == "synthetic_planner_failure"
    assert result.records[1].planner_status is None
    assert result.records[2].reason_code == "synthetic_child_failure"
    assert result.records[2].planner_status is ActiveNullPlanStatus.OBSERVED
    assert len({request.plan_id for request in result.requests}) == 3
    assert len(result.distribution.null_records) == 6
    for request, expected_status in zip(
        result.requests,
        (
            ActiveEdgeScoreStatus.NOT_ESTIMABLE,
            ActiveEdgeScoreStatus.FAILED,
            ActiveEdgeScoreStatus.FAILED,
        ),
        strict=True,
    ):
        column = tuple(
            record
            for record in result.distribution.null_records
            if record.plan_id == request.plan_id
        )
        assert len(column) == 2
        assert {record.status for record in column} == {expected_status}
        assert {record.null_rerun_record_id for record in column} == {
            result.records[request.plan_index].null_rerun_record_id
        }


def test_child_fold_drift_fails_the_entire_preplanned_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(monkeypatch)
    drifted_fold_plan = replace(
        fixture.fold_plan,
        plan_id="drifted-null-fold-plan",
    )

    def runner(
        snapshot: Any,
        config: CrychicConfig,
        resource_bundle: ResourceBundle,
        target_prior: TargetPrior,
        *,
        spec: CrossFitSpec,
    ) -> CrossFitArtifacts:
        del snapshot, config, resource_bundle, spec
        return _child_for_prior(
            fixture,
            target_prior,
            crossfit_id="drifted-child",
            fold_plan=drifted_fold_plan,
        )

    monkeypatch.setattr(rerun_module, "_run_subject_crossfit", runner)

    result = run_active_null_reruns(
        fixture.adata,
        fixture.config,
        fixture.bundle,
        fixture.prior,
        fixture.point_artifacts,
        fixture.universe,
        n_plans=1,
    )

    assert result.status is ActiveNullRerunResultStatus.FAILED
    assert result.records[0].reason_code == "active_null_train_test_leakage"
    assert len(result.distribution.null_records) == 2
    assert {record.status for record in result.distribution.null_records} == {
        ActiveEdgeScoreStatus.FAILED
    }


def test_child_receiver_axis_drift_fails_the_preplanned_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(monkeypatch)

    def runner(
        snapshot: Any,
        config: CrychicConfig,
        resource_bundle: ResourceBundle,
        target_prior: TargetPrior,
        *,
        spec: CrossFitSpec,
    ) -> CrossFitArtifacts:
        del snapshot, config, resource_bundle, spec
        child = _child_for_prior(
            fixture,
            target_prior,
            crossfit_id="receiver-drifted-child",
        )
        object.__setattr__(
            child,
            "receiver_universe",
            SimpleNamespace(
                universe_id="drifted-receiver-universe",
                receiver_axis_id="drifted-receiver-axis",
                receiver_ids=("Receiver",),
            ),
        )
        return child

    monkeypatch.setattr(rerun_module, "_run_subject_crossfit", runner)

    result = run_active_null_reruns(
        fixture.adata,
        fixture.config,
        fixture.bundle,
        fixture.prior,
        fixture.point_artifacts,
        fixture.universe,
        n_plans=1,
    )

    assert result.status is ActiveNullRerunResultStatus.FAILED
    assert result.records[0].reason_code == (
        "active_null_receiver_universe_mismatch"
    )
    assert {record.status for record in result.distribution.null_records} == {
        ActiveEdgeScoreStatus.FAILED
    }


def test_source_resource_mismatch_is_rejected_before_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(monkeypatch)
    planner_called = False

    def planner(*args: Any, **kwargs: Any) -> Any:
        nonlocal planner_called
        planner_called = True
        raise AssertionError("planner must not run")

    monkeypatch.setattr(rerun_module, "plan_active_null_target_prior", planner)

    with pytest.raises(ContractError) as error:
        run_active_null_reruns(
            fixture.adata,
            fixture.config,
            _bundle(manifest_digest="different-resource-manifest"),
            fixture.prior,
            fixture.point_artifacts,
            fixture.universe,
            n_plans=1,
        )

    assert error.value.details.code == "active_null_source_mismatch"
    assert planner_called is False


@pytest.mark.parametrize("n_plans", (0, -1, True))
def test_null_plan_count_requires_one_but_not_two_hundred(
    monkeypatch: pytest.MonkeyPatch,
    n_plans: Any,
) -> None:
    fixture = _fixture(monkeypatch)

    with pytest.raises(ValueError, match="n_plans"):
        run_active_null_reruns(
            fixture.adata,
            fixture.config,
            fixture.bundle,
            fixture.prior,
            fixture.point_artifacts,
            fixture.universe,
            n_plans=n_plans,
        )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "n_jobs", (0, -1, True, 1.5)
)
def test_active_null_n_jobs_requires_a_positive_integer_before_planning(
    monkeypatch: pytest.MonkeyPatch,
    n_jobs: Any,
) -> None:
    fixture = _fixture(monkeypatch)
    planner_called = False

    def planner(*args: Any, **kwargs: Any) -> Any:
        nonlocal planner_called
        planner_called = True
        raise AssertionError("planner must not run")

    monkeypatch.setattr(rerun_module, "plan_active_null_target_prior", planner)
    with pytest.raises(ValueError, match="n_jobs"):
        run_active_null_reruns(
            fixture.adata,
            fixture.config,
            fixture.bundle,
            fixture.prior,
            fixture.point_artifacts,
            fixture.universe,
            n_plans=2,
            n_jobs=n_jobs,
        )
    assert planner_called is False


def test_bounded_parallel_reruns_are_exactly_serial_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(monkeypatch)
    original_planner = plan_active_null_target_prior
    cached_plans: dict[int, Any] = {}
    cache_lock = threading.Lock()
    tracker_lock = threading.Lock()
    delay_enabled = False
    inflight = 0
    maximum_inflight = 0

    def planner(
        prior: TargetPrior,
        *,
        seed_lineage: SeedLineage,
        spec: Any,
    ) -> Any:
        nonlocal inflight, maximum_inflight
        key = seed_lineage.seed
        with cache_lock:
            planned = cached_plans.get(key)
        if planned is None:
            planned = original_planner(
                prior,
                seed_lineage=seed_lineage,
                spec=spec,
            )
            with cache_lock:
                cached_plans[key] = planned
        index = int(seed_lineage.path[-1].split("=")[1])
        if delay_enabled:
            with tracker_lock:
                inflight += 1
                maximum_inflight = max(maximum_inflight, inflight)
            try:
                time.sleep(0.03 * (4 - index))
            finally:
                with tracker_lock:
                    inflight -= 1
        if index == 1:
            raise ContractError(
                "synthetic planner failure",
                code="synthetic_parallel_planner_failure",
                field="planner",
            )
        if index == 2:
            return replace(
                planned,
                status=ActiveNullPlanStatus.NOT_ESTIMABLE,
                reason_code=ActiveNullReasonCode.REASSIGNMENT_INFEASIBLE,
                source_overlap_count=None,
                source_overlap_fraction=None,
                reassigned_links=(),
            )
        return planned

    def runner(
        snapshot: Any,
        config: CrychicConfig,
        resource_bundle: ResourceBundle,
        target_prior: TargetPrior,
        *,
        spec: CrossFitSpec,
    ) -> CrossFitArtifacts:
        del snapshot, config, resource_bundle, spec
        prior_id = rerun_module._target_prior_content_id(target_prior)
        return _child_for_prior(
            fixture,
            target_prior,
            crossfit_id=f"null-child-{prior_id}",
        )

    def adapt(
        child: CrossFitArtifacts,
        frozen: FrozenActiveEdgeUniverse,
    ) -> CrossFitActiveEdgePointRecords:
        return _score_collection(
            child.crossfit_id,
            frozen,
            source_collection_id=f"score-{child.crossfit_id}",
            repeat_id=fixture.spec.repeat_id,
        )

    monkeypatch.setattr(rerun_module, "plan_active_null_target_prior", planner)
    monkeypatch.setattr(rerun_module, "_run_subject_crossfit", runner)
    monkeypatch.setattr(
        rerun_module,
        "adapt_crossfit_active_edge_records_against_universe",
        adapt,
    )

    # Warm the deterministic plan cache so the timing isolates orchestration.
    run_active_null_reruns(
        fixture.adata,
        fixture.config,
        fixture.bundle,
        fixture.prior,
        fixture.point_artifacts,
        fixture.universe,
        n_plans=4,
    )
    delay_enabled = True

    serial_start = time.perf_counter()
    serial = run_active_null_reruns(
        fixture.adata,
        fixture.config,
        fixture.bundle,
        fixture.prior,
        fixture.point_artifacts,
        fixture.universe,
        n_plans=4,
        n_jobs=1,
    )
    serial_seconds = time.perf_counter() - serial_start
    with tracker_lock:
        maximum_inflight = 0

    parallel_start = time.perf_counter()
    parallel = run_active_null_reruns(
        fixture.adata,
        fixture.config,
        fixture.bundle,
        fixture.prior,
        fixture.point_artifacts,
        fixture.universe,
        n_plans=4,
        n_jobs=10,
    )
    parallel_seconds = time.perf_counter() - parallel_start

    assert serial.result_id == parallel.result_id
    assert serial.active_null_id == parallel.active_null_id
    assert tuple(item.plan_id for item in serial.requests) == tuple(
        item.plan_id for item in parallel.requests
    )
    assert tuple(item.null_rerun_record_id for item in serial.records) == tuple(
        item.null_rerun_record_id for item in parallel.records
    )
    assert serial.distribution.distribution_id == (
        parallel.distribution.distribution_id
    )
    assert tuple(item.record_id for item in serial.distribution.null_records) == tuple(
        item.record_id for item in parallel.distribution.null_records
    )
    assert tuple(item.status for item in parallel.records) == (
        ActiveNullRerunStatus.SUCCEEDED,
        ActiveNullRerunStatus.FAILED,
        ActiveNullRerunStatus.NOT_ESTIMABLE,
        ActiveNullRerunStatus.SUCCEEDED,
    )
    assert parallel.records[1].reason_code == "synthetic_parallel_planner_failure"
    assert serial.requested_n_jobs == serial.effective_n_jobs == 1
    assert parallel.requested_n_jobs == 10
    assert parallel.effective_n_jobs == 4
    assert serial.execution_metadata_id != parallel.execution_metadata_id
    assert maximum_inflight == 4
    assert serial.to_manifest()["execution_backend"] == "serial_v1"
    assert parallel.to_manifest()["execution_backend"] == (
        "bounded_shared_snapshot_thread_pool_v1"
    )
    assert parallel.to_manifest()["execution_metadata_id"] == (
        parallel.execution_metadata_id
    )
    assert parallel_seconds < serial_seconds * 0.75
    print(
        "active-null scheduler smoke: "
        f"serial={serial_seconds:.3f}s parallel={parallel_seconds:.3f}s "
        f"speedup={serial_seconds / parallel_seconds:.2f}x"
    )

    scientific_result_id = parallel.result_id
    object.__setattr__(parallel, "requested_n_jobs", 2)
    object.__setattr__(parallel, "effective_n_jobs", 2)
    with pytest.raises(ContractError) as error:
        parallel._require_intact()
    assert error.value.details.code == "active_null_rerun_result_integrity_violation"
    assert parallel.result_id == scientific_result_id
