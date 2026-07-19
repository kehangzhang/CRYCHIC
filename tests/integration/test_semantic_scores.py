from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from tests.integration.test_subject_crossfit import (
    _adata,
    _bundle,
    _config,
    _directional_spec,
    _mixed_adata,
    _persistable_run,
    _persistable_spec,
    _prior,
    _spec,
    _trusted_target_resource,
)

from crychic.core import ContractError, canonical_digest, canonical_json, stable_id
from crychic.results.errors import ResultValidationError
from crychic.workflow import (
    CrossFitArtifacts,
    CrossFitResult,
    SemanticScoreCollection,
    build_crossfit_semantic_scores,
    run_subject_crossfit,
    write_crossfit_result,
)
from crychic.workflow.crossfit_persistence import (
    CROSSFIT_INTEGRATED_LR_QUERY_COLUMNS,
    CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE,
    CROSSFIT_SEMANTIC_TABLE_NAMES,
    _sha256_file,
    _validate_source_response_precision_lineage,
)
from crychic.workflow.semantic_scores import _table_digest


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def untuned_artifacts() -> CrossFitArtifacts:
    return run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def untuned_scores(untuned_artifacts: CrossFitArtifacts) -> SemanticScoreCollection:
    return build_crossfit_semantic_scores(untuned_artifacts)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def tuned_artifacts() -> CrossFitArtifacts:
    return _persistable_run()


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def tuned_scores(tuned_artifacts: CrossFitArtifacts) -> SemanticScoreCollection:
    return build_crossfit_semantic_scores(tuned_artifacts)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def tuned_result(
    tmp_path_factory: pytest.TempPathFactory,
    tuned_artifacts: CrossFitArtifacts,
) -> CrossFitResult:
    return write_crossfit_result(
        tuned_artifacts,
        tmp_path_factory.mktemp("semantic-v8-tuned") / "result",
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def untuned_result(
    tmp_path_factory: pytest.TempPathFactory,
    untuned_artifacts: CrossFitArtifacts,
) -> CrossFitResult:
    return write_crossfit_result(
        untuned_artifacts,
        tmp_path_factory.mktemp("semantic-v8-untuned") / "result",
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def mixed_directional_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> CrossFitResult:
    artifacts = run_subject_crossfit(
        _mixed_adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )
    return write_crossfit_result(
        artifacts,
        tmp_path_factory.mktemp("semantic-v8-mixed-directional") / "result",
    )


def test_untuned_collection_keeps_independent_availability_and_program_views(
    untuned_scores: SemanticScoreCollection,
) -> None:
    manifest = untuned_scores.view_manifest.set_index("semantic_output")

    assert not untuned_scores.availability_score.empty
    assert not untuned_scores.receiver_program_score.empty
    assert untuned_scores.integrated_lr_score.empty
    assert untuned_scores.differential_effect.empty
    assert manifest.loc["availability_score", "status"] == "produced"
    assert manifest.loc["receiver_program_score", "status"] == "produced"
    assert manifest.loc["integrated_lr_score", "status"] == "not_produced"
    assert manifest.loc["differential_effect", "status"] == "not_produced"
    assert set(
        manifest.loc[["integrated_lr_score", "differential_effect"], "reason_code"]
    ) == {"family_common_scoring_not_requested_without_penalty_tuning"}
    assert untuned_scores.formal_inference_allowed is False


def test_four_views_have_distinct_grains_and_bound_source_lineage(
    untuned_scores: SemanticScoreCollection,
) -> None:
    availability = untuned_scores.availability_score
    programs = untuned_scores.receiver_program_score

    assert set(availability["mode"]) == {"state", "ecosystem"}
    assert not availability.duplicated(
        [
            "fold_id",
            "sample_id",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
        ]
    ).any()
    assert not programs.duplicated(
        [
            "fold_id",
            "contrast_id",
            "sample_id",
            "receiver",
            "family_id",
        ]
    ).any()
    assert availability["availability_application_id"].notna().all()
    assert availability["filter_universe_id"].notna().all()
    assert programs["receiver_program_training_artifact_id"].notna().all()
    assert programs["receiver_program_application_id"].notna().all()
    assert availability["source_table_digest"].notna().all()
    assert programs["source_table_digest"].notna().all()
    assert not availability["formal_inference_allowed"].any()
    assert not programs["formal_inference_allowed"].any()
    assert {
        "p_value",
        "q_value",
        "posterior_probability",
        "communication_probability",
    }.isdisjoint(availability.columns) and {
        "p_value",
        "q_value",
        "posterior_probability",
        "communication_probability",
    }.isdisjoint(programs.columns)


def test_tuned_collection_reuses_lr_member_and_subject_effect_producers(
    tuned_scores: SemanticScoreCollection,
) -> None:
    integrated = tuned_scores.integrated_lr_score
    differential = tuned_scores.differential_effect
    source_artifacts = tuned_scores._source_artifacts

    expected_member_rows = sum(
        len(application.member_scores)
        for fold in source_artifacts.folds
        for application in fold.family_common_applications
    )
    expected_effect_rows = sum(
        len(application.subject_differential)
        for fold in source_artifacts.folds
        for application in fold.family_common_applications
    )
    assert len(integrated) == expected_member_rows
    assert len(differential) == expected_effect_rows
    assert set(integrated["source_component"]) == {"sender_unresolved_strength"}
    assert integrated["interaction_id"].notna().all()
    assert integrated["family_common_binding_id"].notna().all()
    assert differential["family_common_binding_id"].notna().all()
    assert set(integrated["status"]).issubset(
        {"observed", "structural_zero", "not_estimable"}
    )
    assert set(differential["status"]).issubset(
        {"observed", "structural_zero", "not_estimable"}
    )
    assert not integrated["formal_inference_allowed"].any()
    assert not differential["formal_inference_allowed"].any()


def test_collection_tables_are_defensive_and_private_mutation_is_rejected(
    untuned_artifacts: CrossFitArtifacts,
) -> None:
    collection = build_crossfit_semantic_scores(untuned_artifacts)
    public = collection.availability_score
    public.loc[:, "availability_score"] = 999.0
    assert not collection.availability_score["availability_score"].eq(999.0).any()

    private = collection._availability_score
    row = private.index[0]
    original = float(private.loc[row, "availability_score"])
    private.loc[row, "availability_score"] = min(original + 0.01, 1.0)
    with pytest.raises(ContractError, match="integrity") as error:
        _ = collection.availability_score
    assert error.value.details.code == ("semantic_score_collection_integrity_violation")


def test_collection_and_builder_reject_caller_construction() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        SemanticScoreCollection()
    with pytest.raises(TypeError, match="producer-owned CrossFitArtifacts"):
        build_crossfit_semantic_scores(object())


def test_named_table_accessor_is_exact_and_rejects_unknown_output(
    untuned_scores: SemanticScoreCollection,
) -> None:
    pd.testing.assert_frame_equal(
        untuned_scores.table("receiver_program_score"),
        untuned_scores.receiver_program_score,
    )
    with pytest.raises(ValueError, match="semantic_output"):
        untuned_scores.table("comm_strength")


def test_batch_table_accessor_validates_once_and_returns_defensive_copies(
    untuned_scores: SemanticScoreCollection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = SemanticScoreCollection._require_intact
    validation_calls = 0

    def counted_require_intact(collection: SemanticScoreCollection) -> None:
        nonlocal validation_calls
        validation_calls += 1
        original(collection)

    monkeypatch.setattr(
        SemanticScoreCollection,
        "_require_intact",
        counted_require_intact,
    )
    observed = untuned_scores.tables()

    assert validation_calls == 1
    private = untuned_scores._tables()
    assert observed.keys() == private.keys()
    for name, table in observed.items():
        assert table is not private[name]
        pd.testing.assert_frame_equal(table, private[name])


def test_v8_roundtrip_replays_all_tuned_semantic_views_exactly(
    tuned_result: CrossFitResult,
    tuned_scores: SemanticScoreCollection,
) -> None:
    loaded = CrossFitResult.load(tuned_result.path)

    assert loaded.manifest["schema_version"] == "9.0.0"
    assert loaded.semantic_score_manifest["collection_id"] == tuned_scores.collection_id
    for semantic_output in (
        "availability_score",
        "receiver_program_score",
        "integrated_lr_score",
        "differential_effect",
    ):
        pd.testing.assert_frame_equal(
            loaded.read_semantic_score(semantic_output),
            tuned_scores.table(semantic_output),
        )
    assert tuple(loaded.query_integrated_lr_scores().columns) == (
        CROSSFIT_INTEGRATED_LR_QUERY_COLUMNS
    )


def test_v8_roundtrip_preserves_untuned_not_produced_views(
    untuned_result: CrossFitResult,
    untuned_scores: SemanticScoreCollection,
) -> None:
    loaded = CrossFitResult.load(untuned_result.path)
    status = {
        str(row[0]): (str(row[1]), cast(str | None, row[2]))
        for row in cast(
            list[list[object]],
            loaded.semantic_score_manifest["view_statuses"],
        )
    }

    pd.testing.assert_frame_equal(
        loaded.read_semantic_availability(), untuned_scores.availability_score
    )
    pd.testing.assert_frame_equal(
        loaded.read_semantic_receiver_programs(),
        untuned_scores.receiver_program_score,
    )
    assert loaded.read_semantic_integrated_lr_scores().empty
    assert loaded.read_semantic_differential_effects().empty
    assert status["integrated_lr_score"] == (
        "not_produced",
        "family_common_scoring_not_requested_without_penalty_tuning",
    )
    assert status["differential_effect"] == status["integrated_lr_score"]


def _rewrite_result_identity(root: Path, manifest: dict[str, object]) -> None:
    manifest["crossfit_result_id"] = stable_id(
        "crossfit_result",
        {key: value for key, value in manifest.items() if key != "crossfit_result_id"},
        schema_version="1",
    )
    (root / "crossfit_manifest.json").write_text(
        f"{canonical_json(manifest)}\n", encoding="utf-8"
    )
    status = {
        "schema_version": manifest["schema_version"],
        "status": "complete",
        "crossfit_result_id": manifest["crossfit_result_id"],
    }
    (root / "_status.json").write_text(
        f"{canonical_json(status)}\n",
        encoding="utf-8",
    )


def test_v8_semantic_replay_rejects_rehashed_row_tamper(
    tmp_path: Path,
    tuned_result: CrossFitResult,
) -> None:
    target = tmp_path / "tampered"
    shutil.copytree(tuned_result.path, target)
    manifest_path = target / "crossfit_manifest.json"
    manifest = cast(
        dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    records = cast(dict[str, dict[str, object]], manifest["tables"])
    table_path = target / cast(
        str, records[CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE]["filename"]
    )
    table = pd.read_parquet(table_path)
    row = int(table.index[0])
    table.loc[row, "interaction_id"] = f"{table.loc[row, 'interaction_id']}_tampered"
    table.to_parquet(table_path, index=False, engine="pyarrow", compression="zstd")
    records[CROSSFIT_SEMANTIC_INTEGRATED_LR_TABLE]["sha256"] = _sha256_file(table_path)
    tampered_digest = _table_digest("integrated_lr_score", table)
    collection = cast(dict[str, object], manifest["semantic_score_collection"])
    for item in cast(list[list[object]], collection["table_digests"]):
        if item[0] == "integrated_lr_score":
            item[1] = tampered_digest
    for view in cast(list[dict[str, object]], collection["views"]):
        if view["semantic_output"] == "integrated_lr_score":
            view["table_digest"] = tampered_digest
    collection["collection_id"] = stable_id(
        "crossfit_semantic_score_collection",
        {
            "crossfit_id": collection["crossfit_id"],
            "crossfit_spec_id": collection["crossfit_spec_id"],
            "formal_inference_allowed": False,
            "repeat_id": collection["repeat_id"],
            "table_digests": collection["table_digests"],
            "table_row_counts": collection["table_row_counts"],
            "view_statuses": collection["view_statuses"],
        },
        schema_version="1",
    )
    _rewrite_result_identity(target, manifest)

    with pytest.raises(ResultValidationError) as error:
        CrossFitResult.load(target)
    assert error.value.details.code == "crossfit_semantic_replay_mismatch"


def test_v7_bundle_remains_readable_without_semantic_tables(
    tmp_path: Path,
    tuned_result: CrossFitResult,
) -> None:
    target = tmp_path / "v7"
    shutil.copytree(tuned_result.path, target)
    manifest = cast(
        dict[str, object],
        json.loads((target / "crossfit_manifest.json").read_text(encoding="utf-8")),
    )
    records = cast(dict[str, dict[str, object]], manifest["tables"])
    for table_name in CROSSFIT_SEMANTIC_TABLE_NAMES:
        (target / cast(str, records.pop(table_name)["filename"])).unlink()
    manifest.pop("semantic_score_collection")
    source = cast(dict[str, object], manifest["source_crossfit_manifest"])
    for field_name in (
        "receiver_family_opportunity_universe_id",
        "family_axis_id",
        "receiver_family_opportunity_axis_id",
        "receiver_family_opportunity_universe",
    ):
        manifest.pop(field_name)
        source.pop(field_name)
    manifest["source_crossfit_manifest_digest"] = canonical_digest(source)
    manifest["schema_version"] = "7.0.0"
    _rewrite_result_identity(target, manifest)

    loaded = CrossFitResult.load(target)
    assert loaded.manifest["schema_version"] == "7.0.0"
    assert not loaded.read_components().empty
    assert tuple(loaded.query_integrated_lr_scores().columns) == (
        CROSSFIT_INTEGRATED_LR_QUERY_COLUMNS
    )
    with pytest.raises(KeyError, match="schema v8"):
        loaded.read_semantic_availability()


def test_v8_bundle_remains_readable_without_root_family_projection(
    tmp_path: Path,
    tuned_result: CrossFitResult,
) -> None:
    target = tmp_path / "v8"
    shutil.copytree(tuned_result.path, target)
    manifest = cast(
        dict[str, object],
        json.loads((target / "crossfit_manifest.json").read_text(encoding="utf-8")),
    )
    source = cast(dict[str, object], manifest["source_crossfit_manifest"])
    for field_name in (
        "receiver_family_opportunity_universe_id",
        "family_axis_id",
        "receiver_family_opportunity_axis_id",
        "receiver_family_opportunity_universe",
    ):
        manifest.pop(field_name)
        source.pop(field_name)
    manifest["source_crossfit_manifest_digest"] = canonical_digest(source)
    manifest["schema_version"] = "8.0.0"
    _rewrite_result_identity(target, manifest)

    loaded = CrossFitResult.load(target)

    assert loaded.manifest["schema_version"] == "8.0.0"
    assert not loaded.read_semantic_availability().empty


def test_v8_cr2_backend_and_partial_precision_support_are_auditable(
    mixed_directional_result: CrossFitResult,
) -> None:
    source = cast(
        dict[str, object],
        mixed_directional_result.manifest["source_crossfit_manifest"],
    )
    _validate_source_response_precision_lineage(source)
    records = [
        record
        for fold in cast(list[dict[str, object]], source["fold_artifacts"])
        for record in cast(
            list[dict[str, object]], fold["receiver_incremental_artifacts"]
        )
        if record["receiver"] == "Receiver"
    ]

    assert records
    for record in records:
        backend = cast(dict[str, object], record["response_backend_manifest"])
        precision_audit = cast(dict[str, object], record["precision_audit_manifest"])
        precision = cast(dict[str, object], precision_audit["provenance"])
        assert backend["artifact_kind"] == "repeated_cr2_fold_response_v1"
        assert backend["n_backend_eligible_features"] < backend["n_features"]
        assert (
            backend["n_precision_supported_features"]
            == precision["n_positive_features"]
        )
        assert precision["method"] == ("repeated_cr2_standardized_inverse_variance_v1")
        assert precision["lineage_mode"] == (
            "repeated_measures_cr2_fold_response_parented_v1"
        )
        assert precision_audit["numeric_values_persisted"] is False


def test_v8_cr2_backend_rejects_rehashed_feature_backend_tamper(
    mixed_directional_result: CrossFitResult,
) -> None:
    source = copy.deepcopy(
        cast(
            dict[str, object],
            mixed_directional_result.manifest["source_crossfit_manifest"],
        )
    )
    folds = cast(list[dict[str, object]], source["fold_artifacts"])
    records = cast(list[dict[str, object]], folds[0]["receiver_incremental_artifacts"])
    record = next(item for item in records if item["receiver"] == "Receiver")
    backend = cast(dict[str, object], record["response_backend_manifest"])
    diagnostics = cast(list[dict[str, object]], backend["feature_diagnostics"])
    diagnostics[0]["effect_id"] = "repeated_measures_feature_effect_poisoned"
    backend["feature_diagnostics_digest"] = stable_id(
        "crossfit_response_feature_diagnostics",
        {
            "feature_diagnostics": diagnostics,
            "feature_ids": backend["feature_ids"],
            "response_artifact_id": backend["response_artifact_id"],
        },
        schema_version="1",
        digest_length=64,
    )
    backend["response_backend_manifest_id"] = stable_id(
        "crossfit_response_backend_manifest",
        {
            key: value
            for key, value in backend.items()
            if key != "response_backend_manifest_id"
        },
        schema_version="1",
    )

    with pytest.raises(ValueError, match=r"feature status|backend"):
        _validate_source_response_precision_lineage(source)


def test_v8_persists_official_descriptive_cr2_without_formal_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = _trusted_target_resource(tmp_path, monkeypatch)
    spec = replace(
        _persistable_spec(),
        autonomous_program_resource=resource,
    )
    artifacts = run_subject_crossfit(
        _mixed_adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )
    applications = [
        application
        for fold in artifacts.folds
        for application in fold.receiver_incremental_applications
    ]
    assert applications
    observed_applications = [
        application
        for application in applications
        if application.official_incremental_status == "observed"
    ]
    assert observed_applications
    assert all(application.is_oof_certified for application in observed_applications)

    result = write_crossfit_result(artifacts, tmp_path / "official-cr2-v8")
    records = [
        record
        for fold in cast(
            list[dict[str, object]],
            cast(dict[str, object], result.manifest["source_crossfit_manifest"])[
                "fold_artifacts"
            ],
        )
        for record in cast(
            list[dict[str, object]], fold["receiver_incremental_artifacts"]
        )
    ]
    assert records
    observed_records = [
        record
        for record in records
        if record["official_incremental_status"] == "observed"
    ]
    assert len(observed_records) == len(observed_applications)
    assert all(
        cast(dict[str, object], record["response_backend_manifest"])["artifact_kind"]
        == "repeated_cr2_fold_response_v1"
        for record in observed_records
    )
    assert result.manifest["formal_inference_status"] == (
        "not_available_descriptive_only"
    )
