from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from crychic.scoring import (
    SCORING_COLLECTION_DERIVED_PLAN_STATUS,
    SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
    SCORING_COLLECTION_EXTENSION_VERSION,
    SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
    PlannedScoringCollectionManifest,
    ReceiverScoringFunctionalManifest,
    ScoringCollectionDocument,
    ScoringCollectionManifest,
)

ROOT = Path(__file__).parents[3]


def _registered(
    receiver: str,
    suffix: str,
    *,
    universe: str = "universe-1",
    support_id: str | None = None,
    contract_version: str = SCORING_COLLECTION_EXTENSION_VERSION,
) -> ReceiverScoringFunctionalManifest:
    support = (
        {
            "receiver_training_support_id": support_id or f"support-{suffix}",
            "receiver_training_support_status": "observed",
        }
        if contract_version == SCORING_COLLECTION_EXTENSION_VERSION
        else {}
    )
    return ReceiverScoringFunctionalManifest.registered(
        receiver=receiver,
        receiver_family_model_id=f"family-{suffix}",
        receiver_incremental_model_id=f"incremental-{suffix}",
        filter_universe_id=universe,
        scoring_functional_id=f"functional-{suffix}",
        score_version="family-common-v3",
        functional_status="observed",
        contract_version=contract_version,
        **support,
    )


def _collection(
    *,
    contrast: str = "stim_vs_control",
    repeat_id: str = "repeat-0",
    fold_id: str = "fold-0",
    universe: str = "universe-1",
    contract_version: str = SCORING_COLLECTION_EXTENSION_VERSION,
) -> ScoringCollectionManifest:
    suffix = f"{contrast}-{repeat_id}-{fold_id}"
    return ScoringCollectionManifest.planned_receiver_registry(
        contrast=contrast,
        repeat_id=repeat_id,
        fold_id=fold_id,
        planned_receivers=("T cell", "B cell"),
        children=(
            _registered(
                "B cell",
                f"b-{suffix}",
                universe=universe,
                support_id=f"support-b-{repeat_id}-{fold_id}",
                contract_version=contract_version,
            ),
            ReceiverScoringFunctionalManifest.not_produced(
                receiver="T cell",
                receiver_family_model_id=f"family-t-{suffix}",
                receiver_incremental_model_id=f"incremental-t-{suffix}",
                filter_universe_id=universe,
                reason_code="family_common_not_requested",
                contract_version=contract_version,
                **(
                    {
                        "receiver_training_support_id": (
                            f"support-t-{repeat_id}-{fold_id}"
                        ),
                        "receiver_training_support_status": "observed",
                    }
                    if contract_version == SCORING_COLLECTION_EXTENSION_VERSION
                    else {}
                ),
            ),
        ),
        contract_version=contract_version,
    )


def _registry(
    collections: tuple[ScoringCollectionManifest, ...] | None = None,
) -> ScoringCollectionDocument:
    resolved = collections or (_collection(),)
    plans = tuple(
        PlannedScoringCollectionManifest.from_collection(
            collection,
            filter_universe_id="universe-1",
        )
        for collection in resolved
    )
    return ScoringCollectionDocument(
        collections=resolved,
        planned_collections=plans,
    )


def test_v4_registry_round_trip_has_exact_authoritative_planned_coverage() -> None:
    document = _registry()
    collection = document.collections[0]

    assert document.extension_schema_version == SCORING_COLLECTION_EXTENSION_VERSION
    assert document.collection_kind == "authoritative_planned_receiver_registry"
    assert document.planning_status == "producer_declared_complete"
    assert document.is_authoritative_registry
    assert document.registry_id
    assert document.planned_collections is not None
    assert document.planned_collections[0].planned_receivers == ("B cell", "T cell")
    assert document.planned_collections[0].filter_universe_id == "universe-1"
    assert collection.planned_receivers == ("B cell", "T cell")
    assert collection.emitted_receivers == ()
    assert collection.composition_status == "planned_receiver_exact"
    assert collection.comparability_scope == "within_receiver_across_contexts_only"
    assert collection.row_union_policy == "cross_receiver_row_union_forbidden"
    assert not collection.common_functional_across_receivers
    assert {child.registry_status for child in collection.children} == {
        "functional_registered",
        "functional_not_produced",
    }

    serialized = document.to_dict()
    schema = json.loads(
        (ROOT / "schemas" / "scoring_collections_v4.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(serialized)
    assert ScoringCollectionDocument.from_dict(serialized) == document


def test_v4_registry_ids_are_invariant_to_all_declaration_row_orders() -> None:
    collections = tuple(
        _collection(contrast=contrast, fold_id=fold_id)
        for contrast in ("stim_vs_control", "treated_vs_control")
        for fold_id in ("fold-0", "fold-1")
    )
    first = _registry(collections)
    reversed_collections = tuple(reversed(collections))
    reversed_plans = tuple(
        PlannedScoringCollectionManifest(
            contrast=collection.contrast,
            repeat_id=collection.repeat_id,
            fold_id=collection.fold_id,
            planned_receivers=tuple(reversed(collection.planned_receivers)),
            filter_universe_id="universe-1",
        )
        for collection in reversed_collections
    )
    second = ScoringCollectionDocument(
        collections=reversed_collections,
        planned_collections=tuple(reversed(reversed_plans)),
    )

    assert second.registry_id == first.registry_id
    assert second.to_dict() == first.to_dict()


def test_v4_registry_rejects_incomplete_opportunities_and_universe_drift() -> None:
    incomplete = tuple(
        _collection(contrast=contrast, fold_id=fold_id)
        for contrast, fold_id in (
            ("stim_vs_control", "fold-0"),
            ("stim_vs_control", "fold-1"),
            ("treated_vs_control", "fold-0"),
        )
    )
    with pytest.raises(ValueError, match="complete contrast by repeat/fold"):
        _registry(incomplete)

    first = _collection(contrast="stim_vs_control")
    drifted = _collection(
        contrast="treated_vs_control",
        universe="universe-2",
    )
    plans = (
        PlannedScoringCollectionManifest.from_collection(
            first, filter_universe_id="universe-1"
        ),
        PlannedScoringCollectionManifest.from_collection(
            drifted, filter_universe_id="universe-2"
        ),
    )
    with pytest.raises(ValueError, match="filter universe must be exact"):
        ScoringCollectionDocument(
            collections=(first, drifted),
            planned_collections=plans,
        )


def test_v4_registry_rejects_missing_planned_receiver_and_function_reuse() -> None:
    child = _registered("B cell", "shared")
    with pytest.raises(ValueError, match="planned_receivers must exactly match"):
        ScoringCollectionManifest.planned_receiver_registry(
            contrast="stim_vs_control",
            repeat_id="repeat-0",
            fold_id="fold-0",
            planned_receivers=("B cell", "T cell"),
            children=(child,),
        )

    reused = ReceiverScoringFunctionalManifest.registered(
        receiver="T cell",
        receiver_family_model_id="family-t",
        receiver_incremental_model_id="incremental-t",
        filter_universe_id="universe-1",
        scoring_functional_id=child.scoring_functional_id or "",
        score_version="family-common-v3",
        functional_status="observed",
        receiver_training_support_id="support-reused",
        receiver_training_support_status="observed",
    )
    with pytest.raises(ValueError, match="unique scoring_functional_id"):
        ScoringCollectionManifest.planned_receiver_registry(
            contrast="stim_vs_control",
            repeat_id="repeat-0",
            fold_id="fold-0",
            planned_receivers=("B cell", "T cell"),
            children=(child, reused),
        )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("path", "replacement", "message"),
    [
        (
            ("planned_collections", 0, "filter_universe_id"),
            "forged-universe",
            "planned scoring collection ID",
        ),
        (
            ("collections", 0, "children", 0, "receiver_family_model_id"),
            "forged-model",
            "receiver child manifest ID",
        ),
        (
            ("collections", 0, "children", 0, "score_version"),
            "forged-version",
            "receiver child manifest ID",
        ),
    ],
)
def test_v4_registry_rejects_tampered_plan_model_and_version(
    path: tuple[str | int, ...],
    replacement: str,
    message: str,
) -> None:
    poisoned = copy.deepcopy(_registry().to_dict())
    target: object = poisoned
    for segment in path[:-1]:
        target = target[segment]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        ScoringCollectionDocument.from_dict(poisoned)


def test_v2_round_trip_remains_derived_and_never_claims_v3_authority() -> None:
    collection = _collection(
        contract_version=SCORING_COLLECTION_DERIVED_REGISTRY_VERSION
    )
    document = ScoringCollectionDocument(collections=(collection,))

    assert document.extension_schema_version == "2.0.0"
    assert document.collection_kind == "planned_receiver_registry"
    assert document.planning_status == SCORING_COLLECTION_DERIVED_PLAN_STATUS
    assert not document.is_authoritative_registry
    assert document.planned_collections is not None
    assert document.planned_collections[0].provenance_status == (
        SCORING_COLLECTION_DERIVED_PLAN_STATUS
    )
    serialized = document.to_dict()
    assert "planning_status" not in serialized
    assert "planned_collections" not in serialized
    assert "is_authoritative_registry" not in serialized
    schema = json.loads(
        (ROOT / "schemas" / "scoring_collections_v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(serialized)
    observed = ScoringCollectionDocument.from_dict(serialized)
    assert observed == document
    assert not observed.is_authoritative_registry


def test_v1_v2_v3_and_v4_schemas_form_explicit_migration_boundaries() -> None:
    v4 = _registry().to_dict()
    v1_schema = json.loads(
        (ROOT / "schemas" / "scoring_collections.schema.json").read_text(
            encoding="utf-8"
        )
    )
    v2_schema = json.loads(
        (ROOT / "schemas" / "scoring_collections_v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    v3_schema = json.loads(
        (ROOT / "schemas" / "scoring_collections_v3.schema.json").read_text(
            encoding="utf-8"
        )
    )
    v4_schema = json.loads(
        (ROOT / "schemas" / "scoring_collections_v4.schema.json").read_text(
            encoding="utf-8"
        )
    )

    with pytest.raises(ValidationError):
        Draft202012Validator(v1_schema).validate(v4)
    with pytest.raises(ValidationError):
        Draft202012Validator(v2_schema).validate(v4)
    with pytest.raises(ValidationError):
        Draft202012Validator(v3_schema).validate(v4)
    poisoned = dict(v4)
    poisoned["extension_schema_version"] = "2.0.0"
    with pytest.raises(ValueError, match="fields are invalid"):
        ScoringCollectionDocument.from_dict(poisoned)
    Draft202012Validator(v4_schema).validate(v4)


def test_v4_outer_training_absence_has_typed_support_and_no_model_or_function() -> None:
    observed = _registered("B cell", "b-absent")
    absent = ReceiverScoringFunctionalManifest.training_not_estimable(
        receiver="T cell",
        filter_universe_id="universe-1",
        receiver_training_support_id="support-t-absent",
    )
    collection = ScoringCollectionManifest.planned_receiver_registry(
        contrast="stim_vs_control",
        repeat_id="repeat-0",
        fold_id="fold-0",
        planned_receivers=("B cell", "T cell"),
        children=(observed, absent),
    )
    document = _registry((collection,))

    child = document.collections[0].children[1]
    assert child.receiver_training_support_status == "not_estimable"
    assert child.receiver_training_support_reason_code == (
        "receiver_absent_in_outer_training"
    )
    assert child.reason_code == "receiver_absent_in_outer_training"
    assert child.receiver_family_model_id is None
    assert child.receiver_incremental_model_id is None
    assert child.scoring_functional_id is None
    assert child.score_version is None
    assert child.functional_status is None
    assert ScoringCollectionDocument.from_dict(document.to_dict()) == document


def test_v4_rejects_cross_fold_axis_drift_and_cross_contrast_support_drift() -> None:
    first = _collection(contrast="stim_vs_control", fold_id="fold-0")
    drifted_axis = ScoringCollectionManifest.planned_receiver_registry(
        contrast="stim_vs_control",
        repeat_id="repeat-0",
        fold_id="fold-1",
        planned_receivers=("B cell",),
        children=(_registered("B cell", "b-fold-1"),),
    )
    with pytest.raises(ValueError, match="exact across every fold"):
        _registry((first, drifted_axis))

    second = ScoringCollectionManifest.planned_receiver_registry(
        contrast="treated_vs_control",
        repeat_id="repeat-0",
        fold_id="fold-0",
        planned_receivers=("B cell", "T cell"),
        children=(
            _registered(
                "B cell",
                "b-treated-fold-0",
                support_id="support-poisoned",
            ),
            ReceiverScoringFunctionalManifest.not_produced(
                receiver="T cell",
                receiver_family_model_id="family-t-treated-fold-0",
                receiver_incremental_model_id="incremental-t-treated-fold-0",
                filter_universe_id="universe-1",
                reason_code="family_common_not_requested",
                receiver_training_support_id="support-t-repeat-0-fold-0",
                receiver_training_support_status="observed",
            ),
        ),
    )
    with pytest.raises(ValueError, match="support must be exact"):
        _registry((first, second))


def test_v3_authoritative_registry_remains_readable_without_v4_support_fields() -> None:
    collection = _collection(
        contract_version=SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION
    )
    document = _registry((collection,))
    serialized = document.to_dict()

    assert serialized["extension_schema_version"] == "3.0.0"
    assert "receiver_training_support_id" not in serialized["collections"][0][
        "children"
    ][0]  # type: ignore[index]
    schema = json.loads(
        (ROOT / "schemas" / "scoring_collections_v3.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(serialized)
    assert ScoringCollectionDocument.from_dict(serialized) == document
