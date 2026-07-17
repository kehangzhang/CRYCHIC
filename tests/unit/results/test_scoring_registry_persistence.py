from __future__ import annotations

from pathlib import Path
from typing import Any

from crychic.results import CrychicResult, write_result
from crychic.scoring import (
    SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
    SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
    PlannedScoringCollectionManifest,
    ReceiverScoringFunctionalManifest,
    ScoringCollectionDocument,
    ScoringCollectionManifest,
    scoring_source_key_digest,
)


def test_v4_registry_persists_outer_training_absent_receiver(
    tmp_path: Path,
    result_payload: dict[str, Any],
) -> None:
    source = result_payload["tables"]["sample_scores"]
    source = source.loc[source["scoring_functional_id"].eq("functional-1")]
    emitted = ReceiverScoringFunctionalManifest.registered(
        receiver="B cell",
        receiver_family_model_id="family-b",
        receiver_incremental_model_id="incremental-b",
        filter_universe_id="universe-1",
        scoring_functional_id="functional-1",
        score_version="family-common-v3",
        functional_status="observed",
        receiver_training_support_id="support-b-fold-0",
        receiver_training_support_status="observed",
        source_score_key_digest=scoring_source_key_digest(source),
        source_score_row_count=len(source),
    )
    planned_only = ReceiverScoringFunctionalManifest.training_not_estimable(
        receiver="T cell",
        filter_universe_id="universe-1",
        receiver_training_support_id="support-t-fold-0",
    )
    collection = ScoringCollectionManifest.planned_receiver_registry(
        contrast="stim_vs_ctrl",
        repeat_id="repeat-0",
        fold_id="fold-0",
        planned_receivers=("B cell", "T cell"),
        children=(emitted, planned_only),
    )
    document = ScoringCollectionDocument(
        collections=(collection,),
        planned_collections=(
            PlannedScoringCollectionManifest.from_collection(
                collection,
                filter_universe_id="universe-1",
            ),
        ),
    )

    result = write_result(
        tmp_path / "result",
        **result_payload,
        scoring_collections=document,
    )

    extension = result.manifest["extensions"]["scoring_collections"]
    assert extension["extension_schema_version"] == "4.0.0"
    assert extension["schema"] == "scoring_collections_v4.schema.json"
    observed = result.read_scoring_collections()[0]
    assert observed.planned_receivers == ("B cell", "T cell")
    assert observed.emitted_receivers == ("B cell",)
    assert observed.source_score_row_count == len(source)
    absent = observed.children[1]
    assert absent.receiver_training_support_status == "not_estimable"
    assert absent.receiver_family_model_id is None
    assert absent.receiver_incremental_model_id is None
    assert absent.scoring_functional_id is None
    assert CrychicResult.load(result.path).read_scoring_collections() == (observed,)


def test_v2_registry_remains_readable_through_result_persistence(
    tmp_path: Path,
    result_payload: dict[str, Any],
) -> None:
    source = result_payload["tables"]["sample_scores"]
    source = source.loc[source["scoring_functional_id"].eq("functional-1")]
    child = ReceiverScoringFunctionalManifest.registered(
        receiver="B cell",
        receiver_family_model_id="family-b",
        receiver_incremental_model_id="incremental-b",
        filter_universe_id="universe-1",
        scoring_functional_id="functional-1",
        score_version="family-common-v2",
        functional_status="observed",
        source_score_key_digest=scoring_source_key_digest(source),
        source_score_row_count=len(source),
        contract_version=SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
    )
    collection = ScoringCollectionManifest.planned_receiver_registry(
        contrast="stim_vs_ctrl",
        repeat_id="repeat-0",
        fold_id="fold-0",
        planned_receivers=("B cell",),
        children=(child,),
        contract_version=SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
    )
    document = ScoringCollectionDocument(collections=(collection,))

    result = write_result(
        tmp_path / "v2-result",
        **result_payload,
        scoring_collections=document,
    )

    extension = result.manifest["extensions"]["scoring_collections"]
    assert extension["extension_schema_version"] == "2.0.0"
    assert extension["schema"] == "scoring_collections_v2.schema.json"
    observed = CrychicResult.load(result.path).read_scoring_collections()
    assert observed == (collection,)


def test_v3_registry_remains_readable_through_result_persistence(
    tmp_path: Path,
    result_payload: dict[str, Any],
) -> None:
    source = result_payload["tables"]["sample_scores"]
    source = source.loc[source["scoring_functional_id"].eq("functional-1")]
    child = ReceiverScoringFunctionalManifest.registered(
        receiver="B cell",
        receiver_family_model_id="family-b",
        receiver_incremental_model_id="incremental-b",
        filter_universe_id="universe-1",
        scoring_functional_id="functional-1",
        score_version="family-common-v3",
        functional_status="observed",
        source_score_key_digest=scoring_source_key_digest(source),
        source_score_row_count=len(source),
        contract_version=SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
    )
    collection = ScoringCollectionManifest.planned_receiver_registry(
        contrast="stim_vs_ctrl",
        repeat_id="repeat-0",
        fold_id="fold-0",
        planned_receivers=("B cell",),
        children=(child,),
        contract_version=SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
    )
    document = ScoringCollectionDocument(
        collections=(collection,),
        planned_collections=(
            PlannedScoringCollectionManifest.from_collection(
                collection,
                filter_universe_id="universe-1",
            ),
        ),
    )

    result = write_result(
        tmp_path / "v3-result",
        **result_payload,
        scoring_collections=document,
    )

    extension = result.manifest["extensions"]["scoring_collections"]
    assert extension["extension_schema_version"] == "3.0.0"
    assert extension["schema"] == "scoring_collections_v3.schema.json"
    assert CrychicResult.load(result.path).read_scoring_collections() == (collection,)
