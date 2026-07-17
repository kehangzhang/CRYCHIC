"""Integrity checks for the directional integrated-LR sidecar."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tests.integration.test_subject_crossfit import (
    _adata,
    _bundle,
    _config,
    _directional_spec,
    _persistable_spec,
    _prior,
)

from crychic.core import canonical_json, stable_id
from crychic.results.errors import (
    IncompleteResultError,
    ResultValidationError,
    ResultWriteError,
)
from crychic.workflow import (
    CrossFitResult,
    DirectionalIntegratedLRResult,
    build_directional_integrated_lr_scores,
    run_subject_crossfit,
    write_crossfit_result,
    write_directional_integrated_lr_result,
)
from crychic.workflow.directional_integrated_lr import _table_digest


@pytest.fixture(scope="module")
def persisted_directional_lr(tmp_path_factory: pytest.TempPathFactory):
    base = _persistable_spec()
    directional = _directional_spec()
    spec = replace(
        base,
        contrasts=directional.contrasts,
        directional_pairs=directional.directional_pairs,
        predeclared_receiver_ids=("Ghost", "Receiver", "Sender"),
        training_spec=replace(base.training_spec, sender_contrasts=None),
    )
    artifacts = run_subject_crossfit(
        _adata(tuple(f"p{i}" for i in range(1, 9))),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )
    pair = artifacts.spec.directional_pairs[0]
    collection = build_directional_integrated_lr_scores(
        artifacts, pair_spec_id=pair.pair_spec_id
    )
    root = tmp_path_factory.mktemp("directional-lr-persistence")
    parent = write_crossfit_result(artifacts, root / "parent")
    result = write_directional_integrated_lr_result(
        collection,
        root / "sidecar",
        source_crossfit_result=parent,
    )
    return root, parent, collection, result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(f"{canonical_json(value)}\n", encoding="utf-8")


def _copy_sidecar(
    tmp_path: Path, persisted_directional_lr: tuple[Any, ...]
) -> tuple[Path, CrossFitResult]:
    root, parent, _collection, _result = persisted_directional_lr
    target = tmp_path / "sidecar"
    shutil.copytree(root / "sidecar", target)
    return target, parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_artifact_identity(root: Path) -> None:
    manifest_path = root / "directional_integrated_lr_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["artifact_id"] = stable_id(
        "directional_integrated_lr_result",
        {key: value for key, value in manifest.items() if key != "artifact_id"},
        schema_version="1",
    )
    _write_json(manifest_path, manifest)
    status = _read_json(root / "_status.json")
    status["artifact_id"] = manifest["artifact_id"]
    _write_json(root / "_status.json", status)


def _rewrite_table(root: Path, table_name: str, table: pd.DataFrame) -> None:
    manifest_path = root / "directional_integrated_lr_manifest.json"
    manifest = _read_json(manifest_path)
    record = manifest["tables"][table_name]
    path = root / record["filename"]
    table.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        compression="zstd",
        row_group_size=131_072,
    )
    record["rows"] = len(table)
    record["sha256"] = _sha256(path)
    record["logical_digest"] = _table_digest(table)
    _write_json(manifest_path, manifest)
    _refresh_artifact_identity(root)


def test_directional_sidecar_roundtrip_ghost_and_defensive_queries(
    persisted_directional_lr,
) -> None:
    root, parent, collection, result = persisted_directional_lr
    loaded = DirectionalIntegratedLRResult.load(
        root / "sidecar", source_crossfit_result=parent.path
    )
    scores = loaded.read_scores()
    opportunities = loaded.read_opportunities()
    assert _table_digest(scores) == _table_digest(collection.scores)
    assert _table_digest(opportunities) == _table_digest(
        collection.opportunity_registry
    )
    assert scores["molecular_lr_equivalence_id"].notna().all()
    manifest = loaded.manifest
    source_universe = parent.manifest["source_crossfit_manifest"][
        "directional_lr_hypothesis_universe"
    ]
    assert (
        manifest["molecular_lr_equivalence_universe_id"]
        == (source_universe["molecular_lr_equivalence_universe_id"])
    )
    assert manifest["molecular_lr_axis_id"] == source_universe["molecular_lr_axis_id"]
    ghost = loaded.query_scores(receiver="Ghost")
    assert not ghost.empty
    assert set(ghost["status"]) == {"not_estimable"}
    assert (
        ghost[
            [
                "integrated_lr_score",
                "family_core_strength",
                "within_family_lr_weight",
                "directional_binding_id",
                "family_common_application_id",
                "incremental_application_id",
            ]
        ]
        .isna()
        .all(axis=None)
    )
    first = scores.iloc[0]
    selected = loaded.query_scores(
        fold_id=str(first["fold_id"]),
        receiver=str(first["receiver"]),
        channel_role=str(first["channel_role"]),
        sample_id=str(first["sample_id"]),
        interaction_id=str(first["interaction_id"]),
        molecular_lr_equivalence_id=str(first["molecular_lr_equivalence_id"]),
        mode=str(first["mode"]),
    )
    assert not selected.empty
    assert set(selected["channel_role"]) == {first["channel_role"]}
    scores.loc[0, "status"] = "forged"
    assert "forged" not in set(result.read_scores()["status"])


def test_directional_sidecar_rejects_wrong_parent(
    tmp_path: Path, persisted_directional_lr
) -> None:
    target, parent = _copy_sidecar(tmp_path, persisted_directional_lr)
    wrong_parent = tmp_path / "wrong-parent"
    shutil.copytree(parent.path, wrong_parent)
    status_path = wrong_parent / "_status.json"
    status = _read_json(status_path)
    status["crossfit_result_id"] = "wrong-parent-result"
    _write_json(status_path, status)
    with pytest.raises(ResultValidationError):
        DirectionalIntegratedLRResult.load(target, source_crossfit_result=wrong_parent)


@pytest.mark.parametrize(
    ("target_file", "field", "value"),
    (
        ("_status.json", "artifact_id", "forged-artifact"),
        (
            "directional_integrated_lr_manifest.json",
            "artifact_schema_version",
            "0.9.0",
        ),
    ),
)
def test_directional_sidecar_rejects_status_and_manifest_tamper(
    tmp_path: Path,
    persisted_directional_lr,
    target_file: str,
    field: str,
    value: object,
) -> None:
    target, parent = _copy_sidecar(tmp_path, persisted_directional_lr)
    path = target / target_file
    payload = _read_json(path)
    payload[field] = value
    _write_json(path, payload)
    with pytest.raises(ResultValidationError):
        DirectionalIntegratedLRResult.load(target, source_crossfit_result=parent)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("columns", ["forged-column"]),
        ("primary_key", ["receiver"]),
    ),
)
def test_directional_sidecar_rejects_table_schema_and_primary_key_tamper(
    tmp_path: Path,
    persisted_directional_lr,
    field: str,
    value: object,
) -> None:
    target, parent = _copy_sidecar(tmp_path, persisted_directional_lr)
    manifest_path = target / "directional_integrated_lr_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["tables"]["scores"][field] = value
    _write_json(manifest_path, manifest)
    _refresh_artifact_identity(target)
    with pytest.raises(ResultValidationError):
        DirectionalIntegratedLRResult.load(target, source_crossfit_result=parent)


def test_directional_sidecar_rejects_parquet_hash_tamper(
    tmp_path: Path, persisted_directional_lr
) -> None:
    target, parent = _copy_sidecar(tmp_path, persisted_directional_lr)
    path = target / "directional_integrated_lr_scores.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ResultValidationError):
        DirectionalIntegratedLRResult.load(target, source_crossfit_result=parent)


def test_directional_sidecar_defensive_read_rechecks_status(
    tmp_path: Path, persisted_directional_lr
) -> None:
    target, parent = _copy_sidecar(tmp_path, persisted_directional_lr)
    loaded = DirectionalIntegratedLRResult.load(target, source_crossfit_result=parent)
    status_path = target / "_status.json"
    status = _read_json(status_path)
    status["artifact_id"] = "changed-after-load"
    _write_json(status_path, status)
    with pytest.raises(ResultValidationError):
        loaded.read_scores()


@pytest.mark.parametrize("mutation", ("missing", "invalid_json"))
def test_directional_sidecar_defensive_read_wraps_manifest_failures(
    tmp_path: Path, persisted_directional_lr, mutation: str
) -> None:
    target, parent = _copy_sidecar(tmp_path, persisted_directional_lr)
    loaded = DirectionalIntegratedLRResult.load(target, source_crossfit_result=parent)
    manifest_path = target / "directional_integrated_lr_manifest.json"
    if mutation == "missing":
        manifest_path.unlink()
    else:
        manifest_path.write_text("{invalid json\n", encoding="utf-8")
    with pytest.raises(ResultValidationError):
        loaded.read_scores()


@pytest.mark.parametrize(
    ("table_name", "column", "value"),
    (
        ("scores", "receiver_training_support_id", "forged-support"),
        ("scores", "family_common_binding_id", "forged-parent"),
        ("scores", "receiver_family_lr_opportunity_id", "forged-universe"),
        ("scores", "molecular_lr_equivalence_id", "forged-molecular-lr"),
        ("opportunities", "n_forward_score_rows", 999),
        ("opportunities", "lr_hypothesis_grid_status", "not-produced"),
    ),
)
def test_directional_sidecar_rejects_coherently_rehashed_semantic_tamper(
    tmp_path: Path,
    persisted_directional_lr,
    table_name: str,
    column: str,
    value: object,
) -> None:
    target, parent = _copy_sidecar(tmp_path, persisted_directional_lr)
    manifest = _read_json(target / "directional_integrated_lr_manifest.json")
    record = manifest["tables"][table_name]
    table = pd.read_parquet(target / record["filename"], engine="pyarrow")
    table.loc[0, column] = value
    _rewrite_table(target, table_name, table)
    with pytest.raises(ResultValidationError):
        DirectionalIntegratedLRResult.load(target, source_crossfit_result=parent)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("directional_lr_hypothesis_universe_id", "forged-universe"),
        ("molecular_lr_equivalence_universe_id", "forged-molecular-universe"),
        ("molecular_lr_axis_id", "forged-molecular-axis"),
        ("pair_spec_id", "forged-pair"),
        ("source_crossfit_result_id", "forged-parent"),
    ),
)
def test_directional_sidecar_rejects_rehashed_manifest_lineage_tamper(
    tmp_path: Path,
    persisted_directional_lr,
    field: str,
    value: object,
) -> None:
    target, parent = _copy_sidecar(tmp_path, persisted_directional_lr)
    manifest_path = target / "directional_integrated_lr_manifest.json"
    manifest = _read_json(manifest_path)
    manifest[field] = value
    _write_json(manifest_path, manifest)
    _refresh_artifact_identity(target)
    with pytest.raises(ResultValidationError):
        DirectionalIntegratedLRResult.load(target, source_crossfit_result=parent)


def test_directional_sidecar_existing_destination_and_incomplete_failure(
    tmp_path: Path, persisted_directional_lr, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, parent, collection, _result = persisted_directional_lr
    with pytest.raises(ResultWriteError) as exists_error:
        write_directional_integrated_lr_result(
            collection,
            root / "sidecar",
            source_crossfit_result=parent,
        )
    assert exists_error.value.details.code == (
        "directional_integrated_lr_destination_exists"
    )

    def fail_parquet(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_parquet)
    destination = tmp_path / "incomplete"
    with pytest.raises(ResultWriteError):
        write_directional_integrated_lr_result(
            collection,
            destination,
            source_crossfit_result=parent,
        )
    assert _read_json(destination / "_status.json")["status"] == "incomplete"
    with pytest.raises(IncompleteResultError):
        DirectionalIntegratedLRResult.load(destination, source_crossfit_result=parent)


def test_directional_queries_validate_filters_without_combining_channels(
    persisted_directional_lr,
) -> None:
    _root, _parent, _collection, result = persisted_directional_lr
    forward = result.query_scores(channel_role="forward")
    reverse = result.query_scores(channel_role="reverse")
    assert set(forward["channel_role"]) == {"forward"}
    assert set(reverse["channel_role"]) == {"reverse"}
    assert len(forward) == len(reverse)
    assert list(forward.columns) == list(reverse.columns)
    with pytest.raises(ValueError):
        result.query_scores(channel_role="merged")
    with pytest.raises(ValueError):
        result.query_opportunities(status="structural_zero")
