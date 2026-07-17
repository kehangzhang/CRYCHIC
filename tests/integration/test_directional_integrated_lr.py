"""Integration checks for the dedicated directional LR score view."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
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

from crychic.attribution import DirectionalContrastPairSpec
from crychic.core import ContractError
from crychic.workflow import (
    build_crossfit_semantic_scores,
    build_directional_integrated_lr_scores,
    run_subject_crossfit,
)
from crychic.workflow import directional_integrated_lr as directional_lr


def _tuned_directional_spec(*, receiver_ids: tuple[str, ...] | None = None):
    base = _persistable_spec()
    directional = _directional_spec()
    return replace(
        base,
        contrasts=directional.contrasts,
        directional_pairs=directional.directional_pairs,
        predeclared_receiver_ids=receiver_ids,
        training_spec=replace(base.training_spec, sender_contrasts=None),
    )


@pytest.fixture(scope="module")
def directional_sources():
    """Fit one tuned directional run and materialize its dedicated collection."""

    spec = _tuned_directional_spec()
    artifacts = run_subject_crossfit(
        _adata(tuple(f"p{i}" for i in range(1, 9))),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )
    pair = artifacts.spec.directional_pairs[0]
    collection = build_directional_integrated_lr_scores(
        artifacts,
        pair_spec_id=pair.pair_spec_id,
    )
    return artifacts, pair, collection


def _source_rows(
    artifacts, pair: DirectionalContrastPairSpec, role: str
) -> pd.DataFrame:
    contrast_name = (
        pair.forward_contrast.name if role == "forward" else pair.reverse_contrast.name
    )
    rows: list[pd.DataFrame] = []
    for fold in artifacts.folds:
        for functional, application in zip(
            fold.family_common_functionals,
            fold.family_common_applications,
            strict=True,
        ):
            if functional.contrast_name != contrast_name:
                continue
            table = application.member_scores.copy()
            table["fold_id"] = fold.fold_id
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


_KEY = [
    "fold_id",
    "receiver",
    "sample_id",
    "subject_id",
    "context_id",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
]

_STATIC_ID_COLUMNS = [
    "directional_lr_hypothesis_universe_id",
    "molecular_lr_equivalence_id",
    "receiver_family_lr_membership_id",
    "receiver_family_lr_hypothesis_id",
    "receiver_family_lr_opportunity_id",
]

_FITTED_PARENT_COLUMNS = [
    "directional_binding_id",
    "family_common_functional_id",
    "family_common_application_id",
    "family_common_binding_id",
    "family_common_member_scores_digest",
    "family_common_family_scores_digest",
    "edge_evidence_digest",
    "incremental_training_artifact_id",
    "incremental_application_id",
]


def test_directional_collection_reuses_independent_source_channels(directional_sources):
    artifacts, pair, collection = directional_sources
    scores = collection.scores
    universe = artifacts.directional_lr_hypothesis_universe
    assert universe is not None

    assert set(scores["channel_role"]) == {"forward", "reverse"}
    assert set(scores["channel"]) == {
        pair.forward_channel_name,
        pair.reverse_channel_name,
    }
    assert not scores.duplicated(["channel_role", "channel", *_KEY]).any()
    assert set(scores["directional_lr_hypothesis_universe_id"]) == {
        universe.universe_id
    }
    assert collection.directional_lr_hypothesis_universe_id == universe.universe_id
    assert collection.molecular_lr_equivalence_universe_id == (
        universe.molecular_lr_equivalence_universe_id
    )
    assert collection.molecular_lr_axis_id == universe.molecular_lr_axis_id
    assert collection.receiver_family_lr_opportunity_axis_id == (
        universe.opportunity_axis_id
    )
    frozen_ids = {
        (receiver, interaction_id, mode): (
            next(
                membership.molecular_lr_equivalence_id
                for membership in universe.memberships
                if membership.membership_id == membership_id
            ),
            membership_id,
            hypothesis_id,
            opportunity_id,
        )
        for (
            receiver,
            interaction_id,
            _driver_id,
            _family_id,
            membership_id,
            mode,
            hypothesis_id,
            opportunity_id,
        ) in universe.opportunity_ids
    }
    assert all(
        (
            row.molecular_lr_equivalence_id,
            row.receiver_family_lr_membership_id,
            row.receiver_family_lr_hypothesis_id,
            row.receiver_family_lr_opportunity_id,
        )
        == frozen_ids[(row.receiver, row.interaction_id, row.mode)]
        for row in scores.itertuples(index=False)
    )
    assert (
        scores[
            [
                "receiver_training_support_id",
                "receiver_training_support_status",
                "design_application_id",
                *_STATIC_ID_COLUMNS,
            ]
        ]
        .notna()
        .all(axis=None)
    )
    binding_status = {
        binding.binding_id: binding.status
        for fold in artifacts.folds
        for binding in fold.directional_response_bindings
    }
    for role in ("forward", "reverse"):
        target = scores.loc[scores["channel_role"] == role].copy()
        source = _source_rows(artifacts, pair, role)
        source = source.rename(
            columns={
                "sender_unresolved_strength": "source_score",
                "family_core_strength": "source_family_core_strength",
                "within_family_lr_weight": "source_within_family_lr_weight",
            }
        )
        merged = target.merge(
            source[
                [
                    *_KEY,
                    "source_score",
                    "source_family_core_strength",
                    "source_within_family_lr_weight",
                ]
            ],
            on=_KEY,
            how="left",
            validate="one_to_one",
        )
        pair_observed = (
            merged["directional_binding_id"].map(binding_status).eq("observed")
        )
        np.testing.assert_allclose(
            merged.loc[pair_observed, "integrated_lr_score"].to_numpy(dtype=float),
            merged.loc[pair_observed, "source_score"].to_numpy(dtype=float),
            equal_nan=True,
        )
        assert merged.loc[~pair_observed, "integrated_lr_score"].isna().all()
        np.testing.assert_allclose(
            merged["family_core_strength"].to_numpy(dtype=float),
            merged["source_family_core_strength"].to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            merged["within_family_lr_weight"].to_numpy(dtype=float),
            merged["source_within_family_lr_weight"].to_numpy(dtype=float),
            equal_nan=True,
        )

    assert not scores["source_agnostic"].any()
    assert not scores["cross_channel_comparable"].any()
    assert not scores["paired_score_comparison_allowed"].any()
    assert not scores["active_inhibition_allowed"].any()
    assert not scores["supports_active_inhibition_claim"].any()
    assert not scores["formal_inference_allowed"].any()
    assert collection.to_manifest()["comparability_scope"] == "within_channel_only"
    assert collection.to_manifest()["opportunity_registry_complete"] is True
    assert collection.score_grid_complete_on_run_receiver_axis is True
    assert "signed_score" not in scores.columns
    assert "difference" not in scores.columns
    assert "inhibition" not in scores.columns


def test_receiver_registry_materializes_training_absence_full_grid() -> None:
    spec = _tuned_directional_spec(receiver_ids=("Ghost", "Receiver", "Sender"))
    artifacts = run_subject_crossfit(
        _adata(tuple(f"p{i}" for i in range(1, 9))),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
    )
    pair = artifacts.spec.directional_pairs[0]
    collection = build_directional_integrated_lr_scores(
        artifacts,
        pair_spec_id=pair.pair_spec_id,
    )
    universe = artifacts.directional_lr_hypothesis_universe
    assert universe is not None

    opportunities = collection.opportunity_registry
    expected = {
        (fold.fold_id, receiver)
        for fold in artifacts.folds
        for receiver in artifacts.receiver_universe.receiver_ids
    }
    assert len(opportunities) == (
        len(artifacts.folds) * len(artifacts.receiver_universe.receiver_ids)
    )
    assert (
        set(opportunities[["fold_id", "receiver"]].itertuples(index=False, name=None))
        == expected
    )
    ghost = opportunities.loc[opportunities["receiver"].eq("Ghost")]
    assert len(ghost) == len(artifacts.folds)
    assert set(ghost["receiver_training_support_status"]) == {"not_estimable"}
    assert set(ghost["receiver_training_support_reason_code"]) == {
        "receiver_absent_in_outer_training"
    }
    assert set(ghost["status"]) == {"not_estimable"}
    assert set(ghost["reason_code"]) == {"receiver_absent_in_outer_training"}
    assert ghost["directional_binding_id"].isna().all()
    assert set(ghost["lr_hypothesis_grid_status"]) == {"produced"}
    assert ghost["lr_hypothesis_grid_reason_code"].isna().all()
    assert set(ghost["directional_lr_hypothesis_universe_id"]) == {universe.universe_id}
    assert set(ghost["receiver_family_lr_opportunity_axis_id"]) == {
        universe.opportunity_axis_id
    }
    parent_columns = [
        column
        for column in opportunities.columns
        if (column.startswith("forward_") or column.startswith("reverse_"))
        and not column.endswith("channel")
    ]
    assert ghost[parent_columns].isna().all(axis=None)
    assert (ghost["n_forward_score_rows"] == ghost["n_reverse_score_rows"]).all()
    assert (ghost["n_forward_score_rows"] > 0).all()

    scores = collection.scores
    ghost_scores = scores.loc[scores["receiver"].eq("Ghost")]
    assert not ghost_scores.empty
    expected_counts: dict[tuple[str, str], int] = {}
    for fold in artifacts.folds:
        for role, _channel, contrast in directional_lr._channel_specs(pair):
            _, design_rows = directional_lr._design_application(
                fold,
                contrast=contrast,
                role=role,
            )
            expected_counts[(fold.fold_id, role)] = len(design_rows) * len(
                universe.hypothesis_ids
            )
    observed_counts = {
        (str(fold_id), str(role)): len(rows)
        for (fold_id, role), rows in ghost_scores.groupby(
            ["fold_id", "channel_role"], sort=False
        )
    }
    assert observed_counts == expected_counts
    assert set(ghost_scores["receiver_training_support_status"]) == {"not_estimable"}
    assert set(ghost_scores["receiver_training_support_reason_code"]) == {
        "receiver_absent_in_outer_training"
    }
    assert set(ghost_scores["status"]) == {"not_estimable"}
    assert set(ghost_scores["reason_code"]) == {"receiver_absent_in_outer_training"}
    assert (
        ghost_scores[
            [
                "integrated_lr_score",
                "family_core_strength",
                "within_family_lr_weight",
                "source_status",
                "source_reason_code",
            ]
        ]
        .isna()
        .all(axis=None)
    )
    assert ghost_scores[_FITTED_PARENT_COLUMNS].isna().all(axis=None)
    assert (
        ghost_scores[
            [
                "receiver_training_support_id",
                "design_application_id",
                *_STATIC_ID_COLUMNS,
            ]
        ]
        .notna()
        .all(axis=None)
    )
    assert set(ghost_scores["directional_lr_hypothesis_universe_id"]) == {
        universe.universe_id
    }

    ghost_parent_rows = [
        row for row in collection.parent_bindings if row["receiver"] == "Ghost"
    ]
    assert len(ghost_parent_rows) == len(artifacts.folds) * 2
    assert all(
        row["directional_lr_hypothesis_universe_id"] == universe.universe_id
        and row["receiver_family_lr_opportunity_axis_id"]
        == universe.opportunity_axis_id
        and row["receiver_training_support_status"] == "not_estimable"
        and row["design_application_id"] is not None
        and all(row[column] is None for column in _FITTED_PARENT_COLUMNS)
        for row in ghost_parent_rows
    )

    supported = opportunities.loc[
        opportunities["receiver_training_support_status"].eq("observed")
    ]
    assert supported["directional_binding_id"].notna().all()
    assert supported[parent_columns].notna().all(axis=None)
    assert set(supported["lr_hypothesis_grid_status"]) == {"produced"}
    assert (
        supported["n_forward_score_rows"] == supported["n_reverse_score_rows"]
    ).all()
    assert (supported["n_forward_score_rows"] > 0).all()
    assert not opportunities["paired_score_comparison_allowed"].any()
    assert not opportunities["active_inhibition_allowed"].any()
    assert not opportunities["supports_active_inhibition_claim"].any()
    assert not opportunities["formal_inference_allowed"].any()

    manifest = collection.to_manifest()
    assert manifest["schema_version"] == "4.0.0"
    assert manifest["opportunity_count"] == len(expected)
    assert manifest["opportunity_registry_complete"] is True
    assert manifest["score_grid_complete_on_run_receiver_axis"] is True
    assert manifest["score_grid_scope"] == (
        "full_run_root_frozen_receiver_family_lr_hypothesis_universe"
    )
    assert manifest["status"] == "complete_descriptive"
    assert manifest["directional_lr_hypothesis_universe_id"] == universe.universe_id
    assert manifest["molecular_lr_equivalence_universe_id"] == (
        universe.molecular_lr_equivalence_universe_id
    )
    assert manifest["molecular_lr_axis_id"] == universe.molecular_lr_axis_id
    assert manifest["receiver_family_lr_opportunity_axis_id"] == (
        universe.opportunity_axis_id
    )

    first = ghost_scores.iloc[0]
    same_static_cell = (
        scores["fold_id"].eq(first["fold_id"])
        & scores["receiver"].eq("Ghost")
        & scores["sample_id"].eq(first["sample_id"])
        & scores["family_id"].eq(first["family_id"])
        & scores["driver_id"].eq(first["driver_id"])
        & scores["interaction_id"].eq(first["interaction_id"])
        & scores["mode"].eq(first["mode"])
    )
    inexact = scores.loc[~same_static_cell].reset_index(drop=True)
    with pytest.raises(ValueError, match="exact frozen source grid"):
        directional_lr._validate_table(inexact, artifacts=artifacts, pair=pair)

    ghost_index = collection._scores.index[collection._scores["receiver"].eq("Ghost")][
        0
    ]
    collection._scores.loc[ghost_index, "family_common_functional_id"] = (
        "forged-fitted-parent"
    )
    with pytest.raises(ContractError):
        _ = collection.scores


def test_pair_grid_and_family_conservation_are_validated(directional_sources):
    artifacts, pair, collection = directional_sources
    scores = collection.scores
    key = [
        "fold_id",
        "receiver",
        "sample_id",
        "subject_id",
        "context_id",
        "family_id",
        "mode",
    ]
    grids = scores.groupby(["channel_role", *key], dropna=False).size()
    assert (grids > 0).all()
    for _, group in scores.groupby(["channel_role", *key], dropna=False):
        observed = group["integrated_lr_score"].dropna()
        if not observed.empty:
            np.testing.assert_allclose(
                observed.sum(), group["family_core_strength"].dropna().iloc[0]
            )

    swapped = scores.copy(deep=True)
    swapped["channel"] = swapped["channel_role"].map(
        {
            "forward": pair.reverse_channel_name,
            "reverse": pair.forward_channel_name,
        }
    )
    with pytest.raises(ValueError, match="role has the wrong channel"):
        directional_lr._validate_table(swapped, artifacts=artifacts, pair=pair)


def test_pair_not_estimable_channel_fails_closed(directional_sources):
    """A pair-level NE binding emits no score-bearing values."""

    artifacts, pair, _ = directional_sources
    fold = artifacts.folds[0]
    common_index = next(
        index
        for index, item in enumerate(fold.family_common_functionals)
        if item.contrast_name == pair.forward_contrast.name
    )
    functional = fold.family_common_functionals[common_index]
    receiver = functional.receiver
    directional_binding = next(
        item
        for item in fold.directional_response_bindings
        if item.pair_spec_id == pair.pair_spec_id and item.receiver == receiver
    )
    parent = directional_lr._parent_chain(
        fold,
        contrast_name=pair.forward_contrast.name,
        contrast=pair.forward_contrast,
        receiver=receiver,
        directional=directional_binding,
        role="forward",
    )
    fake_binding = SimpleNamespace(
        status="not_estimable",
        reason_code="directional_pair_not_estimable",
        binding_id=directional_binding.binding_id,
    )
    receiver_support = next(
        record
        for record in fold.receiver_training_support
        if record.receiver_id == receiver
    )
    design_application, design_rows = directional_lr._design_application(
        fold,
        contrast=pair.forward_contrast,
        role="forward",
    )
    universe = artifacts.directional_lr_hypothesis_universe
    assert universe is not None
    table = directional_lr._channel_rows(
        artifacts=artifacts,
        pair=pair,
        binding=fake_binding,
        functional=parent[0],
        application=parent[1],
        common_binding=parent[2],
        model=parent[3],
        incremental_app=parent[4],
        receiver_support=receiver_support,
        design_application=design_application,
        design_rows=design_rows,
        universe=universe,
        contrast=pair.forward_contrast,
        role="forward",
        channel=pair.forward_channel_name,
    )
    assert (table["status"] == "not_estimable").all()
    assert table["integrated_lr_score"].isna().all()
    source = parent[1].member_scores
    merged = table.merge(
        source[
            [
                *_KEY[1:],
                "family_core_strength",
                "within_family_lr_weight",
            ]
        ],
        on=_KEY[1:],
        suffixes=("", "_source"),
        validate="one_to_one",
    )
    np.testing.assert_allclose(
        merged["family_core_strength"],
        merged["family_core_strength_source"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        merged["within_family_lr_weight"],
        merged["within_family_lr_weight_source"],
        equal_nan=True,
    )


def test_generic_semantic_views_exclude_directional_contrasts(directional_sources):
    artifacts, pair, _ = directional_sources
    semantic = build_crossfit_semantic_scores(artifacts)
    directional_names = {
        pair.forward_contrast.name,
        pair.reverse_contrast.name,
    }
    for view in (semantic.integrated_lr_score, semantic.differential_effect):
        if not view.empty:
            assert directional_names.isdisjoint(set(view["contrast"]))
    manifest = semantic.view_manifest
    for view_name in ("integrated_lr_score", "differential_effect"):
        row = manifest.loc[manifest["semantic_output"] == view_name].iloc[0]
        assert bool(row["dedicated_view"])
        if row["status"] == "not_produced":
            assert row["reason_code"] == (
                "directional_contrasts_require_dedicated_integrated_lr_collection"
            )


def test_collection_tamper_is_detected(directional_sources):
    artifacts, pair, _ = directional_sources
    collection = build_directional_integrated_lr_scores(
        artifacts, pair_spec_id=pair.pair_spec_id
    )
    collection._scores.loc[0, "integrated_lr_score"] = 123.0
    with pytest.raises(ContractError):
        _ = collection.scores


@pytest.mark.parametrize(
    "column",
    (
        "directional_lr_hypothesis_universe_id",
        "molecular_lr_equivalence_id",
        "receiver_family_lr_membership_id",
        "receiver_family_lr_hypothesis_id",
        "receiver_family_lr_opportunity_id",
        "receiver_training_support_id",
        "design_application_id",
        "family_common_binding_id",
    ),
)
def test_static_grid_and_parent_lineage_tamper_is_detected(
    directional_sources,
    column: str,
) -> None:
    artifacts, pair, _ = directional_sources
    collection = build_directional_integrated_lr_scores(
        artifacts, pair_spec_id=pair.pair_spec_id
    )
    collection._scores.loc[0, column] = "forged-lineage"
    with pytest.raises(ContractError):
        _ = collection.scores


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("formal_inference_allowed", True),
        ("directional_lr_hypothesis_universe_id", "forged-universe"),
        ("receiver_family_lr_opportunity_axis_id", "forged-axis"),
    ),
)
def test_opportunity_registry_tamper_is_detected(
    directional_sources,
    column: str,
    value: object,
) -> None:
    artifacts, pair, _ = directional_sources
    collection = build_directional_integrated_lr_scores(
        artifacts, pair_spec_id=pair.pair_spec_id
    )
    collection._opportunity_registry.loc[0, column] = value
    with pytest.raises(ContractError):
        _ = collection.opportunity_registry
