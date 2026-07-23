"""Normalize persisted CRYCHIC sample scores without privileged treatment."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.common import (
    LONG_TABLE_COLUMNS,
    begin_manifest,
    canonical_digest,
    cell_type_support,
    finalize_manifest,
    materialize_fixed_universe,
    prepare_output,
    python_environment,
    validate_long_table,
    validate_prepared_input,
)
from crychic import CrychicResult
from crychic.core import stable_id
from crychic.resources import ResourceBundle
from crychic.scoring import DownstreamEvidencePolicy

from .resource import (
    bundle_resource_table,
    harmonized_resource_bundle,
    load_native_resource_bundle,
)

INFERENTIAL_REASON = "v0_1_inferential_disabled"
METHOD_ID = "crychic"
ANALYSIS_TRACK = "lr_stlr"
SCORE_NAME = "comm_strength"
MECHANISTIC_SCORE_NAME = "mechanistic_sender_lr_score"
MECHANISTIC_SCORE_LAYER_VERSION = "mechanistic_sender_lr_v1"
REQUIRED_SCORE_LAYER_VERSION = "downstream_confirmed_geometric_v1"
SCORE_DIRECTION = "higher"
RECEIVER_ROW_UNION_BRIDGE_VERSION = "receiver_row_union_bridge_v1"


@dataclass(frozen=True, slots=True)
class _ScoreView:
    child_ids: tuple[str, ...]
    contrast_candidates: tuple[str, ...]
    grouped_contrast: str | None
    receiver_partitions: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def grouped(self) -> bool:
        return self.grouped_contrast is not None


class ResultReader(Protocol):
    """Narrow interface needed from :class:`CrychicResult`."""

    manifest: Mapping[str, Any]
    config: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def read_table(
        self,
        name: str,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame: ...


def _edge_id(sender: object, receiver: object, interaction_id: object) -> str:
    return cast(
        str,
        stable_id(
            "communication_edge",
            {
                "interaction_id": str(interaction_id),
                "receiver": str(receiver),
                "sender": str(sender),
            },
        ),
    )


def _edge_map(interactions: pd.DataFrame) -> pd.DataFrame:
    columns = ["sender", "receiver", "interaction_id"]
    result = interactions.loc[:, columns].drop_duplicates(ignore_index=True).copy()
    result["edge_id"] = [
        _edge_id(sender, receiver, interaction_id)
        for sender, receiver, interaction_id in result.itertuples(
            index=False, name=None
        )
    ]
    if result["edge_id"].duplicated().any():
        raise ValueError("CRYCHIC communication edge IDs are not one-to-one")
    return result


def _method_version(result: ResultReader) -> str:
    workflow = result.manifest.get("workflow_parameters", {})
    if isinstance(workflow, Mapping):
        value = workflow.get("method_version")
        if isinstance(value, str) and value:
            return value
    value = result.provenance.get("package_version")
    if not isinstance(value, str) or not value:
        raise ValueError("CRYCHIC result has no method/package version")
    return value


def _min_cells(result: ResultReader) -> int:
    workflow = result.manifest.get("workflow_parameters", {})
    if isinstance(workflow, Mapping):
        pseudobulk = workflow.get("pseudobulk", {})
        if isinstance(pseudobulk, Mapping):
            value = pseudobulk.get("min_cells")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    raise ValueError("CRYCHIC result does not record pseudobulk.min_cells")


def _source_table_digests(result: ResultReader) -> dict[str, str]:
    tables = result.manifest.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError("CRYCHIC result manifest does not record source tables")
    digests: dict[str, str] = {}
    for name in ("sample_scores", "interactions"):
        record = tables.get(name)
        digest = record.get("sha256") if isinstance(record, Mapping) else None
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"CRYCHIC result manifest does not record {name} SHA256")
        digests[name] = digest
    return digests


def _verify_resource(result: ResultReader, bundle: ResourceBundle) -> None:
    key = f"{bundle.resource_id}:{bundle.version}"
    resources = result.manifest.get("resource_digests")
    if (
        not isinstance(resources, Mapping)
        or resources.get(key) != bundle.manifest_digest
    ):
        raise ValueError(
            "CRYCHIC result resource digest does not match the readback bundle"
        )


def _result_samples(
    result: ResultReader,
    sample_scores: pd.DataFrame,
    adata: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = result.config
    context_keys = tuple(map(str, cast(Sequence[object], config["context_keys"])))
    sample_key = str(config["sample_key"])
    subject_key = str(config["subject_key"])
    cell_type_key = str(config["cell_type_key"])
    metadata = validate_prepared_input(
        adata,
        sample_key=sample_key,
        subject_key=subject_key,
        cell_type_key=cell_type_key,
        context_keys=context_keys,
    )
    score_samples = set(sample_scores["sample_id"].astype(str))
    input_samples = set(metadata["sample_id"].astype(str))
    missing_input = score_samples.difference(input_samples)
    if missing_input:
        example = sorted(missing_input)[:5]
        raise ValueError(f"result samples are absent from prepared input: {example}")
    expected = (
        sample_scores.loc[:, ["sample_id", "subject_id", "context_json"]]
        .astype(str)
        .drop_duplicates(ignore_index=True)
    )
    matched = expected.merge(
        metadata,
        on=["sample_id", "subject_id", "context_json"],
        how="left",
        indicator=True,
    )
    if not matched["_merge"].eq("both").all():
        raise ValueError("prepared input subject/context metadata differs from result")
    support = cell_type_support(
        adata,
        sample_key=sample_key,
        cell_type_key=cell_type_key,
    )
    return metadata, support


def _numeric_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    first = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    second = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return pd.Series(
        np.isclose(first, second, rtol=1e-10, atol=1e-12, equal_nan=True),
        index=left.index,
    )


def _unit_component(values: pd.Series, *, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    invalid = values.notna() & numeric.isna()
    finite = numeric.dropna()
    if invalid.any() or (
        not finite.empty
        and ((finite < 0.0).any() or (finite > 1.0).any() or np.isinf(finite).any())
    ):
        raise ValueError(f"{field} must contain values in [0, 1] or missing")
    return numeric


def _context_invariant_mechanistic_rows(
    selected: pd.DataFrame,
    edge_map: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse contrast-trained copies after proving mechanism invariance."""

    keys = [
        "sample_id",
        "subject_id",
        "context_id",
        "context_json",
        "edge_id",
        "repeat_id",
        "fold_id",
        "mode",
    ]
    components = ["availability", "sender_component", "prior_quality"]
    ordered = selected.sort_values(
        [*keys, "scoring_functional_id"], kind="stable", ignore_index=True
    )
    grouped = ordered.groupby(keys, observed=True, sort=False, dropna=False)
    functionals_per_row = grouped["scoring_functional_id"].nunique()
    if functionals_per_row.empty:
        raise ValueError("mechanistic score collapse has no sample-edge rows")
    for component in components:
        distinct = grouped[component].nunique(dropna=False)
        changed = distinct.gt(1)
        if changed.any():
            examples = [tuple(map(str, item)) for item in changed.index[changed][:5]]
            raise ValueError(
                "mechanistic component differs across contrast-trained copies: "
                f"field={component}, sample_edges={examples}; use the required "
                "downstream policy for contrast-specific diagnostics"
            )
    raw = ordered.drop_duplicates(keys, keep="first", ignore_index=True).copy()
    availability = _unit_component(raw["availability"], field="availability")
    prior = _unit_component(raw["prior_quality"], field="prior_quality")
    sender = _unit_component(raw["sender_component"], field="sender_component")
    component_frame = pd.DataFrame(
        {
            "availability": availability,
            "prior_quality": prior,
            "sender_component": sender,
        },
        index=raw.index,
    )
    structural_zero = component_frame.eq(0.0).any(axis=1)
    missing = component_frame.isna().any(axis=1) & ~structural_zero
    raw["score"] = availability * prior * sender
    raw.loc[structural_zero, "score"] = 0.0
    raw.loc[missing, "score"] = np.nan
    raw["status"] = np.where(missing, "missing", "ok")
    raw["reason_code"] = np.where(
        missing,
        "mechanistic_core_component_missing",
        np.where(structural_zero, "mechanistic_structural_zero", ""),
    )
    raw = raw.merge(edge_map, on="edge_id", how="left", validate="many_to_one")
    if raw[["sender", "receiver", "interaction_id"]].isna().any().any():
        raise ValueError("mechanistic rows contain an unmapped communication edge")
    duplicate_key = ["sample_id", "sender", "receiver", "interaction_id"]
    if raw.duplicated(duplicate_key).any():
        raise ValueError("mechanistic collapse contains duplicate sample-event rows")
    raw["target"] = pd.NA
    return raw, {
        "source_rows": len(selected),
        "collapsed_rows": len(raw),
        "source_scoring_functional_count": int(
            selected["scoring_functional_id"].nunique()
        ),
        "copies_per_sample_edge_min": int(functionals_per_row.min()),
        "copies_per_sample_edge_max": int(functionals_per_row.max()),
        "invariant_components": components,
        "exact_invariance_verified": True,
    }


def _contrast_candidates(
    sample_scores: pd.DataFrame,
    interactions: pd.DataFrame,
    edge_map: pd.DataFrame,
    *,
    communication_mode: str,
) -> dict[str, tuple[str, ...]]:
    """Recover functional/contrast provenance from the persisted aggregates."""

    numeric = ["availability", "sender_component", "comm_strength", "prior_quality"]
    selected = sample_scores.loc[sample_scores["mode"].eq(communication_mode)].copy()
    per_subject = (
        selected.groupby(
            ["scoring_functional_id", "context_id", "edge_id", "subject_id"],
            observed=True,
            sort=False,
            dropna=False,
        )[numeric]
        .mean()
        .reset_index()
    )
    summary = (
        per_subject.groupby(
            ["scoring_functional_id", "context_id", "edge_id"],
            observed=True,
            sort=False,
            dropna=False,
        )[numeric]
        .mean()
        .reset_index()
    )
    persisted = interactions.loc[interactions["mode"].eq(communication_mode)].copy()
    persisted = persisted.merge(
        edge_map,
        on=["sender", "receiver", "interaction_id"],
        how="left",
        validate="many_to_one",
    )
    result: dict[str, tuple[str, ...]] = {}
    persisted_numeric = {
        "availability": "availability",
        "sender_component": "assignment_weight",
        "comm_strength": "comm_strength",
        "prior_quality": "prior_quality",
    }
    for functional_id, functional in summary.groupby(
        "scoring_functional_id", sort=True, observed=True
    ):
        candidates: list[str] = []
        support = functional.loc[:, ["context_id", "edge_id"]]
        if support.duplicated().any():
            raise ValueError("functional summary has duplicate context-edge keys")
        for contrast, candidate in persisted.groupby(
            "contrast", sort=True, observed=True
        ):
            candidate_keys = candidate.loc[:, ["context_id", "edge_id"]]
            if candidate_keys.duplicated().any():
                raise ValueError(
                    "persisted interactions contain duplicate contrast context-edge "
                    "keys"
                )
            candidate = candidate.merge(
                support,
                on=["context_id", "edge_id"],
                how="inner",
                validate="one_to_one",
            )
            if len(candidate) != len(functional):
                continue
            left = functional.rename(
                columns={name: f"sample_{name}" for name in numeric}
            )
            right = candidate.rename(
                columns={
                    name: f"persisted_{sample_name}"
                    for sample_name, name in persisted_numeric.items()
                }
            )
            merged = left.merge(
                right,
                on=["context_id", "edge_id"],
                how="inner",
                validate="one_to_one",
                indicator=True,
            )
            if len(merged) != len(functional) or not merged["_merge"].eq("both").all():
                continue
            matches = pd.Series(True, index=merged.index)
            for sample_name in persisted_numeric:
                matches &= _numeric_equal(
                    merged[f"sample_{sample_name}"],
                    merged[f"persisted_{sample_name}"],
                )
            if matches.all():
                candidates.append(str(contrast))
        result[str(functional_id)] = tuple(candidates)
    return result


def _functional_support(
    selected: pd.DataFrame,
    edge_map: pd.DataFrame,
) -> tuple[
    dict[str, frozenset[tuple[str, str]]],
    dict[str, frozenset[tuple[str, str]]],
    dict[str, frozenset[str]],
]:
    annotated = selected.merge(
        edge_map.loc[:, ["edge_id", "receiver"]],
        on="edge_id",
        how="left",
        validate="many_to_one",
    )
    if annotated["receiver"].isna().any():
        raise ValueError("functional support contains an unmapped communication edge")
    edge_support: dict[str, frozenset[tuple[str, str]]] = {}
    context_edge_support: dict[str, frozenset[tuple[str, str]]] = {}
    receiver_support: dict[str, frozenset[str]] = {}
    for functional_id, functional in annotated.groupby(
        "scoring_functional_id", sort=True, observed=True
    ):
        identifier = str(functional_id)
        edge_support[identifier] = frozenset(
            functional.loc[:, ["receiver", "edge_id"]]
            .astype(str)
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        context_edge_support[identifier] = frozenset(
            functional.loc[:, ["context_id", "edge_id"]]
            .astype(str)
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        receiver_support[identifier] = frozenset(functional["receiver"].astype(str))
    return edge_support, context_edge_support, receiver_support


def _persisted_contrast_support(
    interactions: pd.DataFrame,
    edge_map: pd.DataFrame,
    *,
    communication_mode: str,
) -> tuple[
    dict[str, frozenset[tuple[str, str]]],
    dict[str, frozenset[tuple[str, str]]],
]:
    persisted = interactions.loc[interactions["mode"].eq(communication_mode)].merge(
        edge_map,
        on=["sender", "receiver", "interaction_id"],
        how="left",
        validate="many_to_one",
    )
    if persisted["edge_id"].isna().any():
        raise ValueError(
            "persisted interactions contain an unmapped communication edge"
        )
    edge_support: dict[str, frozenset[tuple[str, str]]] = {}
    context_edge_support: dict[str, frozenset[tuple[str, str]]] = {}
    for contrast, candidate in persisted.groupby("contrast", sort=True, observed=True):
        identifier = str(contrast)
        keys = candidate.loc[:, ["context_id", "edge_id"]].astype(str)
        if keys.duplicated().any():
            raise ValueError(
                "persisted interactions contain duplicate contrast context-edge keys"
            )
        edge_support[identifier] = frozenset(
            candidate.loc[:, ["receiver", "edge_id"]]
            .astype(str)
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        context_edge_support[identifier] = frozenset(
            keys.itertuples(index=False, name=None)
        )
    return edge_support, context_edge_support


def _default_score_views(
    selected: pd.DataFrame,
    interactions: pd.DataFrame,
    edge_map: pd.DataFrame,
    candidates: Mapping[str, tuple[str, ...]],
    available_ids: tuple[str, ...],
    *,
    communication_mode: str,
) -> tuple[_ScoreView, ...]:
    if len(available_ids) == 1:
        functional_id = available_ids[0]
        return (
            _ScoreView(
                child_ids=(functional_id,),
                contrast_candidates=candidates.get(functional_id, ()),
                grouped_contrast=None,
                receiver_partitions=(),
            ),
        )

    ambiguous = {
        functional_id: candidates.get(functional_id, ())
        for functional_id in available_ids
        if len(candidates.get(functional_id, ())) != 1
    }
    if ambiguous:
        raise ValueError(
            "default multi-functional readback requires exactly one persisted "
            "contrast candidate per child; select explicit scoring_functional_ids "
            f"for diagnostics: {ambiguous}"
        )

    child_edges, child_context_edges, child_receivers = _functional_support(
        selected, edge_map
    )
    non_receiver_scoped = {
        child_id: sorted(child_receivers[child_id])
        for child_id in available_ids
        if len(child_receivers[child_id]) != 1
    }
    if non_receiver_scoped:
        raise ValueError(
            "default multi-functional readback requires each child to be scoped "
            f"to exactly one receiver: {non_receiver_scoped}"
        )
    expected_edges, expected_context_edges = _persisted_contrast_support(
        interactions,
        edge_map,
        communication_mode=communication_mode,
    )
    grouped: dict[str, list[str]] = {}
    for functional_id in available_ids:
        contrast = candidates[functional_id][0]
        grouped.setdefault(contrast, []).append(functional_id)

    views: list[_ScoreView] = []
    for contrast in sorted(grouped):
        child_ids = tuple(sorted(grouped[contrast]))
        observed_edges: set[tuple[str, str]] = set()
        observed_context_edges: set[tuple[str, str]] = set()
        observed_receivers: set[str] = set()
        for child_id in child_ids:
            receiver = next(iter(child_receivers[child_id]))
            if receiver in observed_receivers:
                raise ValueError(
                    "contrast child scoring functionals have overlapping receiver "
                    f"partitions: contrast={contrast!r}, receiver={receiver!r}"
                )
            observed_receivers.add(receiver)
            overlap = observed_edges.intersection(child_edges[child_id])
            if overlap:
                raise ValueError(
                    "contrast child scoring functionals have overlapping edge support: "
                    f"contrast={contrast!r}, child={child_id!r}, "
                    f"examples={sorted(overlap)[:5]}"
                )
            observed_edges.update(child_edges[child_id])
            observed_context_edges.update(child_context_edges[child_id])
        expected = expected_edges.get(contrast)
        expected_context = expected_context_edges.get(contrast)
        if expected is None or expected_context is None:
            raise ValueError(
                f"persisted contrast is absent from interactions: {contrast}"
            )
        if observed_edges != set(expected):
            missing = sorted(set(expected).difference(observed_edges))[:5]
            extra = sorted(observed_edges.difference(expected))[:5]
            raise ValueError(
                "contrast child scoring functionals do not completely cover the "
                f"expected receiver/edge universe: contrast={contrast!r}, "
                f"missing={missing}, extra={extra}"
            )
        if observed_context_edges != set(expected_context):
            missing = sorted(set(expected_context).difference(observed_context_edges))[
                :5
            ]
            extra = sorted(observed_context_edges.difference(expected_context))[:5]
            raise ValueError(
                "contrast child scoring functionals do not completely cover the "
                f"expected context-edge support: contrast={contrast!r}, "
                f"missing={missing}, extra={extra}"
            )
        views.append(
            _ScoreView(
                child_ids=child_ids,
                contrast_candidates=(contrast,),
                grouped_contrast=contrast,
                receiver_partitions=tuple(
                    (child_id, tuple(sorted(child_receivers[child_id])))
                    for child_id in child_ids
                ),
            )
        )
    return tuple(views)


def _explicit_score_views(
    requested_ids: tuple[str, ...],
    candidates: Mapping[str, tuple[str, ...]],
) -> tuple[_ScoreView, ...]:
    return tuple(
        _ScoreView(
            child_ids=(functional_id,),
            contrast_candidates=candidates.get(functional_id, ()),
            grouped_contrast=None,
            receiver_partitions=(),
        )
        for functional_id in requested_ids
    )


def _apply_crychic_statuses(
    table: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    keys = ["sample_id", "sender", "receiver", "interaction_id", "target"]
    status = raw.loc[:, [*keys, "status", "reason_code"]].rename(
        columns={"status": "crychic_status", "reason_code": "crychic_reason"}
    )
    result = table.merge(status, on=keys, how="left", validate="one_to_one")
    structural = result["status"].isin(
        {"resource_unavailable", "missing", "insufficient_cells", "method_failed"}
    )
    internal_missing = (
        ~structural
        & result["crychic_status"].notna()
        & ~result["crychic_status"].eq("ok")
    )
    result.loc[internal_missing, "status"] = "missing"
    result.loc[internal_missing, "score"] = np.nan

    row_status = result["status"].astype(str)
    detail = row_status.copy()
    detail.loc[row_status.eq("ok")] = ""
    detail.loc[row_status.eq("missing") & structural] = "missing_required_lr_input"
    raw_reason = (
        result["crychic_reason"].astype("string").fillna("missing_score_component")
    )
    raw_reason = raw_reason.str.replace(
        rf"(^|;){INFERENTIAL_REASON}(;|$)",
        ";",
        regex=True,
    ).str.strip(";")
    detail.loc[internal_missing] = raw_reason.loc[internal_missing]
    detail.loc[row_status.eq("insufficient_cells")] = "insufficient_cells"
    detail.loc[row_status.eq("not_returned")] = "not_returned_by_crychic"
    result["reason_code"] = INFERENTIAL_REASON
    has_detail = detail.ne("")
    result.loc[has_detail, "reason_code"] = (
        INFERENTIAL_REASON + ";" + detail.loc[has_detail]
    )
    result = result.drop(columns=["crychic_status", "crychic_reason"])
    ordered = cast(pd.DataFrame, result.loc[:, LONG_TABLE_COLUMNS].copy())
    return validate_long_table(ordered) if validate else ordered


def convert_result_to_long(
    result: ResultReader,
    adata: Any,
    bundle: ResourceBundle,
    *,
    dataset_id: str,
    resource_mode: str,
    communication_mode: str = "state",
    scoring_functional_ids: Sequence[str] | None = None,
    downstream_evidence_policy: DownstreamEvidencePolicy | str = (
        DownstreamEvidencePolicy.REQUIRED
    ),
    min_cells: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Convert one persisted result into fixed-universe benchmark runs.

    ``required`` preserves the historical contrast-specific geometric score.
    ``annotate`` or ``disabled`` emits one context-invariant mechanistic score
    after proving that its components are identical across trained copies.
    """

    if resource_mode not in {"H-common", "native"}:
        raise ValueError("CRYCHIC readback supports H-common or native resource arms")
    if communication_mode not in {"state", "ecosystem"}:
        raise ValueError("communication_mode must be 'state' or 'ecosystem'")
    if not dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")
    policy = DownstreamEvidencePolicy(downstream_evidence_policy)
    if policy is DownstreamEvidencePolicy.MODULATE:
        raise ValueError(
            "baseline readback has no signed held-out downstream support; use "
            "annotate/disabled for canonical mechanism scores or required for "
            "the historical downstream-confirmed diagnostic"
        )
    _verify_resource(result, bundle)

    score_columns = [
        "sample_id",
        "subject_id",
        "context_id",
        "context_json",
        "edge_id",
        "scoring_functional_id",
        "repeat_id",
        "fold_id",
        "mode",
        "availability",
        "sender_component",
        "prior_quality",
        "comm_strength",
        "status",
        "reason_code",
    ]
    sample_scores = result.read_table("sample_scores", columns=score_columns)
    if sample_scores.empty:
        raise ValueError("CRYCHIC result has no sample scores")
    sample_metadata, support = _result_samples(result, sample_scores, adata)
    interactions = result.read_table(
        "interactions",
        columns=[
            "context_id",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
            "contrast",
            "availability",
            "assignment_weight",
            "comm_strength",
            "prior_quality",
        ],
    )
    edges = _edge_map(interactions)
    resource = bundle_resource_table(bundle, input_genes=set(map(str, adata.var_names)))
    unknown = set(edges["interaction_id"]).difference(resource["interaction_id"])
    if unknown:
        example = sorted(unknown)[:5]
        raise ValueError(
            f"result interactions are absent from readback resource: {example}"
        )

    selected = sample_scores.loc[sample_scores["mode"].eq(communication_mode)].copy()
    available_ids = tuple(
        sorted(selected["scoring_functional_id"].astype(str).unique())
    )
    requested_ids = (
        available_ids
        if scoring_functional_ids is None
        else tuple(map(str, scoring_functional_ids))
    )
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("scoring_functional_ids must be unique")
    missing_ids = set(requested_ids).difference(available_ids)
    if missing_ids:
        raise ValueError(f"unknown scoring functional IDs: {sorted(missing_ids)}")
    if not requested_ids:
        raise ValueError("no CRYCHIC scoring functionals selected")
    effective_min_cells = _min_cells(result) if min_cells is None else min_cells
    if effective_min_cells < 1:
        raise ValueError("min_cells must be positive")
    source_run_id = str(result.manifest["run_id"])
    source_table_digests = _source_table_digests(result)

    if policy is not DownstreamEvidencePolicy.REQUIRED:
        if scoring_functional_ids is not None:
            raise ValueError(
                "context-invariant mechanistic readback forbids selecting "
                "individual scoring functionals"
            )
        raw, invariance = _context_invariant_mechanistic_rows(selected, edges)
        observed = raw.loc[raw["status"].eq("ok")].copy()
        run_identity = {
            "source_run_id": source_run_id,
            "communication_mode": communication_mode,
            "resource_mode": resource_mode,
            "dataset_id": dataset_id,
            "score_layer": MECHANISTIC_SCORE_LAYER_VERSION,
            "downstream_evidence_policy": policy.value,
            "source_result_table_digests": source_table_digests,
        }
        run_id = canonical_digest(run_identity, prefix="crychic_benchmark_run")
        table = materialize_fixed_universe(
            observed.loc[
                :,
                [
                    "sample_id",
                    "sender",
                    "receiver",
                    "interaction_id",
                    "target",
                    "score",
                ],
            ],
            sample_metadata=sample_metadata,
            support=support,
            resource=resource,
            dataset_id=dataset_id,
            run_id=run_id,
            method_id=METHOD_ID,
            method_version=MECHANISTIC_SCORE_LAYER_VERSION,
            analysis_track=ANALYSIS_TRACK,
            resource_mode=resource_mode,
            resource_id=bundle.resource_id,
            resource_version=bundle.version,
            score_name=MECHANISTIC_SCORE_NAME,
            score_direction=SCORE_DIRECTION,
            specificity_score_name=None,
            min_cells=effective_min_cells,
            validate=False,
        )
        table = validate_long_table(
            _apply_crychic_statuses(table, raw, validate=False)
        )
        view = {
            "run_id": run_id,
            "source_run_id": source_run_id,
            "view_label": MECHANISTIC_SCORE_NAME,
            "contrast_candidates": [],
            "communication_mode": communication_mode,
            "rows": len(table),
            "view_scope": "context_invariant_mechanistic_sample_score",
            "score_layer": MECHANISTIC_SCORE_LAYER_VERSION,
            "score_name": MECHANISTIC_SCORE_NAME,
            "score_formula": "availability * prior_quality * sender_component",
            "downstream_evidence_policy": policy.value,
            "downstream_evidence_role": (
                "independent_annotation_not_required_for_canonical_score"
                if policy is DownstreamEvidencePolicy.ANNOTATE
                else "disabled"
            ),
            "downstream_evidence_location": "result/sample_scores.parquet:downstream",
            "primary_score": True,
            "component_invariance": invariance,
            "source_result_table_digests": source_table_digests,
        }
        return cast(pd.DataFrame, table.loc[:, LONG_TABLE_COLUMNS]), [view]

    candidates = _contrast_candidates(
        sample_scores,
        interactions,
        edges,
        communication_mode=communication_mode,
    )
    score_views = (
        _default_score_views(
            selected,
            interactions,
            edges,
            candidates,
            available_ids,
            communication_mode=communication_mode,
        )
        if scoring_functional_ids is None
        else _explicit_score_views(requested_ids, candidates)
    )
    tables: list[pd.DataFrame] = []
    views: list[dict[str, Any]] = []
    emitted_run_ids: set[str] = set()
    for view in score_views:
        raw = selected.loc[
            selected["scoring_functional_id"].astype(str).isin(view.child_ids)
        ].merge(edges, on="edge_id", how="left", validate="many_to_one")
        if raw[["sender", "receiver", "interaction_id"]].isna().any().any():
            raise ValueError("sample scores contain an unmapped communication edge")
        duplicate_key = ["sample_id", "sender", "receiver", "interaction_id"]
        if raw.duplicated(duplicate_key).any():
            raise ValueError(
                "one score view has repeated sample-edge rows; receiver child "
                "supports must be disjoint and each child must select one "
                "persisted repeat/fold"
            )
        raw["target"] = pd.NA
        observed = raw.loc[raw["status"].eq("ok")].copy()
        observed["score"] = observed["comm_strength"]
        run_identity: dict[str, Any] = {
            "source_run_id": source_run_id,
            "communication_mode": communication_mode,
            "resource_mode": resource_mode,
            "dataset_id": dataset_id,
        }
        if view.grouped:
            receiver_partitions = {
                child_id: list(receivers)
                for child_id, receivers in view.receiver_partitions
            }
            run_identity.update(
                {
                    "child_scoring_functional_ids": list(view.child_ids),
                    "contrast": view.grouped_contrast,
                    "receiver_partition": receiver_partitions,
                    "bridge_semantics_version": RECEIVER_ROW_UNION_BRIDGE_VERSION,
                    "aggregation": "none_row_union_by_receiver",
                    "common_functional_claim": False,
                    "source_result_table_digests": source_table_digests,
                }
            )
        else:
            run_identity["scoring_functional_id"] = view.child_ids[0]
        run_id = canonical_digest(run_identity, prefix="crychic_benchmark_run")
        table = materialize_fixed_universe(
            observed.loc[
                :,
                [
                    "sample_id",
                    "sender",
                    "receiver",
                    "interaction_id",
                    "target",
                    "score",
                ],
            ],
            sample_metadata=sample_metadata,
            support=support,
            resource=resource,
            dataset_id=dataset_id,
            run_id=run_id,
            method_id=METHOD_ID,
            method_version=REQUIRED_SCORE_LAYER_VERSION,
            analysis_track=ANALYSIS_TRACK,
            resource_mode=resource_mode,
            resource_id=bundle.resource_id,
            resource_version=bundle.version,
            score_name=SCORE_NAME,
            score_direction=SCORE_DIRECTION,
            specificity_score_name=None,
            min_cells=effective_min_cells,
            validate=False,
        )
        table = validate_long_table(_apply_crychic_statuses(table, raw, validate=False))
        if not table["run_id"].astype(str).eq(run_id).all():
            raise ValueError("one CRYCHIC score view emitted multiple run IDs")
        if run_id in emitted_run_ids:
            raise ValueError("CRYCHIC score views must emit distinct run IDs")
        emitted_run_ids.add(run_id)
        tables.append(table)
        view_manifest: dict[str, Any] = {
            "run_id": run_id,
            "source_run_id": source_run_id,
            "contrast_candidates": list(view.contrast_candidates),
            "communication_mode": communication_mode,
            "rows": len(table),
            "score_layer": REQUIRED_SCORE_LAYER_VERSION,
            "score_name": SCORE_NAME,
            "downstream_evidence_policy": DownstreamEvidencePolicy.REQUIRED.value,
            "downstream_evidence_role": "required_historical_diagnostic",
            "primary_score": True,
        }
        if view.grouped:
            view_manifest.update(
                {
                    "child_scoring_functional_ids": list(view.child_ids),
                    "child_receiver_partitions": receiver_partitions,
                    "view_scope": "contrast_level_receiver_row_union",
                    "bridge_semantics_version": RECEIVER_ROW_UNION_BRIDGE_VERSION,
                    "aggregation": "none_row_union_by_receiver",
                    "common_functional_claim": False,
                    "provenance_status": "legacy_reconstructed_fail_closed",
                    "source_result_table_digests": source_table_digests,
                }
            )
        else:
            view_manifest.update(
                {
                    "scoring_functional_id": view.child_ids[0],
                    "view_scope": "single_scoring_functional",
                }
            )
        views.append(view_manifest)
    combined = pd.concat(tables, ignore_index=True)
    return cast(pd.DataFrame, combined.loc[:, LONG_TABLE_COLUMNS]), views


def _safe_output(result_dir: Path, output_dir: Path) -> None:
    result = result_dir.resolve()
    output = output_dir.resolve()
    if result == output or result in output.parents or output in result.parents:
        raise ValueError(
            "adapter output and CRYCHIC result directories must be disjoint"
        )


def export_result(
    result_dir: str | Path,
    input_h5ad: str | Path,
    output_dir: str | Path,
    bundle: ResourceBundle,
    *,
    dataset_id: str,
    resource_mode: str,
    communication_mode: str = "state",
    scoring_functional_ids: Sequence[str] | None = None,
    downstream_evidence_policy: DownstreamEvidencePolicy | str = (
        DownstreamEvidencePolicy.REQUIRED
    ),
    min_cells: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Load, normalize, and write a CRYCHIC benchmark adapter artifact."""

    started = time.perf_counter()
    policy = DownstreamEvidencePolicy(downstream_evidence_policy)
    result_path = Path(result_dir)
    input_path = Path(input_h5ad)
    output_path = Path(output_dir)
    _safe_output(result_path, output_path)
    output = prepare_output(output_path, overwrite=overwrite)
    result = CrychicResult.load(result_path)
    adata = ad.read_h5ad(input_path, backed="r")
    try:
        sample_scores = result.read_table(
            "sample_scores",
            columns=["sample_id", "subject_id", "context_json"],
        )
        sample_metadata, _ = _result_samples(result, sample_scores, adata)
        manifest = begin_manifest(
            repo_root=Path(__file__).resolve().parents[3],
            dataset_id=dataset_id,
            method={"id": METHOD_ID, "version": _method_version(result)},
            environment=python_environment(
                environment_name="crychic_project_uv",
                packages=("CRYCHIC", "anndata", "pandas", "pyarrow"),
                threads=1,
            ),
            input_path=input_path,
            input_shape=adata.shape,
            sample_metadata=sample_metadata,
            input_keys={
                key: result.config[key]
                for key in (
                    "sample_key",
                    "subject_key",
                    "cell_type_key",
                    "context_keys",
                )
            },
            resource={
                "mode": resource_mode,
                "id": bundle.resource_id,
                "version": bundle.version,
                "manifest_digest": bundle.manifest_digest,
                "interactions": len(bundle.interactions),
            },
            parameters={
                "communication_mode": communication_mode,
                "scoring_functional_ids": (
                    None
                    if scoring_functional_ids is None
                    else list(scoring_functional_ids)
                ),
                "downstream_evidence_policy": policy.value,
                "min_cells": min_cells,
                "source_result_run_id": result.manifest["run_id"],
            },
            score_semantics={
                "name": (
                    SCORE_NAME
                    if policy is DownstreamEvidencePolicy.REQUIRED
                    else MECHANISTIC_SCORE_NAME
                ),
                "direction": SCORE_DIRECTION,
                "interpretation": (
                    "exploratory_downstream_confirmed_strength_not_probability"
                    if policy is DownstreamEvidencePolicy.REQUIRED
                    else "exploratory_mechanistic_lr_strength_not_probability"
                ),
                "downstream_evidence_policy": policy.value,
                "probability": None,
                "p_value": None,
                "q_value": None,
                "within_dataset_p_value": "not_emitted",
            },
        )
        table, views = convert_result_to_long(
            result,
            adata,
            bundle,
            dataset_id=dataset_id,
            resource_mode=resource_mode,
            communication_mode=communication_mode,
            scoring_functional_ids=scoring_functional_ids,
            downstream_evidence_policy=policy,
            min_cells=min_cells,
        )
    finally:
        if adata.isbacked:
            adata.file.close()
    score_head_versions = tuple(
        sorted(
            {
                str(view["score_layer"])
                for view in views
                if isinstance(view.get("score_layer"), str)
            }
        )
    )
    if len(score_head_versions) != 1:
        raise ValueError("CRYCHIC long table must have one score-head version")
    if set(table["method_version"].astype(str)) != {score_head_versions[0]}:
        raise ValueError("CRYCHIC long table score-head version is inconsistent")
    cast(dict[str, Any], manifest["method"])["score_head_version"] = (
        score_head_versions[0]
    )
    manifest["source_result"] = {
        "run_id": result.manifest["run_id"],
        "result_schema_version": result.manifest["result_schema_version"],
        "resource_digests": dict(result.manifest["resource_digests"]),
        "score_views": views,
    }
    return cast(
        dict[str, Any], finalize_manifest(manifest, table, output, started=started)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a persisted CRYCHIC result to the benchmark long table"
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--resource-mode", choices=("H-common", "native"), required=True
    )
    parser.add_argument(
        "--communication-mode", choices=("state", "ecosystem"), default="state"
    )
    parser.add_argument("--scoring-functional-id", action="append")
    parser.add_argument(
        "--downstream-evidence-policy",
        choices=(
            DownstreamEvidencePolicy.DISABLED.value,
            DownstreamEvidencePolicy.ANNOTATE.value,
            DownstreamEvidencePolicy.REQUIRED.value,
        ),
        default=DownstreamEvidencePolicy.REQUIRED.value,
    )
    parser.add_argument("--min-cells", type=int)
    parser.add_argument("--harmonized-resource", type=Path)
    parser.add_argument("--harmonized-manifest", type=Path)
    parser.add_argument("--native-adapter", choices=("cellchat", "cellphonedb"))
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--native-manifest", type=Path)
    parser.add_argument("--species", default="human")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _bundle_from_args(args: argparse.Namespace) -> ResourceBundle:
    if args.resource_mode == "H-common":
        if args.harmonized_resource is None or args.harmonized_manifest is None:
            raise ValueError(
                "H-common requires --harmonized-resource and --harmonized-manifest"
            )
        return harmonized_resource_bundle(
            args.harmonized_resource, args.harmonized_manifest
        )
    required = (args.native_adapter, args.database_root, args.native_manifest)
    if any(value is None for value in required):
        raise ValueError(
            "native readback requires --native-adapter, --database-root, and "
            "--native-manifest"
        )
    return load_native_resource_bundle(
        args.native_adapter,
        database_root=args.database_root,
        manifest_path=args.native_manifest,
        species=args.species,
    )


def main() -> None:
    args = _parser().parse_args()
    bundle = _bundle_from_args(args)
    manifest = export_result(
        args.result_dir,
        args.input_h5ad,
        args.output_dir,
        bundle,
        dataset_id=args.dataset_id,
        resource_mode=args.resource_mode,
        communication_mode=args.communication_mode,
        scoring_functional_ids=args.scoring_functional_id,
        downstream_evidence_policy=args.downstream_evidence_policy,
        min_cells=args.min_cells,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": manifest["status"], "output": manifest["output"]}))


if __name__ == "__main__":
    main()
