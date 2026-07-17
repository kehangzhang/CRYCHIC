"""Read persisted cross-fit descriptive scores into the benchmark long schema.

This bridge intentionally exports only native sender-LR rows.  The persisted
collection is receiver balanced, but it is not one common scoring functional
across receivers and therefore is not eligible for a global cross-receiver
ranking endpoint.
"""

from __future__ import annotations

import importlib.metadata
import math
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from benchmarks.adapters.common import (
    LONG_TABLE_COLUMNS,
    LONG_TABLE_SCHEMA,
    canonical_digest,
    validate_long_table,
)
from crychic.attribution import GAIN_CALIBRATION_PERCENTILE_POLICY
from crychic.core import stable_id
from crychic.design import context_fields
from crychic.workflow import CrossFitResult

METHOD_ID = "crychic"
ANALYSIS_TRACK = "crossfit_receiver_balanced_descriptive"
RESOURCE_MODE = "native"
RESOURCE_ID = "crychic_crossfit_native_sender_lr_rows"
SCORE_NAME = "global_sender_lr_score"
SCORE_DIRECTION = "higher"
BENCHMARK_SCOPE = "receiver_stratified_or_receiver_macro_only"
_V3_SCHEMA_VERSION = "3.0.0"
_V4_SCHEMA_VERSION = "4.0.0"
_V5_SCHEMA_VERSION = "5.0.0"
_V6_SCHEMA_VERSION = "6.0.0"
_LEGACY_GAIN_CALIBRATED_SCHEMA_VERSIONS = frozenset(
    {_V4_SCHEMA_VERSION, _V5_SCHEMA_VERSION}
)
_GAIN_CALIBRATED_SCHEMA_VERSIONS = frozenset(
    {*_LEGACY_GAIN_CALIBRATED_SCHEMA_VERSIONS, _V6_SCHEMA_VERSION}
)
_SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {_V3_SCHEMA_VERSION, *_GAIN_CALIBRATED_SCHEMA_VERSIONS}
)
_V4_CALIBRATION_POLICY = "selected_penalty_inner_oof_positive_gain_ecdf_v1"
_V4_SCALE_POLICY = "selected_penalty_inner_oof_gain_percentile_no_heldout_rescaling_v3"
_V4_SENDER_POLICY = "absolute_sender_evidence_without_candidate_normalization_v1"
_V6_SENDER_POLICY = "contrast_common_frozen_softmax_conserved_allocation_v1"
_V6_SCORE_VERSION = "receiver_gain_percentile_mechanistic_conserved_sender_v4"
_V4_RETIRED_CALIBRATION_FIELDS = frozenset(
    {
        "calibration_quantile",
        "global_training_anchor",
        "receiver_training_anchors",
        "receiver_scale_factors",
    }
)
_FORMAL_INFERENCE_STATUS = "not_available_descriptive_only"
_SOURCE_STATUSES = frozenset({"observed", "structural_zero", "not_estimable"})
_GAIN_CALIBRATION_STATUSES = frozenset({"observed", "not_estimable"})
_GAIN_CALIBRATION_BINDING_FIELDS = frozenset(
    {
        "gain_calibration_binding_id",
        "receiver",
        "source_family_common_functional_id",
        "gain_calibration_artifact_id",
        "gain_calibration_spec_id",
        "gain_calibration_status",
        "gain_calibration_reason_code",
        "percentile_policy",
        "positive_gain_source_knots",
        "positive_gain_percentile_knots",
        "n_supported_families",
        "n_positive_observations",
        "n_distinct_positive_gains",
        "tuning_id",
        "outer_incremental_functional_id",
        "outer_selected_resolved_penalty_id",
    }
)
_SOURCE_ROW_KEY = (
    "contrast_common_collection_id",
    "global_common_application_id",
    "fold_id",
    "sample_id",
    "receiver",
    "sender",
    "interaction_id",
    "mode",
)
_LR_ROW_KEY = (
    "contrast_common_collection_id",
    "global_common_application_id",
    "fold_id",
    "sample_id",
    "receiver",
    "interaction_id",
    "mode",
)

_CONTRAST_COMMON_PREFIX_COLUMNS = (
    "crossfit_id",
    "spec_id",
    "repeat_id",
    "fold_id",
    "contrast_id",
    "contrast",
    "contrast_common_collection_id",
    "global_common_application_id",
)
_CONTRAST_COMMON_SUFFIX_COLUMNS = (
    "certification_status",
    "is_oof_certified",
    "formal_inference_status",
    "claim_scope",
    "source_row_id",
)
_V3_GLOBAL_COMMON_LR_SCORE_COLUMNS = (
    "global_common_functional_id",
    "source_family_common_functional_id",
    "source_family_common_application_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "receptor_eligible",
    "ligand_contrast_gate_status",
    "availability",
    "receiver_relative_family_gain",
    "prior_quality",
    "training_family_coefficient",
    "training_receiver_scale_factor",
    "global_lr_core_strength",
    "global_lr_score",
    "status",
    "reason_code",
    "score_version",
)
_V3_GLOBAL_COMMON_SENDER_SCORE_COLUMNS = (
    "global_common_functional_id",
    "source_family_common_functional_id",
    "source_family_common_application_id",
    "source_sender_functional_id",
    "source_sender_application_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "sender",
    "global_lr_score",
    "training_receiver_scale_factor",
    "ligand_availability",
    "training_prevalence_prior",
    "raw_sender_evidence",
    "global_sender_lr_score",
    "status",
    "reason_code",
    "score_version",
)
_V3_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *_V3_GLOBAL_COMMON_LR_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)
_V3_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *_V3_GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)

# Result schemas v4/v5 used gain-percentile LR calibration but retained the
# non-conserving raw-sender multiplier. Keep that exact historical layout local
# to the reader so a runtime schema upgrade cannot reinterpret old bundles.
_V45_GLOBAL_COMMON_LR_SCORE_COLUMNS = (
    "global_common_functional_id",
    "source_family_common_functional_id",
    "source_family_common_application_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "receptor_eligible",
    "ligand_contrast_gate_status",
    "availability",
    "receiver_relative_family_gain",
    "calibrated_family_gain_percentile",
    "gain_calibration_binding_id",
    "gain_calibration_artifact_id",
    "gain_calibration_status",
    "gain_calibration_reason_code",
    "prior_quality",
    "training_family_coefficient",
    "global_lr_core_strength",
    "global_lr_score",
    "status",
    "reason_code",
    "score_version",
)
_V45_GLOBAL_COMMON_SENDER_SCORE_COLUMNS = (
    "global_common_functional_id",
    "source_family_common_functional_id",
    "source_family_common_application_id",
    "source_sender_functional_id",
    "source_sender_application_id",
    "sample_id",
    "subject_id",
    "context_id",
    "receiver",
    "family_id",
    "driver_id",
    "interaction_id",
    "mode",
    "sender",
    "global_lr_score",
    "gain_calibration_binding_id",
    "gain_calibration_artifact_id",
    "gain_calibration_status",
    "gain_calibration_reason_code",
    "ligand_availability",
    "training_prevalence_prior",
    "raw_sender_evidence",
    "global_sender_lr_score",
    "status",
    "reason_code",
    "score_version",
)
_V45_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *_V45_GLOBAL_COMMON_LR_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)
_V45_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *_V45_GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)

# V6 changes only sender resolution: the LR parent remains gain calibrated,
# while the persisted sender table carries the exact conserved softmax weight.
_V6_GLOBAL_COMMON_LR_SCORE_COLUMNS = _V45_GLOBAL_COMMON_LR_SCORE_COLUMNS
_V6_GLOBAL_COMMON_SENDER_SCORE_COLUMNS = (
    *_V45_GLOBAL_COMMON_SENDER_SCORE_COLUMNS[:22],
    "assignment_weight",
    "normalized_entropy",
    *_V45_GLOBAL_COMMON_SENDER_SCORE_COLUMNS[22:],
)
_V6_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *_V6_GLOBAL_COMMON_LR_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)
_V6_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS = (
    *_CONTRAST_COMMON_PREFIX_COLUMNS,
    *_V6_GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    *_CONTRAST_COMMON_SUFFIX_COLUMNS,
)


class CrossFitResultReader(Protocol):
    """Narrow read-only interface accepted by the benchmark bridge."""

    @property
    def manifest(self) -> Mapping[str, object]: ...

    def read_contrast_common_sender_lr_scores(self) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class _GainCalibrationBinding:
    binding_id: str
    receiver: str
    source_family_common_functional_id: str
    artifact_id: str | None
    status: str
    reason_code: str | None
    percentile_policy: str
    source_knots: tuple[float, ...]
    percentile_knots: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _ApplicationLineage:
    schema_version: str
    crossfit_id: str
    spec_id: str
    repeat_id: str
    collection_id: str
    contrast_id: str
    contrast: str
    collection_score_version: str
    collection_certification_status: str
    collection_is_oof_certified: bool
    collection_claim_scope: str
    fold_id: str
    functional_id: str
    application_id: str
    context_ids: frozenset[str]
    heldout_subject_ids: frozenset[str]
    receiver_children: Mapping[str, tuple[str, str]]
    receiver_scale_factors: Mapping[str, float]
    receiver_gain_calibration_bindings: Mapping[str, _GainCalibrationBinding]
    cross_receiver_percentile_rank_eligible: bool
    sender_functional_id: str
    sender_applications: frozenset[tuple[str, str]]
    n_lr_rows: int
    n_sender_rows: int


@dataclass(frozen=True, slots=True)
class _CollectionLineage:
    collection_id: str
    contrast_id: str
    contrast: str
    score_version: str
    applications: tuple[_ApplicationLineage, ...]


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _required_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field=field)


def _required_string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result = tuple(_required_string(item, field=field) for item in value)
    if not result or result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must be a sorted unique non-empty array")
    return result


def _positive_count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _unit_knots(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field} values must be numeric")
        numeric = float(item)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{field} values must lie in [0, 1]")
        result.append(numeric)
    return tuple(result)


def _parse_receiver_children(
    value: object,
    *,
    receiver_ids: tuple[str, ...],
) -> dict[str, tuple[str, str]]:
    if not isinstance(value, list):
        raise ValueError("receiver_children must be an array")
    result: dict[str, tuple[str, str]] = {}
    observed_order: list[str] = []
    for raw_child in value:
        child = _required_mapping(raw_child, field="receiver_children")
        receiver = _required_string(child.get("receiver"), field="receiver")
        functional_id = _required_string(
            child.get("family_common_functional_id"),
            field="family_common_functional_id",
        )
        application_id = _required_string(
            child.get("family_common_application_id"),
            field="family_common_application_id",
        )
        if receiver in result:
            raise ValueError("receiver_children contains duplicate receivers")
        observed_order.append(receiver)
        result[receiver] = (functional_id, application_id)
    if tuple(observed_order) != receiver_ids:
        raise ValueError("receiver_children lacks exact receiver coverage")
    return result


def _parse_receiver_factors(
    value: object,
    *,
    receiver_ids: tuple[str, ...],
) -> dict[str, float]:
    if not isinstance(value, list):
        raise ValueError("receiver_scale_factors must be an array")
    result: dict[str, float] = {}
    observed_order: list[str] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("receiver_scale_factors contains an invalid entry")
        receiver = _required_string(item[0], field="receiver_scale_factors.receiver")
        raw_factor = item[1]
        if isinstance(raw_factor, bool) or not isinstance(raw_factor, (int, float)):
            raise ValueError("receiver scale factor must be numeric")
        factor = float(raw_factor)
        if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
            raise ValueError("receiver scale factor must lie in [0, 1]")
        if receiver in result:
            raise ValueError("receiver_scale_factors contains duplicate receivers")
        observed_order.append(receiver)
        result[receiver] = factor
    if tuple(observed_order) != receiver_ids:
        raise ValueError("receiver_scale_factors lacks exact receiver coverage")
    return result


def _parse_gain_calibration_bindings(
    value: object,
    *,
    receiver_ids: tuple[str, ...],
    receiver_children: Mapping[str, tuple[str, str]],
) -> dict[str, _GainCalibrationBinding]:
    if not isinstance(value, list) or len(value) != len(receiver_ids):
        raise ValueError(
            "receiver_gain_calibration_bindings lacks exact receiver coverage"
        )
    result: dict[str, _GainCalibrationBinding] = {}
    observed_order: list[str] = []
    for expected_receiver, raw_binding in zip(receiver_ids, value, strict=True):
        binding = _required_mapping(
            raw_binding, field="receiver_gain_calibration_bindings"
        )
        if set(binding) != _GAIN_CALIBRATION_BINDING_FIELDS:
            raise ValueError("receiver gain calibration binding fields are invalid")
        receiver = _required_string(binding.get("receiver"), field="receiver")
        source_functional_id = _required_string(
            binding.get("source_family_common_functional_id"),
            field="source_family_common_functional_id",
        )
        if (
            receiver != expected_receiver
            or source_functional_id != receiver_children[receiver][0]
        ):
            raise ValueError("gain calibration binding parent lineage is inconsistent")
        binding_id = _required_string(
            binding.get("gain_calibration_binding_id"),
            field="gain_calibration_binding_id",
        )
        artifact_id = _optional_string(
            binding.get("gain_calibration_artifact_id"),
            field="gain_calibration_artifact_id",
        )
        spec_id = _optional_string(
            binding.get("gain_calibration_spec_id"),
            field="gain_calibration_spec_id",
        )
        status = _required_string(
            binding.get("gain_calibration_status"),
            field="gain_calibration_status",
        )
        reason_code = _optional_string(
            binding.get("gain_calibration_reason_code"),
            field="gain_calibration_reason_code",
        )
        percentile_policy = _required_string(
            binding.get("percentile_policy"), field="percentile_policy"
        )
        if percentile_policy != GAIN_CALIBRATION_PERCENTILE_POLICY:
            raise ValueError("gain calibration percentile policy is inconsistent")
        source_knots = _unit_knots(
            binding.get("positive_gain_source_knots"),
            field="positive_gain_source_knots",
        )
        percentile_knots = _unit_knots(
            binding.get("positive_gain_percentile_knots"),
            field="positive_gain_percentile_knots",
        )
        if len(source_knots) != len(percentile_knots):
            raise ValueError("gain calibration knot arrays do not align")
        supported = _nonnegative_count(
            binding.get("n_supported_families"), field="n_supported_families"
        )
        positives = _nonnegative_count(
            binding.get("n_positive_observations"),
            field="n_positive_observations",
        )
        distinct = _nonnegative_count(
            binding.get("n_distinct_positive_gains"),
            field="n_distinct_positive_gains",
        )
        for field in (
            "tuning_id",
            "outer_incremental_functional_id",
            "outer_selected_resolved_penalty_id",
        ):
            _optional_string(binding.get(field), field=field)
        if status == "observed":
            if (
                reason_code is not None
                or artifact_id is None
                or spec_id is None
                or len(source_knots) < 2
                or source_knots[0] != 0.0
                or percentile_knots[0] != 0.0
                or not math.isclose(percentile_knots[-1], 1.0)
                or any(right <= left for left, right in pairwise(source_knots))
                or any(right <= left for left, right in pairwise(percentile_knots))
                or supported <= 0
                or positives < distinct
                or distinct != len(source_knots) - 1
            ):
                raise ValueError("observed gain calibration binding is invalid")
        elif status == "not_estimable":
            if (
                reason_code is None
                or source_knots
                or percentile_knots
                or (artifact_id is None) != (spec_id is None)
            ):
                raise ValueError("not-estimable gain calibration binding is invalid")
        else:
            raise ValueError(f"unsupported gain calibration status: {status!r}")
        payload = {
            key: item
            for key, item in binding.items()
            if key != "gain_calibration_binding_id"
        }
        if binding_id != stable_id(
            "receiver_gain_calibration_binding", payload, schema_version="1"
        ):
            raise ValueError("gain calibration binding identity is inconsistent")
        if receiver in result:
            raise ValueError(
                "receiver_gain_calibration_bindings contains duplicate receivers"
            )
        observed_order.append(receiver)
        result[receiver] = _GainCalibrationBinding(
            binding_id=binding_id,
            receiver=receiver,
            source_family_common_functional_id=source_functional_id,
            artifact_id=artifact_id,
            status=status,
            reason_code=reason_code,
            percentile_policy=percentile_policy,
            source_knots=source_knots,
            percentile_knots=percentile_knots,
        )
    if tuple(observed_order) != receiver_ids:
        raise ValueError(
            "receiver_gain_calibration_bindings lacks exact receiver coverage"
        )
    return result


def _parse_sender_lineages(
    value: object,
    *,
    receiver_ids: tuple[str, ...],
    sender_functional_id: str,
) -> frozenset[tuple[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("sender_lineages must be a non-empty array")
    canonical: list[tuple[str, str, str]] = []
    result: set[tuple[str, str]] = set()
    for raw_lineage in value:
        lineage = _required_mapping(raw_lineage, field="sender_lineages")
        receiver = _required_string(lineage.get("receiver"), field="receiver")
        functional_id = _required_string(
            lineage.get("sender_functional_id"), field="sender_functional_id"
        )
        application_id = _required_string(
            lineage.get("sender_application_id"), field="sender_application_id"
        )
        if functional_id != sender_functional_id:
            raise ValueError("sender lineage functional identity is inconsistent")
        canonical.append((receiver, functional_id, application_id))
        result.add((receiver, application_id))
    if canonical != sorted(set(canonical)) or {item[0] for item in result} != set(
        receiver_ids
    ):
        raise ValueError("sender_lineages is noncanonical or lacks receiver coverage")
    return frozenset(result)


def _application_lineage(
    raw_application: object,
    *,
    schema_version: str,
    collection_id: str,
    contrast_id: str,
    contrast: str,
    score_version: str,
    certification_status: str,
    is_oof_certified: bool,
    claim_scope: str,
    crossfit_id: str,
    spec_id: str,
    repeat_id: str,
) -> _ApplicationLineage:
    application = _required_mapping(
        raw_application, field="contrast_common fold application"
    )
    if (
        application.get("common_functional_across_receivers") is not False
        or application.get("receiver_balanced_descriptive_collection") is not True
        or application.get("training_only_receiver_calibration") is not True
        or application.get("receiver_scale_amplification") is not False
    ):
        raise ValueError("contrast-common application scope flags are invalid")
    receiver_ids = _required_string_list(
        application.get("receiver_ids"), field="receiver_ids"
    )
    receiver_children = _parse_receiver_children(
        application.get("receiver_children"), receiver_ids=receiver_ids
    )
    factors: Mapping[str, float]
    bindings: Mapping[str, _GainCalibrationBinding]
    cross_receiver_percentile_rank_eligible: bool
    if schema_version == _V3_SCHEMA_VERSION:
        factors = _parse_receiver_factors(
            application.get("receiver_scale_factors"), receiver_ids=receiver_ids
        )
        bindings = {}
        cross_receiver_percentile_rank_eligible = False
    else:
        if set(application).intersection(_V4_RETIRED_CALIBRATION_FIELDS):
            raise ValueError(
                "gain-calibrated application contains retired scale-factor fields"
            )
        expected_sender_policy = (
            _V6_SENDER_POLICY
            if schema_version == _V6_SCHEMA_VERSION
            else _V4_SENDER_POLICY
        )
        if (
            application.get("calibration_policy") != _V4_CALIBRATION_POLICY
            or application.get("scale_policy") != _V4_SCALE_POLICY
            or application.get("sender_policy") != expected_sender_policy
        ):
            raise ValueError(
                "gain-calibrated contrast-common calibration policies are invalid"
            )
        if schema_version == _V6_SCHEMA_VERSION and (
            score_version != _V6_SCORE_VERSION
            or application.get("functional_schema_version") != "4.0.0"
        ):
            raise ValueError(
                "v6 conserved-sender score version or functional schema is invalid"
            )
        for field in ("softmin_power", "epsilon"):
            value = application.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{field} must be a positive finite number")
        factors = {}
        bindings = _parse_gain_calibration_bindings(
            application.get("receiver_gain_calibration_bindings"),
            receiver_ids=receiver_ids,
            receiver_children=receiver_children,
        )
        all_calibrated = all(
            binding.status == "observed" for binding in bindings.values()
        )
        if (
            type(application.get("all_receivers_gain_calibrated")) is not bool
            or application.get("all_receivers_gain_calibrated") is not all_calibrated
            or type(application.get("cross_receiver_percentile_rank_eligible"))
            is not bool
            or application.get("cross_receiver_percentile_rank_eligible")
            is not all_calibrated
        ):
            raise ValueError("receiver gain calibration eligibility flags are invalid")
        cross_receiver_percentile_rank_eligible = all_calibrated
    sender_functional_id = _required_string(
        application.get("sender_functional_id"), field="sender_functional_id"
    )
    sender_applications = _parse_sender_lineages(
        application.get("sender_lineages"),
        receiver_ids=receiver_ids,
        sender_functional_id=sender_functional_id,
    )
    training_subject_ids = frozenset(
        _required_string_list(
            application.get("training_subject_ids"), field="training_subject_ids"
        )
    )
    heldout_subject_ids = frozenset(
        _required_string_list(
            application.get("heldout_subject_ids"), field="heldout_subject_ids"
        )
    )
    if training_subject_ids.intersection(heldout_subject_ids):
        raise ValueError("contrast-common training and held-out subjects overlap")
    return _ApplicationLineage(
        schema_version=schema_version,
        crossfit_id=crossfit_id,
        spec_id=spec_id,
        repeat_id=repeat_id,
        collection_id=collection_id,
        contrast_id=contrast_id,
        contrast=contrast,
        collection_score_version=score_version,
        collection_certification_status=certification_status,
        collection_is_oof_certified=is_oof_certified,
        collection_claim_scope=claim_scope,
        fold_id=_required_string(application.get("fold_id"), field="fold_id"),
        functional_id=_required_string(
            application.get("global_common_functional_id"),
            field="global_common_functional_id",
        ),
        application_id=_required_string(
            application.get("global_common_application_id"),
            field="global_common_application_id",
        ),
        context_ids=frozenset(
            _required_string_list(application.get("context_ids"), field="context_ids")
        ),
        heldout_subject_ids=heldout_subject_ids,
        receiver_children=receiver_children,
        receiver_scale_factors=factors,
        receiver_gain_calibration_bindings=bindings,
        cross_receiver_percentile_rank_eligible=(
            cross_receiver_percentile_rank_eligible
        ),
        sender_functional_id=sender_functional_id,
        sender_applications=sender_applications,
        n_lr_rows=_positive_count(application.get("n_lr_rows"), field="n_lr_rows"),
        n_sender_rows=_positive_count(
            application.get("n_sender_rows"), field="n_sender_rows"
        ),
    )


def _manifest_lineage(
    manifest: Mapping[str, object],
) -> tuple[
    str,
    tuple[_CollectionLineage, ...],
    dict[str, _ApplicationLineage],
]:
    schema_version = manifest.get("schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            "cross-fit benchmark readback requires manifest schema 3.0.0, "
            "4.0.0, 5.0.0, or 6.0.0"
        )
    if manifest.get("status") != "complete":
        raise ValueError("cross-fit result manifest is not complete")
    if manifest.get("contrast_common_stage_connected") is not True:
        raise ValueError("cross-fit result has no connected contrast-common stage")
    crossfit_id = _required_string(manifest.get("crossfit_id"), field="crossfit_id")
    spec_id = _required_string(manifest.get("spec_id"), field="spec_id")
    repeat_id = _required_string(manifest.get("repeat_id"), field="repeat_id")
    raw_collections = manifest.get("contrast_common_collections")
    if not isinstance(raw_collections, list) or not raw_collections:
        raise ValueError("contrast_common_collections must be a non-empty array")
    collections: list[_CollectionLineage] = []
    applications: dict[str, _ApplicationLineage] = {}
    canonical_scopes: list[tuple[str, str]] = []
    for raw_collection in raw_collections:
        collection = _required_mapping(
            raw_collection, field="contrast_common collection"
        )
        collection_id = _required_string(
            collection.get("contrast_common_collection_id"),
            field="contrast_common_collection_id",
        )
        contrast_id = _required_string(
            collection.get("contrast_id"), field="contrast_id"
        )
        contrast = _required_string(collection.get("contrast"), field="contrast")
        score_version = _required_string(
            collection.get("score_version"), field="score_version"
        )
        certification_status = _required_string(
            collection.get("certification_status"), field="certification_status"
        )
        claim_scope = _required_string(
            collection.get("claim_scope"), field="claim_scope"
        )
        is_oof_certified = collection.get("is_oof_certified")
        if type(is_oof_certified) is not bool:
            raise ValueError("collection is_oof_certified must be boolean")
        if (
            collection.get("crossfit_id") != crossfit_id
            or collection.get("spec_id") != spec_id
            or collection.get("repeat_id") != repeat_id
            or collection.get("common_functional_across_receivers") is not False
            or collection.get("receiver_balanced_descriptive_collection") is not True
            or collection.get("formal_inference_allowed") is not False
        ):
            raise ValueError("contrast-common collection scope or lineage is invalid")
        payload = {
            key: value
            for key, value in collection.items()
            if key != "contrast_common_collection_id"
        }
        if collection_id != stable_id(
            "contrast_common_oof_score_collection", payload, schema_version="1"
        ):
            raise ValueError("contrast-common collection identity is inconsistent")
        raw_applications = collection.get("fold_applications")
        if not isinstance(raw_applications, list) or not raw_applications:
            raise ValueError("contrast-common collection has no fold applications")
        parsed_applications: list[_ApplicationLineage] = []
        for raw_application in raw_applications:
            parsed = _application_lineage(
                raw_application,
                schema_version=schema_version,
                collection_id=collection_id,
                contrast_id=contrast_id,
                contrast=contrast,
                score_version=score_version,
                certification_status=certification_status,
                is_oof_certified=is_oof_certified,
                claim_scope=claim_scope,
                crossfit_id=crossfit_id,
                spec_id=spec_id,
                repeat_id=repeat_id,
            )
            if parsed.application_id in applications:
                raise ValueError("global common application IDs must be unique")
            applications[parsed.application_id] = parsed
            parsed_applications.append(parsed)
        fold_ids = tuple(item.fold_id for item in parsed_applications)
        if fold_ids != tuple(sorted(set(fold_ids))):
            raise ValueError("contrast-common fold applications are noncanonical")
        collections.append(
            _CollectionLineage(
                collection_id=collection_id,
                contrast_id=contrast_id,
                contrast=contrast,
                score_version=score_version,
                applications=tuple(parsed_applications),
            )
        )
        canonical_scopes.append((contrast, contrast_id))
    if canonical_scopes != sorted(set(canonical_scopes)):
        raise ValueError("contrast-common collections are noncanonical or duplicated")
    return schema_version, tuple(collections), applications


def _lineage_registry(applications: Mapping[str, _ApplicationLineage]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for application in applications.values():
        for receiver, (
            child_functional,
            child_application,
        ) in application.receiver_children.items():
            binding = application.receiver_gain_calibration_bindings.get(receiver)
            rows.append(
                {
                    "contrast_common_collection_id": application.collection_id,
                    "global_common_application_id": application.application_id,
                    "fold_id": application.fold_id,
                    "receiver": receiver,
                    "_expected_crossfit_id": application.crossfit_id,
                    "_expected_spec_id": application.spec_id,
                    "_expected_repeat_id": application.repeat_id,
                    "_expected_contrast_id": application.contrast_id,
                    "_expected_contrast": application.contrast,
                    "_expected_score_version": (application.collection_score_version),
                    "_expected_functional_id": application.functional_id,
                    "_expected_child_functional_id": child_functional,
                    "_expected_child_application_id": child_application,
                    "_expected_sender_functional_id": (
                        application.sender_functional_id
                    ),
                    "_expected_factor": application.receiver_scale_factors.get(
                        receiver
                    ),
                    "_expected_gain_calibration_binding_id": (
                        None if binding is None else binding.binding_id
                    ),
                    "_expected_gain_calibration_artifact_id": (
                        None if binding is None else binding.artifact_id
                    ),
                    "_expected_gain_calibration_status": (
                        None if binding is None else binding.status
                    ),
                    "_expected_gain_calibration_reason_code": (
                        None if binding is None else binding.reason_code
                    ),
                    "_expected_certification_status": (
                        application.collection_certification_status
                    ),
                    "_expected_is_oof_certified": (
                        application.collection_is_oof_certified
                    ),
                    "_expected_claim_scope": application.collection_claim_scope,
                }
            )
    return pd.DataFrame.from_records(rows)


def _require_exact_source_columns(
    table: pd.DataFrame,
    expected: tuple[str, ...],
    *,
    name: str,
    schema_version: str,
) -> pd.DataFrame:
    if tuple(table.columns) != expected:
        raise ValueError(
            f"{name} does not match the released {schema_version} table schema"
        )
    return table.copy(deep=True)


def _compare_lineage_columns(table: pd.DataFrame) -> None:
    comparisons = {
        "crossfit_id": "_expected_crossfit_id",
        "spec_id": "_expected_spec_id",
        "repeat_id": "_expected_repeat_id",
        "contrast_id": "_expected_contrast_id",
        "contrast": "_expected_contrast",
        "score_version": "_expected_score_version",
        "global_common_functional_id": "_expected_functional_id",
        "source_family_common_functional_id": "_expected_child_functional_id",
        "source_family_common_application_id": "_expected_child_application_id",
        "certification_status": "_expected_certification_status",
        "is_oof_certified": "_expected_is_oof_certified",
        "claim_scope": "_expected_claim_scope",
    }
    for observed, expected in comparisons.items():
        if not table[observed].eq(table[expected]).all():
            raise ValueError(f"contrast-common table {observed} lineage mismatch")


def _validate_scores(table: pd.DataFrame, *, score_column: str) -> None:
    statuses = set(table["status"].astype(str))
    invalid = statuses.difference(_SOURCE_STATUSES)
    if invalid:
        raise ValueError(f"unsupported contrast-common statuses: {sorted(invalid)}")
    numeric = pd.to_numeric(table[score_column], errors="coerce")
    supplied = table[score_column].notna()
    if (supplied & numeric.isna()).any() or np.isinf(numeric.fillna(0.0)).any():
        raise ValueError(f"{score_column} must contain finite numeric scores")
    observed = table["status"].eq("observed")
    structural = table["status"].eq("structural_zero")
    not_estimable = table["status"].eq("not_estimable")
    if (
        numeric[observed].isna().any()
        or ((numeric[observed] < 0.0) | (numeric[observed] > 1.0)).any()
    ):
        raise ValueError("observed contrast-common scores must lie in [0, 1]")
    if (
        numeric[structural].isna().any()
        or not np.equal(numeric[structural].to_numpy(dtype=float), 0.0).all()
    ):
        raise ValueError("structural_zero rows must carry an exact zero score")
    if numeric[not_estimable].notna().any():
        raise ValueError("not_estimable rows must not carry a score")


def _optional_row_string(value: object, *, field: str) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(cast(Any, value))):
            return None
    except (TypeError, ValueError):
        pass
    return _required_string(value, field=field)


def _optional_unit_value(value: object, *, field: str) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(cast(Any, value))):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric or missing")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1] or be missing")
    return numeric


def _same_optional_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _gain_percentile(
    raw_gain: float | None,
    binding: _GainCalibrationBinding,
) -> float | None:
    if raw_gain is None or binding.artifact_id is None:
        return None
    if raw_gain == 0.0:
        return 0.0
    if binding.status != "observed":
        return None
    source = binding.source_knots
    percentiles = binding.percentile_knots
    if raw_gain >= source[-1]:
        return 1.0
    index = bisect_right(source, raw_gain)
    left_x, right_x = source[index - 1], source[index]
    left_y, right_y = percentiles[index - 1], percentiles[index]
    return left_y + (raw_gain - left_x) * (right_y - left_y) / (right_x - left_x)


def _validate_common_table_lineage(
    table: pd.DataFrame,
    *,
    schema_version: str,
    applications: Mapping[str, _ApplicationLineage],
    score_column: str,
    sender_grain: bool,
) -> pd.DataFrame:
    expected_columns: tuple[str, ...]
    if schema_version == _V6_SCHEMA_VERSION:
        expected_columns = (
            _V6_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
            if sender_grain
            else _V6_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS
        )
    elif schema_version in _LEGACY_GAIN_CALIBRATED_SCHEMA_VERSIONS:
        expected_columns = (
            _V45_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
            if sender_grain
            else _V45_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS
        )
    else:
        expected_columns = (
            _V3_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
            if sender_grain
            else _V3_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS
        )
    name = (
        "contrast_common_sender_lr_scores"
        if sender_grain
        else ("contrast_common_lr_scores")
    )
    result = _require_exact_source_columns(
        table,
        expected_columns,
        name=name,
        schema_version=schema_version,
    )
    if result.empty:
        raise ValueError(
            f"{name} cannot be empty for a connected {schema_version} collection"
        )
    required_identifiers = [
        "crossfit_id",
        "spec_id",
        "repeat_id",
        "fold_id",
        "contrast_id",
        "contrast",
        "contrast_common_collection_id",
        "global_common_application_id",
        "global_common_functional_id",
        "source_family_common_functional_id",
        "source_family_common_application_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
        "status",
        "score_version",
        "certification_status",
        "formal_inference_status",
        "claim_scope",
        "source_row_id",
    ]
    if sender_grain:
        required_identifiers.extend(
            [
                "source_sender_functional_id",
                "source_sender_application_id",
                "sender",
            ]
        )
    if schema_version in _GAIN_CALIBRATED_SCHEMA_VERSIONS:
        required_identifiers.extend(
            [
                "gain_calibration_binding_id",
                "gain_calibration_status",
            ]
        )
    for column in required_identifiers:
        values = result[column]
        as_text = values.astype(str)
        if (
            values.isna().any()
            or as_text.eq("").any()
            or as_text.str.strip().ne(as_text).any()
        ):
            raise ValueError(f"{name}.{column} requires canonical identifiers")
    if set(result["formal_inference_status"].astype(str)) != {_FORMAL_INFERENCE_STATUS}:
        raise ValueError(f"{name} cannot claim formal inference")
    duplicate_key = _SOURCE_ROW_KEY if sender_grain else _LR_ROW_KEY
    if result.duplicated(list(duplicate_key), keep=False).any():
        raise ValueError(f"{name} contains duplicate sample/receiver/fold grid rows")
    observed_applications = set(result["global_common_application_id"].astype(str))
    if observed_applications != set(applications):
        raise ValueError(f"{name} does not exactly cover manifest applications")
    counts = result.groupby("global_common_application_id", observed=True).size()
    for application_id, lineage in applications.items():
        expected_count = lineage.n_sender_rows if sender_grain else lineage.n_lr_rows
        if int(counts.get(application_id, 0)) != expected_count:
            raise ValueError(f"{name} row counts disagree with the manifest")
    registry = _lineage_registry(applications)
    result = result.merge(
        registry,
        on=[
            "contrast_common_collection_id",
            "global_common_application_id",
            "fold_id",
            "receiver",
        ],
        how="left",
        validate="many_to_one",
    )
    if result["_expected_functional_id"].isna().any():
        raise ValueError(f"{name} contains rows outside application receiver lineage")
    _compare_lineage_columns(result)
    if schema_version == _V3_SCHEMA_VERSION:
        factors = pd.to_numeric(
            result["training_receiver_scale_factor"], errors="coerce"
        )
        expected_factors = pd.to_numeric(result["_expected_factor"], errors="coerce")
        if (
            factors.isna().any()
            or not np.isclose(
                factors.to_numpy(dtype=float),
                expected_factors.to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-15,
            ).all()
        ):
            raise ValueError("training_receiver_scale_factor disagrees with manifest")
    else:
        binding_comparisons = (
            (
                "gain_calibration_binding_id",
                "_expected_gain_calibration_binding_id",
            ),
            (
                "gain_calibration_artifact_id",
                "_expected_gain_calibration_artifact_id",
            ),
            ("gain_calibration_status", "_expected_gain_calibration_status"),
            (
                "gain_calibration_reason_code",
                "_expected_gain_calibration_reason_code",
            ),
        )
        for observed_name, expected_name in binding_comparisons:
            for observed, expected in zip(
                result[observed_name], result[expected_name], strict=True
            ):
                if _optional_row_string(
                    observed, field=observed_name
                ) != _optional_row_string(expected, field=expected_name):
                    raise ValueError(
                        f"{observed_name} disagrees with the manifest binding"
                    )
    for row in result.loc[
        :,
        [
            "global_common_application_id",
            "subject_id",
            "context_id",
        ],
    ].itertuples(index=False):
        lineage = applications[str(row.global_common_application_id)]
        if (
            str(row.subject_id) not in lineage.heldout_subject_ids
            or str(row.context_id) not in lineage.context_ids
        ):
            raise ValueError("contrast-common row is outside held-out subject/context")
    for grouped_application_id, application_rows in result.groupby(
        "global_common_application_id", sort=False, observed=True
    ):
        expected_receivers = set(
            applications[str(grouped_application_id)].receiver_children
        )
        if set(application_rows["receiver"].astype(str)) != expected_receivers:
            raise ValueError(
                f"{name} lacks exact receiver coverage for one application"
            )
    if schema_version in _GAIN_CALIBRATED_SCHEMA_VERSIONS and not sender_grain:
        for row in result.loc[
            :,
            [
                "global_common_application_id",
                "receiver",
                "receiver_relative_family_gain",
                "calibrated_family_gain_percentile",
            ],
        ].itertuples(index=False):
            application = applications[str(row.global_common_application_id)]
            binding = application.receiver_gain_calibration_bindings[str(row.receiver)]
            raw_gain = _optional_unit_value(
                row.receiver_relative_family_gain,
                field="receiver_relative_family_gain",
            )
            calibrated = _optional_unit_value(
                row.calibrated_family_gain_percentile,
                field="calibrated_family_gain_percentile",
            )
            if not _same_optional_float(
                calibrated, _gain_percentile(raw_gain, binding)
            ):
                raise ValueError(
                    "calibrated_family_gain_percentile disagrees with the "
                    "manifest binding"
                )
    if sender_grain:
        if (
            not result["source_sender_functional_id"]
            .eq(result["_expected_sender_functional_id"])
            .all()
        ):
            raise ValueError("sender functional lineage disagrees with manifest")
        observed_sender_applications = set(
            result.loc[
                :,
                [
                    "global_common_application_id",
                    "receiver",
                    "source_sender_application_id",
                ],
            ].itertuples(index=False, name=None)
        )
        allowed_sender_applications = {
            (application.application_id, receiver, sender_application)
            for application in applications.values()
            for receiver, sender_application in application.sender_applications
        }
        if not observed_sender_applications.issubset(allowed_sender_applications):
            raise ValueError("sender application lineage disagrees with manifest")
    _validate_scores(result, score_column=score_column)
    grid = [
        "contrast_common_collection_id",
        "mode",
        "sample_id",
        "receiver",
    ]
    if (
        result.groupby(grid, observed=True)["fold_id"].nunique().gt(1).any()
        or result.groupby(grid, observed=True)["global_common_application_id"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("one sample/receiver grid appears in multiple folds")
    return result


def _optional_lr_scores(result: CrossFitResultReader) -> pd.DataFrame | None:
    raw_reader = getattr(result, "read_contrast_common_lr_scores", None)
    if raw_reader is None or not callable(raw_reader):
        return None
    reader = cast(Callable[[], pd.DataFrame], raw_reader)
    try:
        table = reader()
    except KeyError:
        return None
    if not isinstance(table, pd.DataFrame):
        raise TypeError("read_contrast_common_lr_scores() must return a DataFrame")
    return table


def _validate_raw_sender_evidence(table: pd.DataFrame) -> None:
    for row in table.itertuples(index=False):
        ligand = _optional_unit_value(
            row.ligand_availability, field="ligand_availability"
        )
        prevalence = _optional_unit_value(
            row.training_prevalence_prior,
            field="training_prevalence_prior",
        )
        raw = _optional_unit_value(row.raw_sender_evidence, field="raw_sender_evidence")
        expected = None if ligand is None or prevalence is None else ligand * prevalence
        if not _same_optional_float(raw, expected):
            raise ValueError(
                "raw_sender_evidence disagrees with ligand-by-prevalence evidence"
            )


def _validate_legacy_sender_formula(bound: pd.DataFrame) -> None:
    """Retain the exact v3-v5 raw-evidence multiplier semantics."""

    _validate_raw_sender_evidence(bound)
    columns = (
        "_parent_status",
        "status",
        "_parent_global_lr_score",
        "raw_sender_evidence",
        "global_sender_lr_score",
    )
    for (
        raw_parent_status,
        raw_sender_status,
        raw_parent_score,
        raw_sender_evidence,
        raw_sender_score,
    ) in bound.loc[:, list(columns)].itertuples(index=False, name=None):
        parent_status = str(raw_parent_status)
        sender_status = str(raw_sender_status)
        parent_score = _optional_unit_value(
            raw_parent_score, field="parent_global_lr_score"
        )
        raw = _optional_unit_value(raw_sender_evidence, field="raw_sender_evidence")
        score = _optional_unit_value(raw_sender_score, field="global_sender_lr_score")
        if parent_status == "structural_zero":
            valid = sender_status == "structural_zero" and score == 0.0
        elif parent_status == "not_estimable":
            valid = sender_status == "not_estimable" and score is None
        elif parent_status != "observed" or parent_score is None:
            valid = False
        elif raw is None:
            valid = sender_status == "not_estimable" and score is None
        elif raw == 0.0:
            valid = sender_status == "structural_zero" and score == 0.0
        else:
            valid = (
                sender_status == "observed"
                and score is not None
                and _same_optional_float(score, parent_score * raw)
            )
        if not valid:
            raise ValueError(
                "legacy global_sender_lr_score violates raw-evidence multiplication"
            )


def _validate_conserved_sender_formula(bound: pd.DataFrame) -> None:
    """Validate complete v6 allocation groups and parent-score conservation."""

    _validate_raw_sender_evidence(bound)
    group_keys = [
        "contrast_common_collection_id",
        "global_common_application_id",
        "fold_id",
        "global_common_functional_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
        "score_version",
    ]
    for _, group in bound.groupby(group_keys, observed=True, sort=False):
        parent_statuses = set(group["_parent_status"].astype(str))
        parent_reasons = {
            _optional_row_string(value, field="parent_reason_code")
            for value in group["_parent_reason_code"]
        }
        parent_scores = [
            _optional_unit_value(value, field="parent_global_lr_score")
            for value in group["_parent_global_lr_score"]
        ]
        if (
            len(parent_statuses) != 1
            or len(parent_reasons) != 1
            or any(
                not _same_optional_float(parent_scores[0], value)
                for value in parent_scores[1:]
            )
        ):
            raise ValueError("v6 sender group does not share one LR parent state")
        parent_status = next(iter(parent_statuses))
        parent_reason = next(iter(parent_reasons))
        parent_score = parent_scores[0]

        raw_evidence = [
            _optional_unit_value(value, field="raw_sender_evidence")
            for value in group["raw_sender_evidence"]
        ]
        weights = [
            _optional_unit_value(value, field="assignment_weight")
            for value in group["assignment_weight"]
        ]
        entropies = [
            _optional_unit_value(value, field="normalized_entropy")
            for value in group["normalized_entropy"]
        ]
        all_weights_missing = all(value is None for value in weights)
        all_weights_present = all(value is not None for value in weights)
        if not (all_weights_missing or all_weights_present):
            raise ValueError(
                "v6 assignment weights must be entirely numeric or entirely missing"
            )
        if all_weights_missing:
            if any(value is not None for value in entropies):
                raise ValueError(
                    "v6 missing assignment weights require missing entropy"
                )
            if all(value is not None for value in raw_evidence):
                raise ValueError(
                    "v6 complete raw sender evidence cannot have missing weights"
                )
        else:
            numeric_weights = cast(list[float], weights)
            if any(value is None for value in raw_evidence):
                raise ValueError(
                    "v6 incomplete raw sender evidence cannot have numeric weights"
                )
            if any(value is None for value in entropies):
                raise ValueError(
                    "v6 numeric assignment weights require numeric entropy"
                )
            if not math.isclose(
                math.fsum(numeric_weights), 1.0, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError("v6 assignment weights do not sum to one")
            expected_entropy = 0.0
            if len(numeric_weights) > 1:
                expected_entropy = -math.fsum(
                    weight * math.log(weight)
                    for weight in numeric_weights
                    if weight > 0.0
                ) / math.log(len(numeric_weights))
            if any(
                value is None
                or not math.isclose(
                    value, expected_entropy, rel_tol=1e-10, abs_tol=1e-12
                )
                for value in entropies
            ):
                raise ValueError(
                    "v6 normalized entropy disagrees with assignment weights"
                )

        statuses = group["status"].astype(str).tolist()
        reasons = [
            _optional_row_string(value, field="reason_code")
            for value in group["reason_code"]
        ]
        scores = [
            _optional_unit_value(value, field="global_sender_lr_score")
            for value in group["global_sender_lr_score"]
        ]
        if parent_status == "structural_zero":
            valid = (
                parent_score == 0.0
                and parent_reason is not None
                and all(status == "structural_zero" for status in statuses)
                and all(score == 0.0 for score in scores)
                and all(reason == parent_reason for reason in reasons)
            )
        elif parent_status == "not_estimable":
            valid = (
                parent_score is None
                and parent_reason is not None
                and all(status == "not_estimable" for status in statuses)
                and all(score is None for score in scores)
                and all(reason == parent_reason for reason in reasons)
            )
        elif (
            parent_status != "observed"
            or parent_reason is not None
            or parent_score is None
            or parent_score <= 0.0
        ):
            valid = False
        elif all_weights_missing:
            valid = (
                all(status == "not_estimable" for status in statuses)
                and all(score is None for score in scores)
                and all(reason is not None for reason in reasons)
                and len(set(reasons)) == 1
            )
        else:
            valid = True
            numeric_weights = cast(list[float], weights)
            for weight, status, reason, score in zip(
                numeric_weights, statuses, reasons, scores, strict=True
            ):
                if weight == 0.0:
                    row_valid = (
                        status == "structural_zero"
                        and reason == "sender_assignment_weight_zero"
                        and score == 0.0
                    )
                else:
                    row_valid = (
                        status == "observed"
                        and reason is None
                        and score is not None
                        and _same_optional_float(score, parent_score * weight)
                    )
                valid = valid and row_valid
            if valid and not math.isclose(
                math.fsum(cast(list[float], scores)),
                parent_score,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError("v6 sender scores do not conserve LR parent strength")
        if not valid:
            raise ValueError(
                "v6 sender status or score violates conserved LR allocation"
            )


def _validate_sender_lr_parents(
    sender: pd.DataFrame,
    lr: pd.DataFrame,
    *,
    schema_version: str,
) -> None:
    parent_keys = [
        "contrast_common_collection_id",
        "global_common_application_id",
        "fold_id",
        "global_common_functional_id",
        "source_family_common_functional_id",
        "source_family_common_application_id",
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "family_id",
        "driver_id",
        "interaction_id",
        "mode",
        "score_version",
    ]
    if schema_version == _V3_SCHEMA_VERSION:
        parent_projection = [
            "global_lr_score",
            "training_receiver_scale_factor",
            "status",
            "reason_code",
        ]
        parent_renames = {
            "global_lr_score": "_parent_global_lr_score",
            "training_receiver_scale_factor": "_parent_receiver_scale_factor",
            "status": "_parent_status",
            "reason_code": "_parent_reason_code",
        }
    else:
        parent_projection = [
            "global_lr_score",
            "calibrated_family_gain_percentile",
            "gain_calibration_binding_id",
            "gain_calibration_artifact_id",
            "gain_calibration_status",
            "gain_calibration_reason_code",
            "status",
            "reason_code",
        ]
        parent_renames = {
            "global_lr_score": "_parent_global_lr_score",
            "calibrated_family_gain_percentile": (
                "_parent_calibrated_family_gain_percentile"
            ),
            "gain_calibration_binding_id": "_parent_gain_calibration_binding_id",
            "gain_calibration_artifact_id": "_parent_gain_calibration_artifact_id",
            "gain_calibration_status": "_parent_gain_calibration_status",
            "gain_calibration_reason_code": ("_parent_gain_calibration_reason_code"),
            "status": "_parent_status",
            "reason_code": "_parent_reason_code",
        }
    parents = lr.loc[:, [*parent_keys, *parent_projection]].rename(
        columns=parent_renames
    )
    parents["_parent_row_present"] = True
    if parents.duplicated(parent_keys, keep=False).any():
        raise ValueError("contrast-common LR parent rows are not unique")
    bound = sender.merge(parents, on=parent_keys, how="left", validate="many_to_one")
    if bound["_parent_row_present"].isna().any():
        raise ValueError("sender-LR row has no contrast-common LR parent")
    numeric_comparisons = [("global_lr_score", "_parent_global_lr_score")]
    if schema_version == _V3_SCHEMA_VERSION:
        numeric_comparisons.append(
            (
                "training_receiver_scale_factor",
                "_parent_receiver_scale_factor",
            )
        )
    for observed_name, expected_name in numeric_comparisons:
        observed = pd.to_numeric(bound[observed_name], errors="coerce")
        expected = pd.to_numeric(bound[expected_name], errors="coerce")
        both_missing = observed.isna() & expected.isna()
        numerically_equal = pd.Series(
            np.isclose(
                observed.fillna(0.0).to_numpy(dtype=float),
                expected.fillna(0.0).to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-15,
            ),
            index=bound.index,
        )
        both_present_equal = observed.notna() & expected.notna() & numerically_equal
        if not (both_missing | both_present_equal).all():
            raise ValueError(f"sender-LR {observed_name} disagrees with LR parent")
    if schema_version in _GAIN_CALIBRATED_SCHEMA_VERSIONS:
        for observed_name, expected_name in (
            (
                "gain_calibration_binding_id",
                "_parent_gain_calibration_binding_id",
            ),
            (
                "gain_calibration_artifact_id",
                "_parent_gain_calibration_artifact_id",
            ),
            ("gain_calibration_status", "_parent_gain_calibration_status"),
            (
                "gain_calibration_reason_code",
                "_parent_gain_calibration_reason_code",
            ),
        ):
            for observed, expected in zip(
                bound[observed_name], bound[expected_name], strict=True
            ):
                if _optional_row_string(
                    observed, field=observed_name
                ) != _optional_row_string(expected, field=expected_name):
                    raise ValueError(
                        f"sender-LR {observed_name} disagrees with LR parent"
                    )
    if schema_version == _V6_SCHEMA_VERSION:
        _validate_conserved_sender_formula(bound)
    else:
        _validate_legacy_sender_formula(bound)


def _prepared_sample_metadata(
    adata: Any,
    *,
    sample_key: str,
    subject_key: str,
    context_keys: Sequence[str],
) -> pd.DataFrame:
    keys = tuple(context_keys)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("context_keys must be a non-empty unique sequence")
    if not hasattr(adata, "obs") or not hasattr(adata, "obs_names"):
        raise TypeError("prepared input must provide AnnData-like obs metadata")
    required = {sample_key, subject_key, *keys}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"prepared h5ad is missing metadata: {sorted(missing)}")
    if not adata.obs_names.is_unique:
        raise ValueError("prepared h5ad observation identifiers must be unique")
    frame = cast(
        pd.DataFrame,
        adata.obs.loc[:, [sample_key, subject_key, *keys]].copy(),
    )
    if frame.isna().any().any():
        raise ValueError("prepared h5ad sample metadata contains null values")
    for field in (sample_key, subject_key):
        values = frame[field].astype(str)
        if values.eq("").any() or values.str.strip().ne(values).any():
            raise ValueError(f"prepared h5ad metadata {field!r} is noncanonical")
        if values.str.contains("|", regex=False).any():
            raise ValueError(f"prepared h5ad metadata {field!r} contains reserved '|'")
    context_pairs = [context_fields(row, keys) for _, row in frame.iterrows()]
    result = pd.DataFrame(
        {
            "sample_id": frame[sample_key].astype(str).to_numpy(),
            "prepared_subject_id": frame[subject_key].astype(str).to_numpy(),
            "prepared_context_id": [item[0] for item in context_pairs],
            "context_json": [item[1] for item in context_pairs],
        }
    ).drop_duplicates(ignore_index=True)
    if result["sample_id"].duplicated(keep=False).any():
        raise ValueError("sample IDs do not map to one subject/context tuple")
    return result.sort_values("sample_id", kind="stable", ignore_index=True)


def _bind_prepared_metadata(
    sender: pd.DataFrame,
    sample_metadata: pd.DataFrame,
) -> pd.DataFrame:
    result = sender.merge(
        sample_metadata, on="sample_id", how="left", validate="many_to_one"
    )
    if result["prepared_subject_id"].isna().any():
        raise ValueError("cross-fit sender scores contain samples absent from AnnData")
    if (
        not result["subject_id"]
        .astype(str)
        .eq(result["prepared_subject_id"].astype(str))
        .all()
    ):
        raise ValueError("cross-fit subject lineage disagrees with AnnData metadata")
    if (
        not result["context_id"]
        .astype(str)
        .eq(result["prepared_context_id"].astype(str))
        .all()
    ):
        raise ValueError("cross-fit context lineage disagrees with AnnData metadata")
    return result


def _method_version(explicit: str | None, *, schema_version: str) -> str:
    if explicit is not None:
        return _required_string(explicit, field="method_version")
    try:
        return importlib.metadata.version("crychic")
    except importlib.metadata.PackageNotFoundError:
        return f"crossfit-result-{schema_version}"


def _coerce_result(
    result_or_path: CrossFitResultReader | str | Path,
) -> CrossFitResultReader:
    if isinstance(result_or_path, (str, Path)):
        return cast(CrossFitResultReader, CrossFitResult.load(result_or_path))
    if not hasattr(result_or_path, "manifest") or not callable(
        getattr(result_or_path, "read_contrast_common_sender_lr_scores", None)
    ):
        raise TypeError(
            "result must be a CrossFitResult v3/v4/v5/v6 path or read-only object"
        )
    return result_or_path


def _native_universe(raw: pd.DataFrame) -> tuple[tuple[str, str, str], ...]:
    reference: tuple[tuple[str, str, str], ...] | None = None
    for _, sample in raw.groupby("sample_id", sort=True, observed=True):
        observed = tuple(
            sorted(
                sample.loc[:, ["sender", "receiver", "interaction_id"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
        if len(observed) != len(set(observed)):
            raise ValueError("one sample contains duplicate native sender-edge rows")
        if reference is None:
            reference = observed
        elif observed != reference:
            raise ValueError(
                "native sender-edge universe differs between samples in one run"
            )
    if not reference:
        raise ValueError("selected contrast-common sender view is empty")
    return reference


def _reason_codes(raw: pd.DataFrame) -> pd.Series:
    reason = raw["reason_code"].astype("string")
    fallback = raw["status"].map(
        {
            "observed": "descriptive_oof_score_no_inference",
            "structural_zero": "structural_zero",
            "not_estimable": "not_estimable",
        }
    )
    return reason.fillna(fallback).astype(str)


def _to_long_view(
    raw: pd.DataFrame,
    *,
    manifest: Mapping[str, object],
    collection: _CollectionLineage,
    dataset_id: str,
    mode: str,
    method_version: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    universe = _native_universe(raw)
    source_result_id = _required_string(
        manifest.get("crossfit_result_id"), field="crossfit_result_id"
    )
    universe_id = canonical_digest(
        {
            "schema_version": LONG_TABLE_SCHEMA,
            "dataset_id": dataset_id,
            "source_crossfit_result_id": source_result_id,
            "contrast_common_collection_id": collection.collection_id,
            "mode": mode,
            "native_sender_edges": universe,
            "candidate_expansion": False,
        },
        prefix="external_universe",
    )
    run_id = canonical_digest(
        {
            "source_crossfit_result_id": source_result_id,
            "contrast_common_collection_id": collection.collection_id,
            "dataset_id": dataset_id,
            "mode": mode,
            "universe_id": universe_id,
            "benchmark_scope": BENCHMARK_SCOPE,
        },
        prefix="crychic_crossfit_benchmark_run",
    )
    output = pd.DataFrame(index=raw.index)
    output["schema_version"] = LONG_TABLE_SCHEMA
    output["run_id"] = run_id
    output["dataset_id"] = dataset_id
    output["method_id"] = METHOD_ID
    output["method_version"] = method_version
    output["analysis_track"] = ANALYSIS_TRACK
    output["resource_mode"] = RESOURCE_MODE
    output["resource_id"] = RESOURCE_ID
    output["resource_version"] = collection.score_version
    output["universe_id"] = universe_id
    output["universe_member"] = True
    output["universe_size"] = len(universe)
    for column in ("sample_id", "subject_id", "sender", "receiver", "interaction_id"):
        output[column] = raw[column].astype(str)
    output["context_json"] = raw["context_json"].astype(str)
    output["native_interaction_id"] = raw["interaction_id"].astype(str)
    output["interaction_direction"] = "ligand_to_receptor"
    output["ligand"] = pd.NA
    output["receptor"] = pd.NA
    output["target"] = pd.NA
    source_score = pd.to_numeric(raw["global_sender_lr_score"], errors="coerce")
    output["score"] = source_score
    output.loc[raw["status"].eq("structural_zero"), "score"] = 0.0
    output.loc[raw["status"].eq("not_estimable"), "score"] = np.nan
    output["score_name"] = SCORE_NAME
    output["score_direction"] = SCORE_DIRECTION
    output["rank"] = np.nan
    output["specificity_score"] = np.nan
    output["specificity_score_name"] = pd.NA
    output["within_dataset_p_value"] = np.nan
    output["within_dataset_p_value_semantics"] = pd.NA
    output["differential_effect"] = np.nan
    output["differential_p_value"] = np.nan
    output["differential_q_value"] = np.nan
    output["status"] = raw["status"].map(
        {
            "observed": "ok",
            "structural_zero": "ok",
            "not_estimable": "missing",
        }
    )
    output["reason_code"] = _reason_codes(raw)
    ok = output["status"].eq("ok")
    for _, indexes in (
        output.loc[ok]
        .groupby(["sample_id", "receiver"], sort=False, observed=True)
        .groups.items()
    ):
        output.loc[indexes, "rank"] = output.loc[indexes, "score"].rank(
            ascending=False, method="average"
        )
    ordered = cast(pd.DataFrame, output.loc[:, LONG_TABLE_COLUMNS].copy())
    ordered = validate_long_table(ordered.reset_index(drop=True))
    view: dict[str, object] = {
        "run_id": run_id,
        "source_crossfit_result_id": source_result_id,
        "contrast_common_collection_id": collection.collection_id,
        "contrast_id": collection.contrast_id,
        "contrast": collection.contrast,
        "mode": mode,
        "score_name": SCORE_NAME,
        "score_version": collection.score_version,
        "rows": len(ordered),
        "samples": int(ordered["sample_id"].nunique()),
        "native_universe_size": len(universe),
        "native_only_no_candidate_expansion": True,
        "source_manifest_schema_version": manifest.get("schema_version"),
        "source_cross_receiver_percentile_rank_eligible": (
            bool(collection.applications)
            and all(
                application.cross_receiver_percentile_rank_eligible
                for application in collection.applications
            )
        ),
        "common_functional_claim": False,
        "receiver_balanced_descriptive_collection": True,
        "global_cross_receiver_endpoint_eligible": False,
        "benchmark_scope": BENCHMARK_SCOPE,
        "rank_scope": "within_sample_receiver_only",
        "formal_inference_allowed": False,
    }
    return ordered, view


def convert_crossfit_result_to_long(
    result_or_path: CrossFitResultReader | str | Path,
    adata: Any,
    *,
    dataset_id: str,
    sample_key: str = "sample_id",
    subject_key: str = "subject_id",
    context_keys: Sequence[str] = ("condition",),
    mode: str = "state",
    collection_id: str | None = None,
    contrast: str | None = None,
    method_version: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Convert v3-v6 receiver-balanced OOF rows without expanding candidates.

    One benchmark run is emitted per selected contrast-common collection.  The
    exact native sender-edge universe must be identical for every sample in a
    run; otherwise readback fails closed instead of inventing missing senders.
    V4-V6 additionally require the LR parent table so calibrated gain
    percentiles and receiver-specific bindings can be verified before export.
    V6 also verifies the complete conserved sender allocation before exposing
    its existing ``global_sender_lr_score`` benchmark projection.
    """

    dataset = _required_string(dataset_id, field="dataset_id")
    selected_mode = _required_string(mode, field="mode")
    selected_collection = (
        None
        if collection_id is None
        else _required_string(collection_id, field="collection_id")
    )
    selected_contrast = (
        None if contrast is None else _required_string(contrast, field="contrast")
    )
    result = _coerce_result(result_or_path)
    manifest = result.manifest
    schema_version, collections, applications = _manifest_lineage(manifest)
    sender_source = result.read_contrast_common_sender_lr_scores()
    if not isinstance(sender_source, pd.DataFrame):
        raise TypeError(
            "read_contrast_common_sender_lr_scores() must return a DataFrame"
        )
    sender = _validate_common_table_lineage(
        sender_source,
        schema_version=schema_version,
        applications=applications,
        score_column="global_sender_lr_score",
        sender_grain=True,
    )
    optional_lr = _optional_lr_scores(result)
    if optional_lr is None and schema_version in _GAIN_CALIBRATED_SCHEMA_VERSIONS:
        raise ValueError(
            "gain-calibrated benchmark readback requires the contrast-common "
            "LR parent table"
        )
    if optional_lr is not None:
        lr = _validate_common_table_lineage(
            optional_lr,
            schema_version=schema_version,
            applications=applications,
            score_column="global_lr_score",
            sender_grain=False,
        )
        _validate_sender_lr_parents(sender, lr, schema_version=schema_version)
    sample_metadata = _prepared_sample_metadata(
        adata,
        sample_key=sample_key,
        subject_key=subject_key,
        context_keys=context_keys,
    )
    sender = _bind_prepared_metadata(sender, sample_metadata)
    chosen = tuple(
        item
        for item in collections
        if (selected_collection is None or item.collection_id == selected_collection)
        and (selected_contrast is None or item.contrast == selected_contrast)
    )
    if not chosen:
        raise ValueError("no contrast-common collection matches the requested filters")
    version = _method_version(method_version, schema_version=schema_version)
    tables: list[pd.DataFrame] = []
    views: list[dict[str, object]] = []
    for item in chosen:
        raw = sender.loc[
            sender["contrast_common_collection_id"].astype(str).eq(item.collection_id)
            & sender["mode"].astype(str).eq(selected_mode)
        ].copy()
        if raw.empty:
            raise ValueError(
                f"collection {item.collection_id!r} has no rows for mode "
                f"{selected_mode!r}"
            )
        table, view = _to_long_view(
            raw,
            manifest=manifest,
            collection=item,
            dataset_id=dataset,
            mode=selected_mode,
            method_version=version,
        )
        tables.append(table)
        views.append(view)
    combined = pd.concat(tables, ignore_index=True)
    return validate_long_table(
        cast(pd.DataFrame, combined.loc[:, LONG_TABLE_COLUMNS].copy())
    ), views


__all__ = [
    "BENCHMARK_SCOPE",
    "CrossFitResultReader",
    "convert_crossfit_result_to_long",
]
