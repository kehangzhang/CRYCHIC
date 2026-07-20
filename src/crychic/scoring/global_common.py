"""Cross-receiver contrast-common mechanistic scoring.

The existing family-common workflow intentionally produces one receiver-local
functional. Those children remain the authoritative decomposition diagnostics,
but their within-family allocations are not a valid global ranking scale. This
module composes the complete receiver universe into one outer-training
functional, evaluates a receiver-balanced LR parent, then uses the frozen
common-sender weights only to allocate that parent:

* interaction availability is used without within-family normalization;
* receiver downstream evidence is a bounded receiver-relative loss reduction;
* prior quality is a neutral-at-one multiplier; and
* sender evidence is converted by the frozen common-sender functional into a
  conserved within-edge allocation.

The resulting score is a receiver-relative mechanistic evidence scale, not an
absolute communication rate, probability, p-value, or causal effect.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.attribution.gain_calibration import (
    GAIN_CALIBRATION_PERCENTILE_POLICY,
    SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact,
)
from crychic.core import ContractError, canonical_json, stable_id
from crychic.core._validation import (
    record_validation,
    validation_is_cached,
    validation_scope,
)
from crychic.sender import (
    CommonSenderApplication,
    CommonSenderApplicationStatus,
    SenderContrastSupportStatus,
)

from .family_common import (
    FamilyCommonScoringApplication,
    FamilyCommonScoringFunctional,
    family_common_sender_application_digest,
)
from .integration import mechanistic_strength

GLOBAL_COMMON_LR_SCORE_COLUMNS = (
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

GLOBAL_COMMON_SENDER_SCORE_COLUMNS = (
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
    "assignment_weight",
    "normalized_entropy",
    "global_sender_lr_score",
    "status",
    "reason_code",
    "score_version",
)
_GLOBAL_LR_DIGEST_KEY_COLUMNS = GLOBAL_COMMON_LR_SCORE_COLUMNS[:11]
_GLOBAL_SENDER_DIGEST_KEY_COLUMNS = GLOBAL_COMMON_SENDER_SCORE_COLUMNS[:14]

_FUNCTIONAL_PRODUCER = "crychic.cross_receiver_common_scoring_functional.v4"
_APPLICATION_PRODUCER = "crychic.cross_receiver_common_scoring_application.v4"
_SCORE_VERSION = "receiver_gain_percentile_mechanistic_conserved_sender_v4"
_ESTIMAND = "receiver_balanced_descriptive_mechanistic_evidence_collection_v4"
_SCALE_POLICY = "selected_penalty_inner_oof_gain_percentile_no_heldout_rescaling_v3"
_CALIBRATION_POLICY = "selected_penalty_inner_oof_positive_gain_ecdf_v1"
_SENDER_POLICY = "contrast_common_frozen_softmax_conserved_allocation_v1"
_CERTIFICATION_STATUS = "heldout_cross_receiver_percentile_diagnostic_not_inference_v4"
_STATUSES = frozenset({"observed", "structural_zero", "not_estimable"})


def _required_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _optional_unit(value: object, *, field_name: str) -> float | None:
    if value is None or value is pd.NA or bool(pd.isna(cast(Any, value))):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    numeric = float(cast(Any, value))
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{field_name} must lie in [0, 1] or be missing")
    return 0.0 if numeric == 0.0 else numeric


def _optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None or value is pd.NA or bool(pd.isna(cast(Any, value))):
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean or missing")
    return value


def _canonical_scalar(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item"):
        try:
            value = cast(Any, value).item()
        except ValueError:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _table_digest(
    name: str,
    table: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    unique_text_prefix: tuple[str, ...],
) -> str:
    if tuple(table.columns) != columns:
        raise ValueError(f"{name} columns do not match the released contract")
    if not unique_text_prefix or columns[: len(unique_text_prefix)] != (
        unique_text_prefix
    ):
        raise ValueError(f"{name} unique digest prefix must lead the contract")
    prefix_table = table.loc[:, list(unique_text_prefix)]
    if prefix_table.isna().any(axis=None) or not all(
        prefix_table[column].map(lambda value: isinstance(value, str)).all()
        for column in unique_text_prefix
    ):
        raise ValueError(f"{name} unique digest prefix must contain strings")
    if prefix_table.duplicated().any():
        raise ValueError(f"{name} unique digest prefix must be unique")
    digest_length = 64
    validation_id: str = stable_id(
        name,
        {},
        schema_version="1",
        digest_length=digest_length,
    )
    id_prefix = validation_id[: -(digest_length + 1)]
    template = canonical_json(
        {
            "components": {"columns": list(columns), "rows": []},
            "kind": name,
            "schema_version": "1",
        }
    )
    rows_marker = '"rows":[]'
    if template.count(rows_marker) != 1:  # pragma: no cover - canonical contract
        raise RuntimeError("canonical stable-ID template lacks one rows marker")
    prefix, suffix = template.split(rows_marker, maxsplit=1)

    def encoded_rows():
        canonical_prefixes = np.fromiter(
            (
                canonical_json([_canonical_scalar(value) for value in row])
                for row in prefix_table.itertuples(index=False, name=None)
            ),
            dtype=object,
            count=len(prefix_table),
        )
        order = np.argsort(canonical_prefixes, kind="stable")
        batch_size = 65_536
        for start in range(0, len(order), batch_size):
            batch = table.iloc[order[start : start + batch_size]]
            for row in batch.itertuples(index=False, name=None):
                yield canonical_json([_canonical_scalar(value) for value in row])

    def digest_rows(rows) -> str:
        digest = hashlib.sha256()
        digest.update(prefix.encode("ascii"))
        digest.update(b'"rows":[')
        for row_index, row in enumerate(rows):
            if row_index:
                digest.update(b",")
            digest.update(row.encode("ascii"))
        digest.update(b"]")
        digest.update(suffix.encode("ascii"))
        return f"{id_prefix}_{digest.hexdigest()[:digest_length]}"

    return digest_rows(encoded_rows())


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossReceiverCommonScoringSpec:
    """Frozen formula for a train-calibrated cross-receiver descriptive rank scale."""

    softmin_power: float = 4.0
    epsilon: float = 1e-12
    schema_version: str = "4.0.0"
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        power = float(self.softmin_power)
        epsilon = float(self.epsilon)
        if not math.isfinite(power) or power <= 0:
            raise ValueError("softmin_power must be finite and positive")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if self.schema_version != "4.0.0":
            raise ValueError("global common scoring schema_version must be 4.0.0")
        payload = {
            "calibration_policy": _CALIBRATION_POLICY,
            "epsilon": epsilon,
            "estimand": _ESTIMAND,
            "scale_policy": _SCALE_POLICY,
            "schema_version": self.schema_version,
            "sender_policy": _SENDER_POLICY,
            "softmin_power": power,
        }
        object.__setattr__(self, "softmin_power", power)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "cross_receiver_common_scoring_spec",
                payload,
                schema_version="4",
            ),
        )

    def _require_intact(self) -> None:
        try:
            repeated = CrossReceiverCommonScoringSpec(
                softmin_power=self.softmin_power,
                epsilon=self.epsilon,
                schema_version=self.schema_version,
            )
        except (TypeError, ValueError) as error:
            raise ContractError(
                "Cross-receiver common scoring spec failed integrity validation",
                code="global_common_spec_integrity_violation",
                field="spec_id",
                remediation="Rebuild the scoring spec from released parameters",
            ) from error
        if repeated.spec_id != self.spec_id:
            raise ContractError(
                "Cross-receiver common scoring spec failed integrity validation",
                code="global_common_spec_integrity_violation",
                field="spec_id",
                remediation="Rebuild the scoring spec from released parameters",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "spec_id": self.spec_id,
            "schema_version": self.schema_version,
            "softmin_power": self.softmin_power,
            "epsilon": self.epsilon,
            "calibration_policy": _CALIBRATION_POLICY,
            "estimand": _ESTIMAND,
            "scale_policy": _SCALE_POLICY,
            "sender_policy": _SENDER_POLICY,
        }


@dataclass(frozen=True, slots=True, init=False)
class CrossReceiverCommonScoringFunctional:
    """One outer-training functional covering every planned receiver."""

    spec: CrossReceiverCommonScoringSpec
    child_functionals: tuple[FamilyCommonScoringFunctional, ...]
    receiver_ids: tuple[str, ...]
    contrast_name: str
    contrast_manifest_id: str
    fold_id: str
    context_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    filter_universe_id: str
    sender_functional_id: str
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    interaction_mapping: tuple[tuple[str, str, str], ...]
    receiver_gain_calibrations: tuple[
        tuple[str, SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact | None], ...
    ]
    score_version: str
    certification_status: str
    global_common_functional_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "CrossReceiverCommonScoringFunctional is producer-owned; use "
            "fit_cross_receiver_common_scoring_functional()"
        )

    @property
    def is_oof_certified(self) -> bool:
        return False

    @property
    def common_functional_across_receivers(self) -> bool:
        return False

    @property
    def receiver_balanced_descriptive_collection(self) -> bool:
        return True

    @property
    def all_receivers_gain_calibrated(self) -> bool:
        return all(
            calibration is not None and calibration.is_estimable
            for _, calibration in self.receiver_gain_calibrations
        )

    @property
    def cross_receiver_percentile_rank_eligible(self) -> bool:
        """Return whether every receiver has an observed train-only ECDF mapping."""

        return self.all_receivers_gain_calibrated

    @property
    def child_functional_ids(self) -> tuple[str, ...]:
        return tuple(
            child.family_common_functional_id for child in self.child_functionals
        )

    def gain_calibration(
        self, receiver: str
    ) -> SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact | None:
        """Return the frozen train-only gain mapping for one receiver."""

        normalized = _required_name(receiver, field_name="receiver")
        calibrations = dict(self.receiver_gain_calibrations)
        if normalized not in calibrations:
            raise KeyError(normalized)
        return calibrations[normalized]

    def gain_calibration_binding(self, receiver: str) -> dict[str, object]:
        """Return the immutable receiver-to-calibration binding record."""

        normalized = _required_name(receiver, field_name="receiver")
        child_by_receiver = {child.receiver: child for child in self.child_functionals}
        if normalized not in child_by_receiver:
            raise KeyError(normalized)
        return _gain_calibration_binding_record(
            child_by_receiver[normalized], self.gain_calibration(normalized)
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "certification_status": self.certification_status,
            "child_functionals": [
                {
                    "family_common_functional_id": (child.family_common_functional_id),
                    "incremental_functional_id": (
                        None
                        if child.incremental_functional is None
                        else child.incremental_functional.incremental_functional_id
                    ),
                    "receiver": child.receiver,
                    "receiver_family_training_artifact_id": (
                        child.receiver_family.training_artifact_id
                    ),
                }
                for child in self.child_functionals
            ],
            "context_ids": list(self.context_ids),
            "contrast_manifest_id": self.contrast_manifest_id,
            "contrast_name": self.contrast_name,
            "estimand": _ESTIMAND,
            "family_ids": list(self.family_ids),
            "feature_ids": list(self.feature_ids),
            "filter_universe_id": self.filter_universe_id,
            "fold_id": self.fold_id,
            "interaction_mapping": [list(item) for item in self.interaction_mapping],
            "calibration_policy": _CALIBRATION_POLICY,
            "receiver_gain_calibration_bindings": [
                self.gain_calibration_binding(receiver)
                for receiver in self.receiver_ids
            ],
            "receiver_ids": list(self.receiver_ids),
            "scale_policy": _SCALE_POLICY,
            "score_version": self.score_version,
            "sender_functional_id": self.sender_functional_id,
            "sender_policy": _SENDER_POLICY,
            "spec_id": self.spec.spec_id,
            "training_subject_ids": list(self.training_subject_ids),
        }

    def _require_intact(self) -> None:
        try:
            repeated = fit_cross_receiver_common_scoring_functional(
                self.child_functionals,
                planned_receiver_ids=self.receiver_ids,
                gain_calibration_artifacts=dict(self.receiver_gain_calibrations),
                spec=self.spec,
            )
        except (ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Cross-receiver common functional failed integrity validation",
                code="global_common_functional_integrity_violation",
                field="global_common_functional_id",
                remediation="Refit from the complete intact receiver child universe",
            ) from error
        if (
            self._producer_marker != _FUNCTIONAL_PRODUCER
            or self.score_version != _SCORE_VERSION
            or self.certification_status != _CERTIFICATION_STATUS
            or repeated.global_common_functional_id != self.global_common_functional_id
            or self._identity_payload() != repeated._identity_payload()
            or self.is_oof_certified
            or self.common_functional_across_receivers
            or not self.receiver_balanced_descriptive_collection
            or self.cross_receiver_percentile_rank_eligible
            != self.all_receivers_gain_calibrated
        ):
            raise ContractError(
                "Cross-receiver common functional failed integrity validation",
                code="global_common_functional_integrity_violation",
                field="global_common_functional_id",
                remediation="Refit from the complete intact receiver child universe",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "global_common_functional_id": self.global_common_functional_id,
            **self._identity_payload(),
            "common_functional_across_receivers": False,
            "receiver_balanced_descriptive_collection": True,
            "receiver_relative": True,
            "training_only_receiver_calibration": True,
            "all_receivers_gain_calibrated": self.all_receivers_gain_calibrated,
            "cross_receiver_percentile_rank_eligible": (
                self.cross_receiver_percentile_rank_eligible
            ),
            "receiver_scale_amplification": False,
            "posthoc_receiver_rescaling": False,
            "formal_inference_allowed": False,
        }


def _shared_child_contract(
    children: tuple[FamilyCommonScoringFunctional, ...],
) -> tuple[
    str,
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str, str], ...],
]:
    first = children[0]
    first._require_intact()
    feature_ids = tuple(first.receiver_family.family_basis.feature_ids)
    family_ids = tuple(first.family_ids)
    mapping = tuple(
        (item.interaction_id, item.family_id, item.driver_id)
        for item in first.interactions
    )
    shared = (
        first.contrast_name,
        first.contrast_manifest_id,
        first.fold_id,
        first.context_ids,
        first.training_subject_ids,
        first.filter_universe_id,
        first.sender_functional.sender_functional_id,
        feature_ids,
        family_ids,
        mapping,
    )
    for child in children[1:]:
        child._require_intact()
        observed = (
            child.contrast_name,
            child.contrast_manifest_id,
            child.fold_id,
            child.context_ids,
            child.training_subject_ids,
            child.filter_universe_id,
            child.sender_functional.sender_functional_id,
            tuple(child.receiver_family.family_basis.feature_ids),
            tuple(child.family_ids),
            tuple(
                (item.interaction_id, item.family_id, item.driver_id)
                for item in child.interactions
            ),
        )
        if observed != shared:
            raise ContractError(
                "Receiver children do not share one contrast-common hypothesis scale",
                code="global_common_child_contract_mismatch",
                field="child_functionals",
                remediation=(
                    "Use one fold, contrast, feature/family/interaction universe, "
                    "and sender functional for every planned receiver"
                ),
            )
    return shared


def _require_shared_training_scale_contract(
    children: tuple[FamilyCommonScoringFunctional, ...],
) -> None:
    observed = tuple(
        child.incremental_functional
        for child in children
        if child.incremental_functional is not None
    )
    if not observed:
        return
    first = observed[0]
    shared = (
        first.loss_design,
        first.family_gain_estimand,
        first.basis_coordinate_transform,
        first.context_regressor_id,
        first.nuisance_design_id,
        first.minimum_scale,
        first.null_loss_floor,
    )
    if any(
        (
            functional.loss_design,
            functional.family_gain_estimand,
            functional.basis_coordinate_transform,
            functional.context_regressor_id,
            functional.nuisance_design_id,
            functional.minimum_scale,
            functional.null_loss_floor,
        )
        != shared
        for functional in observed[1:]
    ):
        raise ContractError(
            "Receiver children do not share one downstream gain scale contract",
            code="global_common_training_scale_contract_mismatch",
            field="child_functionals",
            remediation=(
                "Use the same loss estimand, coordinate transform, nuisance design, "
                "minimum scale, and null-loss floor for every receiver"
            ),
        )


def _validate_gain_calibration_parent(
    child: FamilyCommonScoringFunctional,
    calibration: SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact,
) -> None:
    calibration._require_intact()
    incremental = child.incremental_functional
    if incremental is None:
        raise ContractError(
            "Gain calibration cannot bind an unavailable incremental child",
            code="global_common_gain_calibration_parent_mismatch",
            field="gain_calibration_artifacts",
            remediation="Use None for an unavailable receiver incremental model",
        )
    expected = (
        child.receiver,
        child.contrast_name,
        child.fold_id,
        child.training_subject_ids,
        child.active_family_ids,
        tuple(child.receiver_family.family_basis.feature_ids),
        incremental.null_loss_floor,
        child.tuning_manifest_id,
        incremental.incremental_functional_id,
        child.selected_penalty_id,
    )
    observed = (
        calibration.receiver,
        calibration.contrast_name,
        calibration.outer_fold_id,
        calibration.training_subject_ids,
        calibration.family_ids,
        calibration.feature_ids,
        calibration.null_loss_floor,
        calibration.tuning_id,
        calibration.outer_incremental_functional_id,
        calibration.outer_selected_resolved_penalty_id,
    )
    if observed != expected:
        raise ContractError(
            "Gain calibration does not match its receiver child lineage",
            code="global_common_gain_calibration_parent_mismatch",
            field="gain_calibration_artifacts",
            remediation=(
                "Use the selected-penalty inner-OOF artifact from the same receiver, "
                "contrast, fold, family basis, tuning, and final functional"
            ),
        )


def _gain_calibration_binding_record(
    child: FamilyCommonScoringFunctional,
    calibration: SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact | None,
) -> dict[str, object]:
    if calibration is None:
        status = "not_estimable"
        reason = (
            child.incremental_reason_code
            if child.incremental_functional is None
            else "selected_penalty_inner_oof_gain_calibration_missing"
        ) or "incremental_training_not_estimable"
        payload: dict[str, object] = {
            "receiver": child.receiver,
            "source_family_common_functional_id": (child.family_common_functional_id),
            "gain_calibration_artifact_id": None,
            "gain_calibration_spec_id": None,
            "gain_calibration_status": status,
            "gain_calibration_reason_code": reason,
            "percentile_policy": GAIN_CALIBRATION_PERCENTILE_POLICY,
            "positive_gain_source_knots": [],
            "positive_gain_percentile_knots": [],
            "n_supported_families": 0,
            "n_positive_observations": 0,
            "n_distinct_positive_gains": 0,
            "tuning_id": child.tuning_manifest_id,
            "outer_incremental_functional_id": (
                None
                if child.incremental_functional is None
                else child.incremental_functional.incremental_functional_id
            ),
            "outer_selected_resolved_penalty_id": child.selected_penalty_id,
        }
    else:
        _validate_gain_calibration_parent(child, calibration)
        artifact_payload = calibration.to_dict()
        payload = {
            "receiver": child.receiver,
            "source_family_common_functional_id": (child.family_common_functional_id),
            "gain_calibration_artifact_id": calibration.artifact_id,
            "gain_calibration_spec_id": calibration.spec.spec_id,
            "gain_calibration_status": calibration.status,
            "gain_calibration_reason_code": calibration.reason_code,
            "percentile_policy": artifact_payload["percentile_policy"],
            "positive_gain_source_knots": (
                calibration.positive_gain_source_knots.tolist()
            ),
            "positive_gain_percentile_knots": (
                calibration.positive_gain_percentile_knots.tolist()
            ),
            "n_supported_families": calibration.n_supported_families,
            "n_positive_observations": calibration.n_positive_observations,
            "n_distinct_positive_gains": calibration.n_distinct_positive_gains,
            "tuning_id": calibration.tuning_id,
            "outer_incremental_functional_id": (
                calibration.outer_incremental_functional_id
            ),
            "outer_selected_resolved_penalty_id": (
                calibration.outer_selected_resolved_penalty_id
            ),
        }
    return {
        "gain_calibration_binding_id": stable_id(
            "receiver_gain_calibration_binding", payload, schema_version="1"
        ),
        **payload,
    }


@validation_scope()
def fit_cross_receiver_common_scoring_functional(
    child_functionals: Sequence[FamilyCommonScoringFunctional],
    *,
    planned_receiver_ids: Sequence[str],
    gain_calibration_artifacts: Mapping[
        str, SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact | None
    ]
    | None = None,
    spec: CrossReceiverCommonScoringSpec | None = None,
) -> CrossReceiverCommonScoringFunctional:
    """Compose one global parent from a complete receiver child universe."""

    resolved_spec = CrossReceiverCommonScoringSpec() if spec is None else spec
    if not isinstance(resolved_spec, CrossReceiverCommonScoringSpec):
        raise TypeError("spec must be CrossReceiverCommonScoringSpec or None")
    resolved_spec._require_intact()
    children = tuple(child_functionals)
    if not children or any(
        not isinstance(child, FamilyCommonScoringFunctional) for child in children
    ):
        raise TypeError(
            "child_functionals must contain at least one FamilyCommonScoringFunctional"
        )
    children = tuple(sorted(children, key=lambda child: child.receiver))
    child_receivers = tuple(child.receiver for child in children)
    if len(set(child_receivers)) != len(child_receivers):
        raise ValueError("global common receiver children must be receiver-unique")
    planned = tuple(
        sorted(
            _required_name(value, field_name="receiver")
            for value in planned_receiver_ids
        )
    )
    if not planned or len(set(planned)) != len(planned):
        raise ValueError("planned_receiver_ids must be non-empty and unique")
    if child_receivers != planned:
        raise ContractError(
            "Global common functional lacks exact planned receiver coverage",
            code="global_common_receiver_coverage_mismatch",
            field="planned_receiver_ids",
            remediation="Retain one receiver child for every planned receiver",
        )
    (
        contrast_name,
        contrast_manifest_id,
        fold_id,
        context_ids,
        training_subject_ids,
        filter_universe_id,
        sender_functional_id,
        feature_ids,
        family_ids,
        interaction_mapping,
    ) = _shared_child_contract(children)
    if any(
        child.softmin_power != resolved_spec.softmin_power
        or child.epsilon != resolved_spec.epsilon
        or child.score_version != children[0].score_version
        for child in children
    ):
        raise ContractError(
            "Receiver children use incompatible scoring formula parameters",
            code="global_common_formula_mismatch",
            field="spec",
            remediation="Refit every receiver with the same released score formula",
        )
    if any(child.family_selection_threshold != 0.0 for child in children):
        raise ContractError(
            "Cross-receiver common scoring does not support held-out selection cuts",
            code="global_common_family_selection_threshold_unsupported",
            field="family_selection_threshold",
            remediation=(
                "Refit receiver children with family_selection_threshold=0; apply "
                "any reporting threshold after the complete OOF score collection"
            ),
        )
    _require_shared_training_scale_contract(children)
    if gain_calibration_artifacts is None:
        calibration_by_receiver = dict.fromkeys(planned)
    else:
        calibration_by_receiver = dict(gain_calibration_artifacts)
        if set(calibration_by_receiver) != set(planned):
            raise ContractError(
                "Gain calibration mapping lacks exact planned receiver coverage",
                code="global_common_gain_calibration_coverage_mismatch",
                field="gain_calibration_artifacts",
                remediation=(
                    "Provide one observed, typed NE, or None entry per receiver"
                ),
            )
    calibrations: tuple[
        tuple[str, SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact | None], ...
    ] = tuple((receiver, calibration_by_receiver[receiver]) for receiver in planned)
    for child, (receiver, calibration) in zip(children, calibrations, strict=True):
        if child.receiver != receiver:
            raise RuntimeError("receiver calibration ordering is inconsistent")
        if calibration is not None:
            if not isinstance(
                calibration, SelectedPenaltyInnerOOFFamilyGainCalibrationArtifact
            ):
                raise TypeError(
                    "gain_calibration_artifacts values must be calibration artifacts "
                    "or None"
                )
            _validate_gain_calibration_parent(child, calibration)
    candidate_pairs = {
        (prior.receiver, prior.interaction_id)
        for prior in children[0].sender_functional.candidate_priors
    }
    candidate_receivers = {receiver for receiver, _ in candidate_pairs}
    if candidate_receivers != set(planned):
        unexpected_receivers = tuple(
            sorted(candidate_receivers.symmetric_difference(set(planned)))
        )
        raise ContractError(
            "Global common sender universe does not cover every planned receiver; "
            f"receiver_mismatch={unexpected_receivers!r}",
            code="global_common_sender_universe_mismatch",
            field="sender_functional_id",
            remediation="Freeze a complete candidate sender manifest before scoring",
        )
    self = object.__new__(CrossReceiverCommonScoringFunctional)
    values: dict[str, object] = {
        "spec": resolved_spec,
        "child_functionals": children,
        "receiver_ids": planned,
        "contrast_name": contrast_name,
        "contrast_manifest_id": contrast_manifest_id,
        "fold_id": fold_id,
        "context_ids": context_ids,
        "training_subject_ids": training_subject_ids,
        "filter_universe_id": filter_universe_id,
        "sender_functional_id": sender_functional_id,
        "feature_ids": feature_ids,
        "family_ids": family_ids,
        "interaction_mapping": interaction_mapping,
        "receiver_gain_calibrations": calibrations,
        "score_version": _SCORE_VERSION,
        "certification_status": _CERTIFICATION_STATUS,
        "_producer_marker": _FUNCTIONAL_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "global_common_functional_id",
        stable_id(
            "cross_receiver_common_scoring_functional",
            self._identity_payload(),
            schema_version="4",
        ),
    )
    return self


def _training_family_states(
    child: FamilyCommonScoringFunctional,
) -> dict[str, tuple[float | None, bool | None, str | None]]:
    incremental = child.incremental_functional
    if incremental is None:
        return {
            family_id: (None, None, child.incremental_reason_code)
            for family_id in child.family_ids
        }
    active_index = {
        family_id: index for index, family_id in enumerate(incremental.family_ids)
    }
    result: dict[str, tuple[float | None, bool | None, str | None]] = {}
    for family_id in child.family_ids:
        index = active_index.get(family_id)
        if index is None:
            result[family_id] = (0.0, False, "receptor_family_ineligible")
            continue
        coefficient = float(incremental.family_coefficients[index])
        if not bool(incremental.family_estimable[index]):
            result[family_id] = (
                coefficient,
                None,
                incremental.family_reason_codes[index]
                or "training_family_not_estimable",
            )
        else:
            result[family_id] = (
                coefficient,
                coefficient > 0.0,
                None if coefficient > 0.0 else "family_not_selected_in_training",
            )
    return result


def _lr_row_values(
    *,
    receptor_eligible: bool,
    gate_status: str,
    training_coefficient: float | None,
    training_selected: bool | None,
    training_reason: str | None,
    availability: float | None,
    family_gain: float | None,
    calibrated_family_gain: float | None,
    gain_calibration_status: str,
    gain_calibration_reason: str | None,
    prior_quality: float | None,
    spec: CrossReceiverCommonScoringSpec,
) -> tuple[float | None, float | None, str, str | None]:
    valid_gate_statuses = {
        SenderContrastSupportStatus.SUPPORTED.value,
        SenderContrastSupportStatus.UNSUPPORTED.value,
        SenderContrastSupportStatus.NOT_ESTIMABLE.value,
    }
    if gate_status not in valid_gate_statuses:
        raise ValueError("ligand contrast gate status is not recognized")
    if not receptor_eligible:
        return 0.0, 0.0, "structural_zero", "receptor_interaction_ineligible"
    if gate_status == SenderContrastSupportStatus.UNSUPPORTED.value:
        return 0.0, 0.0, "structural_zero", "ligand_contrast_not_supported"
    if training_selected is False:
        return (
            0.0,
            0.0,
            "structural_zero",
            training_reason or "family_not_selected_in_training",
        )
    if availability == 0.0:
        return 0.0, 0.0, "structural_zero", "interaction_availability_zero"
    if family_gain == 0.0:
        return 0.0, 0.0, "structural_zero", "incremental_downstream_gain_zero"
    if gate_status == SenderContrastSupportStatus.NOT_ESTIMABLE.value:
        return None, None, "not_estimable", "ligand_contrast_not_estimable"
    if training_selected is None:
        return (
            None,
            None,
            "not_estimable",
            training_reason or "training_family_not_estimable",
        )
    if availability is None:
        return None, None, "not_estimable", "interaction_availability_missing"
    if family_gain is None:
        return None, None, "not_estimable", "incremental_downstream_gain_missing"
    if gain_calibration_status != "observed" or calibrated_family_gain is None:
        return (
            None,
            None,
            "not_estimable",
            gain_calibration_reason
            or "selected_penalty_inner_oof_gain_calibration_not_estimable",
        )
    if calibrated_family_gain == 0.0:
        return (
            None,
            None,
            "not_estimable",
            "positive_incremental_gain_calibrated_to_zero",
        )
    core, prior_adjusted, _ = mechanistic_strength(
        availability=availability,
        incremental_downstream=calibrated_family_gain,
        prior_quality=prior_quality,
        sender_weight=1.0,
        softmin_power=spec.softmin_power,
        epsilon=spec.epsilon,
    )
    if core is None:
        return None, None, "not_estimable", "global_lr_core_not_estimable"
    if prior_adjusted is None:
        return core, None, "not_estimable", "prior_quality_missing"
    if prior_adjusted == 0.0:
        return core, 0.0, "structural_zero", "prior_quality_zero"
    return core, prior_adjusted, "observed", None


def _build_global_lr_table(
    functional: CrossReceiverCommonScoringFunctional,
    applications: tuple[FamilyCommonScoringApplication, ...],
) -> pd.DataFrame:
    child_by_receiver = {
        child.receiver: child for child in functional.child_functionals
    }
    reference_grid: set[tuple[object, ...]] | None = None
    rows: list[dict[str, object]] = []
    for application in applications:
        application._require_intact()
        child = child_by_receiver[application.functional.receiver]
        if (
            application.functional.family_common_functional_id
            != child.family_common_functional_id
        ):
            raise ContractError(
                "Global common application received a stale receiver child",
                code="global_common_application_child_mismatch",
                field="source_family_common_application_id",
                remediation=(
                    "Apply the exact child functional bound by the global parent"
                ),
            )
        _, _, family_scores, member_scores, _ = application._validated_tables()
        grid_columns = [
            "sample_id",
            "subject_id",
            "context_id",
            "family_id",
            "driver_id",
            "interaction_id",
            "mode",
        ]
        application_grid = set(
            member_scores.loc[:, grid_columns].itertuples(index=False, name=None)
        )
        if reference_grid is None:
            reference_grid = application_grid
        elif application_grid != reference_grid:
            raise ContractError(
                "Receiver children do not share one exact held-out score grid",
                code="global_common_heldout_grid_mismatch",
                field="child_applications",
                remediation=(
                    "Emit every frozen sample/context/family/interaction/mode row "
                    "for every planned receiver, using explicit not-estimable rows"
                ),
            )
        family_values = family_scores.loc[
            :,
            [
                "sample_id",
                "subject_id",
                "context_id",
                "receiver",
                "family_id",
                "mode",
                "incremental_downstream_gain",
            ],
        ].rename(
            columns={"incremental_downstream_gain": "receiver_relative_family_gain"}
        )
        family_keys = [
            "sample_id",
            "subject_id",
            "context_id",
            "receiver",
            "family_id",
            "mode",
        ]
        expected_family_keys = member_scores.loc[:, family_keys].drop_duplicates()
        observed_family_keys = family_values.loc[:, family_keys].drop_duplicates()
        if not expected_family_keys.equals(observed_family_keys):
            expected_rows = set(expected_family_keys.itertuples(index=False, name=None))
            observed_rows = set(observed_family_keys.itertuples(index=False, name=None))
            if expected_rows != observed_rows:
                raise ValueError(
                    "global common source family rows do not match member rows"
                )
        merged = member_scores.merge(
            family_values,
            how="left",
            on=family_keys,
            validate="many_to_one",
            sort=False,
        )
        training_states = _training_family_states(child)
        calibration = functional.gain_calibration(child.receiver)
        calibration_binding = functional.gain_calibration_binding(child.receiver)
        calibration_status = str(calibration_binding["gain_calibration_status"])
        calibration_reason_value = calibration_binding["gain_calibration_reason_code"]
        calibration_reason = (
            None if calibration_reason_value is None else str(calibration_reason_value)
        )
        for source in merged.itertuples(index=False):
            family_id = str(source.family_id)
            coefficient, selected, training_reason = training_states[family_id]
            receptor_eligible = bool(source.receptor_eligible)
            gate_status = str(source.ligand_contrast_gate_status)
            availability = _optional_unit(
                source.availability, field_name="availability"
            )
            family_gain = _optional_unit(
                source.receiver_relative_family_gain,
                field_name="receiver_relative_family_gain",
            )
            calibrated_family_gain = (
                None if calibration is None else calibration.calibrate_gain(family_gain)
            )
            prior_quality = _optional_unit(
                source.prior_quality,
                field_name="prior_quality",
            )
            core, score, status, reason = _lr_row_values(
                receptor_eligible=receptor_eligible,
                gate_status=gate_status,
                training_coefficient=coefficient,
                training_selected=selected,
                training_reason=training_reason,
                availability=availability,
                family_gain=family_gain,
                calibrated_family_gain=calibrated_family_gain,
                gain_calibration_status=calibration_status,
                gain_calibration_reason=calibration_reason,
                prior_quality=prior_quality,
                spec=functional.spec,
            )
            rows.append(
                {
                    "global_common_functional_id": (
                        functional.global_common_functional_id
                    ),
                    "source_family_common_functional_id": (
                        child.family_common_functional_id
                    ),
                    "source_family_common_application_id": application.application_id,
                    "sample_id": str(source.sample_id),
                    "subject_id": str(source.subject_id),
                    "context_id": str(source.context_id),
                    "receiver": str(source.receiver),
                    "family_id": family_id,
                    "driver_id": str(source.driver_id),
                    "interaction_id": str(source.interaction_id),
                    "mode": str(source.mode),
                    "receptor_eligible": receptor_eligible,
                    "ligand_contrast_gate_status": gate_status,
                    "availability": availability,
                    "receiver_relative_family_gain": family_gain,
                    "calibrated_family_gain_percentile": calibrated_family_gain,
                    "gain_calibration_binding_id": calibration_binding[
                        "gain_calibration_binding_id"
                    ],
                    "gain_calibration_artifact_id": calibration_binding[
                        "gain_calibration_artifact_id"
                    ],
                    "gain_calibration_status": calibration_status,
                    "gain_calibration_reason_code": calibration_reason,
                    "prior_quality": prior_quality,
                    "training_family_coefficient": coefficient,
                    "global_lr_core_strength": core,
                    "global_lr_score": score,
                    "status": status,
                    "reason_code": reason,
                    "score_version": functional.score_version,
                }
            )
    result = pd.DataFrame(rows, columns=GLOBAL_COMMON_LR_SCORE_COLUMNS)
    if result.empty:
        raise ValueError("global common LR application cannot be empty")
    keys = [
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
        "mode",
    ]
    if result.duplicated(keys).any():
        raise ValueError("global common LR score keys must be unique")
    return result.sort_values(keys, kind="stable", ignore_index=True)


def _build_global_sender_table(
    functional: CrossReceiverCommonScoringFunctional,
    lr_scores: pd.DataFrame,
    sender_applications: tuple[CommonSenderApplication, ...],
) -> pd.DataFrame:
    sender_tables: list[pd.DataFrame] = []
    seen_receivers: list[str] = []
    for application in sender_applications:
        if not isinstance(application, CommonSenderApplication):
            raise TypeError("sender_applications must contain CommonSenderApplication")
        application._require_intact()
        if (
            application.functional.sender_functional_id
            != functional.sender_functional_id
        ):
            raise ContractError(
                "Global common sender application uses a different functional",
                code="global_common_sender_application_mismatch",
                field="source_sender_functional_id",
                remediation="Apply the shared contrast-common sender functional",
            )
        application_table = application.table
        receivers = tuple(sorted(set(application_table["receiver"].astype(str))))
        if len(receivers) != 1:
            raise ValueError(
                "each global common sender application must cover one receiver"
            )
        seen_receivers.append(receivers[0])
        sender_tables.append(application_table)
    if tuple(sorted(seen_receivers)) != functional.receiver_ids:
        raise ContractError(
            "Global common sender applications lack exact receiver coverage",
            code="global_common_sender_application_coverage_mismatch",
            field="sender_applications",
            remediation="Retain one complete sender application per planned receiver",
        )
    sender = pd.concat(sender_tables, ignore_index=True)
    sender_keys = [
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
        "sender",
    ]
    if sender.duplicated(sender_keys).any():
        raise ValueError("global common sender source keys must be unique")
    merged = lr_scores.merge(
        sender.loc[
            :,
            [
                "sender_application_id",
                "sender_functional_id",
                *sender_keys,
                "ligand_availability",
                "training_prevalence_prior",
                "raw_sender_evidence",
                "assignment_weight",
                "normalized_entropy",
                "status",
                "reason_code",
            ],
        ].rename(
            columns={
                "status": "sender_status",
                "reason_code": "sender_reason_code",
            }
        ),
        how="inner",
        on=[
            "sample_id",
            "subject_id",
            "context_id",
            "receiver",
            "interaction_id",
        ],
        validate="many_to_many",
        sort=False,
    )
    sender_pairs = set(
        sender.loc[:, ["receiver", "interaction_id"]].itertuples(index=False, name=None)
    )
    lr_pairs = set(
        lr_scores.loc[:, ["receiver", "interaction_id"]].itertuples(
            index=False, name=None
        )
    )
    lr_with_sender = lr_scores.loc[
        [
            (receiver, interaction_id) in sender_pairs
            for receiver, interaction_id in lr_scores.loc[
                :, ["receiver", "interaction_id"]
            ].itertuples(index=False, name=None)
        ]
    ]
    expected_groups = set(
        lr_with_sender.loc[
            :,
            ["sample_id", "subject_id", "context_id", "receiver", "interaction_id"],
        ].itertuples(index=False, name=None)
    )
    source_groups = set(
        sender.loc[
            [
                (receiver, interaction_id) in lr_pairs
                for receiver, interaction_id in sender.loc[
                    :, ["receiver", "interaction_id"]
                ].itertuples(index=False, name=None)
            ],
            [
                "sample_id",
                "subject_id",
                "context_id",
                "receiver",
                "interaction_id",
            ],
        ].itertuples(index=False, name=None)
    )
    if source_groups != expected_groups:
        raise ValueError(
            "global common sender sources do not exactly cover held-out LR rows"
        )
    observed_groups = set(
        merged.loc[
            :,
            ["sample_id", "subject_id", "context_id", "receiver", "interaction_id"],
        ].itertuples(index=False, name=None)
    )
    if observed_groups != expected_groups:
        raise ValueError("global common sender scores lack exact candidate coverage")
    numeric: dict[str, pd.Series] = {}
    for column in (
        "global_lr_score",
        "ligand_availability",
        "training_prevalence_prior",
        "raw_sender_evidence",
        "assignment_weight",
        "normalized_entropy",
    ):
        values = pd.to_numeric(merged[column], errors="coerce").astype(float)
        invalid = merged[column].notna() & values.isna()
        present = values.dropna()
        if invalid.any() or (
            not present.empty
            and (
                (present < 0.0).any()
                or (present > 1.0).any()
                or not present.map(math.isfinite).all()
            )
        ):
            raise ValueError(f"{column} must lie in [0, 1] or be missing")
        numeric[column] = values

    lr_status = merged["status"].astype(str)
    sender_status = merged["sender_status"].astype(str)
    if not set(lr_status).issubset(_STATUSES):
        raise ValueError("global LR source status is not recognized")
    allowed_sender_statuses = {
        CommonSenderApplicationStatus.OK.value,
        CommonSenderApplicationStatus.NOT_ESTIMABLE.value,
    }
    if not set(sender_status).issubset(allowed_sender_statuses):
        raise ValueError("global sender source status is not recognized")

    lr_score = numeric["global_lr_score"]
    assignment_weight = numeric["assignment_weight"]
    structural_parent = lr_status.eq("structural_zero")
    missing_parent = ~structural_parent & (
        lr_status.eq("not_estimable") | lr_score.isna()
    )
    missing_sender = (
        sender_status.eq(CommonSenderApplicationStatus.NOT_ESTIMABLE.value)
        | assignment_weight.isna()
    )
    zero_assignment = assignment_weight.eq(0.0)
    observed = ~(structural_parent | missing_parent | missing_sender | zero_assignment)
    sender_not_estimable = ~structural_parent & ~missing_parent & missing_sender

    score = lr_score * assignment_weight
    score.loc[structural_parent | zero_assignment] = 0.0
    score.loc[missing_parent | sender_not_estimable] = math.nan
    status = pd.Series("observed", index=merged.index, dtype=object)
    status.loc[structural_parent | zero_assignment] = "structural_zero"
    status.loc[missing_parent | sender_not_estimable] = "not_estimable"
    reason = pd.Series(None, index=merged.index, dtype=object)
    reason.loc[structural_parent] = merged.loc[structural_parent, "reason_code"]
    reason.loc[missing_parent] = merged.loc[missing_parent, "reason_code"].where(
        merged.loc[missing_parent, "reason_code"].notna(),
        "global_lr_score_not_estimable",
    )
    reason.loc[sender_not_estimable] = merged.loc[
        sender_not_estimable, "sender_reason_code"
    ].where(
        merged.loc[sender_not_estimable, "sender_reason_code"].notna(),
        "sender_assignment_not_estimable",
    )
    zero_sender = (
        ~structural_parent & ~missing_parent & ~missing_sender & zero_assignment
    )
    reason.loc[zero_sender] = "sender_assignment_weight_zero"
    if score.loc[observed].isna().any():
        raise RuntimeError("observed global sender rows lack a numeric score")

    result = pd.DataFrame(
        {
            "global_common_functional_id": functional.global_common_functional_id,
            "source_family_common_functional_id": merged[
                "source_family_common_functional_id"
            ],
            "source_family_common_application_id": merged[
                "source_family_common_application_id"
            ],
            "source_sender_functional_id": merged["sender_functional_id"],
            "source_sender_application_id": merged["sender_application_id"],
            "sample_id": merged["sample_id"],
            "subject_id": merged["subject_id"],
            "context_id": merged["context_id"],
            "receiver": merged["receiver"],
            "family_id": merged["family_id"],
            "driver_id": merged["driver_id"],
            "interaction_id": merged["interaction_id"],
            "mode": merged["mode"],
            "sender": merged["sender"],
            "global_lr_score": lr_score,
            "gain_calibration_binding_id": merged["gain_calibration_binding_id"],
            "gain_calibration_artifact_id": merged["gain_calibration_artifact_id"],
            "gain_calibration_status": merged["gain_calibration_status"],
            "gain_calibration_reason_code": merged["gain_calibration_reason_code"],
            "ligand_availability": numeric["ligand_availability"],
            "training_prevalence_prior": numeric["training_prevalence_prior"],
            "raw_sender_evidence": numeric["raw_sender_evidence"],
            "assignment_weight": assignment_weight,
            "normalized_entropy": numeric["normalized_entropy"],
            "global_sender_lr_score": score,
            "status": status,
            "reason_code": reason,
            "score_version": functional.score_version,
        },
        columns=GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    )
    keys = [
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
        "mode",
    ]
    if result.empty or result.duplicated(keys).any():
        raise ValueError("global common sender score keys must be non-empty and unique")
    group_keys = [
        "sample_id",
        "subject_id",
        "context_id",
        "receiver",
        "interaction_id",
        "mode",
    ]
    grouped = result.groupby(group_keys, observed=True, sort=False)
    parent_counts = grouped["global_lr_score"].nunique(dropna=False)
    if not parent_counts.eq(1).all():
        raise RuntimeError("sender allocation does not share one LR parent score")
    summary = grouped.agg(
        parent_score=("global_lr_score", "first"),
        group_size=("sender", "size"),
        weight_count=("assignment_weight", "count"),
        weight_sum=("assignment_weight", "sum"),
        score_count=("global_sender_lr_score", "count"),
        score_sum=("global_sender_lr_score", "sum"),
    )
    missing_parent = summary["parent_score"].isna()
    if summary.loc[missing_parent, "score_count"].ne(0).any():
        raise RuntimeError("not-estimable LR parent produced sender scores")
    zero_parent = summary["parent_score"].eq(0.0)
    if (
        summary.loc[zero_parent, "score_count"]
        .ne(summary.loc[zero_parent, "group_size"])
        .any()
        or summary.loc[zero_parent, "score_sum"].ne(0.0).any()
    ):
        raise RuntimeError("structural-zero LR parent did not remain zero")
    positive_parent = ~(missing_parent | zero_parent)
    missing_weights = summary["weight_count"].eq(0)
    if summary.loc[positive_parent & missing_weights, "score_count"].ne(0).any():
        raise RuntimeError("not-estimable sender allocation produced scores")
    complete = positive_parent & ~missing_weights
    if (
        summary.loc[complete, "weight_count"]
        .ne(summary.loc[complete, "group_size"])
        .any()
        or summary.loc[complete, "score_count"]
        .ne(summary.loc[complete, "group_size"])
        .any()
    ):
        raise RuntimeError("sender allocation must fail closed as a complete group")
    if not np.allclose(
        summary.loc[complete, "weight_sum"],
        1.0,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("sender assignment weights do not sum to one")
    if not np.allclose(
        summary.loc[complete, "score_sum"],
        summary.loc[complete, "parent_score"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("global sender scores do not conserve LR strength")
    return result.sort_values(keys, kind="stable", ignore_index=True)


@dataclass(frozen=True, slots=True, init=False)
class CrossReceiverCommonScoringApplication:
    """Held-out global LR and sender-LR scores from one global parent."""

    functional: CrossReceiverCommonScoringFunctional
    child_applications: tuple[FamilyCommonScoringApplication, ...]
    source_sender_application_digests: tuple[tuple[str, str], ...]
    heldout_subject_ids: tuple[str, ...]
    lr_scores_digest: str
    sender_scores_digest: str
    table_row_counts: tuple[int, int]
    application_id: str
    _lr_scores: pd.DataFrame
    _sender_scores: pd.DataFrame
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "CrossReceiverCommonScoringApplication is producer-owned; use "
            "apply_cross_receiver_common_scoring_functional()"
        )

    @property
    def is_oof_certified(self) -> bool:
        return False

    @property
    def global_lr_scores(self) -> pd.DataFrame:
        self._require_intact()
        return self._lr_scores.copy(deep=True)

    @property
    def global_sender_lr_scores(self) -> pd.DataFrame:
        self._require_intact()
        return self._sender_scores.copy(deep=True)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "certification_status": self.functional.certification_status,
            "child_application_ids": [
                application.application_id for application in self.child_applications
            ],
            "global_common_functional_id": (
                self.functional.global_common_functional_id
            ),
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "lr_scores_digest": self.lr_scores_digest,
            "score_version": self.functional.score_version,
            "sender_scores_digest": self.sender_scores_digest,
            "source_sender_application_digests": [
                list(item) for item in self.source_sender_application_digests
            ],
            "table_row_counts": list(self.table_row_counts),
        }

    @validation_scope()
    def _require_intact(self) -> None:
        if validation_is_cached(self):
            return
        try:
            self.functional._require_intact()
            for application in self.child_applications:
                application._require_intact()
            valid = (
                self._producer_marker == _APPLICATION_PRODUCER
                and not self.is_oof_certified
                and tuple(
                    application.functional.family_common_functional_id
                    for application in self.child_applications
                )
                == self.functional.child_functional_ids
                and _table_digest(
                    "cross_receiver_common_lr_scores",
                    self._lr_scores,
                    GLOBAL_COMMON_LR_SCORE_COLUMNS,
                    unique_text_prefix=_GLOBAL_LR_DIGEST_KEY_COLUMNS,
                )
                == self.lr_scores_digest
                and _table_digest(
                    "cross_receiver_common_sender_scores",
                    self._sender_scores,
                    GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
                    unique_text_prefix=_GLOBAL_SENDER_DIGEST_KEY_COLUMNS,
                )
                == self.sender_scores_digest
                and self.table_row_counts
                == (len(self._lr_scores), len(self._sender_scores))
                and not set(self.heldout_subject_ids).intersection(
                    self.functional.training_subject_ids
                )
                and stable_id(
                    "cross_receiver_common_scoring_application",
                    self._identity_payload(),
                    schema_version="4",
                )
                == self.application_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Cross-receiver common application failed integrity validation",
                code="global_common_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact global functional to held-out parents",
            ) from error
        if not valid:
            raise ContractError(
                "Cross-receiver common application failed integrity validation",
                code="global_common_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact global functional to held-out parents",
            )
        record_validation(self)

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "application_id": self.application_id,
            **self._identity_payload(),
            "estimand": _ESTIMAND,
            "common_functional_across_receivers": False,
            "receiver_balanced_descriptive_collection": True,
            "training_only_receiver_calibration": True,
            "all_receivers_gain_calibrated": (
                self.functional.all_receivers_gain_calibrated
            ),
            "cross_receiver_percentile_rank_eligible": (
                self.functional.cross_receiver_percentile_rank_eligible
            ),
            "receiver_scale_amplification": False,
            "formal_inference_allowed": False,
        }


@validation_scope()
def apply_cross_receiver_common_scoring_functional(
    functional: CrossReceiverCommonScoringFunctional,
    child_applications: Sequence[FamilyCommonScoringApplication],
    sender_applications: Sequence[CommonSenderApplication],
) -> CrossReceiverCommonScoringApplication:
    """Apply the global formula and conserve each sender-unresolved LR score."""

    if not isinstance(functional, CrossReceiverCommonScoringFunctional):
        raise TypeError("functional must be CrossReceiverCommonScoringFunctional")
    functional._require_intact()
    children = tuple(child_applications)
    if not children or any(
        not isinstance(application, FamilyCommonScoringApplication)
        for application in children
    ):
        raise TypeError(
            "child_applications must contain FamilyCommonScoringApplication values"
        )
    children = tuple(sorted(children, key=lambda item: item.functional.receiver))
    if (
        tuple(
            application.functional.family_common_functional_id
            for application in children
        )
        != functional.child_functional_ids
    ):
        raise ContractError(
            "Global common child applications do not match the functional",
            code="global_common_application_child_mismatch",
            field="child_applications",
            remediation="Apply every receiver child from the exact global parent",
        )
    heldout_sets = {application.heldout_subject_ids for application in children}
    if len(heldout_sets) != 1:
        raise ContractError(
            "Global common receiver children do not share one held-out subject scope",
            code="global_common_heldout_scope_mismatch",
            field="heldout_subject_ids",
            remediation="Apply every child to the same physical held-out fold",
        )
    heldout_subject_ids = next(iter(heldout_sets))
    if set(heldout_subject_ids).intersection(functional.training_subject_ids):
        raise ContractError(
            "Global common held-out subjects overlap outer training",
            code="global_common_subject_leakage",
            field="heldout_subject_ids",
            remediation="Use subject-disjoint outer folds",
        )
    senders = tuple(sender_applications)
    lr_scores = _build_global_lr_table(functional, children)
    sender_scores = _build_global_sender_table(functional, lr_scores, senders)
    source_sender_digests = tuple(
        sorted(
            (
                str(next(iter(set(application.table["receiver"].astype(str))))),
                family_common_sender_application_digest(application),
            )
            for application in senders
        )
    )
    self = object.__new__(CrossReceiverCommonScoringApplication)
    values: dict[str, object] = {
        "functional": functional,
        "child_applications": children,
        "source_sender_application_digests": source_sender_digests,
        "heldout_subject_ids": heldout_subject_ids,
        "lr_scores_digest": _table_digest(
            "cross_receiver_common_lr_scores",
            lr_scores,
            GLOBAL_COMMON_LR_SCORE_COLUMNS,
            unique_text_prefix=_GLOBAL_LR_DIGEST_KEY_COLUMNS,
        ),
        "sender_scores_digest": _table_digest(
            "cross_receiver_common_sender_scores",
            sender_scores,
            GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
            unique_text_prefix=_GLOBAL_SENDER_DIGEST_KEY_COLUMNS,
        ),
        "table_row_counts": (len(lr_scores), len(sender_scores)),
        "_lr_scores": lr_scores.copy(deep=True),
        "_sender_scores": sender_scores.copy(deep=True),
        "_producer_marker": _APPLICATION_PRODUCER,
    }
    for name, value in values.items():
        object.__setattr__(self, name, value)
    object.__setattr__(
        self,
        "application_id",
        stable_id(
            "cross_receiver_common_scoring_application",
            self._identity_payload(),
            schema_version="4",
        ),
    )
    return self


__all__ = [
    "GLOBAL_COMMON_LR_SCORE_COLUMNS",
    "GLOBAL_COMMON_SENDER_SCORE_COLUMNS",
    "CrossReceiverCommonScoringApplication",
    "CrossReceiverCommonScoringFunctional",
    "CrossReceiverCommonScoringSpec",
    "apply_cross_receiver_common_scoring_functional",
    "fit_cross_receiver_common_scoring_functional",
]
