"""Subject-equal unpenalized effects from common-functional OOF scores."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np
import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id

_BACKEND = "subject_equal_wls_cluster_cr2_v1"
_FORMAL_STATUS = "requires_full_pipeline_resampling_not_released"
_FORMAL_DF_METHOD = "subject_clusters_minus_one_guard_only_v1"
_MIN_FORMAL_CLUSTERS = 6
_VALID_SCORE_STATUSES = frozenset({"observed", "structural_zero"})
_FORMAL_ELIGIBILITY_TOKEN = object()
_OOF_RESULT_TOKEN = object()
_COVARIANCE_TOLERANCE = 1.0e-10
_LEVERAGE_TOLERANCE = 1.0e-10
_REQUIRED_COLUMNS = (
    "subject_id",
    "sample_id",
    "fold_id",
    "context_id",
    "score",
    "score_status",
    "scoring_function_id",
)


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if result == 0.0 else result


def _immutable_array(values: np.ndarray) -> np.ndarray:
    canonical = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    canonical[canonical == 0.0] = 0.0
    result = cast(
        np.ndarray,
        np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(
            canonical.shape
        ),
    )
    result.setflags(write=False)
    return result


def _table_digest(table: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(list(_REQUIRED_COLUMNS)).encode("ascii"))
    ordered = table.sort_values(
        ["fold_id", "subject_id", "context_id", "sample_id"],
        kind="stable",
    )
    for row in ordered.loc[:, list(_REQUIRED_COLUMNS)].itertuples(
        index=False, name=None
    ):
        tokens: list[object] = []
        for value in row:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float):
                tokens.append({"float_hex": value.hex()})
            else:
                tokens.append(value)
        digest.update(canonical_json(tokens).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class OOFEffectSpec:
    """One pre-registered receiver-scoped context contrast."""

    hypothesis_id: str
    contrast_name: str
    contrast_weights: tuple[tuple[str, float], ...]
    minimum_clusters_for_diagnostic_se: int = 8
    maximum_condition_number: float = 1.0e10
    backend: str = _BACKEND
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        hypothesis = _name(self.hypothesis_id, field_name="hypothesis_id")
        contrast_name = _name(self.contrast_name, field_name="contrast_name")
        supplied = tuple(self.contrast_weights)
        if len(supplied) < 2:
            raise ValueError("contrast_weights must contain at least two contexts")
        weights: list[tuple[str, float]] = []
        for context_id, weight in supplied:
            weights.append(
                (
                    _name(context_id, field_name="context_id"),
                    _finite(weight, field_name="contrast_weight"),
                )
            )
        if len({context for context, _ in weights}) != len(weights):
            raise ValueError("contrast context IDs must be unique")
        if not any(weight > 0 for _, weight in weights) or not any(
            weight < 0 for _, weight in weights
        ):
            raise ValueError("contrast must contain positive and negative weights")
        if not math.isclose(sum(weight for _, weight in weights), 0.0, abs_tol=1e-12):
            raise ValueError("contrast weights must sum to zero")
        canonical_weights = tuple(sorted(weights))
        if (
            isinstance(self.minimum_clusters_for_diagnostic_se, bool)
            or not isinstance(self.minimum_clusters_for_diagnostic_se, int)
            or self.minimum_clusters_for_diagnostic_se < 2
        ):
            raise ValueError("minimum clusters must be an integer >= 2")
        maximum_condition_number = _finite(
            self.maximum_condition_number,
            field_name="maximum_condition_number",
        )
        if maximum_condition_number <= 1.0:
            raise ValueError("maximum_condition_number must be > 1")
        if self.backend != _BACKEND:
            raise ValueError(f"backend must be {_BACKEND!r}")
        payload = {
            "hypothesis_id": hypothesis,
            "contrast_name": contrast_name,
            "contrast_weights": [list(item) for item in canonical_weights],
            "minimum_clusters_for_diagnostic_se": (
                self.minimum_clusters_for_diagnostic_se
            ),
            "maximum_condition_number": maximum_condition_number,
            "backend": self.backend,
            "formal_inference_status": _FORMAL_STATUS,
        }
        object.__setattr__(self, "hypothesis_id", hypothesis)
        object.__setattr__(self, "contrast_name", contrast_name)
        object.__setattr__(self, "contrast_weights", canonical_weights)
        object.__setattr__(
            self, "maximum_condition_number", maximum_condition_number
        )
        object.__setattr__(
            self,
            "spec_id",
            stable_id("oof_effect_spec", payload, schema_version="1"),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "contrast_name": self.contrast_name,
            "contrast_weights": [list(item) for item in self.contrast_weights],
            "minimum_clusters_for_diagnostic_se": (
                self.minimum_clusters_for_diagnostic_se
            ),
            "maximum_condition_number": self.maximum_condition_number,
            "backend": self.backend,
            "formal_inference_status": _FORMAL_STATUS,
        }

    def _require_intact(self) -> None:
        try:
            repeated = OOFEffectSpec(
                hypothesis_id=self.hypothesis_id,
                contrast_name=self.contrast_name,
                contrast_weights=self.contrast_weights,
                minimum_clusters_for_diagnostic_se=(
                    self.minimum_clusters_for_diagnostic_se
                ),
                maximum_condition_number=self.maximum_condition_number,
                backend=self.backend,
            )
            valid = (
                self._identity_payload() == repeated._identity_payload()
                and self.spec_id == repeated.spec_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "OOF effect specification failed integrity validation",
                code="oof_effect_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the immutable OOF effect specification",
            ) from error
        if not valid:
            raise ContractError(
                "OOF effect specification failed integrity validation",
                code="oof_effect_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the immutable OOF effect specification",
            )

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(context for context, _ in self.contrast_weights)

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "spec_id": self.spec_id,
            "hypothesis_id": self.hypothesis_id,
            "contrast_name": self.contrast_name,
            "contrast_weights": [list(item) for item in self.contrast_weights],
            "minimum_clusters_for_diagnostic_se": (
                self.minimum_clusters_for_diagnostic_se
            ),
            "maximum_condition_number": self.maximum_condition_number,
            "backend": self.backend,
            "formal_inference_status": _FORMAL_STATUS,
        }


@dataclass(frozen=True, slots=True, init=False)
class OOFContextEffectResult:
    """Producer-owned point effect plus diagnostic CR2 uncertainty."""

    result_id: str
    spec_id: str
    hypothesis_id: str
    source_table_digest: str
    effect: float | None
    standard_error: float | None
    residual_df: float | None
    n_samples: int
    n_subject_context_rows: int
    n_clusters: int
    n_folds: int
    design_rank: int
    design_columns: tuple[str, ...]
    design_condition_number: float | None
    maximum_cluster_leverage: float | None
    minimum_cr2_adjustment_eigenvalue: float | None
    coefficients: np.ndarray
    covariance: np.ndarray
    effect_status: str
    uncertainty_status: str
    reason_code: str | None
    backend: str
    formal_inference_status: str
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "OOFContextEffectResult is producer-owned; use fit_oof_context_effect()"
        )

    @classmethod
    def _from_fit(cls, **values: object) -> OOFContextEffectResult:
        self = object.__new__(cls)
        values = dict(values)
        values["coefficients"] = _immutable_array(
            cast(np.ndarray, values["coefficients"])
        )
        values["covariance"] = _immutable_array(
            cast(np.ndarray, values["covariance"])
        )
        values["design_columns"] = tuple(
            cast(tuple[str, ...], values["design_columns"])
        )
        values["backend"] = _BACKEND
        values["formal_inference_status"] = _FORMAL_STATUS
        values["_producer_token"] = _OOF_RESULT_TOKEN
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "result_id", "pending")
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "oof_context_effect",
                self._result_identity_payload(),
                schema_version="2",
            ),
        )
        self._require_intact()
        return self

    @property
    def formal_inference_allowed(self) -> bool:
        self._require_intact()
        return False

    def _result_identity_payload(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "hypothesis_id": self.hypothesis_id,
            "source_table_digest": self.source_table_digest,
            "effect": self.effect,
            "standard_error": self.standard_error,
            "residual_df": self.residual_df,
            "n_samples": self.n_samples,
            "n_subject_context_rows": self.n_subject_context_rows,
            "n_clusters": self.n_clusters,
            "n_folds": self.n_folds,
            "design_rank": self.design_rank,
            "design_columns": list(self.design_columns),
            "design_condition_number": self.design_condition_number,
            "maximum_cluster_leverage": self.maximum_cluster_leverage,
            "minimum_cr2_adjustment_eigenvalue": (
                self.minimum_cr2_adjustment_eigenvalue
            ),
            "coefficients": self.coefficients.tolist(),
            "covariance": self.covariance.tolist(),
            "effect_status": self.effect_status,
            "uncertainty_status": self.uncertainty_status,
            "reason_code": self.reason_code,
            "backend": self.backend,
            "formal_inference_status": self.formal_inference_status,
        }

    def _require_intact(self) -> None:
        try:
            columns = tuple(
                _name(item, field_name="design_columns")
                for item in self.design_columns
            )
            counts_valid = all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in (
                    self.n_samples,
                    self.n_subject_context_rows,
                    self.n_clusters,
                    self.n_folds,
                    self.design_rank,
                )
            )
            arrays_valid = bool(
                self.coefficients.dtype == np.dtype("<f8")
                and self.covariance.dtype == np.dtype("<f8")
                and not self.coefficients.flags.writeable
                and not self.covariance.flags.writeable
            )
            expected_id = stable_id(
                "oof_context_effect",
                self._result_identity_payload(),
                schema_version="2",
            )
            valid = bool(
                self._producer_token is _OOF_RESULT_TOKEN
                and _name(self.spec_id, field_name="spec_id") == self.spec_id
                and _name(self.hypothesis_id, field_name="hypothesis_id")
                == self.hypothesis_id
                and len(self.source_table_digest) == 64
                and all(
                    item in "0123456789abcdef" for item in self.source_table_digest
                )
                and columns == self.design_columns
                and len(columns) == len(set(columns))
                and counts_valid
                and arrays_valid
                and self.backend == _BACKEND
                and self.formal_inference_status == _FORMAL_STATUS
                and self.result_id == expected_id
            )
            if self.effect_status == "observed":
                covariance_shape = (len(columns), len(columns))
                valid = bool(
                    valid
                    and self.reason_code is None
                    and self.effect is not None
                    and math.isfinite(self.effect)
                    and columns
                    and self.design_rank == len(columns)
                    and self.coefficients.shape == (len(columns),)
                    and np.all(np.isfinite(self.coefficients))
                    and self.design_condition_number is not None
                    and math.isfinite(self.design_condition_number)
                    and self.design_condition_number >= 1.0
                    and self.uncertainty_status
                    in {
                        "observed_cr2_diagnostic",
                        "diagnostic_small_cluster_support",
                        "cr2_not_estimable",
                    }
                )
                if self.uncertainty_status == "cr2_not_estimable":
                    valid = bool(
                        valid
                        and self.standard_error is None
                        and self.residual_df is None
                        and self.covariance.shape == (0, 0)
                        and self.maximum_cluster_leverage is None
                        and self.minimum_cr2_adjustment_eigenvalue is None
                    )
                else:
                    valid = bool(
                        valid
                        and self.standard_error is not None
                        and math.isfinite(self.standard_error)
                        and self.standard_error >= 0.0
                        and self.residual_df == float(self.n_clusters - 1)
                        and self.covariance.shape == covariance_shape
                        and np.all(np.isfinite(self.covariance))
                        and np.allclose(
                            self.covariance,
                            self.covariance.T,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                        and self.maximum_cluster_leverage is not None
                        and math.isfinite(self.maximum_cluster_leverage)
                        and 0.0 <= self.maximum_cluster_leverage < 1.0
                        and self.minimum_cr2_adjustment_eigenvalue is not None
                        and math.isfinite(self.minimum_cr2_adjustment_eigenvalue)
                        and self.minimum_cr2_adjustment_eigenvalue > 0.0
                    )
            elif self.effect_status == "not_estimable":
                condition_valid = bool(
                    self.design_condition_number is None
                    or (
                        math.isfinite(self.design_condition_number)
                        and self.design_condition_number >= 1.0
                    )
                )
                valid = bool(
                    valid
                    and self.reason_code
                    and self.effect is None
                    and self.standard_error is None
                    and self.residual_df is None
                    and self.design_rank == 0
                    and not columns
                    and self.coefficients.shape == (0,)
                    and self.covariance.shape == (0, 0)
                    and self.uncertainty_status == "not_estimable"
                    and condition_valid
                    and self.maximum_cluster_leverage is None
                    and self.minimum_cr2_adjustment_eigenvalue is None
                )
            else:
                valid = False
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "OOF context effect result failed integrity validation",
                code="oof_effect_result_integrity_violation",
                field="result_id",
                remediation="Refit the effect from its intact OOF score table",
            ) from error
        if not valid:
            raise ContractError(
                "OOF context effect result failed integrity validation",
                code="oof_effect_result_integrity_violation",
                field="result_id",
                remediation="Refit the effect from its intact OOF score table",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "result_id": self.result_id,
            "spec_id": self.spec_id,
            "hypothesis_id": self.hypothesis_id,
            "source_table_digest": self.source_table_digest,
            "effect": self.effect,
            "standard_error": self.standard_error,
            "residual_df": self.residual_df,
            "n_samples": self.n_samples,
            "n_subject_context_rows": self.n_subject_context_rows,
            "n_clusters": self.n_clusters,
            "n_folds": self.n_folds,
            "design_rank": self.design_rank,
            "design_columns": list(self.design_columns),
            "design_condition_number": self.design_condition_number,
            "maximum_cluster_leverage": self.maximum_cluster_leverage,
            "minimum_cr2_adjustment_eigenvalue": (
                self.minimum_cr2_adjustment_eigenvalue
            ),
            "effect_status": self.effect_status,
            "uncertainty_status": self.uncertainty_status,
            "reason_code": self.reason_code,
            "backend": self.backend,
            "formal_inference_status": self.formal_inference_status,
            "formal_inference_allowed": False,
        }


class OOFEffectFormalEligibilityStatus(StrEnum):
    """Eligibility of an analytic OOF point fit for the formal resampling chain."""

    ELIGIBLE = "eligible"
    NOT_ESTIMABLE = "not_estimable"


@dataclass(frozen=True, slots=True, init=False)
class OOFEffectFormalEligibility:
    """Producer-owned, fail-closed assessment of one OOF CR2 point fit.

    Eligibility permits the fit to enter full-pipeline bootstrap/permutation
    inference.  It never authorizes an analytic p-value or a release by itself.
    """

    effect_result_id: str
    spec_id: str
    status: OOFEffectFormalEligibilityStatus
    reason_code: str | None
    n_clusters: int
    minimum_clusters: int
    cluster_df: int | None
    degrees_of_freedom_method: str
    eligibility_id: str
    _producer_token: object

    def __init__(self) -> None:
        raise TypeError(
            "OOFEffectFormalEligibility is producer-owned; "
            "use assess_oof_effect_formal_eligibility()"
        )

    @classmethod
    def _from_assessment(
        cls,
        *,
        result: OOFContextEffectResult,
        spec: OOFEffectSpec,
        status: OOFEffectFormalEligibilityStatus,
        reason_code: str | None,
    ) -> OOFEffectFormalEligibility:
        self = object.__new__(cls)
        values: dict[str, object] = {
            "effect_result_id": result.result_id,
            "spec_id": spec.spec_id,
            "status": status,
            "reason_code": reason_code,
            "n_clusters": result.n_clusters,
            "minimum_clusters": max(
                spec.minimum_clusters_for_diagnostic_se,
                _MIN_FORMAL_CLUSTERS,
            ),
            "cluster_df": result.n_clusters - 1 if result.n_clusters > 0 else None,
            "degrees_of_freedom_method": _FORMAL_DF_METHOD,
            "_producer_token": _FORMAL_ELIGIBILITY_TOKEN,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "eligibility_id", "pending")
        object.__setattr__(
            self,
            "eligibility_id",
            stable_id(
                "oof_effect_formal_eligibility",
                self._identity_payload(),
                schema_version="1",
            ),
        )
        self._require_intact()
        return self

    @property
    def formal_backend_eligible(self) -> bool:
        self._require_intact()
        return self.status is OOFEffectFormalEligibilityStatus.ELIGIBLE

    @property
    def formal_inference_allowed(self) -> bool:
        """Release still requires full-pipeline resampling and a G3-F gate."""

        self._require_intact()
        return False

    def _identity_payload(self) -> dict[str, object]:
        return {
            "cluster_df": self.cluster_df,
            "degrees_of_freedom_method": self.degrees_of_freedom_method,
            "effect_result_id": self.effect_result_id,
            "minimum_clusters": self.minimum_clusters,
            "n_clusters": self.n_clusters,
            "reason_code": self.reason_code,
            "release_requirement": "full_pipeline_resampling_and_g3f_gate",
            "spec_id": self.spec_id,
            "status": self.status.value,
        }

    def _require_intact(self) -> None:
        try:
            counts_valid = bool(
                isinstance(self.n_clusters, int)
                and not isinstance(self.n_clusters, bool)
                and self.n_clusters >= 0
                and isinstance(self.minimum_clusters, int)
                and not isinstance(self.minimum_clusters, bool)
                and self.minimum_clusters >= _MIN_FORMAL_CLUSTERS
            )
            valid = bool(
                self._producer_token is _FORMAL_ELIGIBILITY_TOKEN
                and _name(self.effect_result_id, field_name="effect_result_id")
                == self.effect_result_id
                and _name(self.spec_id, field_name="spec_id") == self.spec_id
                and self.degrees_of_freedom_method == _FORMAL_DF_METHOD
                and counts_valid
                and self.eligibility_id
                == stable_id(
                    "oof_effect_formal_eligibility",
                    self._identity_payload(),
                    schema_version="1",
                )
            )
            if self.status is OOFEffectFormalEligibilityStatus.ELIGIBLE:
                valid = bool(
                    valid
                    and self.reason_code is None
                    and self.n_clusters >= self.minimum_clusters
                    and self.cluster_df == self.n_clusters - 1
                )
            else:
                valid = bool(
                    valid
                    and self.status
                    is OOFEffectFormalEligibilityStatus.NOT_ESTIMABLE
                    and self.reason_code
                    and self.cluster_df
                    == (self.n_clusters - 1 if self.n_clusters > 0 else None)
                )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "OOF formal eligibility failed integrity validation",
                code="oof_effect_formal_eligibility_integrity_violation",
                field="eligibility_id",
                remediation="Reassess the intact OOF effect and specification",
            ) from error
        if not valid:
            raise ContractError(
                "OOF formal eligibility failed integrity validation",
                code="oof_effect_formal_eligibility_integrity_violation",
                field="eligibility_id",
                remediation="Reassess the intact OOF effect and specification",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "eligibility_id": self.eligibility_id,
            "effect_result_id": self.effect_result_id,
            "spec_id": self.spec_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "n_clusters": self.n_clusters,
            "minimum_clusters": self.minimum_clusters,
            "cluster_df": self.cluster_df,
            "degrees_of_freedom_method": self.degrees_of_freedom_method,
            "formal_backend_eligible": self.formal_backend_eligible,
            "formal_inference_allowed": False,
            "release_requirement": "full_pipeline_resampling_and_g3f_gate",
        }


def _result_contrast_vector(
    result: OOFContextEffectResult,
    spec: OOFEffectSpec,
) -> np.ndarray | None:
    weights = dict(spec.contrast_weights)
    contrast: np.ndarray = np.zeros(len(result.design_columns), dtype=float)
    observed_contexts: set[str] = set()
    for index, column in enumerate(result.design_columns):
        if column.startswith("context:"):
            context = column.removeprefix("context:")
            if context not in weights or context in observed_contexts:
                return None
            observed_contexts.add(context)
            contrast[index] = weights[context]
        elif not column.startswith("fold:"):
            return None
    if observed_contexts != set(spec.context_ids):
        return None
    return contrast


def _covariance_is_psd(covariance: np.ndarray) -> bool:
    if covariance.size == 0:
        return False
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = float(np.max(np.abs(eigenvalues)))
    if scale == 0.0:
        return True
    return bool(float(np.min(eigenvalues)) >= -_COVARIANCE_TOLERANCE * scale)


def assess_oof_effect_formal_eligibility(
    result: OOFContextEffectResult,
    spec: OOFEffectSpec,
) -> OOFEffectFormalEligibility:
    """Assess whether an OOF CR2 fit can enter formal resampling inference."""

    if not isinstance(result, OOFContextEffectResult):
        raise TypeError("result must be an OOFContextEffectResult")
    if not isinstance(spec, OOFEffectSpec):
        raise TypeError("spec must be an OOFEffectSpec")
    result._require_intact()
    spec._require_intact()
    if result.spec_id != spec.spec_id or result.hypothesis_id != spec.hypothesis_id:
        raise ContractError(
            "OOF effect and formal eligibility specification do not match",
            code="oof_effect_formal_spec_mismatch",
            field="spec_id,hypothesis_id",
            remediation="Assess the result with the specification used to fit it",
        )
    reason: str | None = None
    minimum_clusters = max(
        spec.minimum_clusters_for_diagnostic_se,
        _MIN_FORMAL_CLUSTERS,
    )
    if result.effect_status != "observed":
        reason = result.reason_code or "oof_effect_not_estimable"
    elif result.n_clusters < minimum_clusters:
        reason = "oof_effect_insufficient_subject_clusters_for_formal_cr2"
    elif result.design_rank != len(result.design_columns):
        reason = "oof_effect_rank_deficient_design"
    elif (
        result.design_condition_number is None
        or not math.isfinite(result.design_condition_number)
        or result.design_condition_number > spec.maximum_condition_number
    ):
        reason = "oof_effect_ill_conditioned_design"
    elif result.uncertainty_status != "observed_cr2_diagnostic":
        reason = (
            "oof_effect_cr2_covariance_not_estimable"
            if result.uncertainty_status == "cr2_not_estimable"
            else "oof_effect_cr2_uncertainty_not_formal_eligible"
        )
    elif (
        result.maximum_cluster_leverage is None
        or not math.isfinite(result.maximum_cluster_leverage)
        or result.maximum_cluster_leverage < 0.0
        or result.maximum_cluster_leverage >= 1.0
        or result.minimum_cr2_adjustment_eigenvalue is None
        or not math.isfinite(result.minimum_cr2_adjustment_eigenvalue)
        or result.minimum_cr2_adjustment_eigenvalue <= _LEVERAGE_TOLERANCE
    ):
        reason = "oof_effect_invalid_cr2_leverage_diagnostics"
    elif (
        result.covariance.shape
        != (len(result.design_columns), len(result.design_columns))
        or not np.all(np.isfinite(result.covariance))
    ):
        reason = "oof_effect_non_finite_cr2_covariance"
    elif not _covariance_is_psd(result.covariance):
        reason = "oof_effect_indefinite_cr2_covariance"
    else:
        contrast = _result_contrast_vector(result, spec)
        if contrast is None:
            reason = "oof_effect_design_contrast_mismatch"
        else:
            expected_effect = float(contrast @ result.coefficients)
            expected_variance = float(contrast @ result.covariance @ contrast)
            expected_standard_error = (
                math.sqrt(expected_variance) if expected_variance > 0.0 else None
            )
            if (
                result.effect is None
                or not math.isfinite(result.effect)
                or not math.isclose(
                    result.effect,
                    expected_effect,
                    rel_tol=1.0e-10,
                    abs_tol=0.0,
                )
            ):
                reason = "oof_effect_contrast_coefficient_mismatch"
            elif (
                expected_standard_error is None
                or result.standard_error is None
                or not math.isfinite(result.standard_error)
                or result.standard_error <= 0.0
                or not math.isclose(
                    result.standard_error,
                    expected_standard_error,
                    rel_tol=1.0e-10,
                    abs_tol=0.0,
                )
            ):
                reason = "oof_effect_degenerate_cr2_contrast_variance"
    if reason is None and result.residual_df != float(result.n_clusters - 1):
        reason = "oof_effect_invalid_cluster_degrees_of_freedom"
    status = (
        OOFEffectFormalEligibilityStatus.ELIGIBLE
        if reason is None
        else OOFEffectFormalEligibilityStatus.NOT_ESTIMABLE
    )
    return OOFEffectFormalEligibility._from_assessment(
        result=result,
        spec=spec,
        status=status,
        reason_code=reason,
    )


def _not_estimable(
    spec: OOFEffectSpec,
    *,
    table_digest: str,
    n_samples: int,
    n_subject_context_rows: int,
    n_clusters: int,
    n_folds: int,
    reason_code: str,
    design_condition_number: float | None = None,
) -> OOFContextEffectResult:
    if design_condition_number is not None and not math.isfinite(
        design_condition_number
    ):
        design_condition_number = None
    return OOFContextEffectResult._from_fit(
        spec_id=spec.spec_id,
        hypothesis_id=spec.hypothesis_id,
        source_table_digest=table_digest,
        effect=None,
        standard_error=None,
        residual_df=None,
        n_samples=n_samples,
        n_subject_context_rows=n_subject_context_rows,
        n_clusters=n_clusters,
        n_folds=n_folds,
        design_rank=0,
        design_columns=(),
        design_condition_number=design_condition_number,
        maximum_cluster_leverage=None,
        minimum_cr2_adjustment_eigenvalue=None,
        coefficients=np.empty(0),
        covariance=np.empty((0, 0)),
        effect_status="not_estimable",
        uncertainty_status="not_estimable",
        reason_code=reason_code,
    )


def _cr2_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    bread: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    weighted_design = np.sqrt(weights)[:, np.newaxis] * design
    weighted_residual = np.sqrt(weights) * residual
    meat = np.zeros_like(bread)
    maximum_cluster_leverage = 0.0
    minimum_adjustment_eigenvalue = math.inf
    for cluster in sorted(set(clusters.tolist())):
        selected = clusters == cluster
        cluster_design = weighted_design[selected]
        cluster_residual = weighted_residual[selected]
        leverage = cluster_design @ bread @ cluster_design.T
        leverage = 0.5 * (leverage + leverage.T)
        leverage_eigenvalues = np.linalg.eigvalsh(leverage)
        maximum_cluster_leverage = max(
            maximum_cluster_leverage,
            max(0.0, float(np.max(leverage_eigenvalues))),
        )
        adjustment_matrix = np.eye(int(selected.sum())) - leverage
        adjustment_matrix = 0.5 * (adjustment_matrix + adjustment_matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(adjustment_matrix)
        cluster_minimum = float(np.min(eigenvalues))
        minimum_adjustment_eigenvalue = min(
            minimum_adjustment_eigenvalue,
            cluster_minimum,
        )
        if cluster_minimum <= _LEVERAGE_TOLERANCE:
            raise np.linalg.LinAlgError("cluster leverage leaves no CR2 residual space")
        adjustment = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        score = cluster_design.T @ (adjustment @ cluster_residual)
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    covariance = cast(np.ndarray, 0.5 * (covariance + covariance.T))
    if not math.isfinite(minimum_adjustment_eigenvalue):
        raise np.linalg.LinAlgError("CR2 requires at least one subject cluster")
    return (
        covariance,
        maximum_cluster_leverage,
        minimum_adjustment_eigenvalue,
    )


def fit_oof_context_effect(
    score_table: pd.DataFrame,
    spec: OOFEffectSpec,
) -> OOFContextEffectResult:
    """Fit a subject-equal context contrast after averaging repeated samples."""

    if not isinstance(score_table, pd.DataFrame):
        raise TypeError("score_table must be a pandas DataFrame")
    if not isinstance(spec, OOFEffectSpec):
        raise TypeError("spec must be an OOFEffectSpec")
    spec._require_intact()
    missing = set(_REQUIRED_COLUMNS).difference(score_table.columns)
    if missing:
        raise ValueError(f"score_table is missing columns: {sorted(missing)}")
    table = score_table.loc[:, list(_REQUIRED_COLUMNS)].copy()
    for column in (
        "subject_id",
        "sample_id",
        "fold_id",
        "context_id",
        "score_status",
        "scoring_function_id",
    ):
        if table[column].isna().any():
            raise ValueError(f"{column} cannot contain missing values")
        table[column] = table[column].astype(str)
        if table[column].eq("").any():
            raise ValueError(f"{column} cannot contain empty values")
    if table["sample_id"].duplicated().any():
        raise ContractError(
            "OOF effect input must contain one row per sample",
            code="oof_effect_duplicate_sample",
            field="sample_id",
            remediation="Select one receiver-family score before effect fitting",
        )
    numeric = pd.to_numeric(table["score"], errors="coerce")
    table_digest = _table_digest(table.assign(score=numeric))
    contexts = set(table["context_id"])
    if contexts != set(spec.context_ids):
        return _not_estimable(
            spec,
            table_digest=table_digest,
            n_samples=len(table),
            n_subject_context_rows=0,
            n_clusters=table["subject_id"].nunique(),
            n_folds=table["fold_id"].nunique(),
            reason_code="oof_effect_context_universe_mismatch",
        )
    usable_status = table["score_status"].isin(_VALID_SCORE_STATUSES)
    if (
        not usable_status.all()
        or numeric.isna().any()
        or not np.isfinite(numeric).all()
    ):
        return _not_estimable(
            spec,
            table_digest=table_digest,
            n_samples=len(table),
            n_subject_context_rows=0,
            n_clusters=table["subject_id"].nunique(),
            n_folds=table["fold_id"].nunique(),
            reason_code="incomplete_oof_score_coverage",
        )
    table["score"] = numeric.astype(float)
    if table.loc[table["score_status"].eq("structural_zero"), "score"].ne(0).any():
        raise ContractError(
            "Structural-zero OOF scores must be exactly zero",
            code="oof_effect_invalid_structural_zero",
            field="score",
            remediation="Preserve structural-zero values from scoring",
        )
    if table.groupby("fold_id")["scoring_function_id"].nunique().ne(1).any():
        raise ContractError(
            "Every fold must use one common scoring functional across contexts",
            code="oof_effect_noncommon_functional",
            field="scoring_function_id",
            remediation="Apply one frozen functional to all held-out contexts",
        )
    if table.groupby("subject_id")["fold_id"].nunique().ne(1).any():
        raise ContractError(
            "Each subject must belong to exactly one OOF fold",
            code="oof_effect_subject_fold_leakage",
            field="subject_id,fold_id",
            remediation="Repair the subject-block OOF table",
        )
    fold_contexts = table.groupby("fold_id")["context_id"].agg(set)
    if any(values != set(spec.context_ids) for values in fold_contexts):
        return _not_estimable(
            spec,
            table_digest=table_digest,
            n_samples=len(table),
            n_subject_context_rows=0,
            n_clusters=table["subject_id"].nunique(),
            n_folds=table["fold_id"].nunique(),
            reason_code="oof_effect_fold_context_support_incomplete",
        )

    collapsed = (
        table.groupby(
            ["subject_id", "fold_id", "context_id"],
            sort=True,
            observed=True,
        )["score"]
        .mean()
        .reset_index()
    )
    subjects = tuple(sorted(collapsed["subject_id"].unique()))
    folds = tuple(sorted(collapsed["fold_id"].unique()))
    contexts_ordered = spec.context_ids
    n_clusters = len(subjects)
    if n_clusters < 2:
        return _not_estimable(
            spec,
            table_digest=table_digest,
            n_samples=len(table),
            n_subject_context_rows=len(collapsed),
            n_clusters=n_clusters,
            n_folds=len(folds),
            reason_code="oof_effect_insufficient_subject_clusters",
        )
    context_index = {context: index for index, context in enumerate(contexts_ordered)}
    nuisance_folds = folds[1:]
    fold_index = {
        fold: len(contexts_ordered) + index for index, fold in enumerate(nuisance_folds)
    }
    design_columns = (
        *(f"context:{context}" for context in contexts_ordered),
        *(f"fold:{fold}" for fold in nuisance_folds),
    )
    design: np.ndarray = np.zeros((len(collapsed), len(design_columns)), dtype=float)
    for row_index, row in collapsed.iterrows():
        design[row_index, context_index[str(row["context_id"])]] = 1.0
        fold = str(row["fold_id"])
        if fold in fold_index:
            design[row_index, fold_index[fold]] = 1.0
    response = collapsed["score"].to_numpy(dtype=float)
    clusters = collapsed["subject_id"].astype(str).to_numpy()
    cluster_counts = collapsed.groupby("subject_id").size().to_dict()
    weights = np.asarray(
        [1.0 / int(cluster_counts[subject]) for subject in clusters], dtype=float
    )
    weighted_design = np.sqrt(weights)[:, np.newaxis] * design
    weighted_response = np.sqrt(weights) * response
    rank = int(np.linalg.matrix_rank(weighted_design))
    if rank != design.shape[1]:
        return _not_estimable(
            spec,
            table_digest=table_digest,
            n_samples=len(table),
            n_subject_context_rows=len(collapsed),
            n_clusters=n_clusters,
            n_folds=len(folds),
            reason_code="oof_effect_rank_deficient_design",
        )
    design_condition_number = float(np.linalg.cond(weighted_design))
    if (
        not math.isfinite(design_condition_number)
        or design_condition_number > spec.maximum_condition_number
    ):
        return _not_estimable(
            spec,
            table_digest=table_digest,
            n_samples=len(table),
            n_subject_context_rows=len(collapsed),
            n_clusters=n_clusters,
            n_folds=len(folds),
            reason_code="oof_effect_ill_conditioned_design",
            design_condition_number=design_condition_number,
        )
    gram = weighted_design.T @ weighted_design
    try:
        bread = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        return _not_estimable(
            spec,
            table_digest=table_digest,
            n_samples=len(table),
            n_subject_context_rows=len(collapsed),
            n_clusters=n_clusters,
            n_folds=len(folds),
            reason_code="oof_effect_singular_weighted_gram",
            design_condition_number=design_condition_number,
        )
    coefficients = bread @ weighted_design.T @ weighted_response
    residual = response - design @ coefficients
    contrast: np.ndarray = np.zeros(design.shape[1], dtype=float)
    weights_by_context = dict(spec.contrast_weights)
    for context, index in context_index.items():
        contrast[index] = weights_by_context[context]
    effect = float(contrast @ coefficients)
    uncertainty_status = "observed_cr2_diagnostic"
    standard_error: float | None
    residual_df: float | None
    covariance: np.ndarray
    maximum_cluster_leverage: float | None
    minimum_cr2_adjustment_eigenvalue: float | None
    try:
        (
            covariance,
            maximum_cluster_leverage,
            minimum_cr2_adjustment_eigenvalue,
        ) = _cr2_covariance(design, residual, weights, clusters, bread)
        if not np.all(np.isfinite(covariance)) or not _covariance_is_psd(covariance):
            raise np.linalg.LinAlgError("indefinite or non-finite CR2 covariance")
        variance = float(contrast @ covariance @ contrast)
        variance_scale = float(
            np.linalg.norm(contrast) ** 2 * np.linalg.norm(covariance, ord=2)
        )
        if (
            not math.isfinite(variance)
            or variance < -_COVARIANCE_TOLERANCE * variance_scale
        ):
            raise np.linalg.LinAlgError("negative CR2 contrast variance")
        standard_error = math.sqrt(max(0.0, variance))
        residual_df = float(n_clusters - 1)
        if n_clusters < max(
            spec.minimum_clusters_for_diagnostic_se,
            _MIN_FORMAL_CLUSTERS,
        ):
            uncertainty_status = "diagnostic_small_cluster_support"
    except np.linalg.LinAlgError:
        covariance = np.empty((0, 0))
        standard_error = None
        residual_df = None
        maximum_cluster_leverage = None
        minimum_cr2_adjustment_eigenvalue = None
        uncertainty_status = "cr2_not_estimable"
    return OOFContextEffectResult._from_fit(
        spec_id=spec.spec_id,
        hypothesis_id=spec.hypothesis_id,
        source_table_digest=table_digest,
        effect=effect,
        standard_error=standard_error,
        residual_df=residual_df,
        n_samples=len(table),
        n_subject_context_rows=len(collapsed),
        n_clusters=n_clusters,
        n_folds=len(folds),
        design_rank=rank,
        design_columns=tuple(design_columns),
        design_condition_number=design_condition_number,
        maximum_cluster_leverage=maximum_cluster_leverage,
        minimum_cr2_adjustment_eigenvalue=(
            minimum_cr2_adjustment_eigenvalue
        ),
        coefficients=coefficients,
        covariance=covariance,
        effect_status="observed",
        uncertainty_status=uncertainty_status,
        reason_code=None,
    )


__all__ = [
    "OOFContextEffectResult",
    "OOFEffectFormalEligibility",
    "OOFEffectFormalEligibilityStatus",
    "OOFEffectSpec",
    "assess_oof_effect_formal_eligibility",
    "fit_oof_context_effect",
]
