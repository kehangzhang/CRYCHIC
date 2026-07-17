"""Fold-scoped directional bindings across response and incremental artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from crychic.attribution import (
    DirectionalContrastPairSpec,
    DirectionalResponsePair,
    build_directional_response_pair,
)
from crychic.core import ContractError, stable_id
from crychic.response import (
    FoldGeneResponseArtifact,
    RepeatedMeasuresCR2FeatureEffect,
    RepeatedMeasuresFeatureEffect,
)
from crychic.response.repeated_fold import RepeatedMeasuresFoldResponseArtifact

from .receiver_incremental import (
    ReceiverIncrementalApplication,
    ReceiverIncrementalTrainingArtifact,
)

_ResponseArtifact: TypeAlias = (
    FoldGeneResponseArtifact | RepeatedMeasuresFoldResponseArtifact
)
_RepeatedFeatureEffect: TypeAlias = (
    RepeatedMeasuresFeatureEffect | RepeatedMeasuresCR2FeatureEffect
)
_PRODUCER_MARKER = "crychic.workflow.directional_crossfit_binding.v1"
_SCHEMA_VERSION = "1"
_STATUS_OBSERVED = "observed"
_STATUS_NOT_ESTIMABLE = "not_estimable"
_COMBINATION_RULE = "independent_contrast_views_not_additive_v1"
_ALIGNMENT_ATOL = 1e-12


def _parent_mismatch(message: str, *, field: str) -> ContractError:
    return ContractError(
        message,
        code="directional_crossfit_parent_mismatch",
        field=field,
        remediation=(
            "Map forward and reverse artifacts by the exact registered contrast "
            "names and retain their original fold-scoped parents"
        ),
    )


def _response_is_usable(response: _ResponseArtifact) -> bool:
    if isinstance(response, FoldGeneResponseArtifact):
        return bool(response.status == "ok")
    return bool(
        response.status == response.observed_status
        and np.isfinite(response.effect).all()
    )


def _response_reason(response: _ResponseArtifact) -> str:
    if (
        isinstance(response, RepeatedMeasuresFoldResponseArtifact)
        and response.status == response.observed_status
        and not np.isfinite(response.effect).all()
    ):
        return "partial_feature_response_not_supported_by_directional_pair"
    return response.reason_code or "response_not_estimable"


def _diagnostic_float_token(value: float | None) -> str | None:
    if value is None:
        return None
    numeric = float(value)
    if np.isnan(numeric):
        return "nan"
    if numeric == np.inf:
        return "+inf"
    if numeric == -np.inf:
        return "-inf"
    return numeric.hex()


def _repeated_feature_diagnostics(
    effect: _RepeatedFeatureEffect,
) -> tuple[object, ...]:
    if isinstance(effect, RepeatedMeasuresCR2FeatureEffect):
        cr2_diagnostics: tuple[object, ...] = (
            effect.n_effective_clusters,
            _diagnostic_float_token(effect.maximum_cluster_leverage),
            _diagnostic_float_token(effect.minimum_cr2_adjustment_eigenvalue),
            effect.backend,
            effect.degrees_of_freedom_method,
        )
    else:
        cr2_diagnostics = (None, None, None, None, None)
    return (
        type(effect),
        effect.feature_id,
        str(effect.status),
        effect.reason_code,
        effect.n_input_samples,
        effect.n_model_cells,
        effect.n_subject_clusters,
        effect.n_contrast_subject_clusters,
        effect.n_context_subjects_min,
        effect.n_repeated_subject_clusters,
        effect.n_complete_contrast_subjects,
        effect.design_rank,
        effect.residual_df,
        effect.cluster_df,
        _diagnostic_float_token(effect.condition_number),
        tuple(effect.diagnostic_codes),
        *cr2_diagnostics,
    )


def _require_repeated_response_symmetry(
    forward: RepeatedMeasuresFoldResponseArtifact,
    reverse: RepeatedMeasuresFoldResponseArtifact,
) -> None:
    if type(forward.repeated_effect) is not type(reverse.repeated_effect):
        raise _parent_mismatch(
            "Directional repeated responses use different response backends",
            field="reverse_response.repeated_effect",
        )

    forward_design = forward.repeated_design
    reverse_design = reverse.repeated_design
    if forward_design.spec.to_dict() != reverse_design.spec.to_dict():
        raise _parent_mismatch(
            "Directional repeated responses use different frozen design policies",
            field="reverse_response.repeated_design.spec",
        )

    forward_cell_lineage = (
        forward_design.sample_ids,
        forward_design.sample_subject_ids,
        forward_design.cell_ids,
        forward_design.cell_subject_ids,
        forward_design.cell_context_ids,
        forward_design.cell_sample_counts,
        forward_design.contrast_context_ids,
        forward_design.model_columns,
    )
    reverse_cell_lineage = (
        reverse_design.sample_ids,
        reverse_design.sample_subject_ids,
        reverse_design.cell_ids,
        reverse_design.cell_subject_ids,
        reverse_design.cell_context_ids,
        reverse_design.cell_sample_counts,
        reverse_design.contrast_context_ids,
        reverse_design.model_columns,
    )
    cell_arrays_symmetric = bool(
        np.array_equal(
            forward_design.sample_cell_indices,
            reverse_design.sample_cell_indices,
        )
        and np.array_equal(
            forward_design.design_matrix,
            reverse_design.design_matrix,
        )
        and np.allclose(
            reverse_design.contrast_vector,
            -forward_design.contrast_vector,
            rtol=0.0,
            atol=_ALIGNMENT_ATOL,
        )
        and np.allclose(
            np.asarray(reverse_design.contrast_context_weights, dtype=np.float64),
            -np.asarray(forward_design.contrast_context_weights, dtype=np.float64),
            rtol=0.0,
            atol=_ALIGNMENT_ATOL,
        )
    )
    if forward_cell_lineage != reverse_cell_lineage or not cell_arrays_symmetric:
        raise _parent_mismatch(
            "Directional repeated responses do not share the same frozen cells and "
            "opposite design contrast",
            field="reverse_response.repeated_design",
        )

    forward_support = (
        forward_design.design_rank,
        _diagnostic_float_token(forward_design.condition_number),
        forward_design.residual_df,
        forward_design.n_subject_clusters,
        forward_design.n_contrast_subject_clusters,
        forward_design.n_repeated_subject_clusters,
        forward_design.n_complete_contrast_subjects,
        forward_design.context_subject_counts,
        forward_design.status,
        forward_design.reason_code,
    )
    reverse_support = (
        reverse_design.design_rank,
        _diagnostic_float_token(reverse_design.condition_number),
        reverse_design.residual_df,
        reverse_design.n_subject_clusters,
        reverse_design.n_contrast_subject_clusters,
        reverse_design.n_repeated_subject_clusters,
        reverse_design.n_complete_contrast_subjects,
        reverse_design.context_subject_counts,
        reverse_design.status,
        reverse_design.reason_code,
    )
    if forward_support != reverse_support:
        raise _parent_mismatch(
            "Directional repeated responses have asymmetric design support",
            field="reverse_response.repeated_design.support",
        )

    forward_feature_diagnostics = tuple(
        _repeated_feature_diagnostics(effect)
        for effect in forward.repeated_effect.feature_effects
    )
    reverse_feature_diagnostics = tuple(
        _repeated_feature_diagnostics(effect)
        for effect in reverse.repeated_effect.feature_effects
    )
    if forward_feature_diagnostics != reverse_feature_diagnostics:
        raise _parent_mismatch(
            "Directional repeated responses have asymmetric feature diagnostics",
            field="reverse_response.feature_effects",
        )


def _require_response_pair_scope(
    pair_spec: DirectionalContrastPairSpec,
    forward: _ResponseArtifact,
    reverse: _ResponseArtifact,
) -> None:
    if type(forward) is not type(reverse):
        raise _parent_mismatch(
            "Directional responses must use the same response estimator class",
            field="reverse_response",
        )
    if forward.contrast_name != pair_spec.forward_contrast.name:
        raise _parent_mismatch(
            "Forward response does not match the registered forward contrast name",
            field="forward_response",
        )
    if reverse.contrast_name != pair_spec.reverse_contrast.name:
        raise _parent_mismatch(
            "Reverse response does not match the registered reverse contrast name",
            field="reverse_response",
        )
    if isinstance(forward, RepeatedMeasuresFoldResponseArtifact) and isinstance(
        reverse, RepeatedMeasuresFoldResponseArtifact
    ):
        _require_repeated_response_symmetry(forward, reverse)
    shared_forward = (
        forward.receiver,
        forward.fold_id,
        forward.feature_ids,
        forward.training_input_digest,
        forward.training_sample_manifest_digest,
        forward.training_design_digest,
        forward.context_keys,
        forward.training_sample_ids,
        forward.training_sample_subject_ids,
        forward.training_sample_context_ids,
        forward.training_subject_ids,
        forward.sample_ids,
        forward.sample_subject_ids,
        forward.sample_context_ids,
        forward.subject_ids,
        forward.missing_sample_ids,
        forward.sample_values_digest,
        forward.method,
        forward.value_scale,
        forward.min_subjects_per_context,
        forward.n_model_samples,
        forward.n_model_subjects,
    )
    shared_reverse = (
        reverse.receiver,
        reverse.fold_id,
        reverse.feature_ids,
        reverse.training_input_digest,
        reverse.training_sample_manifest_digest,
        reverse.training_design_digest,
        reverse.context_keys,
        reverse.training_sample_ids,
        reverse.training_sample_subject_ids,
        reverse.training_sample_context_ids,
        reverse.training_subject_ids,
        reverse.sample_ids,
        reverse.sample_subject_ids,
        reverse.sample_context_ids,
        reverse.subject_ids,
        reverse.missing_sample_ids,
        reverse.sample_values_digest,
        reverse.method,
        reverse.value_scale,
        reverse.min_subjects_per_context,
        reverse.n_model_samples,
        reverse.n_model_subjects,
    )
    if shared_forward != shared_reverse:
        raise _parent_mismatch(
            "Directional responses do not share the exact receiver, fold, rows, "
            "subjects, features, and input lineage",
            field="reverse_response",
        )
    if _response_is_usable(forward) and _response_is_usable(reverse):
        if not np.allclose(
            reverse.effect,
            -forward.effect,
            rtol=0.0,
            atol=_ALIGNMENT_ATOL,
            equal_nan=True,
        ):
            raise _parent_mismatch(
                "Reverse response effects do not negate the forward effects",
                field="reverse_response.effect",
            )
        if not np.allclose(
            reverse.standard_error,
            forward.standard_error,
            rtol=0.0,
            atol=_ALIGNMENT_ATOL,
            equal_nan=True,
        ):
            raise _parent_mismatch(
                "Directional response standard errors are not symmetric",
                field="reverse_response.standard_error",
            )
        if not np.allclose(
            reverse.raw_precision,
            forward.raw_precision,
            rtol=0.0,
            atol=_ALIGNMENT_ATOL,
            equal_nan=True,
        ):
            raise _parent_mismatch(
                "Directional response raw precision is not symmetric",
                field="reverse_response.raw_precision",
            )


def _require_model_parent(
    *,
    channel: str,
    expected_contrast_name: str,
    response: _ResponseArtifact,
    model: ReceiverIncrementalTrainingArtifact,
) -> None:
    expected = (
        response.artifact_id,
        expected_contrast_name,
        response.receiver,
        response.fold_id,
        response.feature_ids,
        response.training_subject_ids,
        response.sample_ids,
        response.sample_context_ids,
    )
    observed = (
        model.response_artifact_id,
        model.contrast_name,
        model.receiver,
        model.fold_id,
        model.feature_ids,
        model.training_subject_ids,
        model.training_sample_ids,
        model.training_sample_context_ids,
    )
    if observed != expected:
        raise _parent_mismatch(
            f"{channel.capitalize()} incremental model does not derive from its "
            "directional response parent",
            field=f"{channel}_model",
        )


def _require_application_parent(
    *,
    channel: str,
    model: ReceiverIncrementalTrainingArtifact,
    application: ReceiverIncrementalApplication,
) -> None:
    expected_functional_id = (
        None
        if model.diagnostic_functional is None
        else model.diagnostic_functional.incremental_functional_id
    )
    expected = (
        model.training_artifact_id,
        expected_functional_id,
        model.training_subject_ids,
    )
    observed = (
        application.training_artifact_id,
        application.diagnostic_functional_id,
        application.training_subject_ids,
    )
    if observed != expected:
        raise _parent_mismatch(
            f"{channel.capitalize()} incremental application does not derive from "
            "its directional model parent",
            field=f"{channel}_application",
        )
    if set(application.heldout_subject_ids).intersection(
        application.training_subject_ids
    ):
        raise _parent_mismatch(
            f"{channel.capitalize()} incremental application overlaps training "
            "subjects",
            field=f"{channel}_application.heldout_subject_ids",
        )


def _unavailable_components(
    *,
    forward_response: _ResponseArtifact,
    reverse_response: _ResponseArtifact,
    forward_model: ReceiverIncrementalTrainingArtifact,
    reverse_model: ReceiverIncrementalTrainingArtifact,
    forward_application: ReceiverIncrementalApplication,
    reverse_application: ReceiverIncrementalApplication,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not _response_is_usable(forward_response):
        reasons.append("forward_response:" + _response_reason(forward_response))
    if not _response_is_usable(reverse_response):
        reasons.append("reverse_response:" + _response_reason(reverse_response))
    if forward_model.diagnostic_status != _STATUS_OBSERVED:
        reasons.append(
            "forward_model:"
            + (
                forward_model.diagnostic_reason_code
                or "incremental_training_not_estimable"
            )
        )
    if reverse_model.diagnostic_status != _STATUS_OBSERVED:
        reasons.append(
            "reverse_model:"
            + (
                reverse_model.diagnostic_reason_code
                or "incremental_training_not_estimable"
            )
        )
    if forward_application.diagnostic_status != _STATUS_OBSERVED:
        reasons.append(
            "forward_application:"
            + (
                forward_application.diagnostic_reason_code
                or "incremental_application_not_estimable"
            )
        )
    if reverse_application.diagnostic_status != _STATUS_OBSERVED:
        reasons.append(
            "reverse_application:"
            + (
                reverse_application.diagnostic_reason_code
                or "incremental_application_not_estimable"
            )
        )
    return tuple(reasons)


@dataclass(frozen=True, slots=True, init=False)
class DirectionalCrossFitBinding:
    """Producer-owned two-channel diagnostic binding for one fold and receiver."""

    pair_spec: DirectionalContrastPairSpec
    pair_spec_id: str
    forward_channel_name: str
    reverse_channel_name: str
    receiver: str
    fold_id: str
    feature_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    heldout_subject_ids: tuple[str, ...]
    forward_response_id: str
    reverse_response_id: str
    forward_training_artifact_id: str
    reverse_training_artifact_id: str
    forward_application_id: str
    reverse_application_id: str
    status: str
    reason_code: str | None
    response_pair: DirectionalResponsePair | None
    formal_inference_allowed: bool
    active_inhibition_allowed: bool
    supports_active_inhibition_claim: bool
    paired_score_comparison_allowed: bool
    combination_rule: str
    binding_id: str
    _forward_response: _ResponseArtifact
    _reverse_response: _ResponseArtifact
    _forward_model: ReceiverIncrementalTrainingArtifact
    _reverse_model: ReceiverIncrementalTrainingArtifact
    _forward_application: ReceiverIncrementalApplication
    _reverse_application: ReceiverIncrementalApplication
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "DirectionalCrossFitBinding is producer-owned; "
            "use build_directional_crossfit_binding()"
        )

    @classmethod
    def _from_parents(
        cls,
        pair_spec: DirectionalContrastPairSpec,
        forward_response: _ResponseArtifact,
        reverse_response: _ResponseArtifact,
        forward_model: ReceiverIncrementalTrainingArtifact,
        reverse_model: ReceiverIncrementalTrainingArtifact,
        forward_application: ReceiverIncrementalApplication,
        reverse_application: ReceiverIncrementalApplication,
    ) -> DirectionalCrossFitBinding:
        values = _validated_binding_values(
            pair_spec,
            forward_response,
            reverse_response,
            forward_model,
            reverse_model,
            forward_application,
            reverse_application,
        )
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_producer_marker", _PRODUCER_MARKER)
        object.__setattr__(
            self,
            "binding_id",
            stable_id(
                "directional_crossfit_binding",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "combination_rule": self.combination_rule,
            "active_inhibition_allowed": self.active_inhibition_allowed,
            "feature_ids": list(self.feature_ids),
            "fold_id": self.fold_id,
            "formal_inference_allowed": self.formal_inference_allowed,
            "forward_application_id": self.forward_application_id,
            "forward_channel_name": self.forward_channel_name,
            "forward_response_id": self.forward_response_id,
            "forward_training_artifact_id": self.forward_training_artifact_id,
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "paired_score_comparison_allowed": (self.paired_score_comparison_allowed),
            "pair_spec_id": self.pair_spec_id,
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "response_pair_id": (
                None
                if self.response_pair is None
                else self.response_pair.response_pair_id
            ),
            "reverse_application_id": self.reverse_application_id,
            "reverse_channel_name": self.reverse_channel_name,
            "reverse_response_id": self.reverse_response_id,
            "reverse_training_artifact_id": self.reverse_training_artifact_id,
            "status": self.status,
            "supports_active_inhibition_claim": (self.supports_active_inhibition_claim),
            "training_subject_ids": list(self.training_subject_ids),
        }

    def _require_intact(self) -> None:
        try:
            expected_values = _validated_binding_values(
                self.pair_spec,
                self._forward_response,
                self._reverse_response,
                self._forward_model,
                self._reverse_model,
                self._forward_application,
                self._reverse_application,
            )
            for name, expected in expected_values.items():
                observed = getattr(self, name)
                if name == "response_pair":
                    if not _response_pairs_match(observed, expected):
                        raise ValueError("directional response pair changed")
                elif name.startswith("_"):
                    if observed is not expected:
                        raise ValueError(f"{name} parent changed")
                elif observed != expected:
                    raise ValueError(f"{name} changed")
            expected_id = stable_id(
                "directional_crossfit_binding",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and expected_id == self.binding_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Directional cross-fit binding failed integrity validation",
                code="directional_crossfit_binding_integrity_violation",
                field="binding_id",
                remediation="Rebuild the binding from intact directional parents",
            ) from error
        if not valid:
            raise ContractError(
                "Directional cross-fit binding failed integrity validation",
                code="directional_crossfit_binding_integrity_violation",
                field="binding_id",
                remediation="Rebuild the binding from intact directional parents",
            )

    def to_dict(self) -> dict[str, object]:
        """Return complete directional lineage without expanding numeric vectors."""

        self._require_intact()
        return {
            "binding_id": self.binding_id,
            **self._identity_payload(),
            "pair_spec": self.pair_spec.to_dict(),
            "response_pair": (
                None if self.response_pair is None else self.response_pair.to_dict()
            ),
        }


def _response_pairs_match(
    observed: DirectionalResponsePair | None,
    expected: DirectionalResponsePair | None,
) -> bool:
    if observed is None or expected is None:
        return observed is expected
    return bool(
        observed.to_dict() == expected.to_dict()
        and np.array_equal(
            observed.forward_signed_response,
            expected.forward_signed_response,
        )
        and np.array_equal(
            observed.reverse_signed_response,
            expected.reverse_signed_response,
        )
        and np.array_equal(
            observed.forward_solver_response,
            expected.forward_solver_response,
        )
        and np.array_equal(
            observed.reverse_solver_response,
            expected.reverse_solver_response,
        )
    )


def _validated_binding_values(
    pair_spec: DirectionalContrastPairSpec,
    forward_response: _ResponseArtifact,
    reverse_response: _ResponseArtifact,
    forward_model: ReceiverIncrementalTrainingArtifact,
    reverse_model: ReceiverIncrementalTrainingArtifact,
    forward_application: ReceiverIncrementalApplication,
    reverse_application: ReceiverIncrementalApplication,
) -> dict[str, Any]:
    if not isinstance(pair_spec, DirectionalContrastPairSpec):
        raise TypeError("pair_spec must be a DirectionalContrastPairSpec")
    if not isinstance(
        forward_response,
        (FoldGeneResponseArtifact, RepeatedMeasuresFoldResponseArtifact),
    ) or not isinstance(
        reverse_response,
        (FoldGeneResponseArtifact, RepeatedMeasuresFoldResponseArtifact),
    ):
        raise TypeError("directional responses must be fold response artifacts")
    if not isinstance(forward_model, ReceiverIncrementalTrainingArtifact) or not (
        isinstance(reverse_model, ReceiverIncrementalTrainingArtifact)
    ):
        raise TypeError(
            "directional models must be ReceiverIncrementalTrainingArtifact"
        )
    if not isinstance(forward_application, ReceiverIncrementalApplication) or not (
        isinstance(reverse_application, ReceiverIncrementalApplication)
    ):
        raise TypeError(
            "directional applications must be ReceiverIncrementalApplication"
        )
    pair_spec._require_intact()
    forward_response._require_intact()
    reverse_response._require_intact()
    forward_model._require_intact()
    reverse_model._require_intact()
    forward_application._require_intact()
    reverse_application._require_intact()
    _require_response_pair_scope(pair_spec, forward_response, reverse_response)
    _require_model_parent(
        channel="forward",
        expected_contrast_name=pair_spec.forward_contrast.name,
        response=forward_response,
        model=forward_model,
    )
    _require_model_parent(
        channel="reverse",
        expected_contrast_name=pair_spec.reverse_contrast.name,
        response=reverse_response,
        model=reverse_model,
    )
    _require_application_parent(
        channel="forward", model=forward_model, application=forward_application
    )
    _require_application_parent(
        channel="reverse", model=reverse_model, application=reverse_application
    )
    model_scope_forward = (
        forward_model.receiver,
        forward_model.fold_id,
        forward_model.feature_ids,
        forward_model.family_ids,
        forward_model.receiver_family_training_artifact_id,
        forward_model.training_subject_ids,
    )
    model_scope_reverse = (
        reverse_model.receiver,
        reverse_model.fold_id,
        reverse_model.feature_ids,
        reverse_model.family_ids,
        reverse_model.receiver_family_training_artifact_id,
        reverse_model.training_subject_ids,
    )
    if model_scope_forward != model_scope_reverse:
        raise _parent_mismatch(
            "Directional incremental models do not share receiver, fold, features, "
            "frozen family parent, family axis, and training subjects",
            field="reverse_model",
        )
    heldout_forward = (
        forward_application.heldout_sample_ids,
        forward_application.heldout_context_ids,
        forward_application.heldout_subject_ids,
    )
    heldout_reverse = (
        reverse_application.heldout_sample_ids,
        reverse_application.heldout_context_ids,
        reverse_application.heldout_subject_ids,
    )
    if heldout_forward != heldout_reverse:
        raise _parent_mismatch(
            "Directional incremental applications do not cover the same held-out "
            "rows and subjects",
            field="reverse_application",
        )
    reasons = _unavailable_components(
        forward_response=forward_response,
        reverse_response=reverse_response,
        forward_model=forward_model,
        reverse_model=reverse_model,
        forward_application=forward_application,
        reverse_application=reverse_application,
    )
    response_pair: DirectionalResponsePair | None = None
    status = _STATUS_NOT_ESTIMABLE if reasons else _STATUS_OBSERVED
    reason_code = (
        None
        if not reasons
        else "directional_components_not_estimable[" + ";".join(reasons) + "]"
    )
    if status == _STATUS_OBSERVED:
        response_pair = build_directional_response_pair(
            forward_response.feature_ids,
            pair_spec.forward_contrast,
            pair_spec.reverse_contrast,
            forward_response.effect,
            reverse_response.effect,
            forward_response_id=forward_response.artifact_id,
            reverse_response_id=reverse_response.artifact_id,
        )
    return {
        "pair_spec": pair_spec,
        "pair_spec_id": pair_spec.pair_spec_id,
        "forward_channel_name": pair_spec.forward_channel_name,
        "reverse_channel_name": pair_spec.reverse_channel_name,
        "receiver": forward_response.receiver,
        "fold_id": forward_response.fold_id,
        "feature_ids": forward_response.feature_ids,
        "training_subject_ids": forward_response.training_subject_ids,
        "heldout_subject_ids": forward_application.heldout_subject_ids,
        "forward_response_id": forward_response.artifact_id,
        "reverse_response_id": reverse_response.artifact_id,
        "forward_training_artifact_id": forward_model.training_artifact_id,
        "reverse_training_artifact_id": reverse_model.training_artifact_id,
        "forward_application_id": forward_application.application_id,
        "reverse_application_id": reverse_application.application_id,
        "status": status,
        "reason_code": reason_code,
        "response_pair": response_pair,
        "formal_inference_allowed": False,
        "active_inhibition_allowed": False,
        "supports_active_inhibition_claim": False,
        "paired_score_comparison_allowed": False,
        "combination_rule": _COMBINATION_RULE,
        "_forward_response": forward_response,
        "_reverse_response": reverse_response,
        "_forward_model": forward_model,
        "_reverse_model": reverse_model,
        "_forward_application": forward_application,
        "_reverse_application": reverse_application,
    }


def build_directional_crossfit_binding(
    pair_spec: DirectionalContrastPairSpec,
    forward_response: _ResponseArtifact,
    reverse_response: _ResponseArtifact,
    forward_model: ReceiverIncrementalTrainingArtifact,
    reverse_model: ReceiverIncrementalTrainingArtifact,
    forward_application: ReceiverIncrementalApplication,
    reverse_application: ReceiverIncrementalApplication,
) -> DirectionalCrossFitBinding:
    """Bind two explicitly named directional chains without combining scores."""

    return DirectionalCrossFitBinding._from_parents(
        pair_spec,
        forward_response,
        reverse_response,
        forward_model,
        reverse_model,
        forward_application,
        reverse_application,
    )


__all__ = [
    "DirectionalCrossFitBinding",
    "build_directional_crossfit_binding",
]
