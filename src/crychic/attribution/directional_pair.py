"""Paired forward/reverse response semantics for non-negative target priors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from crychic.core import ContractError, stable_id
from crychic.design import ContrastSpec

_PAIR_SCHEMA_VERSION = "1"
_RECONSTRUCTION_SCHEMA_VERSION = "1"
_CONTRAST_PAIR_SPEC_SCHEMA_VERSION = "1.0.0"
_ALIGNMENT_ATOL = 1e-12
_FORWARD_CLAIM = "direction_compatible_activation"
_REVERSE_CLAIM = "attenuation_of_activation_from_explicit_reverse_contrast"
_PRIOR_SEMANTICS = "nonnegative_activation_prior_v1"
_COMBINATION_RULE = "independent_contrast_views_not_additive_v1"
_FORWARD_CHANNEL_NAME = "increased_activation_compatible"
_REVERSE_CHANNEL_NAME = "reduced_activation_compatible"


def _feature_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError("feature_ids must be a sequence, not a string")
    result = tuple(values)
    if not result or any(
        not isinstance(value, str) or not value.strip() for value in result
    ):
        raise ContractError(
            "Directional response features must be non-empty strings",
            code="invalid_directional_feature_alignment",
            field="feature_ids",
            remediation="Provide one unique feature ID per response position",
        )
    if len(result) != len(set(result)):
        raise ContractError(
            "Directional response features must be unique",
            code="invalid_directional_feature_alignment",
            field="feature_ids",
            remediation="Resolve duplicate features before response attribution",
        )
    return result


def _identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(
            f"{field_name} must be a non-empty producer identifier",
            code="invalid_directional_response_provenance",
            field=field_name,
            remediation="Retain the exact response artifact identity",
        )
    return value


def _immutable_vector(values: np.ndarray) -> np.ndarray:
    """Return a canonical float64 vector backed by immutable bytes."""

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


def _finite_aligned_vector(
    values: np.ndarray,
    *,
    length: int,
    field_name: str,
) -> np.ndarray:
    result = _immutable_vector(values)
    if result.shape != (length,) or np.any(~np.isfinite(result)):
        raise ContractError(
            f"{field_name} must be a finite vector aligned to feature_ids",
            code="invalid_directional_feature_alignment",
            field=field_name,
            remediation="Align one finite response value per declared feature",
        )
    return result


def _contrast_id(contrast: ContrastSpec) -> str:
    identifier: str = stable_id("contrast", contrast.to_dict())
    return identifier


def _validate_reverse_contrast(
    forward: ContrastSpec,
    reverse: ContrastSpec,
) -> None:
    if not isinstance(forward, ContrastSpec) or not isinstance(reverse, ContrastSpec):
        raise TypeError("forward_contrast and reverse_contrast must be ContrastSpec")
    if (
        not forward.estimable
        or not reverse.estimable
        or forward.reason_code is not None
        or reverse.reason_code is not None
    ):
        raise ContractError(
            "Directional channels require two estimable contrasts",
            code="invalid_directional_contrast_pair",
            field="forward_contrast",
            remediation="Resolve contrast estimability before attribution",
        )
    if forward.name == reverse.name:
        raise ContractError(
            "Reverse attenuation requires a separately named contrast",
            code="invalid_directional_contrast_pair",
            field="reverse_contrast",
            remediation="Register an explicit reverse contrast with distinct identity",
        )
    if forward.family != reverse.family or forward.mode != reverse.mode:
        raise ContractError(
            "Forward and reverse contrasts must share family and mode",
            code="invalid_directional_contrast_pair",
            field="reverse_contrast",
            remediation="Reverse only the weights of the registered forward contrast",
        )
    if set(forward.weights) != set(reverse.weights) or any(
        float(reverse.weights[node]) != -float(weight)
        for node, weight in forward.weights.items()
    ):
        raise ContractError(
            "Reverse contrast weights must exactly negate the forward contrast",
            code="invalid_directional_contrast_pair",
            field="reverse_contrast",
            remediation=(
                "Create the reverse contrast over the identical context support"
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectionalContrastPairSpec:
    """Explicitly ordered forward/reverse contrasts for two-channel scoring."""

    forward_contrast: ContrastSpec
    reverse_contrast: ContrastSpec
    schema_version: str = _CONTRAST_PAIR_SPEC_SCHEMA_VERSION
    forward_contrast_id: str = field(init=False)
    reverse_contrast_id: str = field(init=False)
    forward_channel_name: str = field(init=False)
    reverse_channel_name: str = field(init=False)
    supports_active_inhibition_claim: bool = field(init=False)
    combination_rule: str = field(init=False)
    pair_spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_reverse_contrast(self.forward_contrast, self.reverse_contrast)
        if self.schema_version != _CONTRAST_PAIR_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "directional contrast pair schema_version must be "
                f"{_CONTRAST_PAIR_SPEC_SCHEMA_VERSION}"
            )
        forward_contrast_id = _contrast_id(self.forward_contrast)
        reverse_contrast_id = _contrast_id(self.reverse_contrast)
        pair_spec_id = stable_id(
            "directional_contrast_pair_spec",
            {
                "combination_rule": _COMBINATION_RULE,
                "forward_channel_name": _FORWARD_CHANNEL_NAME,
                "forward_contrast_id": forward_contrast_id,
                "reverse_channel_name": _REVERSE_CHANNEL_NAME,
                "reverse_contrast_id": reverse_contrast_id,
                "schema_version": self.schema_version,
                "supports_active_inhibition_claim": False,
            },
            schema_version="1",
        )
        object.__setattr__(self, "forward_contrast_id", forward_contrast_id)
        object.__setattr__(self, "reverse_contrast_id", reverse_contrast_id)
        object.__setattr__(self, "forward_channel_name", _FORWARD_CHANNEL_NAME)
        object.__setattr__(self, "reverse_channel_name", _REVERSE_CHANNEL_NAME)
        object.__setattr__(self, "supports_active_inhibition_claim", False)
        object.__setattr__(self, "combination_rule", _COMBINATION_RULE)
        object.__setattr__(self, "pair_spec_id", pair_spec_id)

    def _require_intact(self) -> None:
        try:
            repeated = DirectionalContrastPairSpec(
                forward_contrast=self.forward_contrast,
                reverse_contrast=self.reverse_contrast,
                schema_version=self.schema_version,
            )
            valid = (
                self.forward_contrast_id == repeated.forward_contrast_id
                and self.reverse_contrast_id == repeated.reverse_contrast_id
                and self.forward_channel_name == repeated.forward_channel_name
                and self.reverse_channel_name == repeated.reverse_channel_name
                and self.supports_active_inhibition_claim
                == repeated.supports_active_inhibition_claim
                and self.combination_rule == repeated.combination_rule
                and self.pair_spec_id == repeated.pair_spec_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Directional contrast pair specification failed integrity validation",
                code="directional_contrast_pair_spec_integrity_violation",
                field="pair_spec_id",
                remediation="Recreate the explicit directional contrast pair",
            ) from error
        if not valid:
            raise ContractError(
                "Directional contrast pair specification failed integrity validation",
                code="directional_contrast_pair_spec_integrity_violation",
                field="pair_spec_id",
                remediation="Recreate the explicit directional contrast pair",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "pair_spec_id": self.pair_spec_id,
            "schema_version": self.schema_version,
            "forward_contrast_id": self.forward_contrast_id,
            "reverse_contrast_id": self.reverse_contrast_id,
            "forward_contrast": self.forward_contrast.to_dict(),
            "reverse_contrast": self.reverse_contrast.to_dict(),
            "forward_channel_name": self.forward_channel_name,
            "reverse_channel_name": self.reverse_channel_name,
            "supports_active_inhibition_claim": (
                self.supports_active_inhibition_claim
            ),
            "combination_rule": self.combination_rule,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectionalResponsePair:
    """Two independent positive channels for one signed activation contrast.

    The reverse channel represents reduced activation under an explicitly
    registered reverse contrast. It never authorizes an active-inhibition claim.
    """

    feature_ids: tuple[str, ...]
    forward_contrast: ContrastSpec
    reverse_contrast: ContrastSpec
    forward_response_id: str
    reverse_response_id: str
    forward_signed_response: np.ndarray
    reverse_signed_response: np.ndarray
    prior_direction: int = 1
    forward_solver_response: np.ndarray = field(init=False, repr=False)
    reverse_solver_response: np.ndarray = field(init=False, repr=False)
    forward_contrast_id: str = field(init=False)
    reverse_contrast_id: str = field(init=False)
    forward_channel_id: str = field(init=False)
    reverse_channel_id: str = field(init=False)
    response_pair_id: str = field(init=False)

    def __post_init__(self) -> None:
        features = _feature_ids(self.feature_ids)
        _validate_reverse_contrast(self.forward_contrast, self.reverse_contrast)
        forward_response_id = _identifier(
            self.forward_response_id,
            field_name="forward_response_id",
        )
        reverse_response_id = _identifier(
            self.reverse_response_id,
            field_name="reverse_response_id",
        )
        if forward_response_id == reverse_response_id:
            raise ContractError(
                "Forward and reverse channels require distinct response provenance",
                code="invalid_directional_response_provenance",
                field="reverse_response_id",
                remediation="Fit and register the explicit reverse response artifact",
            )
        if (
            isinstance(self.prior_direction, (bool, np.bool_))
            or not isinstance(self.prior_direction, (int, np.integer))
            or self.prior_direction != 1
        ):
            raise ContractError(
                "This response pair accepts only a non-negative activation prior",
                code="unsupported_signed_prior_semantics",
                field="prior_direction",
                remediation=(
                    "Use prior direction +1 or define a separately reviewed "
                    "signed prior"
                ),
            )
        forward = _finite_aligned_vector(
            self.forward_signed_response,
            length=len(features),
            field_name="forward_signed_response",
        )
        reverse = _finite_aligned_vector(
            self.reverse_signed_response,
            length=len(features),
            field_name="reverse_signed_response",
        )
        if not np.allclose(
            reverse,
            -forward,
            rtol=0.0,
            atol=_ALIGNMENT_ATOL,
        ):
            raise ContractError(
                "Reverse response must be the aligned effect of the reverse contrast",
                code="invalid_directional_response_pair",
                field="reverse_signed_response",
                remediation=(
                    "Preserve feature order and reverse the exact registered contrast"
                ),
            )
        forward_solver = _immutable_vector(np.maximum(forward, 0.0))
        reverse_solver = _immutable_vector(np.maximum(reverse, 0.0))
        forward_contrast_id = _contrast_id(self.forward_contrast)
        reverse_contrast_id = _contrast_id(self.reverse_contrast)
        forward_channel_id = stable_id(
            "directional_response_channel",
            {
                "biological_claim": _FORWARD_CLAIM,
                "contrast_id": forward_contrast_id,
                "feature_ids": list(features),
                "response_id": forward_response_id,
                "signed_response": forward.tolist(),
                "solver_response": forward_solver.tolist(),
            },
            schema_version=_PAIR_SCHEMA_VERSION,
        )
        reverse_channel_id = stable_id(
            "directional_response_channel",
            {
                "biological_claim": _REVERSE_CLAIM,
                "contrast_id": reverse_contrast_id,
                "feature_ids": list(features),
                "response_id": reverse_response_id,
                "signed_response": reverse.tolist(),
                "solver_response": reverse_solver.tolist(),
            },
            schema_version=_PAIR_SCHEMA_VERSION,
        )
        response_pair_id = stable_id(
            "directional_response_pair",
            {
                "combination_rule": _COMBINATION_RULE,
                "forward_channel_id": forward_channel_id,
                "prior_semantics": _PRIOR_SEMANTICS,
                "reverse_channel_id": reverse_channel_id,
            },
            schema_version=_PAIR_SCHEMA_VERSION,
        )
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "forward_response_id", forward_response_id)
        object.__setattr__(self, "reverse_response_id", reverse_response_id)
        object.__setattr__(self, "forward_signed_response", forward)
        object.__setattr__(self, "reverse_signed_response", reverse)
        object.__setattr__(self, "prior_direction", 1)
        object.__setattr__(self, "forward_solver_response", forward_solver)
        object.__setattr__(self, "reverse_solver_response", reverse_solver)
        object.__setattr__(self, "forward_contrast_id", forward_contrast_id)
        object.__setattr__(self, "reverse_contrast_id", reverse_contrast_id)
        object.__setattr__(self, "forward_channel_id", forward_channel_id)
        object.__setattr__(self, "reverse_channel_id", reverse_channel_id)
        object.__setattr__(self, "response_pair_id", response_pair_id)

    @property
    def forward_biological_claim(self) -> str:
        return _FORWARD_CLAIM

    @property
    def reverse_biological_claim(self) -> str:
        return _REVERSE_CLAIM

    @property
    def supports_active_inhibition_claim(self) -> bool:
        return False

    def reconstruct(
        self,
        forward_prediction: np.ndarray,
        reverse_prediction: np.ndarray,
    ) -> DirectionalPairReconstruction:
        """Reconstruct each contrast independently with a complete signed residual."""

        forward = _finite_aligned_vector(
            forward_prediction,
            length=len(self.feature_ids),
            field_name="forward_prediction",
        )
        reverse = _finite_aligned_vector(
            reverse_prediction,
            length=len(self.feature_ids),
            field_name="reverse_prediction",
        )
        if np.any(forward < 0) or np.any(reverse < 0):
            raise ContractError(
                "Directional channel predictions must be non-negative",
                code="invalid_directional_prediction",
                field="forward_prediction",
                remediation="Use a non-negative basis and non-negative coefficients",
            )
        return DirectionalPairReconstruction(
            response_pair_id=self.response_pair_id,
            forward_channel_id=self.forward_channel_id,
            reverse_channel_id=self.reverse_channel_id,
            feature_ids=self.feature_ids,
            observed_forward_response=self.forward_signed_response,
            observed_reverse_response=self.reverse_signed_response,
            forward_prediction=forward,
            reverse_prediction=reverse,
            forward_signed_residual=self.forward_signed_response - forward,
            reverse_signed_residual=self.reverse_signed_response - reverse,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "response_pair_id": self.response_pair_id,
            "forward_channel_id": self.forward_channel_id,
            "reverse_channel_id": self.reverse_channel_id,
            "forward_contrast_id": self.forward_contrast_id,
            "reverse_contrast_id": self.reverse_contrast_id,
            "forward_response_id": self.forward_response_id,
            "reverse_response_id": self.reverse_response_id,
            "feature_ids": list(self.feature_ids),
            "prior_semantics": _PRIOR_SEMANTICS,
            "forward_biological_claim": _FORWARD_CLAIM,
            "reverse_biological_claim": _REVERSE_CLAIM,
            "supports_active_inhibition_claim": False,
            "combination_rule": _COMBINATION_RULE,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectionalPairReconstruction:
    """Two non-additive channel reconstructions of the same signed effect."""

    response_pair_id: str
    forward_channel_id: str
    reverse_channel_id: str
    feature_ids: tuple[str, ...]
    observed_forward_response: np.ndarray
    observed_reverse_response: np.ndarray
    forward_prediction: np.ndarray
    reverse_prediction: np.ndarray
    forward_signed_residual: np.ndarray
    reverse_signed_residual: np.ndarray
    reconstruction_id: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = {
            name: _identifier(getattr(self, name), field_name=name)
            for name in (
                "response_pair_id",
                "forward_channel_id",
                "reverse_channel_id",
            )
        }
        if identifiers["forward_channel_id"] == identifiers["reverse_channel_id"]:
            raise ContractError(
                "Directional reconstructions require distinct channel identities",
                code="invalid_directional_reconstruction",
                field="reverse_channel_id",
                remediation="Retain both channels from the paired response contract",
            )
        features = _feature_ids(self.feature_ids)
        vectors = {
            name: _finite_aligned_vector(
                getattr(self, name),
                length=len(features),
                field_name=name,
            )
            for name in (
                "observed_forward_response",
                "observed_reverse_response",
                "forward_prediction",
                "reverse_prediction",
                "forward_signed_residual",
                "reverse_signed_residual",
            )
        }
        if np.any(vectors["forward_prediction"] < 0) or np.any(
            vectors["reverse_prediction"] < 0
        ):
            raise ContractError(
                "Directional reconstruction predictions must be non-negative",
                code="invalid_directional_reconstruction",
                field="forward_prediction",
                remediation="Retain the two non-negative solver predictions",
            )
        if not np.allclose(
            vectors["observed_reverse_response"],
            -vectors["observed_forward_response"],
            rtol=0.0,
            atol=_ALIGNMENT_ATOL,
        ):
            raise ContractError(
                "Directional reconstruction responses are not reverse contrasts",
                code="invalid_directional_reconstruction",
                field="observed_reverse_response",
                remediation="Reconstruct the exact paired response artifact",
            )
        if not np.allclose(
            vectors["forward_prediction"] + vectors["forward_signed_residual"],
            vectors["observed_forward_response"],
            rtol=0.0,
            atol=_ALIGNMENT_ATOL,
        ) or not np.allclose(
            vectors["reverse_prediction"] + vectors["reverse_signed_residual"],
            vectors["observed_reverse_response"],
            rtol=0.0,
            atol=_ALIGNMENT_ATOL,
        ):
            raise ContractError(
                "Prediction plus signed residual must reconstruct each response",
                code="invalid_directional_reconstruction",
                field="forward_signed_residual",
                remediation="Compute residuals against the complete signed response",
            )
        reconstruction_id = stable_id(
            "directional_pair_reconstruction",
            {
                "feature_ids": list(features),
                "forward_channel_id": identifiers["forward_channel_id"],
                "forward_prediction": vectors["forward_prediction"].tolist(),
                "forward_signed_residual": vectors["forward_signed_residual"].tolist(),
                "observed_forward_response": vectors[
                    "observed_forward_response"
                ].tolist(),
                "observed_reverse_response": vectors[
                    "observed_reverse_response"
                ].tolist(),
                "response_pair_id": identifiers["response_pair_id"],
                "reverse_channel_id": identifiers["reverse_channel_id"],
                "reverse_prediction": vectors["reverse_prediction"].tolist(),
                "reverse_signed_residual": vectors["reverse_signed_residual"].tolist(),
            },
            schema_version=_RECONSTRUCTION_SCHEMA_VERSION,
        )
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "feature_ids", features)
        for name, vector in vectors.items():
            object.__setattr__(self, name, vector)
        object.__setattr__(self, "reconstruction_id", reconstruction_id)

    @property
    def reverse_prediction_on_forward_scale(self) -> np.ndarray:
        """Map reverse-channel attenuation back to the forward signed scale."""

        return _immutable_vector(-self.reverse_prediction)

    @property
    def reverse_residual_on_forward_scale(self) -> np.ndarray:
        return _immutable_vector(-self.reverse_signed_residual)

    @property
    def forward_reconstructed_response(self) -> np.ndarray:
        return _immutable_vector(self.forward_prediction + self.forward_signed_residual)

    @property
    def reverse_reconstructed_on_forward_scale(self) -> np.ndarray:
        return _immutable_vector(
            -(self.reverse_prediction + self.reverse_signed_residual)
        )

    @property
    def supports_active_inhibition_claim(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "reconstruction_id": self.reconstruction_id,
            "response_pair_id": self.response_pair_id,
            "forward_channel_id": self.forward_channel_id,
            "reverse_channel_id": self.reverse_channel_id,
            "feature_ids": list(self.feature_ids),
            "reverse_biological_claim": _REVERSE_CLAIM,
            "supports_active_inhibition_claim": False,
            "combination_rule": _COMBINATION_RULE,
        }


def build_directional_response_pair(
    feature_ids: Sequence[str],
    forward_contrast: ContrastSpec,
    reverse_contrast: ContrastSpec,
    forward_signed_response: np.ndarray,
    reverse_signed_response: np.ndarray,
    *,
    forward_response_id: str,
    reverse_response_id: str,
    prior_direction: int = 1,
) -> DirectionalResponsePair:
    """Build reviewed forward/reverse positive channels with strict provenance."""

    features = _feature_ids(feature_ids)
    return DirectionalResponsePair(
        feature_ids=features,
        forward_contrast=forward_contrast,
        reverse_contrast=reverse_contrast,
        forward_response_id=forward_response_id,
        reverse_response_id=reverse_response_id,
        forward_signed_response=forward_signed_response,
        reverse_signed_response=reverse_signed_response,
        prior_direction=prior_direction,
    )


__all__ = [
    "DirectionalContrastPairSpec",
    "DirectionalPairReconstruction",
    "DirectionalResponsePair",
    "build_directional_response_pair",
]
