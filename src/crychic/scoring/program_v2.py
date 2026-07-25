"""Fold-fitted signed sample-level receiver mechanism programs for M1."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from crychic.core import canonical_digest, stable_id
from crychic.pseudobulk import PseudobulkDataset
from crychic.resources import ResourceBundle, TargetPrior

from .absolute_v2 import AbsoluteActivityV2Transform
from .sample_edge_v2 import SampleEdgeScoreV2

SIGNED_PROGRAM_V2_VERSION = "fold_residualized_signed_program_m1_v2"
_SCHEMA_VERSION = "2.0.0"
_PROGRAM_TABLE_COLUMNS = (
    "sample_id",
    "subject_id",
    "receiver",
    "interaction_id",
    "program_unaligned_raw",
    "program_direction",
    "program_signed",
    "program_status",
    "program_reason_code",
)


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _names(
    values: Sequence[str], *, field_name: str, allow_empty: bool = True
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(sorted(_name(value, field_name=field_name) for value in values))
    if (not allow_empty and not result) or len(result) != len(set(result)):
        raise ValueError(f"{field_name} must be unique and have valid cardinality")
    return result


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _immutable_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).copy(order="C")
    if np.any(~np.isfinite(result)):
        raise ValueError("signed-program arrays must be finite")
    result.setflags(write=False)
    return result


def _weighted_targets(
    values: Sequence[tuple[str, float]], *, field_name: str
) -> tuple[tuple[str, float], ...]:
    supplied = tuple(values)
    result = tuple(
        sorted(
            (
                _name(gene, field_name=f"{field_name}_gene"),
                _finite(weight, field_name=f"{field_name}_weight"),
            )
            for gene, weight in supplied
        )
    )
    if len({gene for gene, _ in result}) != len(result) or any(
        weight <= 0.0 for _, weight in result
    ):
        raise ValueError(f"{field_name} must contain unique positive weights")
    total = sum(weight for _, weight in result)
    return tuple((gene, weight / total) for gene, weight in result) if result else ()


class SignedMechanismDirection(StrEnum):
    """How the unaligned target program maps to mechanism-consistent response."""

    ACTIVATION = "activation"
    ATTENUATION = "attenuation"
    UNKNOWN = "unknown"

    @property
    def sign(self) -> int | None:
        if self is SignedMechanismDirection.ACTIVATION:
            return 1
        if self is SignedMechanismDirection.ATTENUATION:
            return -1
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedProgramDefinitionV2:
    """Prior-only positive/negative target definition for one LR interaction."""

    interaction_id: str
    positive_targets: tuple[tuple[str, float], ...] = ()
    negative_targets: tuple[tuple[str, float], ...] = ()
    mechanism_direction: SignedMechanismDirection | str = (
        SignedMechanismDirection.UNKNOWN
    )
    source_id: str = "user_declared"
    definition_id: str = field(init=False)

    def __post_init__(self) -> None:
        interaction = _name(self.interaction_id, field_name="interaction_id")
        positive = _weighted_targets(
            self.positive_targets, field_name="positive_targets"
        )
        negative = _weighted_targets(
            self.negative_targets, field_name="negative_targets"
        )
        if not positive and not negative:
            raise ValueError("signed program definition requires at least one target")
        if set(gene for gene, _ in positive).intersection(gene for gene, _ in negative):
            raise ValueError("positive and negative target sets must be disjoint")
        direction = SignedMechanismDirection(self.mechanism_direction)
        source = _name(self.source_id, field_name="source_id")
        object.__setattr__(self, "interaction_id", interaction)
        object.__setattr__(self, "positive_targets", positive)
        object.__setattr__(self, "negative_targets", negative)
        object.__setattr__(self, "mechanism_direction", direction)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(
            self,
            "definition_id",
            stable_id(
                "signed_program_definition_v2",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *(gene for gene, _ in self.positive_targets),
                    *(gene for gene, _ in self.negative_targets),
                }
            )
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "interaction_id": self.interaction_id,
            "mechanism_direction": SignedMechanismDirection(
                self.mechanism_direction
            ).value,
            "negative_targets": [list(item) for item in self.negative_targets],
            "positive_targets": [list(item) for item in self.positive_targets],
            "source_id": self.source_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {"definition_id": self.definition_id, **self._identity_payload()}


def signed_program_definitions_from_resources_v2(
    bundle: ResourceBundle,
    target_prior: TargetPrior,
) -> tuple[SignedProgramDefinitionV2, ...]:
    """Map a frozen TargetPrior to interaction programs without expression input."""

    if not isinstance(bundle, ResourceBundle):
        raise TypeError("bundle must be ResourceBundle")
    if not isinstance(target_prior, TargetPrior):
        raise TypeError("target_prior must be TargetPrior")
    direction = (
        SignedMechanismDirection.ACTIVATION
        if target_prior.direction == 1
        else SignedMechanismDirection.ATTENUATION
    )
    known_drivers = set(target_prior.driver_ids)
    definitions: list[SignedProgramDefinitionV2] = []
    for interaction in bundle.interactions:
        candidates = (
            {interaction.interaction_id}
            if target_prior.driver_kind == "interaction"
            else {
                interaction.ligand_name,
                *(
                    interaction.ligand_subunits
                    if len(interaction.ligand_subunits) == 1
                    else ()
                ),
            }
        )
        matches = tuple(sorted(candidates.intersection(known_drivers)))
        if len(matches) != 1:
            continue
        links = tuple(
            link
            for link in target_prior.links_for_driver(matches[0])
            if link.weight > 0.0
        )
        if not links:
            continue
        definitions.append(
            SignedProgramDefinitionV2(
                interaction_id=interaction.interaction_id,
                positive_targets=tuple((link.target, link.weight) for link in links),
                mechanism_direction=direction,
                source_id=(
                    f"{target_prior.resource_id}:{target_prior.version}:"
                    f"{target_prior.manifest_digest}"
                ),
            )
        )
    return tuple(sorted(definitions, key=lambda item: item.interaction_id))


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedProgramV2Spec:
    """Pre-registered nuisance and target-coverage policy for M1."""

    continuous_covariates: tuple[str, ...] = ()
    categorical_covariates: tuple[str, ...] = ()
    generic_state_feature_ids: tuple[str, ...] = ()
    mechanism_direction_overrides: tuple[
        tuple[str, SignedMechanismDirection | str], ...
    ] = ()
    minimum_training_samples: int = 6
    minimum_training_subjects: int = 4
    minimum_matched_targets: int = 1
    nuisance_scale_floor: float = 1.0e-6
    maximum_condition_number: float = 1.0e8
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        continuous = _names(
            self.continuous_covariates, field_name="continuous_covariates"
        )
        categorical = _names(
            self.categorical_covariates, field_name="categorical_covariates"
        )
        generic = _names(
            self.generic_state_feature_ids,
            field_name="generic_state_feature_ids",
        )
        supplied_overrides = tuple(self.mechanism_direction_overrides)
        overrides = tuple(
            sorted(
                (
                    _name(interaction_id, field_name="override_interaction_id"),
                    SignedMechanismDirection(direction),
                )
                for interaction_id, direction in supplied_overrides
            )
        )
        if len({interaction_id for interaction_id, _ in overrides}) != len(overrides):
            raise ValueError("mechanism direction overrides must be interaction-unique")
        if set(continuous).intersection(categorical):
            raise ValueError("signed-program covariate roles overlap")
        for field_name, minimum in (
            ("minimum_training_samples", 4),
            ("minimum_training_subjects", 3),
            ("minimum_matched_targets", 1),
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field_name} must be an integer >= {minimum}")
        scale_floor = _finite(
            self.nuisance_scale_floor, field_name="nuisance_scale_floor"
        )
        condition = _finite(
            self.maximum_condition_number,
            field_name="maximum_condition_number",
        )
        if scale_floor <= 0.0 or condition <= 1.0:
            raise ValueError("signed-program numerical policy is invalid")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "continuous_covariates", continuous)
        object.__setattr__(self, "categorical_covariates", categorical)
        object.__setattr__(self, "generic_state_feature_ids", generic)
        object.__setattr__(self, "mechanism_direction_overrides", overrides)
        object.__setattr__(self, "nuisance_scale_floor", scale_floor)
        object.__setattr__(self, "maximum_condition_number", condition)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "signed_program_v2_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "categorical_covariates": list(self.categorical_covariates),
            "continuous_covariates": list(self.continuous_covariates),
            "generic_state_feature_ids": list(self.generic_state_feature_ids),
            "mechanism_direction_overrides": [
                [interaction_id, SignedMechanismDirection(direction).value]
                for interaction_id, direction in self.mechanism_direction_overrides
            ],
            "maximum_condition_number": self.maximum_condition_number,
            "minimum_matched_targets": self.minimum_matched_targets,
            "minimum_training_samples": self.minimum_training_samples,
            "minimum_training_subjects": self.minimum_training_subjects,
            "nuisance_scale_floor": self.nuisance_scale_floor,
            "schema_version": self.schema_version,
            "version": SIGNED_PROGRAM_V2_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {"spec_id": self.spec_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedProgramReceiverTransformV2:
    """One receiver's subject-equal nuisance regression on target z-scores."""

    receiver: str
    training_sample_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    target_feature_ids: tuple[str, ...]
    background_feature_ids: tuple[str, ...]
    numeric_encodings: tuple[tuple[str, float, float], ...]
    categorical_levels: tuple[tuple[str, tuple[str, ...]], ...]
    design_columns: tuple[str, ...]
    coefficients: npt.NDArray[np.float64]
    design_rank: int
    design_condition_number: float | None
    status: str
    reason_code: str | None
    transform_id: str = field(init=False)

    def __post_init__(self) -> None:
        receiver = _name(self.receiver, field_name="receiver")
        samples = _names(
            self.training_sample_ids,
            field_name="training_sample_ids",
            allow_empty=self.status != "observed",
        )
        subjects = _names(
            self.training_subject_ids,
            field_name="training_subject_ids",
            allow_empty=self.status != "observed",
        )
        targets = _names(
            self.target_feature_ids,
            field_name="target_feature_ids",
            allow_empty=False,
        )
        background = _names(
            self.background_feature_ids,
            field_name="background_feature_ids",
        )
        encodings = tuple(
            (
                _name(name, field_name="numeric_encoding_name"),
                _finite(center, field_name="numeric_encoding_center"),
                _finite(scale, field_name="numeric_encoding_scale"),
            )
            for name, center, scale in self.numeric_encodings
        )
        if len({name for name, _, _ in encodings}) != len(encodings) or any(
            scale <= 0.0 for _, _, scale in encodings
        ):
            raise ValueError("numeric encodings must be unique with positive scale")
        categories = tuple(
            (
                _name(name, field_name="categorical_covariate"),
                _names(levels, field_name="categorical_levels", allow_empty=False),
            )
            for name, levels in self.categorical_levels
        )
        if len({name for name, _ in categories}) != len(categories):
            raise ValueError("categorical encoding columns must be unique")
        design_columns = tuple(
            _name(value, field_name="design_columns") for value in self.design_columns
        )
        if len(design_columns) != len(set(design_columns)):
            raise ValueError("design_columns must be unique")
        coefficients = _immutable_array(self.coefficients)
        if coefficients.shape != (len(design_columns), len(targets)):
            raise ValueError("signed-program coefficient shape is invalid")
        if (
            isinstance(self.design_rank, bool)
            or not isinstance(self.design_rank, int)
            or self.design_rank < 0
        ):
            raise ValueError("design_rank must be a non-negative integer")
        condition = self.design_condition_number
        if condition is not None:
            condition = _finite(condition, field_name="design_condition_number")
            if condition < 1.0:
                raise ValueError("design_condition_number must be >= 1")
        if self.status == "observed":
            if (
                self.reason_code is not None
                or not samples
                or not subjects
                or not design_columns
                or self.design_rank != len(design_columns)
                or condition is None
            ):
                raise ValueError("observed signed-program transform is incomplete")
        elif self.status != "not_estimable" or self.reason_code is None:
            raise ValueError("receiver transform status is invalid")
        object.__setattr__(self, "receiver", receiver)
        object.__setattr__(self, "training_sample_ids", samples)
        object.__setattr__(self, "training_subject_ids", subjects)
        object.__setattr__(self, "target_feature_ids", targets)
        object.__setattr__(self, "background_feature_ids", background)
        object.__setattr__(self, "numeric_encodings", encodings)
        object.__setattr__(self, "categorical_levels", categories)
        object.__setattr__(self, "design_columns", design_columns)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "design_condition_number", condition)
        object.__setattr__(
            self,
            "transform_id",
            stable_id(
                "signed_program_receiver_transform_v2",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "background_feature_ids": list(self.background_feature_ids),
            "categorical_levels": [
                [name, list(levels)] for name, levels in self.categorical_levels
            ],
            "coefficients_digest": canonical_digest(self.coefficients.tolist()),
            "design_columns": list(self.design_columns),
            "design_condition_number": self.design_condition_number,
            "design_rank": self.design_rank,
            "numeric_encodings": [list(item) for item in self.numeric_encodings],
            "reason_code": self.reason_code,
            "receiver": self.receiver,
            "status": self.status,
            "target_feature_ids": list(self.target_feature_ids),
            "training_sample_ids": list(self.training_sample_ids),
            "training_subject_ids": list(self.training_subject_ids),
        }

    def to_dict(self) -> dict[str, object]:
        return {"transform_id": self.transform_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedProgramFunctionalV2:
    """Fold-training signed-program definitions and receiver nuisance transforms."""

    fold_id: str
    training_subject_ids: tuple[str, ...]
    training_input_digest: str
    activity_transform_id: str
    definitions: tuple[SignedProgramDefinitionV2, ...]
    receiver_transforms: tuple[SignedProgramReceiverTransformV2, ...]
    matched_target_feature_ids: tuple[str, ...]
    generic_state_feature_ids: tuple[str, ...]
    spec: SignedProgramV2Spec
    outcome_agnostic: bool = True
    formal_inference_allowed: bool = False
    functional_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in ("fold_id", "training_input_digest", "activity_transform_id"):
            object.__setattr__(
                self,
                field_name,
                _name(getattr(self, field_name), field_name=field_name),
            )
        subjects = _names(
            self.training_subject_ids,
            field_name="training_subject_ids",
            allow_empty=False,
        )
        supplied_definitions = tuple(self.definitions)
        supplied_transforms = tuple(self.receiver_transforms)
        if not supplied_definitions or any(
            not isinstance(item, SignedProgramDefinitionV2)
            for item in supplied_definitions
        ):
            raise TypeError("definitions must contain signed program definitions")
        if not supplied_transforms or any(
            not isinstance(item, SignedProgramReceiverTransformV2)
            for item in supplied_transforms
        ):
            raise TypeError("receiver_transforms must contain receiver transforms")
        definitions = tuple(
            sorted(supplied_definitions, key=lambda item: item.interaction_id)
        )
        transforms = tuple(sorted(supplied_transforms, key=lambda item: item.receiver))
        if len({item.interaction_id for item in definitions}) != len(definitions):
            raise ValueError("signed program definitions must be interaction-unique")
        if len({item.receiver for item in transforms}) != len(transforms):
            raise ValueError("receiver transforms must be receiver-unique")
        matched = _names(
            self.matched_target_feature_ids,
            field_name="matched_target_feature_ids",
            allow_empty=False,
        )
        generic = _names(
            self.generic_state_feature_ids,
            field_name="generic_state_feature_ids",
        )
        if not isinstance(self.spec, SignedProgramV2Spec):
            raise TypeError("spec must be SignedProgramV2Spec")
        if generic != self.spec.generic_state_feature_ids:
            raise ValueError("generic state features do not match signed-program spec")
        if any(item.target_feature_ids != matched for item in transforms):
            raise ValueError("receiver transforms do not share the matched target axis")
        if self.outcome_agnostic is not True:
            raise ValueError("signed-program functional must remain outcome-agnostic")
        if self.formal_inference_allowed is not False:
            raise ValueError("signed-program head cannot claim formal inference")
        object.__setattr__(self, "training_subject_ids", subjects)
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "receiver_transforms", transforms)
        object.__setattr__(self, "matched_target_feature_ids", matched)
        object.__setattr__(self, "generic_state_feature_ids", generic)
        object.__setattr__(
            self,
            "functional_id",
            stable_id(
                "signed_program_functional_v2",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "activity_transform_id": self.activity_transform_id,
            "definitions": [item.to_dict() for item in self.definitions],
            "fold_id": self.fold_id,
            "formal_inference_allowed": self.formal_inference_allowed,
            "generic_state_feature_ids": list(self.generic_state_feature_ids),
            "matched_target_feature_ids": list(self.matched_target_feature_ids),
            "outcome_agnostic": self.outcome_agnostic,
            "receiver_transforms": [
                item.to_dict() for item in self.receiver_transforms
            ],
            "spec_id": self.spec.spec_id,
            "training_input_digest": self.training_input_digest,
            "training_subject_ids": list(self.training_subject_ids),
            "version": SIGNED_PROGRAM_V2_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "functional_id": self.functional_id,
            **self._identity_payload(),
            "spec": self.spec.to_dict(),
        }


def _matrix_metadata(aggregate: PseudobulkDataset) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        aggregate.unit_metadata.set_index("unit_id", drop=False)
        .loc[list(aggregate.matrix_unit_ids)]
        .reset_index(drop=True),
    )


def _receiver_z_expression(
    activity_transform: AbsoluteActivityV2Transform,
    aggregate: PseudobulkDataset,
    receiver: str,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    metadata = _matrix_metadata(aggregate)
    selected = metadata["cell_type"].astype(str).eq(receiver)
    selected &= metadata["state_eligible"].astype(bool)
    rows = np.flatnonzero(selected.to_numpy())
    if not len(rows):
        return metadata.iloc[[]].copy(), np.empty(
            (0, len(activity_transform.feature_ids))
        )
    aggregate_index = {name: index for index, name in enumerate(aggregate.feature_ids)}
    columns = np.asarray(
        [aggregate_index[name] for name in activity_transform.feature_ids], dtype=int
    )
    counts = aggregate.counts[rows]
    libraries = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    usable = libraries > 0.0
    rows = rows[usable]
    libraries = libraries[usable]
    if not len(rows):
        return metadata.iloc[[]].copy(), np.empty(
            (0, len(activity_transform.feature_ids))
        )
    counts = aggregate.counts[rows][:, columns].toarray().astype(float)
    receiver_index = activity_transform.cell_type_ids.index(receiver)
    size_factor = libraries / float(
        activity_transform.median_library_sizes[receiver_index]
    )
    expression = np.log2(counts / size_factor[:, np.newaxis] + 1.0)
    center = activity_transform.feature_medians[receiver_index]
    scale = activity_transform.feature_mads[receiver_index]
    z_expression = (expression - center[np.newaxis, :]) / scale[np.newaxis, :]
    return metadata.iloc[rows].reset_index(drop=True), z_expression


def _numeric_raw_states(
    metadata: pd.DataFrame,
    z_expression: np.ndarray,
    *,
    background_indices: np.ndarray,
    generic_indices: np.ndarray,
    spec: SignedProgramV2Spec,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        "log_cell_count": np.log1p(
            pd.to_numeric(metadata["n_cells"], errors="coerce").to_numpy(dtype=float)
        ),
        "global_receiver_state": (
            np.median(z_expression[:, background_indices], axis=1)
            if len(background_indices)
            else np.zeros(len(metadata), dtype=float)
        ),
    }
    if len(generic_indices):
        values["generic_receiver_state"] = np.mean(
            z_expression[:, generic_indices], axis=1
        )
    for column in spec.continuous_covariates:
        values[f"covariate:{column}"] = pd.to_numeric(
            metadata[column], errors="coerce"
        ).to_numpy(dtype=float)
    return values


def _fit_receiver_transform(
    receiver: str,
    metadata: pd.DataFrame,
    z_expression: np.ndarray,
    *,
    feature_ids: tuple[str, ...],
    target_feature_ids: tuple[str, ...],
    generic_feature_ids: tuple[str, ...],
    spec: SignedProgramV2Spec,
) -> SignedProgramReceiverTransformV2:
    feature_index = {name: index for index, name in enumerate(feature_ids)}
    target_indices = np.asarray(
        [feature_index[name] for name in target_feature_ids], dtype=int
    )
    generic_indices = np.asarray(
        [feature_index[name] for name in generic_feature_ids], dtype=int
    )
    target_set = set(target_feature_ids).union(generic_feature_ids)
    background_ids = tuple(name for name in feature_ids if name not in target_set)
    background_indices = np.asarray(
        [feature_index[name] for name in background_ids], dtype=int
    )
    if metadata.empty:
        return SignedProgramReceiverTransformV2(
            receiver=receiver,
            training_sample_ids=(),
            training_subject_ids=(),
            target_feature_ids=target_feature_ids,
            background_feature_ids=background_ids,
            numeric_encodings=(),
            categorical_levels=(),
            design_columns=(),
            coefficients=np.empty((0, len(target_feature_ids))),
            design_rank=0,
            design_condition_number=None,
            status="not_estimable",
            reason_code="receiver_training_expression_unavailable",
        )
    missing_covariates = set(
        (*spec.continuous_covariates, *spec.categorical_covariates)
    ).difference(metadata.columns)
    if missing_covariates:
        raise ValueError(
            f"receiver metadata lacks program covariates: {sorted(missing_covariates)}"
        )
    raw_numeric = _numeric_raw_states(
        metadata,
        z_expression,
        background_indices=background_indices,
        generic_indices=generic_indices,
        spec=spec,
    )
    complete: npt.NDArray[np.bool_] = np.ones(len(metadata), dtype=bool)
    for values in raw_numeric.values():
        complete &= np.isfinite(values)
    for column in spec.categorical_covariates:
        complete &= metadata[column].notna().to_numpy()
    selected_metadata = metadata.loc[complete].reset_index(drop=True)
    selected_z = z_expression[complete]
    if (
        len(selected_metadata) < spec.minimum_training_samples
        or selected_metadata["subject_id"].astype(str).nunique()
        < spec.minimum_training_subjects
    ):
        return SignedProgramReceiverTransformV2(
            receiver=receiver,
            training_sample_ids=tuple(
                sorted(selected_metadata["sample_id"].astype(str).unique())
            ),
            training_subject_ids=tuple(
                sorted(selected_metadata["subject_id"].astype(str).unique())
            ),
            target_feature_ids=target_feature_ids,
            background_feature_ids=background_ids,
            numeric_encodings=(),
            categorical_levels=(),
            design_columns=(),
            coefficients=np.empty((0, len(target_feature_ids))),
            design_rank=0,
            design_condition_number=None,
            status="not_estimable",
            reason_code="insufficient_receiver_training_replication",
        )
    parts: list[np.ndarray] = [np.ones((len(selected_metadata), 1))]
    design_columns = ["intercept"]
    numeric_encodings: list[tuple[str, float, float]] = []
    for name, all_values in raw_numeric.items():
        values = all_values[complete]
        center = float(values.mean())
        raw_scale = float(np.std(values, ddof=0))
        if raw_scale < spec.nuisance_scale_floor:
            continue
        scale = raw_scale
        parts.append(((values - center) / scale)[:, np.newaxis])
        design_columns.append(f"numeric:{name}")
        numeric_encodings.append((name, center, scale))
    categorical_levels: list[tuple[str, tuple[str, ...]]] = []
    for column in spec.categorical_covariates:
        categorical_values = selected_metadata[column].astype(str)
        levels = tuple(sorted(categorical_values.unique()))
        categorical_levels.append((column, levels))
        for level in levels[1:]:
            parts.append(
                categorical_values.eq(level).to_numpy(dtype=float)[:, np.newaxis]
            )
            design_columns.append(f"categorical:{column}={level}")
    design = np.hstack(parts)
    subject_counts = selected_metadata["subject_id"].astype(str).value_counts()
    weights = (
        selected_metadata["subject_id"]
        .astype(str)
        .map(lambda subject: 1.0 / float(subject_counts[subject]))
        .to_numpy(dtype=float)
    )
    weighted_design = np.sqrt(weights)[:, np.newaxis] * design
    rank = int(np.linalg.matrix_rank(weighted_design))
    condition = float(np.linalg.cond(weighted_design))
    if (
        rank != design.shape[1]
        or not math.isfinite(condition)
        or condition > spec.maximum_condition_number
        or len(selected_metadata) <= design.shape[1]
    ):
        return SignedProgramReceiverTransformV2(
            receiver=receiver,
            training_sample_ids=tuple(
                sorted(selected_metadata["sample_id"].astype(str).unique())
            ),
            training_subject_ids=tuple(
                sorted(selected_metadata["subject_id"].astype(str).unique())
            ),
            target_feature_ids=target_feature_ids,
            background_feature_ids=background_ids,
            numeric_encodings=tuple(numeric_encodings),
            categorical_levels=tuple(categorical_levels),
            design_columns=tuple(design_columns),
            coefficients=np.zeros((len(design_columns), len(target_feature_ids))),
            design_rank=rank,
            design_condition_number=(condition if math.isfinite(condition) else None),
            status="not_estimable",
            reason_code="rank_deficient_or_ill_conditioned_program_nuisance",
        )
    target_values = selected_z[:, target_indices]
    weighted_target = np.sqrt(weights)[:, np.newaxis] * target_values
    coefficients = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
    return SignedProgramReceiverTransformV2(
        receiver=receiver,
        training_sample_ids=tuple(
            sorted(selected_metadata["sample_id"].astype(str).unique())
        ),
        training_subject_ids=tuple(
            sorted(selected_metadata["subject_id"].astype(str).unique())
        ),
        target_feature_ids=target_feature_ids,
        background_feature_ids=background_ids,
        numeric_encodings=tuple(numeric_encodings),
        categorical_levels=tuple(categorical_levels),
        design_columns=tuple(design_columns),
        coefficients=coefficients,
        design_rank=rank,
        design_condition_number=condition,
        status="observed",
        reason_code=None,
    )


def fit_signed_program_v2(
    activity_transform: AbsoluteActivityV2Transform,
    training_aggregate: PseudobulkDataset,
    definitions: Sequence[SignedProgramDefinitionV2],
    *,
    training_input_digest: str,
    spec: SignedProgramV2Spec | None = None,
) -> SignedProgramFunctionalV2:
    """Fit receiver nuisance regressions without consuming condition labels."""

    if not isinstance(activity_transform, AbsoluteActivityV2Transform):
        raise TypeError("activity_transform must be AbsoluteActivityV2Transform")
    if not isinstance(training_aggregate, PseudobulkDataset):
        raise TypeError("training_aggregate must be PseudobulkDataset")
    resolved = spec or SignedProgramV2Spec()
    if not isinstance(resolved, SignedProgramV2Spec):
        raise TypeError("spec must be SignedProgramV2Spec or None")
    supplied_definitions = tuple(definitions)
    if not supplied_definitions or any(
        not isinstance(item, SignedProgramDefinitionV2) for item in supplied_definitions
    ):
        raise TypeError("definitions must contain SignedProgramDefinitionV2 values")
    canonical_definitions = tuple(
        sorted(supplied_definitions, key=lambda item: item.interaction_id)
    )
    if len({item.interaction_id for item in canonical_definitions}) != len(
        canonical_definitions
    ):
        raise ValueError("signed program definitions must be interaction-unique")
    aggregate_subjects = tuple(
        sorted(training_aggregate.unit_metadata["subject_id"].astype(str).unique())
    )
    if aggregate_subjects != activity_transform.training_subject_ids:
        raise ValueError("program training subjects differ from M0 activity transform")
    if not set(activity_transform.feature_ids).issubset(training_aggregate.feature_ids):
        raise ValueError("program training lacks M0 transform features")
    all_targets = tuple(
        sorted(
            {
                gene
                for definition in canonical_definitions
                for gene in definition.target_ids
            }
        )
    )
    matched_targets = tuple(
        gene for gene in all_targets if gene in activity_transform.feature_ids
    )
    if not matched_targets:
        raise ValueError("no signed-program target is present on the M0 feature axis")
    missing_generic = set(resolved.generic_state_feature_ids).difference(
        activity_transform.feature_ids
    )
    if missing_generic:
        raise ValueError(
            f"generic state features are absent from M0: {sorted(missing_generic)}"
        )
    overlapping_generic = set(resolved.generic_state_feature_ids).intersection(
        matched_targets
    )
    if overlapping_generic:
        raise ValueError(
            "generic state features cannot overlap mechanism targets: "
            f"{sorted(overlapping_generic)}"
        )
    receiver_transforms = []
    for receiver in activity_transform.cell_type_ids:
        metadata, z_expression = _receiver_z_expression(
            activity_transform, training_aggregate, receiver
        )
        receiver_transforms.append(
            _fit_receiver_transform(
                receiver,
                metadata,
                z_expression,
                feature_ids=activity_transform.feature_ids,
                target_feature_ids=matched_targets,
                generic_feature_ids=resolved.generic_state_feature_ids,
                spec=resolved,
            )
        )
    return SignedProgramFunctionalV2(
        fold_id=activity_transform.fold_id,
        training_subject_ids=activity_transform.training_subject_ids,
        training_input_digest=_name(
            training_input_digest, field_name="training_input_digest"
        ),
        activity_transform_id=activity_transform.transform_manifest_id,
        definitions=canonical_definitions,
        receiver_transforms=tuple(receiver_transforms),
        matched_target_feature_ids=matched_targets,
        generic_state_feature_ids=resolved.generic_state_feature_ids,
        spec=resolved,
    )


def _application_design(
    receiver_transform: SignedProgramReceiverTransformV2,
    metadata: pd.DataFrame,
    z_expression: np.ndarray,
    *,
    feature_ids: tuple[str, ...],
    generic_feature_ids: tuple[str, ...],
    spec: SignedProgramV2Spec,
) -> tuple[np.ndarray, np.ndarray]:
    feature_index = {name: index for index, name in enumerate(feature_ids)}
    background_indices = np.asarray(
        [feature_index[name] for name in receiver_transform.background_feature_ids],
        dtype=int,
    )
    generic_indices = np.asarray(
        [feature_index[name] for name in generic_feature_ids], dtype=int
    )
    raw_numeric = _numeric_raw_states(
        metadata,
        z_expression,
        background_indices=background_indices,
        generic_indices=generic_indices,
        spec=spec,
    )
    valid: npt.NDArray[np.bool_] = np.ones(len(metadata), dtype=bool)
    parts: list[np.ndarray] = [np.ones((len(metadata), 1))]
    for name, center, scale in receiver_transform.numeric_encodings:
        values = raw_numeric[name]
        valid &= np.isfinite(values)
        parts.append(((values - center) / scale)[:, np.newaxis])
    for column, levels in receiver_transform.categorical_levels:
        categorical_values = metadata[column]
        valid &= categorical_values.notna().to_numpy()
        categorical_strings = categorical_values.astype(str)
        valid &= categorical_strings.isin(levels).to_numpy()
        for level in levels[1:]:
            parts.append(
                categorical_strings.eq(level).to_numpy(dtype=float)[:, np.newaxis]
            )
    return np.hstack(parts), valid


def _program_application_table(
    functional: SignedProgramFunctionalV2,
    activity_transform: AbsoluteActivityV2Transform,
    aggregate: PseudobulkDataset,
) -> pd.DataFrame:
    if functional.activity_transform_id != activity_transform.transform_manifest_id:
        raise ValueError("signed-program functional does not match activity transform")
    definitions = functional.definitions
    feature_index = {
        name: index for index, name in enumerate(activity_transform.feature_ids)
    }
    rows: list[dict[str, object]] = []
    for receiver_transform in functional.receiver_transforms:
        receiver = receiver_transform.receiver
        metadata, z_expression = _receiver_z_expression(
            activity_transform, aggregate, receiver
        )
        if metadata.empty:
            continue
        if receiver_transform.status == "observed":
            design, valid = _application_design(
                receiver_transform,
                metadata,
                z_expression,
                feature_ids=activity_transform.feature_ids,
                generic_feature_ids=functional.generic_state_feature_ids,
                spec=functional.spec,
            )
            target_indices = np.asarray(
                [feature_index[name] for name in receiver_transform.target_feature_ids],
                dtype=int,
            )
            residual = z_expression[:, target_indices] - (
                design @ receiver_transform.coefficients
            )
            residual_index = {
                name: index
                for index, name in enumerate(receiver_transform.target_feature_ids)
            }
        else:
            valid = np.zeros(len(metadata), dtype=bool)
            residual = np.empty((len(metadata), 0))
            residual_index = {}
        for definition in definitions:
            positive = tuple(
                (gene, weight)
                for gene, weight in definition.positive_targets
                if gene in residual_index
            )
            negative = tuple(
                (gene, weight)
                for gene, weight in definition.negative_targets
                if gene in residual_index
            )
            matched_count = len(positive) + len(negative)
            for row_index, sample in metadata.iterrows():
                raw: float | None
                signed: float | None
                reason: str | None
                status: str
                if receiver_transform.status != "observed":
                    raw = signed = None
                    status = "not_estimable"
                    reason = receiver_transform.reason_code
                elif not valid[row_index]:
                    raw = signed = None
                    status = "not_estimable"
                    reason = "heldout_program_nuisance_not_estimable"
                elif matched_count < functional.spec.minimum_matched_targets:
                    raw = signed = None
                    status = "not_estimable"
                    reason = "insufficient_matched_program_targets"
                else:
                    positive_total = sum(weight for _, weight in positive)
                    negative_total = sum(weight for _, weight in negative)
                    positive_score = (
                        sum(
                            weight * residual[row_index, residual_index[gene]]
                            for gene, weight in positive
                        )
                        / positive_total
                        if positive
                        else 0.0
                    )
                    negative_score = (
                        sum(
                            weight * residual[row_index, residual_index[gene]]
                            for gene, weight in negative
                        )
                        / negative_total
                        if negative
                        else 0.0
                    )
                    raw = float(positive_score - negative_score)
                    direction_sign = SignedMechanismDirection(
                        definition.mechanism_direction
                    ).sign
                    if direction_sign is None:
                        signed = None
                        status = "not_estimable"
                        reason = "unknown_mechanism_direction"
                    else:
                        signed = float(direction_sign * raw)
                        status = "observed"
                        reason = None
                rows.append(
                    {
                        "sample_id": str(sample["sample_id"]),
                        "subject_id": str(sample["subject_id"]),
                        "receiver": receiver,
                        "interaction_id": definition.interaction_id,
                        "program_unaligned_raw": raw,
                        "program_direction": SignedMechanismDirection(
                            definition.mechanism_direction
                        ).value,
                        "program_signed": signed,
                        "program_status": status,
                        "program_reason_code": reason,
                    }
                )
    result = pd.DataFrame(rows, columns=_PROGRAM_TABLE_COLUMNS)
    if result.duplicated(["sample_id", "receiver", "interaction_id"]).any():
        raise ValueError("signed-program application keys must be unique")
    return result.sort_values(
        ["sample_id", "receiver", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )


def build_signed_program_v2_training_table(
    functional: SignedProgramFunctionalV2,
    activity_transform: AbsoluteActivityV2Transform,
    training_aggregate: PseudobulkDataset,
) -> pd.DataFrame:
    """Apply the frozen program residualizer to training rows for M2 inputs."""

    if not isinstance(functional, SignedProgramFunctionalV2):
        raise TypeError("functional must be SignedProgramFunctionalV2")
    subjects = tuple(
        sorted(training_aggregate.unit_metadata["subject_id"].astype(str).unique())
    )
    if subjects != functional.training_subject_ids:
        raise ValueError("program training application subjects differ from functional")
    return _program_application_table(
        functional, activity_transform, training_aggregate
    )


def apply_signed_program_v2_to_sample_edges(
    functional: SignedProgramFunctionalV2,
    activity_transform: AbsoluteActivityV2Transform,
    heldout_aggregate: PseudobulkDataset,
    scores: SampleEdgeScoreV2,
) -> SampleEdgeScoreV2:
    """Attach parent-level raw/direction/signed M1 values to held-out v2 rows."""

    if not isinstance(functional, SignedProgramFunctionalV2):
        raise TypeError("functional must be SignedProgramFunctionalV2")
    if not isinstance(scores, SampleEdgeScoreV2):
        raise TypeError("scores must be SampleEdgeScoreV2")
    if (
        scores.provenance.fold_id != functional.fold_id
        or scores.provenance.training_subject_ids != functional.training_subject_ids
        or scores.provenance.training_input_digest != functional.training_input_digest
        or scores.provenance.transform_manifest_id != functional.activity_transform_id
    ):
        raise ValueError("sample-edge lineage does not match signed-program functional")
    if not scores.table["program_status"].eq("not_computed").all():
        raise ValueError("signed program refuses to overwrite an applied program head")
    program = _program_application_table(
        functional, activity_transform, heldout_aggregate
    )
    table = scores.table.copy(deep=True)
    program_columns = [
        "program_unaligned_raw",
        "program_direction",
        "program_signed",
        "program_status",
        "program_reason_code",
        "program_functional_id",
    ]
    table = table.drop(columns=program_columns)
    program["program_functional_id"] = functional.functional_id
    table["_row_order"] = np.arange(len(table))
    table = table.merge(
        program,
        on=["sample_id", "subject_id", "receiver", "interaction_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    missing = table["program_status"].isna()
    direction_by_interaction = {
        item.interaction_id: SignedMechanismDirection(item.mechanism_direction).value
        for item in functional.definitions
    }
    table.loc[missing, "program_unaligned_raw"] = np.nan
    table.loc[missing, "program_signed"] = np.nan
    table.loc[missing, "program_direction"] = (
        table.loc[missing, "interaction_id"]
        .map(direction_by_interaction)
        .fillna("unknown")
    )
    table.loc[missing, "program_status"] = "not_estimable"
    table.loc[missing, "program_reason_code"] = "receiver_program_not_measured"
    table.loc[missing, "program_functional_id"] = functional.functional_id
    table = table.sort_values("_row_order", kind="stable").drop(columns="_row_order")
    table = table.loc[:, scores.table.columns].reset_index(drop=True)
    return SampleEdgeScoreV2(table=table, provenance=scores.provenance)


__all__ = [
    "SIGNED_PROGRAM_V2_VERSION",
    "SignedMechanismDirection",
    "SignedProgramDefinitionV2",
    "SignedProgramFunctionalV2",
    "SignedProgramReceiverTransformV2",
    "SignedProgramV2Spec",
    "apply_signed_program_v2_to_sample_edges",
    "build_signed_program_v2_training_table",
    "fit_signed_program_v2",
    "signed_program_definitions_from_resources_v2",
]
