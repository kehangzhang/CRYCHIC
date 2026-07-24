"""Fold-fitted sample-comparable LR activity for the v7 M0 estimator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from crychic.availability import (
    BatchAvailability,
    FrozenInteractionUniverse,
    InteractionFilterApplication,
)
from crychic.core import canonical_digest, canonical_json, stable_id
from crychic.pseudobulk import PseudobulkDataset
from crychic.resources import Interaction, ResourceBundle

from .contracts import float64_array_digest
from .sample_edge_v2 import (
    SAMPLE_EDGE_SCORE_V2_COLUMNS,
    SampleEdgeScoreV2,
    SampleEdgeScoreV2Provenance,
)

ABSOLUTE_ACTIVITY_V2_SCORE_VERSION = "sample_comparable_log_geometric_activity_m0_v2"
ABSOLUTE_ACTIVITY_V2_TRANSFORM_VERSION = "fold_library_robust_expression_v2"
ABSOLUTE_ACTIVITY_V2_CALIBRATION_COLUMNS = (
    "sample_id",
    "subject_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
    "sender_detection_raw",
    "parent_peak_raw",
    "parent_total_raw",
    "parent_mean_raw",
    "status",
)

_PARENT_KEY = (
    "sample_id",
    "context_id",
    "receiver",
    "interaction_id",
)
_CANDIDATE_KEY = (
    "sample_id",
    "context_id",
    "sender",
    "receiver",
    "interaction_id",
)
_AVAILABILITY_COLUMNS = {
    *_CANDIDATE_KEY,
    "subject_id",
    "availability_state",
}


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(_name(value, field_name=field_name) for value in values))
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return normalized


def _immutable_float_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).copy(order="C")
    if np.any(~np.isfinite(result)):
        raise ValueError("absolute-activity transform arrays must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class AbsoluteActivityV2Spec:
    """Pre-registered transform and LR bottleneck policy."""

    complex_temperature: float = 0.25
    bottleneck_penalty: float = 0.0
    mad_scale: float = 1.4826
    robust_scale_floor: float = 1e-6
    schema_version: str = "2.0.0"
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        temperature = float(self.complex_temperature)
        penalty = float(self.bottleneck_penalty)
        mad_scale = float(self.mad_scale)
        scale_floor = float(self.robust_scale_floor)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("complex_temperature must be finite and positive")
        if penalty not in {0.0, 0.25}:
            raise ValueError("bottleneck_penalty must be one of 0 or 0.25")
        if not math.isfinite(mad_scale) or mad_scale <= 0.0:
            raise ValueError("mad_scale must be finite and positive")
        if not math.isfinite(scale_floor) or scale_floor <= 0.0:
            raise ValueError("robust_scale_floor must be finite and positive")
        if self.schema_version != "2.0.0":
            raise ValueError("AbsoluteActivityV2Spec schema_version must be 2.0.0")
        object.__setattr__(self, "complex_temperature", temperature)
        object.__setattr__(self, "bottleneck_penalty", penalty)
        object.__setattr__(self, "mad_scale", mad_scale)
        object.__setattr__(self, "robust_scale_floor", scale_floor)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "absolute_activity_v2_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "bottleneck_penalty": self.bottleneck_penalty,
            "complex_temperature": self.complex_temperature,
            "mad_scale": self.mad_scale,
            "robust_scale_floor": self.robust_scale_floor,
            "schema_version": self.schema_version,
            "score_version": ABSOLUTE_ACTIVITY_V2_SCORE_VERSION,
            "transform_version": ABSOLUTE_ACTIVITY_V2_TRANSFORM_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {"spec_id": self.spec_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenActivityInteractionV2:
    """Only the frozen molecular fields consumed by M0 v2."""

    interaction_id: str
    ligand_name: str
    receptor_name: str
    ligand_subunits: tuple[str, ...]
    receptor_subunits: tuple[str, ...]
    ligand_is_complex: bool
    receptor_is_complex: bool

    def __post_init__(self) -> None:
        for field_name in ("interaction_id", "ligand_name", "receptor_name"):
            object.__setattr__(
                self,
                field_name,
                _name(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "ligand_subunits",
            _names(self.ligand_subunits, field_name="ligand_subunits"),
        )
        object.__setattr__(
            self,
            "receptor_subunits",
            _names(self.receptor_subunits, field_name="receptor_subunits"),
        )
        if (
            type(self.ligand_is_complex) is not bool
            or type(self.receptor_is_complex) is not bool
        ):
            raise TypeError("interaction complex flags must be boolean")

    @classmethod
    def from_interaction(cls, interaction: Interaction) -> FrozenActivityInteractionV2:
        return cls(
            interaction_id=interaction.interaction_id,
            ligand_name=interaction.ligand_name,
            receptor_name=interaction.receptor_name,
            ligand_subunits=interaction.ligand_subunits,
            receptor_subunits=interaction.receptor_subunits,
            ligand_is_complex=interaction.ligand_is_complex,
            receptor_is_complex=interaction.receptor_is_complex,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "interaction_id": self.interaction_id,
            "ligand_name": self.ligand_name,
            "receptor_name": self.receptor_name,
            "ligand_subunits": list(self.ligand_subunits),
            "receptor_subunits": list(self.receptor_subunits),
            "ligand_is_complex": self.ligand_is_complex,
            "receptor_is_complex": self.receptor_is_complex,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AbsoluteActivityV2Transform:
    """Training-fold expression and reliability transform."""

    fold_id: str
    training_subject_ids: tuple[str, ...]
    training_input_digest: str
    resource_id: str
    resource_version: str
    resource_manifest_digest: str
    interaction_universe_id: str
    interactions: tuple[FrozenActivityInteractionV2, ...]
    cell_type_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    median_library_sizes: npt.NDArray[np.float64]
    cell_count_half_saturation: npt.NDArray[np.float64]
    feature_medians: npt.NDArray[np.float64]
    feature_mads: npt.NDArray[np.float64]
    spec: AbsoluteActivityV2Spec
    transform_manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "fold_id",
            "training_input_digest",
            "resource_id",
            "resource_version",
            "resource_manifest_digest",
            "interaction_universe_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _name(getattr(self, field_name), field_name=field_name),
            )
        subjects = _names(self.training_subject_ids, field_name="training_subject_ids")
        cell_types = _names(self.cell_type_ids, field_name="cell_type_ids")
        features = _names(self.feature_ids, field_name="feature_ids")
        interactions = tuple(
            sorted(tuple(self.interactions), key=lambda item: item.interaction_id)
        )
        if not interactions or any(
            not isinstance(item, FrozenActivityInteractionV2) for item in interactions
        ):
            raise TypeError("interactions must contain frozen activity interactions")
        if len({item.interaction_id for item in interactions}) != len(interactions):
            raise ValueError("activity interactions must be unique")
        if not isinstance(self.spec, AbsoluteActivityV2Spec):
            raise TypeError("spec must be AbsoluteActivityV2Spec")
        libraries = _immutable_float_array(self.median_library_sizes)
        half_saturation = _immutable_float_array(self.cell_count_half_saturation)
        medians = _immutable_float_array(self.feature_medians)
        mads = _immutable_float_array(self.feature_mads)
        expected_vector = (len(cell_types),)
        expected_matrix = (len(cell_types), len(features))
        if (
            libraries.shape != expected_vector
            or half_saturation.shape != expected_vector
        ):
            raise ValueError("cell-type transform vectors have incompatible shape")
        if medians.shape != expected_matrix or mads.shape != expected_matrix:
            raise ValueError("feature transform matrices have incompatible shape")
        if np.any(libraries <= 0.0) or np.any(half_saturation <= 0.0):
            raise ValueError("library and cell-count reference values must be positive")
        if np.any(mads < self.spec.robust_scale_floor):
            raise ValueError("feature MAD values must respect robust_scale_floor")
        object.__setattr__(self, "training_subject_ids", subjects)
        object.__setattr__(self, "cell_type_ids", cell_types)
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "interactions", interactions)
        object.__setattr__(self, "median_library_sizes", libraries)
        object.__setattr__(self, "cell_count_half_saturation", half_saturation)
        object.__setattr__(self, "feature_medians", medians)
        object.__setattr__(self, "feature_mads", mads)
        object.__setattr__(
            self,
            "transform_manifest_id",
            stable_id(
                "absolute_activity_v2_transform",
                self._identity_payload(),
                schema_version=self.spec.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm_version": ABSOLUTE_ACTIVITY_V2_TRANSFORM_VERSION,
            "cell_count_half_saturation_digest": float64_array_digest(
                self.cell_count_half_saturation
            ),
            "cell_type_ids": list(self.cell_type_ids),
            "feature_ids": list(self.feature_ids),
            "feature_mads_digest": float64_array_digest(self.feature_mads),
            "feature_medians_digest": float64_array_digest(self.feature_medians),
            "fold_id": self.fold_id,
            "interaction_digest": canonical_digest(
                [item.to_dict() for item in self.interactions]
            ),
            "interaction_universe_id": self.interaction_universe_id,
            "median_library_sizes_digest": float64_array_digest(
                self.median_library_sizes
            ),
            "resource_id": self.resource_id,
            "resource_manifest_digest": self.resource_manifest_digest,
            "resource_version": self.resource_version,
            "spec_id": self.spec.spec_id,
            "training_input_digest": self.training_input_digest,
            "training_subject_ids": list(self.training_subject_ids),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "transform_manifest_id": self.transform_manifest_id,
            **self._identity_payload(),
            "interactions": [item.to_dict() for item in self.interactions],
            "spec": self.spec.to_dict(),
        }


def _matrix_metadata(aggregate: PseudobulkDataset) -> pd.DataFrame:
    metadata = aggregate.unit_metadata.set_index("unit_id", drop=False)
    result = metadata.loc[list(aggregate.matrix_unit_ids)].reset_index(drop=True)
    return cast(pd.DataFrame, result)


def _require_resource_binding(
    bundle: ResourceBundle,
    universe: FrozenInteractionUniverse,
) -> tuple[FrozenActivityInteractionV2, ...]:
    if not isinstance(bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    if not isinstance(universe, FrozenInteractionUniverse):
        raise TypeError("interaction_universe must be FrozenInteractionUniverse")
    universe._require_intact()
    if (
        universe.resource_id != bundle.resource_id
        or universe.resource_version != bundle.version
        or universe.resource_manifest_digest != bundle.manifest_digest
    ):
        raise ValueError("frozen interaction universe does not match resource bundle")
    by_id = {item.interaction_id: item for item in bundle.interactions}
    unknown = set(universe.interaction_ids).difference(by_id)
    if unknown:
        raise ValueError(
            f"frozen interaction universe has unknown IDs: {sorted(unknown)}"
        )
    return tuple(
        FrozenActivityInteractionV2.from_interaction(by_id[interaction_id])
        for interaction_id in universe.interaction_ids
    )


def fit_absolute_activity_v2_transform(
    aggregate: PseudobulkDataset,
    resource_bundle: ResourceBundle,
    interaction_universe: FrozenInteractionUniverse,
    *,
    fold_id: str,
    training_input_digest: str,
    spec: AbsoluteActivityV2Spec | None = None,
) -> AbsoluteActivityV2Transform:
    """Fit outcome-independent expression transforms on training subjects only."""

    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("M0 v2 training requires a raw-count PseudobulkDataset")
    resolved = spec or AbsoluteActivityV2Spec()
    if not isinstance(resolved, AbsoluteActivityV2Spec):
        raise TypeError("spec must be AbsoluteActivityV2Spec or None")
    interactions = _require_resource_binding(resource_bundle, interaction_universe)
    feature_ids = tuple(
        sorted(
            {
                gene
                for item in interactions
                for gene in (*item.ligand_subunits, *item.receptor_subunits)
            }
        )
    )
    feature_index = {name: index for index, name in enumerate(aggregate.feature_ids)}
    missing_features = set(feature_ids).difference(feature_index)
    if missing_features:
        raise ValueError(
            "training aggregate lacks frozen interaction features: "
            f"{sorted(missing_features)}"
        )
    selected_columns = np.asarray(
        [feature_index[name] for name in feature_ids], dtype=np.int64
    )
    metadata = _matrix_metadata(aggregate)
    eligible = metadata["state_eligible"].astype(bool)
    cell_type_ids = tuple(
        sorted(metadata.loc[eligible, "cell_type"].astype(str).unique())
    )
    if not cell_type_ids:
        raise ValueError("training aggregate has no state-eligible cell types")
    library_sizes = np.asarray(aggregate.counts.sum(axis=1)).ravel().astype(float)
    medians: list[np.ndarray] = []
    mads: list[np.ndarray] = []
    median_libraries: list[float] = []
    cell_halves: list[float] = []
    for cell_type in cell_type_ids:
        mask = eligible & metadata["cell_type"].astype(str).eq(cell_type)
        rows = np.flatnonzero(mask.to_numpy())
        libraries = library_sizes[rows]
        if len(rows) == 0 or np.any(libraries <= 0.0):
            raise ValueError(f"cell type {cell_type!r} has invalid training libraries")
        median_library = float(np.median(libraries))
        size_factors = libraries / median_library
        counts = aggregate.counts[rows][:, selected_columns].toarray().astype(float)
        normalized = np.divide(
            counts,
            size_factors[:, np.newaxis],
            out=np.zeros_like(counts),
            where=size_factors[:, np.newaxis] > 0.0,
        )
        transformed = np.log2(normalized + 1.0)
        feature_median = np.median(transformed, axis=0)
        feature_mad = resolved.mad_scale * np.median(
            np.abs(transformed - feature_median), axis=0
        )
        feature_mad = np.maximum(feature_mad, resolved.robust_scale_floor)
        n_cells = metadata.iloc[rows]["n_cells"].to_numpy(dtype=float)
        median_libraries.append(median_library)
        cell_halves.append(max(1.0, float(np.median(n_cells))))
        medians.append(feature_median)
        mads.append(feature_mad)
    training_subject_ids = tuple(
        sorted(set(aggregate.unit_metadata["subject_id"].astype(str)))
    )
    return AbsoluteActivityV2Transform(
        fold_id=_name(fold_id, field_name="fold_id"),
        training_subject_ids=training_subject_ids,
        training_input_digest=_name(
            training_input_digest, field_name="training_input_digest"
        ),
        resource_id=resource_bundle.resource_id,
        resource_version=resource_bundle.version,
        resource_manifest_digest=resource_bundle.manifest_digest,
        interaction_universe_id=interaction_universe.filter_universe_id,
        interactions=interactions,
        cell_type_ids=cell_type_ids,
        feature_ids=feature_ids,
        median_library_sizes=np.asarray(median_libraries),
        cell_count_half_saturation=np.asarray(cell_halves),
        feature_medians=np.vstack(medians),
        feature_mads=np.vstack(mads),
        spec=resolved,
    )


def _aggregate_entity(
    expression: npt.NDArray[np.float64],
    feature_indices: Sequence[int],
    *,
    is_complex: bool,
    temperature: float,
) -> float:
    values = expression[np.asarray(feature_indices, dtype=np.int64)]
    if np.any(~np.isfinite(values)):
        return math.nan
    if len(values) == 1:
        return float(values[0])
    if is_complex:
        minimum = float(values.min())
        shifted = np.exp(-(values - minimum) / temperature)
        return float(minimum - temperature * math.log(float(shifted.mean())))
    original_scale = np.maximum(np.exp2(values) - 1.0, 0.0)
    return float(np.log2(float(original_scale.sum()) + 1.0))


def _condition_value(row: pd.Series, columns: tuple[str, ...]) -> str:
    if len(columns) == 1:
        return _name(str(row[columns[0]]), field_name="condition")
    return str(canonical_json({column: row[column] for column in columns}))


def _compute_absolute_activity_v2_source(
    transform: AbsoluteActivityV2Transform,
    aggregate: PseudobulkDataset,
    availability: BatchAvailability,
    *,
    condition_columns: Sequence[str],
) -> pd.DataFrame:
    """Compute activity rows without assigning inferential application lineage."""

    if not isinstance(transform, AbsoluteActivityV2Transform):
        raise TypeError("transform must be AbsoluteActivityV2Transform")
    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("M0 v2 application requires a raw-count PseudobulkDataset")
    if not isinstance(availability, BatchAvailability):
        raise TypeError("availability must be BatchAvailability")
    universe = availability.frozen_interaction_universe
    if (
        universe.filter_universe_id != transform.interaction_universe_id
        or availability.resource_id != transform.resource_id
        or availability.resource_version != transform.resource_version
    ):
        raise ValueError("availability does not match the activity transform")
    conditions = _names(tuple(condition_columns), field_name="condition_columns")
    source = availability.sample_interactions.copy(deep=True)
    missing = _AVAILABILITY_COLUMNS.union(conditions).difference(source.columns)
    if missing:
        raise ValueError(f"availability lacks M0 v2 columns: {sorted(missing)}")
    if source.duplicated(list(_CANDIDATE_KEY)).any():
        raise ValueError("activity availability candidate keys must be unique")

    feature_index = {name: index for index, name in enumerate(aggregate.feature_ids)}
    missing_features = set(transform.feature_ids).difference(feature_index)
    if missing_features:
        raise ValueError(
            "application aggregate lacks transform features: "
            f"{sorted(missing_features)}"
        )
    selected_columns = np.asarray(
        [feature_index[name] for name in transform.feature_ids], dtype=np.int64
    )
    cell_type_index = {
        cell_type: index for index, cell_type in enumerate(transform.cell_type_ids)
    }
    metadata = _matrix_metadata(aggregate)
    libraries = np.asarray(aggregate.counts.sum(axis=1)).ravel().astype(float)
    selected_counts = aggregate.counts[:, selected_columns].toarray().astype(float)
    unit_expression: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
    unit_reliability: dict[tuple[str, str], float] = {}
    for row_index, row in metadata.iterrows():
        sample_id = str(row["sample_id"])
        cell_type = str(row["cell_type"])
        transform_row = cell_type_index.get(cell_type)
        if transform_row is None:
            continue
        n_cells = float(row["n_cells"])
        half = float(transform.cell_count_half_saturation[transform_row])
        unit_reliability[(sample_id, cell_type)] = n_cells / (n_cells + half)
        if not bool(row["state_eligible"]) or libraries[row_index] <= 0.0:
            continue
        size_factor = libraries[row_index] / float(
            transform.median_library_sizes[transform_row]
        )
        normalized = selected_counts[row_index] / size_factor
        unit_expression[(sample_id, cell_type)] = np.log2(normalized + 1.0)

    interactions = {item.interaction_id: item for item in transform.interactions}
    transform_feature_index = {
        name: index for index, name in enumerate(transform.feature_ids)
    }
    entity_cache: dict[tuple[str, str, str, str], float] = {}

    def entity_value(
        sample_id: str,
        cell_type: str,
        interaction_id: str,
        side: str,
    ) -> float:
        key = (sample_id, cell_type, interaction_id, side)
        cached = entity_cache.get(key)
        if cached is not None:
            return cached
        expression = unit_expression.get((sample_id, cell_type))
        interaction = interactions.get(interaction_id)
        if expression is None or interaction is None:
            value = math.nan
        else:
            subunits = (
                interaction.ligand_subunits
                if side == "ligand"
                else interaction.receptor_subunits
            )
            is_complex = (
                interaction.ligand_is_complex
                if side == "ligand"
                else interaction.receptor_is_complex
            )
            value = _aggregate_entity(
                expression,
                [transform_feature_index[gene] for gene in subunits],
                is_complex=is_complex,
                temperature=transform.spec.complex_temperature,
            )
        entity_cache[key] = value
        return value

    ligand_values: list[float] = []
    receptor_values: list[float] = []
    reliability_values: list[float] = []
    ligand_names: list[str] = []
    receptor_names: list[str] = []
    for row in source.itertuples(index=False):
        sample_id = str(row.sample_id)
        sender = str(row.sender)
        receiver = str(row.receiver)
        interaction_id = str(row.interaction_id)
        interaction = interactions.get(interaction_id)
        if interaction is None:
            raise ValueError(f"availability has unknown interaction {interaction_id!r}")
        ligand_values.append(entity_value(sample_id, sender, interaction_id, "ligand"))
        receptor_values.append(
            entity_value(sample_id, receiver, interaction_id, "receptor")
        )
        sender_reliability = unit_reliability.get((sample_id, sender), math.nan)
        receiver_reliability = unit_reliability.get((sample_id, receiver), math.nan)
        reliability_values.append(
            math.sqrt(sender_reliability * receiver_reliability)
            if math.isfinite(sender_reliability) and math.isfinite(receiver_reliability)
            else math.nan
        )
        ligand_names.append(interaction.ligand_name)
        receptor_names.append(interaction.receptor_name)
    source["ligand_activity_raw"] = ligand_values
    source["receptor_activity_raw"] = receptor_values
    average = 0.5 * (source["ligand_activity_raw"] + source["receptor_activity_raw"])
    imbalance = (source["ligand_activity_raw"] - source["receptor_activity_raw"]).abs()
    source["sender_detection_raw"] = average - (
        transform.spec.bottleneck_penalty * imbalance
    )
    source["ligand"] = ligand_names
    source["receptor"] = receptor_names
    source["condition"] = source.apply(_condition_value, axis=1, columns=conditions)
    source["cell_count_reliability"] = reliability_values
    source["reliability_weight"] = reliability_values

    grouped = source.groupby(list(_PARENT_KEY), observed=True, sort=False)
    source["candidate_sender_count"] = grouped["sender"].transform("size").astype(int)
    source["effective_candidate_count"] = (
        grouped["sender_detection_raw"].transform("count").astype(int)
    )
    summaries = grouped["sender_detection_raw"].agg(
        parent_peak_raw="max",
        parent_total_raw=lambda values: values.sum(min_count=1),
        parent_mean_raw="mean",
    )
    source = source.merge(
        summaries.reset_index(),
        on=list(_PARENT_KEY),
        how="left",
        validate="many_to_one",
        sort=False,
    )
    measured = source["sender_detection_raw"].notna()
    low_evidence = measured & source["sender_detection_raw"].eq(0.0)
    source["coverage_status"] = np.where(
        measured, "measured", "not_measured_or_not_estimable"
    )
    source["status"] = np.where(
        ~measured,
        "not_estimable",
        np.where(low_evidence, "low_evidence", "observed"),
    )
    source["reason_code"] = np.where(
        ~measured,
        "sender_or_receiver_not_measured",
        np.where(low_evidence, "measured_zero_low_evidence", None),
    )
    source["structural_impossibility"] = False
    mechanism_support = pd.to_numeric(source["availability_state"], errors="coerce")
    finite_mechanism = mechanism_support.dropna()
    if (
        source["availability_state"].notna().any()
        and mechanism_support.isna().sum() > source["availability_state"].isna().sum()
    ) or ((finite_mechanism < 0.0) | (finite_mechanism > 1.0)).any():
        raise ValueError("availability_state must contain values in [0, 1] or NA")
    source["mechanism_support"] = mechanism_support.astype(float)

    for head, status, reason, functional_id in (
        (
            "program_signed",
            "program_status",
            "program_reason_code",
            "program_functional_id",
        ),
        (
            "coupling_prior",
            "coupling_status",
            "coupling_reason_code",
            "coupling_functional_id",
        ),
        (
            "active_probability",
            "occurrence_status",
            "occurrence_reason_code",
            "occurrence_functional_id",
        ),
        (
            "sender_attribution",
            "attribution_status",
            "attribution_reason_code",
            "attribution_functional_id",
        ),
    ):
        source[head] = np.nan
        source[status] = "not_computed"
        source[reason] = f"{head}_not_computed"
        source[functional_id] = None
    source["null_sender_attribution"] = np.nan
    source["attribution_entropy"] = np.nan
    source["selection_stability"] = np.nan
    return cast(pd.DataFrame, source)


def build_absolute_activity_v2_training_table(
    transform: AbsoluteActivityV2Transform,
    aggregate: PseudobulkDataset,
    availability: BatchAvailability,
    *,
    condition_columns: Sequence[str],
) -> pd.DataFrame:
    """Build non-OOF activity rows solely for fold-training calibration."""

    if not isinstance(transform, AbsoluteActivityV2Transform):
        raise TypeError("transform must be AbsoluteActivityV2Transform")
    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("M0 v2 calibration requires a raw-count PseudobulkDataset")
    if not isinstance(availability, BatchAvailability) or (
        availability.filter_application
        is not InteractionFilterApplication.TRAINING_SELECTION_V1
    ):
        raise ValueError(
            "M0 v2 training calibration requires training-selection availability"
        )
    aggregate_subjects = tuple(
        sorted(set(aggregate.unit_metadata["subject_id"].astype(str)))
    )
    if (
        availability.application_subject_ids != transform.training_subject_ids
        or aggregate_subjects != transform.training_subject_ids
    ):
        raise ValueError(
            "M0 v2 training calibration subjects must match the fitted transform"
        )
    source = _compute_absolute_activity_v2_source(
        transform,
        aggregate,
        availability,
        condition_columns=condition_columns,
    )
    return source.loc[:, list(ABSOLUTE_ACTIVITY_V2_CALIBRATION_COLUMNS)].sort_values(
        list(_CANDIDATE_KEY), kind="stable", ignore_index=True
    )


def apply_absolute_activity_v2_transform(
    transform: AbsoluteActivityV2Transform,
    aggregate: PseudobulkDataset,
    availability: BatchAvailability,
    *,
    condition_columns: Sequence[str],
    repeat_id: str,
    application_input_digest: str,
    config_digest: str,
    seed_lineage_id: str,
    package_version: str,
) -> SampleEdgeScoreV2:
    """Apply one frozen transform to held-out sample x sender x LR x receiver rows."""

    if not isinstance(availability, BatchAvailability) or (
        availability.filter_application
        is not InteractionFilterApplication.FROZEN_APPLICATION_V1
    ):
        raise ValueError(
            "M0 v2 held-out application requires a frozen interaction universe"
        )
    source = _compute_absolute_activity_v2_source(
        transform,
        aggregate,
        availability,
        condition_columns=condition_columns,
    )

    application_subjects = tuple(
        sorted(set(aggregate.unit_metadata["subject_id"].astype(str)))
    )
    if set(source["subject_id"].astype(str)).difference(application_subjects):
        raise ValueError("availability contains subjects outside held-out aggregate")
    provenance = SampleEdgeScoreV2Provenance(
        fold_id=transform.fold_id,
        repeat_id=_name(repeat_id, field_name="repeat_id"),
        training_subject_ids=transform.training_subject_ids,
        application_subject_ids=application_subjects,
        training_input_digest=transform.training_input_digest,
        application_input_digest=_name(
            application_input_digest, field_name="application_input_digest"
        ),
        config_digest=_name(config_digest, field_name="config_digest"),
        resource_id=transform.resource_id,
        resource_version=transform.resource_version,
        resource_manifest_digest=transform.resource_manifest_digest,
        interaction_universe_id=transform.interaction_universe_id,
        transform_manifest_id=transform.transform_manifest_id,
        seed_lineage_id=_name(seed_lineage_id, field_name="seed_lineage_id"),
        score_version=ABSOLUTE_ACTIVITY_V2_SCORE_VERSION,
        package_version=_name(package_version, field_name="package_version"),
    )
    source["fold_id"] = transform.fold_id
    source["score_version"] = ABSOLUTE_ACTIVITY_V2_SCORE_VERSION
    source["activity_functional_id"] = provenance.provenance_id
    source["out_of_fold"] = True
    source["provenance"] = canonical_json(provenance.to_dict())
    result = source.loc[:, list(SAMPLE_EDGE_SCORE_V2_COLUMNS)].sort_values(
        list(_CANDIDATE_KEY), kind="stable", ignore_index=True
    )
    return SampleEdgeScoreV2(table=result, provenance=provenance)


__all__ = [
    "ABSOLUTE_ACTIVITY_V2_CALIBRATION_COLUMNS",
    "ABSOLUTE_ACTIVITY_V2_SCORE_VERSION",
    "ABSOLUTE_ACTIVITY_V2_TRANSFORM_VERSION",
    "AbsoluteActivityV2Spec",
    "AbsoluteActivityV2Transform",
    "FrozenActivityInteractionV2",
    "apply_absolute_activity_v2_transform",
    "build_absolute_activity_v2_training_table",
    "fit_absolute_activity_v2_transform",
]
