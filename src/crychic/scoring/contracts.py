"""Frozen scoring functionals and sample communication score contracts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd

from crychic.core import canonical_digest, canonical_json, stable_id

CORE_COMPONENTS = ("availability", "downstream", "sender", "prior_quality")
LEGACY_UNTRACKED_SCORE_VERSION = "geometric_v1_untracked"
SCORING_COLLECTION_EXTENSION_VERSION = "1.0.0"
SCORING_COLLECTION_DIGEST_METHOD = "sha256_canonical_csv_v1"
SCORING_SOURCE_KEY_COLUMNS = (
    "subject_id",
    "sample_id",
    "context_id",
    "design_row_id",
    "edge_id",
    "scoring_functional_id",
    "repeat_id",
    "fold_id",
    "mode",
)


class _DigestSink:
    """Minimal text sink that streams canonical CSV into SHA-256."""

    __slots__ = ("_digest",)

    def __init__(self, digest: Any) -> None:
        self._digest = digest

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8")
        self._digest.update(encoded)
        return len(value)


def scoring_source_key_digest(table: pd.DataFrame) -> str:
    """Hash a source-score key set independently of input row order."""

    if not isinstance(table, pd.DataFrame):
        raise TypeError("source score keys must be a pandas DataFrame")
    missing = set(SCORING_SOURCE_KEY_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"source score keys are missing: {sorted(missing)}")
    keys = table.loc[:, list(SCORING_SOURCE_KEY_COLUMNS)].copy()
    for column in SCORING_SOURCE_KEY_COLUMNS:
        if keys[column].isna().any():
            raise ValueError(f"source score key {column} must not contain NA")
        values = keys[column].tolist()
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(
                f"source score key {column} must contain non-empty strings"
            )
    if keys.duplicated(list(SCORING_SOURCE_KEY_COLUMNS)).any():
        raise ValueError("source score keys must be unique")
    keys = keys.sort_values(
        list(SCORING_SOURCE_KEY_COLUMNS),
        kind="stable",
        ignore_index=True,
    )
    digest = hashlib.sha256()
    digest.update(f"{SCORING_COLLECTION_DIGEST_METHOD}\n".encode())
    keys.to_csv(
        cast(Any, _DigestSink(digest)),
        index=False,
        header=True,
        lineterminator="\n",
    )
    return digest.hexdigest()


def _required_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sha256_digest(value: object, *, field_name: str) -> str:
    normalized = _required_identifier(value, field_name=field_name)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True, slots=True, kw_only=True)
class ReceiverScoringFunctionalManifest:
    """One emitted receiver partition and its verifiable source-score keys."""

    receiver: str
    scoring_functional_id: str
    source_score_key_digest: str
    source_score_row_count: int
    provenance_status: str = "functional_metadata_not_persisted_unverified"
    child_manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = ("receiver", "scoring_functional_id")
        payload: dict[str, object] = {}
        for field_name in identifiers:
            normalized = _required_identifier(
                getattr(self, field_name), field_name=field_name
            )
            object.__setattr__(self, field_name, normalized)
            payload[field_name] = normalized
        if self.provenance_status != "functional_metadata_not_persisted_unverified":
            raise ValueError(
                "provenance_status must declare unverified functional metadata"
            )
        payload["provenance_status"] = self.provenance_status
        source_digest = _sha256_digest(
            self.source_score_key_digest,
            field_name="source_score_key_digest",
        )
        if (
            isinstance(self.source_score_row_count, bool)
            or not isinstance(self.source_score_row_count, int)
            or self.source_score_row_count <= 0
        ):
            raise ValueError("source_score_row_count must be a positive integer")
        object.__setattr__(self, "source_score_key_digest", source_digest)
        payload.update(
            {
                "source_score_key_digest": source_digest,
                "source_score_row_count": self.source_score_row_count,
            }
        )
        object.__setattr__(
            self,
            "child_manifest_id",
            stable_id("receiver_scoring_functional", payload),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical child-manifest representation."""

        return {
            "child_manifest_id": self.child_manifest_id,
            "receiver": self.receiver,
            "scoring_functional_id": self.scoring_functional_id,
            "provenance_status": self.provenance_status,
            "source_score_key_digest": self.source_score_key_digest,
            "source_score_row_count": self.source_score_row_count,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> ReceiverScoringFunctionalManifest:
        """Parse a child manifest while verifying its derived stable ID."""

        expected = {
            "child_manifest_id",
            "receiver",
            "scoring_functional_id",
            "provenance_status",
            "source_score_key_digest",
            "source_score_row_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("receiver child manifest fields are invalid")
        result = cls(
            receiver=value["receiver"],  # type: ignore[arg-type]
            scoring_functional_id=value["scoring_functional_id"],  # type: ignore[arg-type]
            provenance_status=value["provenance_status"],  # type: ignore[arg-type]
            source_score_key_digest=value["source_score_key_digest"],  # type: ignore[arg-type]
            source_score_row_count=value["source_score_row_count"],  # type: ignore[arg-type]
        )
        if value["child_manifest_id"] != result.child_manifest_id:
            raise ValueError("receiver child manifest ID does not match its payload")
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoringCollectionManifest:
    """Partial emitted receiver partition for one contrast, repeat, and fold.

    This contract deliberately does not claim that receiver children share one
    scoring functional, nor that emitted children cover a planned receiver
    universe.
    """

    contrast: str
    repeat_id: str
    fold_id: str
    emitted_receivers: tuple[str, ...]
    children: tuple[ReceiverScoringFunctionalManifest, ...]
    common_functional_across_receivers: bool = False
    partition_key: str = "receiver"
    receiver_scope: str = "emitted_sample_scores"
    composition_status: str = "partial_emitted_only"
    scoring_collection_id: str = field(init=False)
    source_score_key_digest: str = field(init=False)
    source_score_row_count: int = field(init=False)

    def __post_init__(self) -> None:
        contrast = _required_identifier(self.contrast, field_name="contrast")
        repeat_id = _required_identifier(self.repeat_id, field_name="repeat_id")
        fold_id = _required_identifier(self.fold_id, field_name="fold_id")
        if self.common_functional_across_receivers is not False:
            raise ValueError(
                "receiver-partition collections must declare "
                "common_functional_across_receivers=false"
            )
        if self.partition_key != "receiver":
            raise ValueError("partition_key must be 'receiver'")
        if self.receiver_scope != "emitted_sample_scores":
            raise ValueError("receiver_scope must be 'emitted_sample_scores'")
        if self.composition_status != "partial_emitted_only":
            raise ValueError("composition_status must be 'partial_emitted_only'")
        receivers = _stable_names(
            tuple(self.emitted_receivers), field_name="emitted_receivers"
        )
        children = tuple(self.children)
        if not children or any(
            not isinstance(child, ReceiverScoringFunctionalManifest)
            for child in children
        ):
            raise ValueError(
                "children must contain ReceiverScoringFunctionalManifest values"
            )
        children = tuple(sorted(children, key=lambda child: child.receiver))
        child_receivers = tuple(child.receiver for child in children)
        if child_receivers != receivers:
            raise ValueError(
                "emitted_receivers must exactly match receiver child partitions"
            )
        for field_name in ("child_manifest_id", "scoring_functional_id"):
            values = [getattr(child, field_name) for child in children]
            if len(set(values)) != len(values):
                raise ValueError(
                    f"receiver children must have unique {field_name} values"
                )
        source_row_count = sum(child.source_score_row_count for child in children)
        source_digest = canonical_digest(
            {
                "digest_method": SCORING_COLLECTION_DIGEST_METHOD,
                "receiver_children": [
                    {
                        "receiver": child.receiver,
                        "source_score_key_digest": child.source_score_key_digest,
                        "source_score_row_count": child.source_score_row_count,
                    }
                    for child in children
                ],
            }
        )
        payload = {
            "common_functional_across_receivers": False,
            "composition_status": "partial_emitted_only",
            "contrast": contrast,
            "fold_id": fold_id,
            "partition_key": "receiver",
            "receiver_scope": "emitted_sample_scores",
            "repeat_id": repeat_id,
            "emitted_receivers": list(receivers),
            "children": [child.to_dict() for child in children],
            "source_score_key_digest": source_digest,
            "source_score_row_count": source_row_count,
        }
        object.__setattr__(self, "contrast", contrast)
        object.__setattr__(self, "repeat_id", repeat_id)
        object.__setattr__(self, "fold_id", fold_id)
        object.__setattr__(self, "emitted_receivers", receivers)
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "source_score_key_digest", source_digest)
        object.__setattr__(self, "source_score_row_count", source_row_count)
        object.__setattr__(
            self,
            "scoring_collection_id",
            stable_id("scoring_collection", payload),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical receiver-collection representation."""

        return {
            "scoring_collection_id": self.scoring_collection_id,
            "contrast": self.contrast,
            "repeat_id": self.repeat_id,
            "fold_id": self.fold_id,
            "partition_key": self.partition_key,
            "receiver_scope": self.receiver_scope,
            "composition_status": self.composition_status,
            "common_functional_across_receivers": (
                self.common_functional_across_receivers
            ),
            "emitted_receivers": list(self.emitted_receivers),
            "children": [child.to_dict() for child in self.children],
            "source_score_key_digest": self.source_score_key_digest,
            "source_score_row_count": self.source_score_row_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ScoringCollectionManifest:
        """Parse a collection while verifying all derived identities."""

        expected = {
            "scoring_collection_id",
            "contrast",
            "repeat_id",
            "fold_id",
            "partition_key",
            "receiver_scope",
            "composition_status",
            "common_functional_across_receivers",
            "emitted_receivers",
            "children",
            "source_score_key_digest",
            "source_score_row_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("scoring collection manifest fields are invalid")
        raw_receivers = value["emitted_receivers"]
        raw_children = value["children"]
        if (
            not isinstance(raw_receivers, Sequence)
            or isinstance(raw_receivers, str)
            or not isinstance(raw_children, Sequence)
            or isinstance(raw_children, str)
        ):
            raise ValueError("scoring collection receivers and children must be arrays")
        result = cls(
            contrast=value["contrast"],  # type: ignore[arg-type]
            repeat_id=value["repeat_id"],  # type: ignore[arg-type]
            fold_id=value["fold_id"],  # type: ignore[arg-type]
            partition_key=value["partition_key"],  # type: ignore[arg-type]
            receiver_scope=value["receiver_scope"],  # type: ignore[arg-type]
            composition_status=value["composition_status"],  # type: ignore[arg-type]
            common_functional_across_receivers=value[  # type: ignore[arg-type]
                "common_functional_across_receivers"
            ],
            emitted_receivers=cast(tuple[str, ...], tuple(raw_receivers)),
            children=tuple(
                ReceiverScoringFunctionalManifest.from_dict(child)
                for child in raw_children
                if isinstance(child, Mapping)
            ),
        )
        if len(result.children) != len(raw_children):
            raise ValueError("scoring collection children must be objects")
        derived = result.to_dict()
        for field_name in (
            "scoring_collection_id",
            "source_score_key_digest",
            "source_score_row_count",
        ):
            if value[field_name] != derived[field_name]:
                raise ValueError(f"{field_name} does not match the collection payload")
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoringCollectionDocument:
    """Versioned persisted document containing receiver collections."""

    collections: tuple[ScoringCollectionManifest, ...]
    extension_schema_version: str = SCORING_COLLECTION_EXTENSION_VERSION
    result_schema_version: str = "0.1.0"
    collection_kind: str = "receiver_partition"
    digest_method: str = SCORING_COLLECTION_DIGEST_METHOD

    def __post_init__(self) -> None:
        if self.extension_schema_version != SCORING_COLLECTION_EXTENSION_VERSION:
            raise ValueError("unsupported scoring collection extension version")
        if self.result_schema_version != "0.1.0":
            raise ValueError("scoring collections require result schema v0.1.0")
        if self.collection_kind != "receiver_partition":
            raise ValueError("collection_kind must be 'receiver_partition'")
        if self.digest_method != SCORING_COLLECTION_DIGEST_METHOD:
            raise ValueError("unsupported scoring collection digest method")
        collections = tuple(self.collections)
        if not collections or any(
            not isinstance(collection, ScoringCollectionManifest)
            for collection in collections
        ):
            raise ValueError("scoring collection document must not be empty")
        collections = tuple(
            sorted(
                collections,
                key=lambda item: (item.contrast, item.repeat_id, item.fold_id),
            )
        )
        group_keys = [
            (item.contrast, item.repeat_id, item.fold_id) for item in collections
        ]
        if len(set(group_keys)) != len(group_keys):
            raise ValueError(
                "scoring collections must be unique by contrast, repeat, and fold"
            )
        child_ids = [
            child.scoring_functional_id
            for collection in collections
            for child in collection.children
        ]
        if len(set(child_ids)) != len(child_ids):
            raise ValueError(
                "a receiver child functional must belong to exactly one collection"
            )
        object.__setattr__(self, "collections", collections)

    def to_dict(self) -> dict[str, object]:
        """Return a schema-ready versioned collection document."""

        return {
            "extension_schema_version": self.extension_schema_version,
            "result_schema_version": self.result_schema_version,
            "collection_kind": self.collection_kind,
            "digest_method": self.digest_method,
            "collections": [collection.to_dict() for collection in self.collections],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ScoringCollectionDocument:
        """Parse and fully validate a persisted collection document."""

        expected = {
            "extension_schema_version",
            "result_schema_version",
            "collection_kind",
            "digest_method",
            "collections",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("scoring collection document fields are invalid")
        raw_collections = value["collections"]
        if not isinstance(raw_collections, Sequence) or isinstance(
            raw_collections, str
        ):
            raise ValueError("scoring collection document collections must be an array")
        collections = tuple(
            ScoringCollectionManifest.from_dict(collection)
            for collection in raw_collections
            if isinstance(collection, Mapping)
        )
        if len(collections) != len(raw_collections):
            raise ValueError("scoring collections must be objects")
        return cls(
            extension_schema_version=value["extension_schema_version"],  # type: ignore[arg-type]
            result_schema_version=value["result_schema_version"],  # type: ignore[arg-type]
            collection_kind=value["collection_kind"],  # type: ignore[arg-type]
            digest_method=value["digest_method"],  # type: ignore[arg-type]
            collections=collections,
        )


def float64_array_digest(values: np.ndarray) -> str:
    """Hash an array after canonical C-order, little-endian float64 conversion."""

    canonical = np.asarray(values, dtype="<f8", order="C")
    if np.any(~np.isfinite(canonical)):
        raise ValueError("array digest requires finite values")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoringModelManifest:
    """Stable identity of every learned artifact consumed by a score."""

    score_version: str
    basis_id: str
    coefficient_digest: str
    receptor_gate_manifest_id: str
    target_weight_manifest_id: str
    downstream_functional_id: str
    sender_functional_id: str
    availability_transform_id: str
    precision_transform_id: str
    filter_universe_id: str
    tuning_manifest_id: str
    model_manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        artifact_fields = (
            "score_version",
            "basis_id",
            "coefficient_digest",
            "receptor_gate_manifest_id",
            "target_weight_manifest_id",
            "downstream_functional_id",
            "sender_functional_id",
            "availability_transform_id",
            "precision_transform_id",
            "filter_universe_id",
            "tuning_manifest_id",
        )
        payload: dict[str, str] = {}
        for field_name in artifact_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            normalized = value.strip()
            object.__setattr__(self, field_name, normalized)
            payload[field_name] = normalized
        object.__setattr__(
            self,
            "model_manifest_id",
            stable_id("scoring_model_manifest", payload),
        )

    def to_dict(self) -> dict[str, str]:
        """Return a serialization-ready learned-artifact manifest."""

        return {
            "model_manifest_id": self.model_manifest_id,
            "score_version": self.score_version,
            "basis_id": self.basis_id,
            "coefficient_digest": self.coefficient_digest,
            "receptor_gate_manifest_id": self.receptor_gate_manifest_id,
            "target_weight_manifest_id": self.target_weight_manifest_id,
            "downstream_functional_id": self.downstream_functional_id,
            "sender_functional_id": self.sender_functional_id,
            "availability_transform_id": self.availability_transform_id,
            "precision_transform_id": self.precision_transform_id,
            "filter_universe_id": self.filter_universe_id,
            "tuning_manifest_id": self.tuning_manifest_id,
        }


class ScoringFunctionalStatus(StrEnum):
    """Whether a functional is exploratory in-sample or genuinely OOF."""

    EXPLORATORY_IN_SAMPLE = "exploratory_in_sample"
    OUT_OF_FOLD = "out_of_fold"


class CommunicationScoreStatus(StrEnum):
    """Validity of an integrated communication strength."""

    OK = "ok"
    MISSING_CORE_EVIDENCE = "missing_core_evidence"


def _stable_tuple(
    values: tuple[Hashable, ...], *, field_name: str
) -> tuple[Hashable, ...]:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
    try:
        return tuple(sorted(values, key=canonical_json))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} values must support canonical JSON serialization"
        ) from error


def _stable_names(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = tuple(value.strip() for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")
    return tuple(sorted(normalized))


def _component_mapping(
    values: Mapping[str, float],
    *,
    field_name: str,
    allow_zero: bool,
) -> Mapping[str, float]:
    if set(values) != set(CORE_COMPONENTS):
        raise ValueError(f"{field_name} must define exactly {list(CORE_COMPONENTS)}")
    normalized: dict[str, float] = {}
    for component in CORE_COMPONENTS:
        value = float(values[component])
        valid = math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
        if not valid:
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(
                f"{field_name}[{component!r}] must be finite and {relation}"
            )
        normalized[component] = value
    if allow_zero and not any(normalized.values()):
        raise ValueError("at least one component weight must be positive")
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoringFunctional:
    """One immutable, contrast-level communication scoring functional."""

    contrast_name: str
    contrast_contexts: tuple[Hashable, ...]
    training_subject_ids: tuple[str, ...]
    interaction_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    score_version: str = LEGACY_UNTRACKED_SCORE_VERSION
    model_manifest: ScoringModelManifest | None = None
    status: ScoringFunctionalStatus = ScoringFunctionalStatus.EXPLORATORY_IN_SAMPLE
    fold_id: str | None = None
    component_weights: Mapping[str, float] = field(
        default_factory=lambda: dict.fromkeys(CORE_COMPONENTS, 1.0)
    )
    component_scales: Mapping[str, float] = field(
        default_factory=lambda: dict.fromkeys(CORE_COMPONENTS, 1.0)
    )
    frozen: bool = True
    output_scale: str = "unit_interval"
    scoring_function_id: str = field(init=False)
    model_manifest_id: str | None = field(init=False)
    interaction_universe_id: str = field(init=False)
    target_universe_id: str = field(init=False)
    reason_code: str | None = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.contrast_name, str) or not self.contrast_name.strip():
            raise ValueError("contrast_name must be a non-empty string")
        if not self.frozen:
            raise ValueError("a ScoringFunctional must be frozen before application")
        if self.output_scale != "unit_interval":
            raise ValueError("v0.1 scoring supports only output_scale='unit_interval'")
        if not isinstance(self.score_version, str) or not self.score_version.strip():
            raise ValueError("score_version must be a non-empty string")
        score_version = self.score_version.strip()
        if self.model_manifest is None:
            if score_version != LEGACY_UNTRACKED_SCORE_VERSION:
                raise ValueError(
                    "a tracked score_version requires a ScoringModelManifest"
                )
            model_manifest_id = None
        else:
            if not isinstance(self.model_manifest, ScoringModelManifest):
                raise TypeError("model_manifest must be a ScoringModelManifest or None")
            if score_version != self.model_manifest.score_version:
                raise ValueError(
                    "score_version must match the supplied ScoringModelManifest"
                )
            model_manifest_id = self.model_manifest.model_manifest_id
        status = ScoringFunctionalStatus(self.status)
        contexts = _stable_tuple(
            tuple(self.contrast_contexts), field_name="contrast_contexts"
        )
        if len(contexts) < 2:
            raise ValueError("contrast_contexts must contain at least two contexts")
        training_subjects = _stable_names(
            tuple(self.training_subject_ids), field_name="training_subject_ids"
        )
        interactions = _stable_names(
            tuple(self.interaction_ids), field_name="interaction_ids"
        )
        targets = _stable_names(tuple(self.target_ids), field_name="target_ids")
        weights = _component_mapping(
            self.component_weights,
            field_name="component_weights",
            allow_zero=True,
        )
        scales = _component_mapping(
            self.component_scales,
            field_name="component_scales",
            allow_zero=False,
        )
        if status is ScoringFunctionalStatus.OUT_OF_FOLD:
            if not isinstance(self.fold_id, str) or not self.fold_id.strip():
                raise ValueError("out-of-fold functional requires a non-empty fold_id")
            reason_code = None
        else:
            if self.fold_id is not None:
                raise ValueError("in-sample functional must not declare an OOF fold_id")
            reason_code = "exploratory_not_cross_fitted"

        interaction_universe_id = stable_id(
            "interaction_universe", {"interaction_ids": list(interactions)}
        )
        target_universe_id = stable_id("target_universe", {"target_ids": list(targets)})
        payload = {
            "component_scales": [[key, scales[key]] for key in CORE_COMPONENTS],
            "component_weights": [[key, weights[key]] for key in CORE_COMPONENTS],
            "contrast_contexts": list(contexts),
            "contrast_name": self.contrast_name.strip(),
            "fold_id": self.fold_id,
            "frozen": True,
            "interaction_universe_id": interaction_universe_id,
            "model_manifest_id": model_manifest_id,
            "output_scale": self.output_scale,
            "score_version": score_version,
            "status": status.value,
            "target_universe_id": target_universe_id,
            "training_subject_ids": list(training_subjects),
        }
        function_id = stable_id("scoring_function", payload)

        object.__setattr__(self, "contrast_name", self.contrast_name.strip())
        object.__setattr__(self, "contrast_contexts", contexts)
        object.__setattr__(self, "training_subject_ids", training_subjects)
        object.__setattr__(self, "interaction_ids", interactions)
        object.__setattr__(self, "target_ids", targets)
        object.__setattr__(self, "score_version", score_version)
        object.__setattr__(self, "component_weights", weights)
        object.__setattr__(self, "component_scales", scales)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "model_manifest_id", model_manifest_id)
        object.__setattr__(self, "interaction_universe_id", interaction_universe_id)
        object.__setattr__(self, "target_universe_id", target_universe_id)
        object.__setattr__(self, "scoring_function_id", function_id)

    @property
    def is_out_of_fold(self) -> bool:
        return self.status is ScoringFunctionalStatus.OUT_OF_FOLD

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready frozen functional definition."""

        return {
            "scoring_function_id": self.scoring_function_id,
            "contrast_name": self.contrast_name,
            "contrast_contexts": list(self.contrast_contexts),
            "training_subject_ids": list(self.training_subject_ids),
            "fold_id": self.fold_id,
            "interaction_ids": list(self.interaction_ids),
            "interaction_universe_id": self.interaction_universe_id,
            "target_ids": list(self.target_ids),
            "target_universe_id": self.target_universe_id,
            "score_version": self.score_version,
            "model_manifest_id": self.model_manifest_id,
            "model_manifest": (
                None if self.model_manifest is None else self.model_manifest.to_dict()
            ),
            "component_weights": dict(self.component_weights),
            "component_scales": dict(self.component_scales),
            "status": self.status.value,
            "reason_code": self.reason_code,
            "frozen": self.frozen,
            "output_scale": self.output_scale,
        }


SCORE_COLUMNS = {
    "sample_id",
    "subject_id",
    "context",
    "sender",
    "receiver",
    "interaction_id",
    "mode",
    "availability",
    "downstream_activity",
    "sender_component",
    "prior_quality",
    "abundance_component",
    "comm_strength",
    "status",
    "reason_code",
    "functional_status",
    "functional_reason_code",
    "contrast",
    "fold_id",
    "scoring_function_id",
    "score_version",
    "model_manifest_id",
    "interaction_universe_id",
    "target_universe_id",
}


def validate_common_functional(table: pd.DataFrame) -> None:
    """Reject context-specific functionals within one contrast/fold score group."""

    required = {
        "contrast",
        "fold_id",
        "context",
        "functional_status",
        "scoring_function_id",
        "score_version",
        "model_manifest_id",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"score table is missing functional fields: {sorted(missing)}")
    formal = table["functional_status"] == ScoringFunctionalStatus.OUT_OF_FOLD.value
    for (contrast, fold_id), group in table.loc[formal].groupby(
        ["contrast", "fold_id"], dropna=False, sort=False
    ):
        if group["scoring_function_id"].nunique(dropna=False) != 1:
            raise ValueError(
                "out-of-fold contexts use different scoring_function_id values "
                f"for contrast={contrast!r}, fold_id={fold_id!r}"
            )
        for column in ("score_version", "model_manifest_id"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    "out-of-fold contexts use different "
                    f"{column} values for contrast={contrast!r}, "
                    f"fold_id={fold_id!r}"
                )


@dataclass(frozen=True, slots=True)
class CommunicationScores:
    """Component-preserving sample communication strengths."""

    table: pd.DataFrame
    functional: ScoringFunctional

    def __post_init__(self) -> None:
        table = self.table.copy(deep=True)
        missing = SCORE_COLUMNS.difference(table.columns)
        if missing:
            raise ValueError(f"communication score table is missing: {sorted(missing)}")
        forbidden_names = {
            "p",
            "p_value",
            "q",
            "q_value",
            "posterior",
            "posterior_probability",
        }
        forbidden = [
            str(column)
            for column in table
            if "probability" in str(column).lower()
            or str(column).lower() in forbidden_names
        ]
        if forbidden:
            raise ValueError(
                "scoring contains no probability or hypothesis-test fields; "
                "forbidden columns: "
                f"{sorted(forbidden)}"
            )
        if not table.empty:
            function_ids = set(table["scoring_function_id"])
            if function_ids != {self.functional.scoring_function_id}:
                raise ValueError(
                    "CommunicationScores rows must use their functional's stable ID"
                )
            if set(table["contrast"]) != {self.functional.contrast_name}:
                raise ValueError("score rows must use their functional's contrast")
            expected_provenance = {
                "functional_status": self.functional.status.value,
                "fold_id": self.functional.fold_id or "in_sample",
                "score_version": self.functional.score_version,
                "model_manifest_id": self.functional.model_manifest_id,
                "interaction_universe_id": self.functional.interaction_universe_id,
                "target_universe_id": self.functional.target_universe_id,
            }
            for column, expected in expected_provenance.items():
                if set(table[column]) != {expected}:
                    raise ValueError(
                        f"score rows have incompatible {column} provenance"
                    )
            observed_contexts = set(table["context"])
            unknown_contexts = observed_contexts.difference(
                self.functional.contrast_contexts
            )
            if unknown_contexts:
                raise ValueError(
                    "score rows contain contexts outside the functional: "
                    f"{unknown_contexts}"
                )
            if self.functional.is_out_of_fold and observed_contexts != set(
                self.functional.contrast_contexts
            ):
                raise ValueError(
                    "out-of-fold score table must contain every contrast context"
                )
        key = [
            "sample_id",
            "context",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
            "scoring_function_id",
        ]
        if table.duplicated(key).any():
            raise ValueError("communication score primary key must be unique")
        if not set(table["mode"]).issubset({"state", "ecosystem"}):
            raise ValueError("communication score mode must be state or ecosystem")
        allowed_status = {status.value for status in CommunicationScoreStatus}
        if not set(table["status"]).issubset(allowed_status):
            raise ValueError("communication score status is not recognized")
        for column in (
            "availability",
            "downstream_activity",
            "sender_component",
            "prior_quality",
            "abundance_component",
            "comm_strength",
        ):
            numeric = pd.to_numeric(table[column], errors="coerce")
            invalid_type = table[column].notna() & numeric.isna()
            if invalid_type.any():
                raise ValueError(f"{column} must contain numeric values or NA")
            present = numeric.dropna()
            if ((present < 0) | (present > 1)).any():
                raise ValueError(f"{column} values must lie in [0, 1]")
        ok = table["status"] == CommunicationScoreStatus.OK.value
        if table.loc[ok, "comm_strength"].isna().any():
            raise ValueError("status=ok requires a finite comm_strength")
        if table.loc[~ok, "comm_strength"].notna().any():
            raise ValueError("missing score status requires comm_strength=NA")
        if table.loc[ok, "reason_code"].notna().any():
            raise ValueError("status=ok must not carry a missing-evidence reason")
        if table.loc[~ok, "reason_code"].isna().any():
            raise ValueError("missing score status requires an explicit reason_code")
        validate_common_functional(table)
        object.__setattr__(self, "table", table)

    @property
    def score_semantics(self) -> str:
        return "strength_not_probability"

    def for_mode(self, mode: str) -> pd.DataFrame:
        """Return an independent copy of state or ecosystem score rows."""

        if mode not in {"state", "ecosystem"}:
            raise ValueError("mode must be 'state' or 'ecosystem'")
        return self.table.loc[self.table["mode"] == mode].copy()
