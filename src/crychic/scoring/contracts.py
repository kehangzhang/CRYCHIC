"""Frozen scoring functionals and sample communication score contracts."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
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
SCORING_COLLECTION_LEGACY_EXTENSION_VERSION = "1.0.0"
SCORING_COLLECTION_DERIVED_REGISTRY_VERSION = "2.0.0"
SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION = "3.0.0"
SCORING_COLLECTION_EXTENSION_VERSION = "4.0.0"
SCORING_COLLECTION_SUPPORTED_VERSIONS = (
    SCORING_COLLECTION_LEGACY_EXTENSION_VERSION,
    SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
    SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
    SCORING_COLLECTION_EXTENSION_VERSION,
)
SCORING_COLLECTION_DIGEST_METHOD = "sha256_canonical_csv_v1"
SCORING_COLLECTION_AUTHORITATIVE_PLAN_STATUS = "producer_declared_complete"
SCORING_COLLECTION_DERIVED_PLAN_STATUS = (
    "derived_from_v2_collections_legacy_compatible"
)
SCORING_COLLECTION_UNAVAILABLE_PLAN_STATUS = "unavailable_legacy_v1"
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
class PlannedScoringCollectionManifest:
    """One planned contrast/repeat/fold receiver universe.

    Producer-declared entries are persisted by authoritative v3/v4 documents.
    Derived entries only expose the weaker scope recoverable from a v2 document;
    they are never serialized back as authoritative plans.
    """

    contrast: str
    repeat_id: str
    fold_id: str
    planned_receivers: tuple[str, ...]
    filter_universe_id: str | None
    provenance_status: str = SCORING_COLLECTION_AUTHORITATIVE_PLAN_STATUS
    contract_version: str = SCORING_COLLECTION_EXTENSION_VERSION
    plan_manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        contrast = _required_identifier(self.contrast, field_name="contrast")
        repeat_id = _required_identifier(self.repeat_id, field_name="repeat_id")
        fold_id = _required_identifier(self.fold_id, field_name="fold_id")
        planned_receivers = _stable_names(
            tuple(self.planned_receivers), field_name="planned_receivers"
        )
        if self.contract_version in {
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        }:
            if (
                self.provenance_status
                != SCORING_COLLECTION_AUTHORITATIVE_PLAN_STATUS
            ):
                raise ValueError(
                    "authoritative planned collections require producer-declared "
                    "provenance"
                )
            filter_universe_id: str | None = _required_identifier(
                self.filter_universe_id, field_name="filter_universe_id"
            )
        elif self.contract_version == SCORING_COLLECTION_DERIVED_REGISTRY_VERSION:
            if self.provenance_status != SCORING_COLLECTION_DERIVED_PLAN_STATUS:
                raise ValueError(
                    "v2 derived planned collections require legacy-compatible "
                    "provenance"
                )
            filter_universe_id = (
                None
                if self.filter_universe_id is None
                else _required_identifier(
                    self.filter_universe_id, field_name="filter_universe_id"
                )
            )
        else:
            raise ValueError("unsupported planned collection contract version")
        object.__setattr__(self, "contrast", contrast)
        object.__setattr__(self, "repeat_id", repeat_id)
        object.__setattr__(self, "fold_id", fold_id)
        object.__setattr__(self, "planned_receivers", planned_receivers)
        object.__setattr__(self, "filter_universe_id", filter_universe_id)
        payload = {
            "contract_version": self.contract_version,
            "contrast": contrast,
            "filter_universe_id": filter_universe_id,
            "fold_id": fold_id,
            "planned_receivers": list(planned_receivers),
            "provenance_status": self.provenance_status,
            "repeat_id": repeat_id,
        }
        object.__setattr__(
            self,
            "plan_manifest_id",
            stable_id(
                "planned_receiver_scoring_collection",
                payload,
                schema_version=self.contract_version.split(".", maxsplit=1)[0],
            ),
        )

    @property
    def scope_key(self) -> tuple[str, str, str]:
        """Return the exact contrast/repeat/fold collection key."""

        return (self.contrast, self.repeat_id, self.fold_id)

    @classmethod
    def derived_from_v2(
        cls,
        collection: ScoringCollectionManifest,
    ) -> PlannedScoringCollectionManifest:
        """Expose, without upgrading, the plan recoverable from one v2 collection."""

        if collection.contract_version != SCORING_COLLECTION_DERIVED_REGISTRY_VERSION:
            raise ValueError("derived plans require a v2 scoring collection")
        universe_ids = {
            child.filter_universe_id
            for child in collection.children
            if child.filter_universe_id is not None
        }
        filter_universe_id = (
            next(iter(universe_ids)) if len(universe_ids) == 1 else None
        )
        return cls(
            contrast=collection.contrast,
            repeat_id=collection.repeat_id,
            fold_id=collection.fold_id,
            planned_receivers=collection.planned_receivers,
            filter_universe_id=filter_universe_id,
            provenance_status=SCORING_COLLECTION_DERIVED_PLAN_STATUS,
            contract_version=SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
        )

    @classmethod
    def from_collection(
        cls,
        collection: ScoringCollectionManifest,
        *,
        filter_universe_id: str,
    ) -> PlannedScoringCollectionManifest:
        """Declare the producer-owned plan for one authoritative collection."""

        if collection.contract_version not in {
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        }:
            raise ValueError("authoritative plans require a v3/v4 scoring collection")
        return cls(
            contrast=collection.contrast,
            repeat_id=collection.repeat_id,
            fold_id=collection.fold_id,
            planned_receivers=collection.planned_receivers,
            filter_universe_id=filter_universe_id,
            contract_version=collection.contract_version,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical authoritative planned-collection representation."""

        if self.contract_version not in {
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        }:
            raise ValueError("derived v2 plans cannot be serialized as authoritative")
        return {
            "plan_entry_version": self.contract_version,
            "plan_manifest_id": self.plan_manifest_id,
            "contrast": self.contrast,
            "repeat_id": self.repeat_id,
            "fold_id": self.fold_id,
            "planned_receivers": list(self.planned_receivers),
            "filter_universe_id": self.filter_universe_id,
            "provenance_status": self.provenance_status,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> PlannedScoringCollectionManifest:
        """Parse an authoritative plan entry while verifying its stable identity."""

        expected = {
            "plan_entry_version",
            "plan_manifest_id",
            "contrast",
            "repeat_id",
            "fold_id",
            "planned_receivers",
            "filter_universe_id",
            "provenance_status",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("planned scoring collection fields are invalid")
        raw_receivers = value["planned_receivers"]
        if not isinstance(raw_receivers, Sequence) or isinstance(raw_receivers, str):
            raise ValueError("planned scoring collection receivers must be an array")
        result = cls(
            contrast=value["contrast"],  # type: ignore[arg-type]
            repeat_id=value["repeat_id"],  # type: ignore[arg-type]
            fold_id=value["fold_id"],  # type: ignore[arg-type]
            planned_receivers=cast(tuple[str, ...], tuple(raw_receivers)),
            filter_universe_id=value["filter_universe_id"],  # type: ignore[arg-type]
            provenance_status=value["provenance_status"],  # type: ignore[arg-type]
            contract_version=value["plan_entry_version"],  # type: ignore[arg-type]
        )
        if value["plan_manifest_id"] != result.plan_manifest_id:
            raise ValueError("planned scoring collection ID does not match its payload")
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class ReceiverScoringFunctionalManifest:
    """One receiver child in either the legacy or authoritative registry contract."""

    receiver: str
    scoring_functional_id: str | None
    source_score_key_digest: str | None = None
    source_score_row_count: int = 0
    provenance_status: str = "functional_metadata_not_persisted_unverified"
    contract_version: str = SCORING_COLLECTION_LEGACY_EXTENSION_VERSION
    receiver_family_model_id: str | None = None
    receiver_incremental_model_id: str | None = None
    filter_universe_id: str | None = None
    score_version: str | None = None
    functional_status: str | None = None
    registry_status: str | None = None
    emission_status: str | None = None
    reason_code: str | None = None
    receiver_training_support_id: str | None = None
    receiver_training_support_status: str | None = None
    receiver_training_support_reason_code: str | None = None
    child_manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        receiver = _required_identifier(self.receiver, field_name="receiver")
        object.__setattr__(self, "receiver", receiver)
        if self.contract_version == SCORING_COLLECTION_LEGACY_EXTENSION_VERSION:
            self._validate_legacy(receiver)
            return
        if self.contract_version not in {
            SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        }:
            raise ValueError("unsupported receiver child contract version")
        self._validate_registry(receiver)

    def _validate_legacy(self, receiver: str) -> None:
        functional_id = _required_identifier(
            self.scoring_functional_id, field_name="scoring_functional_id"
        )
        object.__setattr__(self, "scoring_functional_id", functional_id)
        if self.provenance_status != "functional_metadata_not_persisted_unverified":
            raise ValueError(
                "provenance_status must declare unverified functional metadata"
            )
        if any(
            value is not None
            for value in (
                self.receiver_family_model_id,
                self.receiver_incremental_model_id,
                self.filter_universe_id,
                self.score_version,
                self.functional_status,
                self.registry_status,
                self.emission_status,
                self.reason_code,
                self.receiver_training_support_id,
                self.receiver_training_support_status,
                self.receiver_training_support_reason_code,
            )
        ):
            raise ValueError("legacy receiver children cannot claim registry metadata")
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
        payload = {
            "receiver": receiver,
            "scoring_functional_id": functional_id,
            "provenance_status": self.provenance_status,
            "source_score_key_digest": source_digest,
            "source_score_row_count": self.source_score_row_count,
        }
        object.__setattr__(
            self,
            "child_manifest_id",
            stable_id("receiver_scoring_functional", payload),
        )

    def _validate_registry(self, receiver: str) -> None:
        if self.provenance_status != "producer_lineage_verified":
            raise ValueError(
                "registry provenance_status must be 'producer_lineage_verified'"
            )
        is_v4 = self.contract_version == SCORING_COLLECTION_EXTENSION_VERSION
        support_id: str | None = None
        support_status: str | None = None
        support_reason: str | None = None
        if is_v4:
            support_id = _required_identifier(
                self.receiver_training_support_id,
                field_name="receiver_training_support_id",
            )
            if self.receiver_training_support_status not in {
                "observed",
                "not_estimable",
            }:
                raise ValueError("receiver_training_support_status is invalid")
            support_status = self.receiver_training_support_status
            if support_status == "observed":
                if self.receiver_training_support_reason_code is not None:
                    raise ValueError(
                        "observed receiver training support cannot have a reason"
                    )
            else:
                support_reason = _required_identifier(
                    self.receiver_training_support_reason_code,
                    field_name="receiver_training_support_reason_code",
                )
                if support_reason != "receiver_absent_in_outer_training":
                    raise ValueError(
                        "not-estimable receiver training support requires the "
                        "receiver_absent_in_outer_training reason"
                    )
        elif any(
            value is not None
            for value in (
                self.receiver_training_support_id,
                self.receiver_training_support_status,
                self.receiver_training_support_reason_code,
            )
        ):
            raise ValueError("v1-v3 receiver children cannot claim v4 support lineage")

        filter_universe_id = _required_identifier(
            self.filter_universe_id, field_name="filter_universe_id"
        )
        object.__setattr__(self, "filter_universe_id", filter_universe_id)
        model_identifiers: dict[str, str | None] = {}
        for field_name in (
            "receiver_family_model_id",
            "receiver_incremental_model_id",
        ):
            raw_value = getattr(self, field_name)
            if is_v4 and support_status == "not_estimable":
                if raw_value is not None:
                    raise ValueError(
                        "an outer-training-absent receiver cannot claim model lineage"
                    )
                model_identifiers[field_name] = None
            else:
                model_identifiers[field_name] = _required_identifier(
                    raw_value, field_name=field_name
                )
            object.__setattr__(self, field_name, model_identifiers[field_name])
        if self.registry_status not in {
            "functional_registered",
            "functional_not_produced",
        }:
            raise ValueError("registry_status is invalid")
        if self.emission_status not in {"emitted", "not_emitted"}:
            raise ValueError("emission_status is invalid")
        if is_v4 and support_status == "not_estimable" and (
            self.registry_status != "functional_not_produced"
            or self.emission_status != "not_emitted"
        ):
            raise ValueError(
                "an outer-training-absent receiver must be unproduced and not emitted"
            )

        functional_id: str | None
        score_version: str | None
        functional_status: str | None
        reason_code: str | None
        if self.registry_status == "functional_registered":
            functional_id = _required_identifier(
                self.scoring_functional_id, field_name="scoring_functional_id"
            )
            score_version = _required_identifier(
                self.score_version, field_name="score_version"
            )
            if self.functional_status not in {"observed", "not_estimable"}:
                raise ValueError(
                    "registered children require an observed or not_estimable "
                    "functional_status"
                )
            functional_status = self.functional_status
            if functional_status == "observed":
                if self.reason_code is not None:
                    raise ValueError(
                        "observed functionals cannot have a reason_code"
                    )
                reason_code = None
            else:
                reason_code = _required_identifier(
                    self.reason_code, field_name="reason_code"
                )
        else:
            if any(
                value is not None
                for value in (
                    self.scoring_functional_id,
                    self.score_version,
                    self.functional_status,
                )
            ):
                raise ValueError(
                    "unproduced children cannot claim functional metadata"
                )
            functional_id = None
            score_version = None
            functional_status = None
            reason_code = _required_identifier(
                self.reason_code, field_name="reason_code"
            )
        if (
            is_v4
            and support_status == "not_estimable"
            and reason_code != support_reason
        ):
            raise ValueError(
                "outer-training-absent child and support reason codes must match"
            )

        if self.emission_status == "emitted":
            if self.registry_status != "functional_registered":
                raise ValueError("only registered functionals can be emitted")
            source_digest: str | None = _sha256_digest(
                self.source_score_key_digest,
                field_name="source_score_key_digest",
            )
            if (
                isinstance(self.source_score_row_count, bool)
                or not isinstance(self.source_score_row_count, int)
                or self.source_score_row_count <= 0
            ):
                raise ValueError(
                    "emitted children require a positive source_score_row_count"
                )
        else:
            if (
                self.source_score_key_digest is not None
                or self.source_score_row_count != 0
            ):
                raise ValueError(
                    "non-emitted children require null digest and zero source rows"
                )
            source_digest = None

        object.__setattr__(self, "scoring_functional_id", functional_id)
        object.__setattr__(self, "score_version", score_version)
        object.__setattr__(self, "functional_status", functional_status)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "receiver_training_support_id", support_id)
        object.__setattr__(self, "receiver_training_support_status", support_status)
        object.__setattr__(
            self,
            "receiver_training_support_reason_code",
            support_reason,
        )
        object.__setattr__(self, "source_score_key_digest", source_digest)
        payload: dict[str, object] = {
            "contract_version": self.contract_version,
            "emission_status": self.emission_status,
            "filter_universe_id": filter_universe_id,
            "functional_status": functional_status,
            "provenance_status": self.provenance_status,
            "reason_code": reason_code,
            "receiver": receiver,
            "receiver_family_model_id": model_identifiers["receiver_family_model_id"],
            "receiver_incremental_model_id": model_identifiers[
                "receiver_incremental_model_id"
            ],
            "registry_status": self.registry_status,
            "score_version": score_version,
            "scoring_functional_id": functional_id,
            "source_score_key_digest": source_digest,
            "source_score_row_count": self.source_score_row_count,
        }
        if is_v4:
            payload.update(
                {
                    "receiver_training_support_id": support_id,
                    "receiver_training_support_status": support_status,
                    "receiver_training_support_reason_code": support_reason,
                }
            )
        object.__setattr__(
            self,
            "child_manifest_id",
            stable_id(
                "receiver_scoring_registry_entry",
                payload,
                schema_version=self.contract_version.split(".", maxsplit=1)[0],
            ),
        )

    @classmethod
    def registered(
        cls,
        *,
        receiver: str,
        receiver_family_model_id: str,
        receiver_incremental_model_id: str,
        filter_universe_id: str,
        scoring_functional_id: str,
        score_version: str,
        functional_status: str,
        reason_code: str | None = None,
        source_score_key_digest: str | None = None,
        source_score_row_count: int = 0,
        receiver_training_support_id: str | None = None,
        receiver_training_support_status: str | None = None,
        receiver_training_support_reason_code: str | None = None,
        contract_version: str = SCORING_COLLECTION_EXTENSION_VERSION,
    ) -> ReceiverScoringFunctionalManifest:
        """Create one producer-verified functional registry entry."""

        emitted = source_score_key_digest is not None or source_score_row_count != 0
        return cls(
            receiver=receiver,
            scoring_functional_id=scoring_functional_id,
            source_score_key_digest=source_score_key_digest,
            source_score_row_count=source_score_row_count,
            provenance_status="producer_lineage_verified",
            contract_version=contract_version,
            receiver_family_model_id=receiver_family_model_id,
            receiver_incremental_model_id=receiver_incremental_model_id,
            filter_universe_id=filter_universe_id,
            score_version=score_version,
            functional_status=functional_status,
            registry_status="functional_registered",
            emission_status="emitted" if emitted else "not_emitted",
            reason_code=reason_code,
            receiver_training_support_id=receiver_training_support_id,
            receiver_training_support_status=receiver_training_support_status,
            receiver_training_support_reason_code=(
                receiver_training_support_reason_code
            ),
        )

    @classmethod
    def not_produced(
        cls,
        *,
        receiver: str,
        receiver_family_model_id: str,
        receiver_incremental_model_id: str,
        filter_universe_id: str,
        reason_code: str,
        receiver_training_support_id: str | None = None,
        receiver_training_support_status: str | None = None,
        receiver_training_support_reason_code: str | None = None,
        contract_version: str = SCORING_COLLECTION_EXTENSION_VERSION,
    ) -> ReceiverScoringFunctionalManifest:
        """Register a planned receiver whose scoring functional was not produced."""

        return cls(
            receiver=receiver,
            scoring_functional_id=None,
            provenance_status="producer_lineage_verified",
            contract_version=contract_version,
            receiver_family_model_id=receiver_family_model_id,
            receiver_incremental_model_id=receiver_incremental_model_id,
            filter_universe_id=filter_universe_id,
            registry_status="functional_not_produced",
            emission_status="not_emitted",
            reason_code=reason_code,
            receiver_training_support_id=receiver_training_support_id,
            receiver_training_support_status=receiver_training_support_status,
            receiver_training_support_reason_code=(
                receiver_training_support_reason_code
            ),
        )

    @classmethod
    def training_not_estimable(
        cls,
        *,
        receiver: str,
        filter_universe_id: str,
        receiver_training_support_id: str,
        reason_code: str = "receiver_absent_in_outer_training",
    ) -> ReceiverScoringFunctionalManifest:
        """Register a v4 opportunity absent from one outer-training fold."""

        return cls(
            receiver=receiver,
            scoring_functional_id=None,
            provenance_status="producer_lineage_verified",
            contract_version=SCORING_COLLECTION_EXTENSION_VERSION,
            receiver_family_model_id=None,
            receiver_incremental_model_id=None,
            filter_universe_id=filter_universe_id,
            registry_status="functional_not_produced",
            emission_status="not_emitted",
            reason_code=reason_code,
            receiver_training_support_id=receiver_training_support_id,
            receiver_training_support_status="not_estimable",
            receiver_training_support_reason_code=reason_code,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical child-manifest representation."""

        if self.contract_version == SCORING_COLLECTION_LEGACY_EXTENSION_VERSION:
            return {
                "child_manifest_id": self.child_manifest_id,
                "receiver": self.receiver,
                "scoring_functional_id": self.scoring_functional_id,
                "provenance_status": self.provenance_status,
                "source_score_key_digest": self.source_score_key_digest,
                "source_score_row_count": self.source_score_row_count,
            }
        result: dict[str, object] = {
            "registry_entry_version": self.contract_version,
            "child_manifest_id": self.child_manifest_id,
            "receiver": self.receiver,
            "receiver_family_model_id": self.receiver_family_model_id,
            "receiver_incremental_model_id": self.receiver_incremental_model_id,
            "filter_universe_id": self.filter_universe_id,
            "scoring_functional_id": self.scoring_functional_id,
            "score_version": self.score_version,
            "functional_status": self.functional_status,
            "registry_status": self.registry_status,
            "emission_status": self.emission_status,
            "reason_code": self.reason_code,
            "provenance_status": self.provenance_status,
            "source_score_key_digest": self.source_score_key_digest,
            "source_score_row_count": self.source_score_row_count,
        }
        if self.contract_version == SCORING_COLLECTION_EXTENSION_VERSION:
            result.update(
                {
                    "receiver_training_support_id": (
                        self.receiver_training_support_id
                    ),
                    "receiver_training_support_status": (
                        self.receiver_training_support_status
                    ),
                    "receiver_training_support_reason_code": (
                        self.receiver_training_support_reason_code
                    ),
                }
            )
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> ReceiverScoringFunctionalManifest:
        """Parse a child manifest while verifying its derived stable ID."""

        legacy_expected = {
            "child_manifest_id",
            "receiver",
            "scoring_functional_id",
            "provenance_status",
            "source_score_key_digest",
            "source_score_row_count",
        }
        registry_expected = {
            "registry_entry_version",
            "child_manifest_id",
            "receiver",
            "receiver_family_model_id",
            "receiver_incremental_model_id",
            "filter_universe_id",
            "scoring_functional_id",
            "score_version",
            "functional_status",
            "registry_status",
            "emission_status",
            "reason_code",
            "provenance_status",
            "source_score_key_digest",
            "source_score_row_count",
        }
        v4_registry_expected = {
            *registry_expected,
            "receiver_training_support_id",
            "receiver_training_support_status",
            "receiver_training_support_reason_code",
        }
        if not isinstance(value, Mapping):
            raise ValueError("receiver child manifest fields are invalid")
        if set(value) == legacy_expected:
            result = cls(
                receiver=value["receiver"],  # type: ignore[arg-type]
                scoring_functional_id=value["scoring_functional_id"],  # type: ignore[arg-type]
                provenance_status=value["provenance_status"],  # type: ignore[arg-type]
                source_score_key_digest=value["source_score_key_digest"],  # type: ignore[arg-type]
                source_score_row_count=value["source_score_row_count"],  # type: ignore[arg-type]
            )
        elif (
            value.get("registry_entry_version")
            == SCORING_COLLECTION_EXTENSION_VERSION
            and set(value) == v4_registry_expected
        ) or (
            value.get("registry_entry_version")
            in {
                SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
                SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            }
            and set(value) == registry_expected
        ):
            result = cls(
                receiver=value["receiver"],  # type: ignore[arg-type]
                scoring_functional_id=value["scoring_functional_id"],  # type: ignore[arg-type]
                source_score_key_digest=value["source_score_key_digest"],  # type: ignore[arg-type]
                source_score_row_count=value["source_score_row_count"],  # type: ignore[arg-type]
                provenance_status=value["provenance_status"],  # type: ignore[arg-type]
                contract_version=value["registry_entry_version"],  # type: ignore[arg-type]
                receiver_family_model_id=value["receiver_family_model_id"],  # type: ignore[arg-type]
                receiver_incremental_model_id=value[  # type: ignore[arg-type]
                    "receiver_incremental_model_id"
                ],
                filter_universe_id=value["filter_universe_id"],  # type: ignore[arg-type]
                score_version=value["score_version"],  # type: ignore[arg-type]
                functional_status=value["functional_status"],  # type: ignore[arg-type]
                registry_status=value["registry_status"],  # type: ignore[arg-type]
                emission_status=value["emission_status"],  # type: ignore[arg-type]
                reason_code=value["reason_code"],  # type: ignore[arg-type]
                receiver_training_support_id=value.get(  # type: ignore[arg-type]
                    "receiver_training_support_id"
                ),
                receiver_training_support_status=value.get(  # type: ignore[arg-type]
                    "receiver_training_support_status"
                ),
                receiver_training_support_reason_code=value.get(  # type: ignore[arg-type]
                    "receiver_training_support_reason_code"
                ),
            )
        else:
            raise ValueError("receiver child manifest fields are invalid")
        if value["child_manifest_id"] != result.child_manifest_id:
            raise ValueError("receiver child manifest ID does not match its payload")
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoringCollectionManifest:
    """Receiver-scoped collection for one contrast, repeat, and fold."""

    contrast: str
    repeat_id: str
    fold_id: str
    emitted_receivers: tuple[str, ...]
    children: tuple[ReceiverScoringFunctionalManifest, ...]
    common_functional_across_receivers: bool = False
    partition_key: str = "receiver"
    receiver_scope: str = "emitted_sample_scores"
    composition_status: str = "partial_emitted_only"
    contract_version: str = SCORING_COLLECTION_LEGACY_EXTENSION_VERSION
    planned_receivers: tuple[str, ...] = ()
    comparability_scope: str | None = None
    row_union_policy: str | None = None
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
        children = tuple(self.children)
        if not children or any(
            not isinstance(child, ReceiverScoringFunctionalManifest)
            for child in children
        ):
            raise ValueError(
                "children must contain ReceiverScoringFunctionalManifest values"
            )
        children = tuple(sorted(children, key=lambda child: child.receiver))
        if self.contract_version == SCORING_COLLECTION_LEGACY_EXTENSION_VERSION:
            self._validate_legacy_collection(contrast, repeat_id, fold_id, children)
            return
        if self.contract_version not in {
            SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        }:
            raise ValueError("unsupported scoring collection contract version")
        self._validate_registry_collection(contrast, repeat_id, fold_id, children)

    def _set_derived(
        self,
        *,
        contrast: str,
        repeat_id: str,
        fold_id: str,
        emitted_receivers: tuple[str, ...],
        planned_receivers: tuple[str, ...],
        children: tuple[ReceiverScoringFunctionalManifest, ...],
        source_digest: str,
        source_row_count: int,
        payload: Mapping[str, object],
    ) -> None:
        object.__setattr__(self, "contrast", contrast)
        object.__setattr__(self, "repeat_id", repeat_id)
        object.__setattr__(self, "fold_id", fold_id)
        object.__setattr__(self, "emitted_receivers", emitted_receivers)
        object.__setattr__(self, "planned_receivers", planned_receivers)
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "source_score_key_digest", source_digest)
        object.__setattr__(self, "source_score_row_count", source_row_count)
        object.__setattr__(
            self,
            "scoring_collection_id",
            stable_id(
                "scoring_collection",
                dict(payload),
                schema_version=self.contract_version.split(".", maxsplit=1)[0],
            ),
        )

    def _validate_legacy_collection(
        self,
        contrast: str,
        repeat_id: str,
        fold_id: str,
        children: tuple[ReceiverScoringFunctionalManifest, ...],
    ) -> None:
        if self.receiver_scope != "emitted_sample_scores":
            raise ValueError("receiver_scope must be 'emitted_sample_scores'")
        if self.composition_status != "partial_emitted_only":
            raise ValueError("composition_status must be 'partial_emitted_only'")
        if self.planned_receivers or any(
            value is not None
            for value in (self.comparability_scope, self.row_union_policy)
        ):
            raise ValueError(
                "legacy collections cannot claim planned registry metadata"
            )
        if any(
            child.contract_version != SCORING_COLLECTION_LEGACY_EXTENSION_VERSION
            for child in children
        ):
            raise ValueError("legacy collections require legacy receiver children")
        receivers = _stable_names(
            tuple(self.emitted_receivers), field_name="emitted_receivers"
        )
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
        self._set_derived(
            contrast=contrast,
            repeat_id=repeat_id,
            fold_id=fold_id,
            emitted_receivers=receivers,
            planned_receivers=(),
            children=children,
            source_digest=source_digest,
            source_row_count=source_row_count,
            payload=payload,
        )

    def _validate_registry_collection(
        self,
        contrast: str,
        repeat_id: str,
        fold_id: str,
        children: tuple[ReceiverScoringFunctionalManifest, ...],
    ) -> None:
        expected_policy = {
            "receiver_scope": "planned_receiver_universe",
            "composition_status": "planned_receiver_exact",
            "comparability_scope": "within_receiver_across_contexts_only",
            "row_union_policy": "cross_receiver_row_union_forbidden",
        }
        for field_name, expected in expected_policy.items():
            if getattr(self, field_name) != expected:
                raise ValueError(
                    f"{field_name} must be {expected!r} for registry collections"
                )
        planned = _stable_names(
            tuple(self.planned_receivers), field_name="planned_receivers"
        )
        raw_emitted = tuple(self.emitted_receivers)
        if any(
            not isinstance(value, str) or not value.strip() for value in raw_emitted
        ):
            raise ValueError("emitted_receivers must contain non-empty strings")
        emitted = tuple(sorted(value.strip() for value in raw_emitted))
        if len(set(emitted)) != len(emitted):
            raise ValueError("emitted_receivers must contain unique values")
        if any(
            child.contract_version != self.contract_version
            for child in children
        ):
            raise ValueError(
                "registry collections require matching receiver child versions"
            )
        child_receivers = tuple(child.receiver for child in children)
        if child_receivers != planned:
            raise ValueError(
                "planned_receivers must exactly match receiver registry children"
            )
        derived_emitted = tuple(
            child.receiver
            for child in children
            if child.emission_status == "emitted"
        )
        if emitted != derived_emitted:
            raise ValueError(
                "emitted_receivers must exactly match emitted registry children"
            )
        child_ids = [child.child_manifest_id for child in children]
        if len(set(child_ids)) != len(child_ids):
            raise ValueError(
                "receiver children must have unique child_manifest_id values"
            )
        functional_ids = [
            child.scoring_functional_id
            for child in children
            if child.scoring_functional_id is not None
        ]
        if len(set(functional_ids)) != len(functional_ids):
            raise ValueError(
                "receiver children must have unique scoring_functional_id values"
            )
        emitted_children = tuple(
            child for child in children if child.emission_status == "emitted"
        )
        source_row_count = sum(
            child.source_score_row_count for child in emitted_children
        )
        source_digest = canonical_digest(
            {
                "digest_method": SCORING_COLLECTION_DIGEST_METHOD,
                "receiver_children": [
                    {
                        "receiver": child.receiver,
                        "source_score_key_digest": child.source_score_key_digest,
                        "source_score_row_count": child.source_score_row_count,
                    }
                    for child in emitted_children
                ],
            }
        )
        payload = {
            "collection_contract_version": self.contract_version,
            "common_functional_across_receivers": False,
            "composition_status": self.composition_status,
            "contrast": contrast,
            "fold_id": fold_id,
            "partition_key": "receiver",
            "receiver_scope": self.receiver_scope,
            "repeat_id": repeat_id,
            "planned_receivers": list(planned),
            "emitted_receivers": list(emitted),
            "comparability_scope": self.comparability_scope,
            "row_union_policy": self.row_union_policy,
            "children": [child.to_dict() for child in children],
            "source_score_key_digest": source_digest,
            "source_score_row_count": source_row_count,
        }
        self._set_derived(
            contrast=contrast,
            repeat_id=repeat_id,
            fold_id=fold_id,
            emitted_receivers=emitted,
            planned_receivers=planned,
            children=children,
            source_digest=source_digest,
            source_row_count=source_row_count,
            payload=payload,
        )

    @classmethod
    def planned_receiver_registry(
        cls,
        *,
        contrast: str,
        repeat_id: str,
        fold_id: str,
        planned_receivers: tuple[str, ...],
        children: tuple[ReceiverScoringFunctionalManifest, ...],
        contract_version: str = SCORING_COLLECTION_EXTENSION_VERSION,
    ) -> ScoringCollectionManifest:
        """Create an exact planned-receiver collection."""

        emitted = tuple(
            child.receiver for child in children if child.emission_status == "emitted"
        )
        return cls(
            contrast=contrast,
            repeat_id=repeat_id,
            fold_id=fold_id,
            emitted_receivers=emitted,
            planned_receivers=planned_receivers,
            children=children,
            receiver_scope="planned_receiver_universe",
            composition_status="planned_receiver_exact",
            comparability_scope="within_receiver_across_contexts_only",
            row_union_policy="cross_receiver_row_union_forbidden",
            contract_version=contract_version,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical receiver-collection representation."""

        if self.contract_version == SCORING_COLLECTION_LEGACY_EXTENSION_VERSION:
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
        return {
            "collection_contract_version": self.contract_version,
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
            "planned_receivers": list(self.planned_receivers),
            "emitted_receivers": list(self.emitted_receivers),
            "comparability_scope": self.comparability_scope,
            "row_union_policy": self.row_union_policy,
            "children": [child.to_dict() for child in self.children],
            "source_score_key_digest": self.source_score_key_digest,
            "source_score_row_count": self.source_score_row_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ScoringCollectionManifest:
        """Parse a collection while verifying all derived identities."""

        legacy_expected = {
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
        registry_expected = {
            *legacy_expected,
            "collection_contract_version",
            "planned_receivers",
            "comparability_scope",
            "row_union_policy",
        }
        if not isinstance(value, Mapping) or frozenset(value) not in {
            frozenset(legacy_expected),
            frozenset(registry_expected),
        }:
            raise ValueError("scoring collection manifest fields are invalid")
        raw_receivers = value["emitted_receivers"]
        raw_planned = value.get("planned_receivers", ())
        raw_children = value["children"]
        if (
            not isinstance(raw_receivers, Sequence)
            or isinstance(raw_receivers, str)
            or not isinstance(raw_planned, Sequence)
            or isinstance(raw_planned, str)
            or not isinstance(raw_children, Sequence)
            or isinstance(raw_children, str)
        ):
            raise ValueError("scoring collection receivers and children must be arrays")
        contract_version = value.get(
            "collection_contract_version",
            SCORING_COLLECTION_LEGACY_EXTENSION_VERSION,
        )
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
            planned_receivers=cast(tuple[str, ...], tuple(raw_planned)),
            children=tuple(
                ReceiverScoringFunctionalManifest.from_dict(child)
                for child in raw_children
                if isinstance(child, Mapping)
            ),
            comparability_scope=value.get("comparability_scope"),  # type: ignore[arg-type]
            row_union_policy=value.get("row_union_policy"),  # type: ignore[arg-type]
            contract_version=contract_version,  # type: ignore[arg-type]
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
    planned_collections: tuple[PlannedScoringCollectionManifest, ...] | None = None
    extension_schema_version: str | None = None
    result_schema_version: str = "0.1.0"
    collection_kind: str | None = None
    digest_method: str = SCORING_COLLECTION_DIGEST_METHOD
    registry_id: str | None = field(init=False)
    planning_status: str = field(init=False)
    is_authoritative_registry: bool = field(init=False)

    def __post_init__(self) -> None:
        collections = tuple(self.collections)
        if not collections or any(
            not isinstance(collection, ScoringCollectionManifest)
            for collection in collections
        ):
            raise ValueError("scoring collection document must not be empty")
        versions = {collection.contract_version for collection in collections}
        if len(versions) != 1:
            raise ValueError(
                "scoring collection documents cannot mix contract versions"
            )
        inferred_version = next(iter(versions))
        extension_version = self.extension_schema_version or inferred_version
        if extension_version not in SCORING_COLLECTION_SUPPORTED_VERSIONS:
            raise ValueError("unsupported scoring collection extension version")
        if extension_version != inferred_version:
            raise ValueError("document and collection contract versions do not match")
        if self.result_schema_version != "0.1.0":
            raise ValueError("scoring collections require result schema v0.1.0")
        expected_kind = {
            SCORING_COLLECTION_LEGACY_EXTENSION_VERSION: "receiver_partition",
            SCORING_COLLECTION_DERIVED_REGISTRY_VERSION: (
                "planned_receiver_registry"
            ),
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION: (
                "authoritative_planned_receiver_registry"
            ),
            SCORING_COLLECTION_EXTENSION_VERSION: (
                "authoritative_planned_receiver_registry"
            ),
        }[extension_version]
        collection_kind = self.collection_kind or expected_kind
        if collection_kind != expected_kind:
            raise ValueError(f"collection_kind must be {expected_kind!r}")
        if self.digest_method != SCORING_COLLECTION_DIGEST_METHOD:
            raise ValueError("unsupported scoring collection digest method")
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
        child_manifest_entries = [
            (child.child_manifest_id, child)
            for collection in collections
            for child in collection.children
        ]
        child_manifest_counts = Counter(
            child_id for child_id, _ in child_manifest_entries
        )
        duplicated_child_ids = {
            child_id for child_id, count in child_manifest_counts.items() if count > 1
        }
        if duplicated_child_ids and (
            extension_version != SCORING_COLLECTION_EXTENSION_VERSION
            or any(
                child_id in duplicated_child_ids
                and child.receiver_training_support_status != "not_estimable"
                for child_id, child in child_manifest_entries
            )
        ):
            raise ValueError("a receiver child must belong to exactly one collection")
        child_ids = [
            child.scoring_functional_id
            for collection in collections
            for child in collection.children
            if child.scoring_functional_id is not None
        ]
        if len(set(child_ids)) != len(child_ids):
            raise ValueError(
                "a receiver child functional must belong to exactly one collection"
            )
        if extension_version in {
            SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        }:
            planned_by_fold: dict[tuple[str, str], tuple[str, ...]] = {}
            for collection in collections:
                key = (collection.repeat_id, collection.fold_id)
                previous = planned_by_fold.setdefault(key, collection.planned_receivers)
                if previous != collection.planned_receivers:
                    raise ValueError(
                        "planned receiver universe must be exact across contrasts in "
                        "one repeat/fold"
                    )
        if extension_version == SCORING_COLLECTION_EXTENSION_VERSION:
            run_receiver_axis = collections[0].planned_receivers
            if any(
                collection.planned_receivers != run_receiver_axis
                for collection in collections[1:]
            ):
                raise ValueError(
                    "v4 planned receiver universe must be exact across every fold"
                )
            support_by_fold_receiver: dict[
                tuple[str, str, str], tuple[str | None, str | None, str | None]
            ] = {}
            support_scope_by_id: dict[str, tuple[str, str, str]] = {}
            for collection in collections:
                for child in collection.children:
                    support_key = (
                        collection.repeat_id,
                        collection.fold_id,
                        child.receiver,
                    )
                    support = (
                        child.receiver_training_support_id,
                        child.receiver_training_support_status,
                        child.receiver_training_support_reason_code,
                    )
                    previous_support = support_by_fold_receiver.setdefault(
                        support_key, support
                    )
                    if previous_support != support:
                        raise ValueError(
                            "v4 receiver training support must be exact across "
                            "contrasts in one repeat/fold"
                        )
                    support_id = cast(str, child.receiver_training_support_id)
                    previous_scope = support_scope_by_id.setdefault(
                        support_id, support_key
                    )
                    if previous_scope != support_key:
                        raise ValueError(
                            "v4 receiver training support IDs cannot be reused "
                            "across fold/receiver scopes"
                        )
        registry_id: str | None
        if extension_version == SCORING_COLLECTION_LEGACY_EXTENSION_VERSION:
            if self.planned_collections not in (None, ()):
                raise ValueError("v1 documents cannot claim a planned collection set")
            planned_collections: tuple[PlannedScoringCollectionManifest, ...] = ()
            planning_status = SCORING_COLLECTION_UNAVAILABLE_PLAN_STATUS
            registry_id = None
        elif extension_version == SCORING_COLLECTION_DERIVED_REGISTRY_VERSION:
            if self.planned_collections not in (None, ()):
                raise ValueError(
                    "v2 documents cannot claim producer-declared collection plans"
                )
            planned_collections = tuple(
                PlannedScoringCollectionManifest.derived_from_v2(collection)
                for collection in collections
            )
            planning_status = SCORING_COLLECTION_DERIVED_PLAN_STATUS
            registry_id = stable_id(
                "receiver_scoring_registry",
                {
                    "collection_ids": [
                        collection.scoring_collection_id for collection in collections
                    ],
                    "digest_method": self.digest_method,
                    "extension_schema_version": extension_version,
                    "result_schema_version": self.result_schema_version,
                },
                schema_version="2",
            )
        else:
            raw_plans = self.planned_collections
            if (
                raw_plans is None
                or not raw_plans
                or any(
                    not isinstance(plan, PlannedScoringCollectionManifest)
                    for plan in raw_plans
                )
            ):
                raise ValueError(
                    "authoritative documents require explicit planned collection "
                    "manifests"
                )
            planned_collections = tuple(
                sorted(raw_plans, key=lambda plan: plan.scope_key)
            )
            if any(
                plan.contract_version != extension_version
                or plan.provenance_status
                != SCORING_COLLECTION_AUTHORITATIVE_PLAN_STATUS
                for plan in planned_collections
            ):
                raise ValueError(
                    "authoritative planned collections must use the document version"
                )
            planned_keys = [plan.scope_key for plan in planned_collections]
            if len(set(planned_keys)) != len(planned_keys):
                raise ValueError(
                    "planned collections must be unique by contrast, repeat, and fold"
                )
            planned_ids = [plan.plan_manifest_id for plan in planned_collections]
            if len(set(planned_ids)) != len(planned_ids):
                raise ValueError("planned collections must have unique stable IDs")
            if set(planned_keys) != set(group_keys):
                raise ValueError(
                    "planned collections must exactly cover persisted collections"
                )
            contrasts = {plan.contrast for plan in planned_collections}
            repeat_folds = {
                (plan.repeat_id, plan.fold_id) for plan in planned_collections
            }
            expected_keys = {
                (contrast, repeat_id, fold_id)
                for contrast in contrasts
                for repeat_id, fold_id in repeat_folds
            }
            if set(planned_keys) != expected_keys:
                raise ValueError(
                    "planned collections must form the complete contrast by "
                    "repeat/fold opportunity set"
                )
            plans_by_key = {plan.scope_key: plan for plan in planned_collections}
            filter_universe_by_fold: dict[tuple[str, str], str] = {}
            for collection in collections:
                collection_key = (
                    collection.contrast,
                    collection.repeat_id,
                    collection.fold_id,
                )
                plan = plans_by_key[collection_key]
                if collection.planned_receivers != plan.planned_receivers:
                    raise ValueError(
                        "planned receiver universe does not match its collection"
                    )
                child_universes = {
                    child.filter_universe_id for child in collection.children
                }
                if child_universes != {plan.filter_universe_id}:
                    raise ValueError(
                        "receiver children must exactly match the planned filter "
                        "universe"
                    )
                fold_key = (collection.repeat_id, collection.fold_id)
                previous_universe = filter_universe_by_fold.setdefault(
                    fold_key, cast(str, plan.filter_universe_id)
                )
                if previous_universe != plan.filter_universe_id:
                    raise ValueError(
                        "filter universe must be exact across contrasts in one "
                        "repeat/fold"
                    )
            planning_status = SCORING_COLLECTION_AUTHORITATIVE_PLAN_STATUS
            registry_id = stable_id(
                "authoritative_receiver_scoring_registry",
                {
                    "collection_ids": [
                        collection.scoring_collection_id for collection in collections
                    ],
                    "digest_method": self.digest_method,
                    "extension_schema_version": extension_version,
                    "planned_collection_ids": planned_ids,
                    "planning_status": planning_status,
                    "result_schema_version": self.result_schema_version,
                },
                schema_version=extension_version.split(".", maxsplit=1)[0],
            )
        object.__setattr__(self, "extension_schema_version", extension_version)
        object.__setattr__(self, "collection_kind", collection_kind)
        object.__setattr__(self, "registry_id", registry_id)
        object.__setattr__(self, "planning_status", planning_status)
        object.__setattr__(
            self,
            "is_authoritative_registry",
            extension_version
            in {
                SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
                SCORING_COLLECTION_EXTENSION_VERSION,
            },
        )
        object.__setattr__(self, "planned_collections", planned_collections)
        object.__setattr__(self, "collections", collections)

    def to_dict(self) -> dict[str, object]:
        """Return a schema-ready versioned collection document."""

        result: dict[str, object] = {
            "extension_schema_version": self.extension_schema_version,
            "result_schema_version": self.result_schema_version,
            "collection_kind": self.collection_kind,
            "digest_method": self.digest_method,
            "collections": [collection.to_dict() for collection in self.collections],
        }
        if self.extension_schema_version in {
            SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        }:
            result["registry_id"] = self.registry_id
        if self.extension_schema_version in {
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        }:
            result["planning_status"] = self.planning_status
            result["is_authoritative_registry"] = self.is_authoritative_registry
            result["planned_collections"] = [
                plan.to_dict() for plan in cast(
                    tuple[PlannedScoringCollectionManifest, ...],
                    self.planned_collections,
                )
            ]
        return result

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
        if not isinstance(value, Mapping):
            raise ValueError("scoring collection document fields are invalid")
        version = value.get("extension_schema_version")
        if version == SCORING_COLLECTION_LEGACY_EXTENSION_VERSION:
            allowed_fields = expected
        elif version == SCORING_COLLECTION_DERIVED_REGISTRY_VERSION:
            allowed_fields = {*expected, "registry_id"}
        elif version in {
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        }:
            allowed_fields = {
                *expected,
                "registry_id",
                "planning_status",
                "is_authoritative_registry",
                "planned_collections",
            }
        else:
            raise ValueError("unsupported scoring collection extension version")
        if set(value) != allowed_fields:
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
        planned_collections: tuple[PlannedScoringCollectionManifest, ...] | None = None
        if version in {
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        }:
            raw_plans = value["planned_collections"]
            if not isinstance(raw_plans, Sequence) or isinstance(raw_plans, str):
                raise ValueError("planned collections must be an array")
            planned_collections = tuple(
                PlannedScoringCollectionManifest.from_dict(plan)
                for plan in raw_plans
                if isinstance(plan, Mapping)
            )
            if len(planned_collections) != len(raw_plans):
                raise ValueError("planned collections must be objects")
        result = cls(
            extension_schema_version=value["extension_schema_version"],  # type: ignore[arg-type]
            result_schema_version=value["result_schema_version"],  # type: ignore[arg-type]
            collection_kind=value["collection_kind"],  # type: ignore[arg-type]
            digest_method=value["digest_method"],  # type: ignore[arg-type]
            collections=collections,
            planned_collections=planned_collections,
        )
        if version in {
            SCORING_COLLECTION_DERIVED_REGISTRY_VERSION,
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        } and value["registry_id"] != result.registry_id:
            raise ValueError("receiver scoring registry ID does not match its payload")
        if version in {
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        } and (
            value["planning_status"] != result.planning_status
        ):
            raise ValueError("receiver scoring planning status is invalid")
        if version in {
            SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
            SCORING_COLLECTION_EXTENSION_VERSION,
        } and (
            value["is_authoritative_registry"] is not True
            or not result.is_authoritative_registry
        ):
            raise ValueError("receiver scoring authority status is invalid")
        return result


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
