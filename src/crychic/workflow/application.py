"""Frozen application boundary for partial fold training artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from anndata import AnnData

from crychic.availability import (
    BatchAvailability,
    InteractionFilterApplication,
    estimate_bundle_availability,
)
from crychic.core import ContractError, canonical_json, stable_id
from crychic.core._validation import (
    record_validation,
    validation_is_cached,
    validation_scope,
)
from crychic.sender import (
    CommonSenderApplication,
    apply_contrast_common_sender_functional,
)

from .training import TrainingArtifacts, _prepare_raw_fold, _PreparedRawFold

_APPLICATION_STATUS = "frozen_application_partial_not_oof"


def _table_cell_token(value: object) -> dict[str, object]:
    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "missing", "value": None}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "missing", "value": None}
        if not math.isfinite(value):
            raise ValueError("application tables cannot contain infinite values")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": canonical_json(value),
    }


def _table_digest(table_name: str, table: pd.DataFrame) -> str:
    validation_id = stable_id(
        "training_application_table",
        {},
        schema_version="1",
        digest_length=64,
    )
    id_prefix = validation_id[:-(64 + 1)]
    template = canonical_json(
        {
            "components": {
                "columns": [str(column) for column in table.columns],
                "rows": [],
                "table_name": table_name,
            },
            "kind": "training_application_table",
            "schema_version": "1",
        }
    )
    rows_marker = '"rows":[]'
    prefix, suffix = template.split(rows_marker, maxsplit=1)

    def encoded_rows():
        for row in table.itertuples(index=False, name=None):
            yield json.dumps(
                [_table_cell_token(value) for value in row],
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )

    def digest_rows(rows) -> str:
        digest = hashlib.sha256()
        digest.update(prefix.encode("ascii"))
        digest.update(b'"rows":[')
        for index, row in enumerate(rows):
            if index:
                digest.update(b",")
            digest.update(row.encode("ascii"))
        digest.update(b"]")
        digest.update(suffix.encode("ascii"))
        return f"{id_prefix}_{digest.hexdigest()}"

    digest = hashlib.sha256()
    digest.update(prefix.encode("ascii"))
    digest.update(b'"rows":[')
    previous: str | None = None
    row_count = 0
    for row in encoded_rows():
        if previous is not None and row < previous:
            break
        if row_count:
            digest.update(b",")
        digest.update(row.encode("ascii"))
        previous = row
        row_count += 1
    else:
        digest.update(b"]")
        digest.update(suffix.encode("ascii"))
        return f"{id_prefix}_{digest.hexdigest()}"
    return digest_rows(sorted(encoded_rows()))


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingArtifactApplication:
    """Held-out frozen-universe availability without an OOF scoring claim."""

    training_artifact_id: str
    heldout_subject_ids: tuple[str, ...]
    heldout_sample_ids: tuple[str, ...]
    heldout_input_digest: str
    availability: BatchAvailability
    sender_assignments: tuple[CommonSenderApplication, ...]
    excluded_cell_type_ids: tuple[str, ...] = ()
    application_status: str = _APPLICATION_STATUS
    availability_mapping_summary_digest: str = field(init=False)
    availability_sample_interactions_digest: str = field(init=False)
    sender_assignment_table_digests: tuple[tuple[str, str], ...] = field(init=False)
    application_id: str = field(init=False)

    @validation_scope()
    def __post_init__(self) -> None:
        if validation_is_cached(self):
            return
        if not self.training_artifact_id:
            raise ValueError("training_artifact_id must not be empty")
        subjects = tuple(sorted(self.heldout_subject_ids))
        samples = tuple(sorted(self.heldout_sample_ids))
        if (
            not subjects
            or not samples
            or len(subjects) != len(set(subjects))
            or len(samples) != len(set(samples))
        ):
            raise ValueError("held-out subject and sample IDs must not be empty")
        if not self.heldout_input_digest:
            raise ValueError("heldout_input_digest must not be empty")
        if not isinstance(self.availability, BatchAvailability):
            raise TypeError("availability must be a BatchAvailability")
        self.availability.frozen_interaction_universe._require_intact()
        availability = BatchAvailability(
            sample_interactions=self.availability.sample_interactions.copy(deep=True),
            mapping_summary=self.availability.mapping_summary.copy(deep=True),
            resource_id=self.availability.resource_id,
            resource_version=self.availability.resource_version,
            detection_available=self.availability.detection_available,
            frozen_interaction_universe=self.availability.frozen_interaction_universe,
            filter_application=self.availability.filter_application,
            application_subject_ids=self.availability.application_subject_ids,
        )
        if availability.filter_application is not (
            InteractionFilterApplication.FROZEN_APPLICATION_V1
        ):
            raise ValueError("application availability must use a frozen universe")
        if availability.application_subject_ids != subjects:
            raise ValueError("availability subjects must match held-out subjects")
        observed_samples = set(
            availability.sample_interactions.get(
                "sample_id", pd.Series(dtype="object")
            ).astype(str)
        )
        if not observed_samples.issubset(samples):
            raise ValueError("availability rows contain unknown held-out samples")
        if self.application_status != _APPLICATION_STATUS:
            raise ValueError("partial application must not claim OOF certification")
        excluded_cell_types = tuple(sorted(self.excluded_cell_type_ids))
        if len(excluded_cell_types) != len(set(excluded_cell_types)):
            raise ValueError("excluded_cell_type_ids must be unique")
        assignments: list[CommonSenderApplication] = []
        for item in self.sender_assignments:
            if not isinstance(item, CommonSenderApplication):
                raise TypeError(
                    "sender_assignments must contain CommonSenderApplication"
                )
            item._require_intact()
            validated = item
            assignment_subjects = set(validated.table["subject_id"].astype(str))
            assignment_samples = set(validated.table["sample_id"].astype(str))
            if not assignment_subjects.issubset(subjects) or not (
                assignment_samples.issubset(samples)
            ):
                raise ValueError(
                    "sender assignments contain rows outside held-out scope"
                )
            assignments.append(validated)
        frozen_assignments = tuple(
            sorted(
                assignments,
                key=lambda item: item.functional.sender_functional_id,
            )
        )
        functional_ids = tuple(
            item.functional.sender_functional_id for item in frozen_assignments
        )
        if len(set(functional_ids)) != len(frozen_assignments):
            raise ValueError("sender assignments must bind unique functionals")
        if any(item.is_oof_certified for item in frozen_assignments):
            raise ValueError("partial sender application cannot claim OOF status")
        object.__setattr__(self, "heldout_subject_ids", subjects)
        object.__setattr__(self, "heldout_sample_ids", samples)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "sender_assignments", frozen_assignments)
        object.__setattr__(self, "excluded_cell_type_ids", excluded_cell_types)
        object.__setattr__(
            self,
            "availability_mapping_summary_digest",
            _table_digest(
                "availability_mapping_summary",
                availability.mapping_summary,
            ),
        )
        object.__setattr__(
            self,
            "availability_sample_interactions_digest",
            _table_digest(
                "availability_sample_interactions",
                availability.sample_interactions,
            ),
        )
        object.__setattr__(
            self,
            "sender_assignment_table_digests",
            tuple(
                (
                    item.functional.sender_functional_id,
                    _table_digest("common_sender_application", item.table),
                )
                for item in frozen_assignments
            ),
        )
        object.__setattr__(
            self,
            "application_id",
            stable_id("training_artifact_application", self._identity_payload()),
        )
        record_validation(self)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "application_status": self.application_status,
            "availability": {
                "application_subject_ids": list(
                    self.availability.application_subject_ids
                ),
                "detection_available": self.availability.detection_available,
                "filter_application": self.availability.filter_application.value,
                "filter_universe_id": self.availability.filter_universe_id,
                "mapping_summary_digest": self.availability_mapping_summary_digest,
                "resource_id": self.availability.resource_id,
                "resource_version": self.availability.resource_version,
                "sample_interactions_digest": (
                    self.availability_sample_interactions_digest
                ),
            },
            "heldout_input_digest": self.heldout_input_digest,
            "excluded_cell_type_ids": list(self.excluded_cell_type_ids),
            "heldout_sample_ids": list(self.heldout_sample_ids),
            "heldout_subject_ids": list(self.heldout_subject_ids),
            "sender_assignments": [
                {
                    "sender_functional_id": sender_functional_id,
                    "table_digest": table_digest,
                }
                for sender_functional_id, table_digest in (
                    self.sender_assignment_table_digests
                )
            ],
            "training_artifact_id": self.training_artifact_id,
        }

    @validation_scope()
    def _require_intact(self) -> None:
        """Reject mutation of held-out availability or sender assignment children."""

        if validation_is_cached(self):
            return
        try:
            repeated = TrainingArtifactApplication(
                training_artifact_id=self.training_artifact_id,
                heldout_subject_ids=self.heldout_subject_ids,
                heldout_sample_ids=self.heldout_sample_ids,
                heldout_input_digest=self.heldout_input_digest,
                availability=self.availability,
                sender_assignments=self.sender_assignments,
                excluded_cell_type_ids=self.excluded_cell_type_ids,
                application_status=self.application_status,
            )
            valid = (
                isinstance(self.heldout_subject_ids, tuple)
                and isinstance(self.heldout_sample_ids, tuple)
                and isinstance(self.sender_assignments, tuple)
                and isinstance(self.excluded_cell_type_ids, tuple)
                and self.heldout_subject_ids == repeated.heldout_subject_ids
                and self.heldout_sample_ids == repeated.heldout_sample_ids
                and self.excluded_cell_type_ids
                == repeated.excluded_cell_type_ids
                and self.application_status == repeated.application_status
                and self.availability_mapping_summary_digest
                == repeated.availability_mapping_summary_digest
                and self.availability_sample_interactions_digest
                == repeated.availability_sample_interactions_digest
                and self.sender_assignment_table_digests
                == repeated.sender_assignment_table_digests
                and self.application_id == repeated.application_id
                and not self.is_oof_certified
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Training application failed child integrity validation",
                code="training_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact training artifact to raw held-out data",
            ) from error
        if not valid:
            raise ContractError(
                "Training application failed child integrity validation",
                code="training_application_integrity_violation",
                field="application_id",
                remediation="Reapply the intact training artifact to raw held-out data",
            )
        record_validation(self)

    @property
    def is_oof_certified(self) -> bool:
        """Return false until all learned stages are train/apply separated."""

        return False


def _apply_training_artifacts_from_prepared(
    artifacts: TrainingArtifacts,
    prepared: _PreparedRawFold,
) -> TrainingArtifactApplication:
    if not isinstance(artifacts, TrainingArtifacts):
        raise TypeError("artifacts must be TrainingArtifacts")
    artifacts._require_producer_owned()
    if not isinstance(prepared, _PreparedRawFold):
        raise TypeError("prepared must be a _PreparedRawFold")
    prepared.require_compatible(
        artifacts.config,
        min_cells=artifacts.spec.min_cells,
        cell_types=artifacts.cell_type_ids,
    )
    overlap = set(prepared.subject_ids).intersection(artifacts.training_subject_ids)
    if overlap:
        raise ValueError(
            "held-out AnnData overlaps training subjects: " + ", ".join(sorted(overlap))
        )
    availability = estimate_bundle_availability(
        prepared.aggregate,
        artifacts.resource_bundle,
        context_keys=artifacts.config.context_keys,
        parameters=artifacts.spec.availability_parameters,
        min_pooled_availability=artifacts.spec.min_pooled_availability,
        frozen_interaction_universe=artifacts.frozen_interaction_universe,
    )
    if availability.application_subject_ids != prepared.subject_ids:
        availability = BatchAvailability(
            sample_interactions=availability.sample_interactions,
            mapping_summary=availability.mapping_summary,
            resource_id=availability.resource_id,
            resource_version=availability.resource_version,
            detection_available=availability.detection_available,
            frozen_interaction_universe=availability.frozen_interaction_universe,
            filter_application=availability.filter_application,
            application_subject_ids=prepared.subject_ids,
        )
    sender_assignments = tuple(
        apply_contrast_common_sender_functional(
            functional,
            availability.sample_interactions,
        )
        for functional in artifacts.sender_functionals
    )
    return TrainingArtifactApplication(
        training_artifact_id=artifacts.training_artifact_id,
        heldout_subject_ids=prepared.subject_ids,
        heldout_sample_ids=prepared.sample_ids,
        heldout_input_digest=prepared.input_digest,
        availability=availability,
        sender_assignments=sender_assignments,
        excluded_cell_type_ids=prepared.excluded_cell_type_ids,
    )


def apply_training_artifacts(
    artifacts: TrainingArtifacts,
    heldout_adata: AnnData,
) -> TrainingArtifactApplication:
    """Apply a producer-owned frozen universe to raw held-out AnnData.

    No fitting function from :mod:`crychic.workflow.training` is called here.
    The held-out cell-type grid is restricted to the training-fold universe.
    """

    if not isinstance(artifacts, TrainingArtifacts):
        raise TypeError("artifacts must be TrainingArtifacts")
    artifacts._require_producer_owned()
    prepared = _prepare_raw_fold(
        heldout_adata,
        artifacts.config,
        min_cells=artifacts.spec.min_cells,
        cell_types=artifacts.cell_type_ids,
    )
    return _apply_training_artifacts_from_prepared(artifacts, prepared)


__all__ = ["TrainingArtifactApplication", "apply_training_artifacts"]
