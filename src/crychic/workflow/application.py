"""Frozen application boundary for partial fold training artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from anndata import AnnData

from crychic.availability import (
    BatchAvailability,
    InteractionFilterApplication,
    estimate_bundle_availability,
)
from crychic.sender import (
    CommonSenderApplication,
    apply_contrast_common_sender_functional,
)

from .training import TrainingArtifacts, _prepare_raw_fold

_APPLICATION_STATUS = "frozen_application_partial_not_oof"


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingArtifactApplication:
    """Held-out frozen-universe availability without an OOF scoring claim."""

    training_artifact_id: str
    heldout_subject_ids: tuple[str, ...]
    heldout_sample_ids: tuple[str, ...]
    heldout_input_digest: str
    availability: BatchAvailability
    sender_assignments: tuple[CommonSenderApplication, ...]
    application_status: str = _APPLICATION_STATUS

    def __post_init__(self) -> None:
        if not self.training_artifact_id:
            raise ValueError("training_artifact_id must not be empty")
        if not self.heldout_subject_ids or not self.heldout_sample_ids:
            raise ValueError("held-out subject and sample IDs must not be empty")
        if not self.heldout_input_digest:
            raise ValueError("heldout_input_digest must not be empty")
        if self.availability.filter_application is not (
            InteractionFilterApplication.FROZEN_APPLICATION_V1
        ):
            raise ValueError("application availability must use a frozen universe")
        if self.application_status != _APPLICATION_STATUS:
            raise ValueError("partial application must not claim OOF certification")
        if any(item.is_oof_certified for item in self.sender_assignments):
            raise ValueError("partial sender application cannot claim OOF status")

    @property
    def is_oof_certified(self) -> bool:
        """Return false until all learned stages are train/apply separated."""

        return False


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
    )


__all__ = ["TrainingArtifactApplication", "apply_training_artifacts"]
