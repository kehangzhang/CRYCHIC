"""Post-cross-fit orchestration for formal M4 two-part occurrence inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crychic.core import stable_id
from crychic.inference import (
    TwoPartOccurrenceV2Result,
    TwoPartOccurrenceV2Spec,
    build_sample_edge_two_part_input,
    fit_two_part_occurrence_v2,
)

from .crossfit import CrossFitArtifacts


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossFitTwoPartOccurrenceV2Result:
    """M4 result bound to one exact subject-cross-fit run."""

    crossfit_id: str
    occurrence: TwoPartOccurrenceV2Result
    fold_score_provenance_ids: tuple[str, ...]
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.crossfit_id, str)
            or not self.crossfit_id
            or self.crossfit_id != self.crossfit_id.strip()
        ):
            raise ValueError("crossfit_id must be a canonical identifier")
        if not isinstance(self.occurrence, TwoPartOccurrenceV2Result):
            raise TypeError("occurrence must be TwoPartOccurrenceV2Result")
        provenance = tuple(sorted(self.fold_score_provenance_ids))
        if not provenance or len(provenance) != len(set(provenance)):
            raise ValueError("fold score provenance IDs must be non-empty and unique")
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in provenance
        ):
            raise ValueError("fold score provenance IDs must be canonical")
        object.__setattr__(self, "fold_score_provenance_ids", provenance)
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "crossfit_two_part_occurrence_v2",
                {
                    "crossfit_id": self.crossfit_id,
                    "effect_ids": list(self.occurrence.effects["effect_id"]),
                    "fold_score_provenance_ids": list(provenance),
                    "occurrence_spec_id": self.occurrence.spec.spec_id,
                },
                schema_version="2",
            ),
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "crossfit_id": self.crossfit_id,
            "occurrence_spec": self.occurrence.spec.to_dict(),
            "fold_score_provenance_ids": list(self.fold_score_provenance_ids),
            "subject_event_rows": len(self.occurrence.subject_events),
            "effect_rows": len(self.occurrence.effects),
            "observed_effects": int(
                self.occurrence.effects["status"].eq("observed").sum()
            ),
            "formal_inference_scope": "fixed_threshold_oof_occurrence_only",
            "conditional_intensity_formal_inference_allowed": False,
        }


def fit_crossfit_two_part_occurrence_v2(
    crossfit: CrossFitArtifacts,
    spec: TwoPartOccurrenceV2Spec,
    *,
    sample_metadata: pd.DataFrame | None = None,
) -> CrossFitTwoPartOccurrenceV2Result:
    """Fit M4 from exact held-out v2 score rows across all outer folds."""

    if not isinstance(crossfit, CrossFitArtifacts):
        raise TypeError("crossfit must be CrossFitArtifacts")
    crossfit._require_intact()
    if not isinstance(spec, TwoPartOccurrenceV2Spec):
        raise TypeError("spec must be TwoPartOccurrenceV2Spec")
    parts: list[pd.DataFrame] = []
    provenance_ids: list[str] = []
    for fold in crossfit.folds:
        scores = fold.sample_edge_scores_v2
        if scores is None:
            raise ValueError("M4 requires M0 v2 scores in every cross-fit fold")
        provenance_ids.append(scores.provenance.provenance_id)
        parts.append(
            build_sample_edge_two_part_input(
                scores,
                spec=spec,
                sample_metadata=sample_metadata,
            )
        )
    activity = pd.concat(parts, ignore_index=True)
    design = spec.design
    if activity.duplicated(["event_id", design.sample_column]).any():
        raise ValueError("cross-fit M4 rows overlap across outer folds")
    observed_samples = set(activity["sample_id"].astype(str))
    expected_samples = set(crossfit.oof_sample_edge_scores_v2["sample_id"].astype(str))
    if observed_samples != expected_samples:
        raise ValueError("cross-fit M4 input does not cover the exact OOF sample axis")
    occurrence = fit_two_part_occurrence_v2(activity, spec)
    return CrossFitTwoPartOccurrenceV2Result(
        crossfit_id=crossfit.crossfit_id,
        occurrence=occurrence,
        fold_score_provenance_ids=tuple(provenance_ids),
    )


__all__ = [
    "CrossFitTwoPartOccurrenceV2Result",
    "fit_crossfit_two_part_occurrence_v2",
]
