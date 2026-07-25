"""Bind uncertainty-aware M5 shrinkage to design-aware continuous effects."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crychic.core import stable_id
from crychic.scoring import (
    FrozenHypergraphPrior,
    UncertaintyAwareHypergraphShrinkageV2Fit,
    UncertaintyAwareHypergraphShrinkageV2Spec,
    fit_uncertainty_aware_hypergraph_shrinkage_v2,
)

from .design_aware import DesignAwareDifferentialResult


@dataclass(frozen=True, slots=True, kw_only=True)
class DesignAwareHypergraphShrinkageV2Result:
    """M5 table bound to one raw-effect design and contrast lineage."""

    shrinkage: pd.DataFrame
    fit: UncertaintyAwareHypergraphShrinkageV2Fit
    design_spec_id: str
    contrast_name: str
    source_effect_ids: tuple[str, ...]
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fit, UncertaintyAwareHypergraphShrinkageV2Fit):
            raise TypeError("fit must be UncertaintyAwareHypergraphShrinkageV2Fit")
        for field_name in ("design_spec_id", "contrast_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} must be a canonical identifier")
        effect_ids = tuple(sorted(self.source_effect_ids))
        if not effect_ids or len(effect_ids) != len(set(effect_ids)):
            raise ValueError("source_effect_ids must be non-empty and unique")
        if len(effect_ids) != len(self.shrinkage):
            raise ValueError("M5 source effects and shrinkage rows differ")
        if self.shrinkage["formal_inference_allowed"].any():
            raise ValueError("M5 shrinkage cannot claim formal inference")
        object.__setattr__(self, "source_effect_ids", effect_ids)
        object.__setattr__(self, "shrinkage", self.shrinkage.copy(deep=True))
        object.__setattr__(
            self,
            "result_id",
            stable_id(
                "design_aware_hypergraph_shrinkage_v2",
                {
                    "contrast_name": self.contrast_name,
                    "design_spec_id": self.design_spec_id,
                    "fit_id": self.fit.fit_id,
                    "source_effect_ids": list(effect_ids),
                },
                schema_version="2",
            ),
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "design_spec_id": self.design_spec_id,
            "contrast_name": self.contrast_name,
            "source_effect_ids": list(self.source_effect_ids),
            "fit": self.fit.to_dict(),
            "effect_rows": len(self.shrinkage),
            "formal_inference_allowed": False,
            "required_formal_next_stage": "full_pipeline_subject_resampling",
        }


def fit_design_aware_hypergraph_shrinkage_v2(
    differential: DesignAwareDifferentialResult,
    *,
    prior: FrozenHypergraphPrior,
    contrast_name: str,
    spec: UncertaintyAwareHypergraphShrinkageV2Spec | None = None,
) -> DesignAwareHypergraphShrinkageV2Result:
    """Apply M5 to one exact design-aware contrast without creating p/q values."""

    if not isinstance(differential, DesignAwareDifferentialResult):
        raise TypeError("differential must be DesignAwareDifferentialResult")
    if not isinstance(prior, FrozenHypergraphPrior):
        raise TypeError("prior must be FrozenHypergraphPrior")
    if (
        not isinstance(contrast_name, str)
        or not contrast_name
        or contrast_name != contrast_name.strip()
    ):
        raise ValueError("contrast_name must be canonical")
    selected = differential.effects.loc[
        differential.effects["contrast_name"].eq(contrast_name)
    ].copy()
    if selected.empty:
        raise ValueError("design-aware result does not contain the requested contrast")
    if selected["event_id"].duplicated().any():
        raise ValueError("design-aware contrast must contain one effect per event")
    prior_ids = tuple(edge.edge_id for edge in prior.edges)
    selected = selected.sort_values("event_id", kind="stable", ignore_index=True)
    if tuple(selected["event_id"].astype(str)) != prior_ids:
        raise ValueError("design-aware event universe differs from frozen H_prior")
    estimates = selected.loc[:, ["event_id", "effect", "standard_error"]].rename(
        columns={"event_id": "edge_id"}
    )
    shrinkage, fit = fit_uncertainty_aware_hypergraph_shrinkage_v2(
        estimates,
        prior=prior,
        spec=spec,
    )
    return DesignAwareHypergraphShrinkageV2Result(
        shrinkage=shrinkage,
        fit=fit,
        design_spec_id=differential.spec.spec_id,
        contrast_name=contrast_name,
        source_effect_ids=tuple(selected["effect_id"].astype(str)),
    )


__all__ = [
    "DesignAwareHypergraphShrinkageV2Result",
    "fit_design_aware_hypergraph_shrinkage_v2",
]
