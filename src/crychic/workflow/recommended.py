"""Versioned, fail-closed defaults for descriptive subject cross-fit."""

from __future__ import annotations

from collections.abc import Sequence

from crychic.attribution import PenaltyTuningSpec
from crychic.design import ContrastSpec
from crychic.scoring import FrozenLatentNuisanceSpec

from .crossfit import CrossFitSpec
from .training import FoldTrainingSpec


def validate_recommended_crossfit_spec(spec: CrossFitSpec) -> CrossFitSpec:
    """Require the complete fail-closed policy used by high-level analysis."""

    if not isinstance(spec, CrossFitSpec):
        raise TypeError("spec must be a CrossFitSpec instance")
    spec._require_intact()
    if spec.training_spec.max_interactions is not None:
        raise ValueError("crossfit_descriptive_v1 requires max_interactions=None")
    if spec.outer_fold_partition_seed is None:
        raise ValueError(
            "crossfit_descriptive_v1 requires an outer fold partition seed"
        )
    if spec.penalty_tuning_spec is None:
        raise ValueError("crossfit_descriptive_v1 requires a PenaltyTuningSpec")
    autonomous = spec.autonomous_program_resource
    trusted_static = (
        autonomous is not None and autonomous.is_manifest_verified_trusted
    )
    if spec.latent_nuisance_spec is None and not trusted_static:
        raise ValueError(
            "crossfit_descriptive_v1 requires FrozenLatentNuisanceSpec or a "
            "manifest-verified static autonomous program resource"
        )
    if (
        spec.latent_nuisance_spec is not None
        and autonomous is not None
        and not trusted_static
    ):
        raise ValueError(
            "crossfit_descriptive_v1 cannot combine latent nuisance with an "
            "untrusted static autonomous resource"
        )
    return spec


def recommended_crossfit_spec(
    *,
    contrasts: Sequence[ContrastSpec],
    training_spec: FoldTrainingSpec | None = None,
    strata_keys: Sequence[str] = (),
    root_seed: int = 0,
) -> CrossFitSpec:
    """Build the descriptive cross-fit profile from explicit contrasts.

    This convenience builder never infers a contrast from observed data. It uses
    an uncapped interaction universe, nested train-fold latent nuisance learning,
    and subject-blocked relative-penalty tuning. Advanced policies should
    construct :class:`CrossFitSpec` directly.
    """

    declared_training = FoldTrainingSpec() if training_spec is None else training_spec
    declared_contrasts = tuple(contrasts)
    if not declared_contrasts:
        raise ValueError("contrasts must contain at least one ContrastSpec")
    if not isinstance(declared_training, FoldTrainingSpec):
        raise TypeError("training_spec must be a FoldTrainingSpec instance or None")
    declared_training._require_intact()
    if declared_training.max_interactions is not None:
        raise ValueError("recommended_crossfit_spec requires max_interactions=None")
    return validate_recommended_crossfit_spec(
        CrossFitSpec(
            contrasts=declared_contrasts,
            outer_fold_partition_seed=root_seed,
            training_spec=declared_training,
            strata_keys=tuple(strata_keys),
            latent_nuisance_spec=FrozenLatentNuisanceSpec(),
            penalty_tuning_spec=PenaltyTuningSpec(root_seed=root_seed),
        )
    )


__all__ = ["recommended_crossfit_spec", "validate_recommended_crossfit_spec"]
