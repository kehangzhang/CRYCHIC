from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from crychic import (
    AnalysisProfile,
    CrossFitSpec,
    Crychic,
    CrychicConfig,
    FoldTrainingSpec,
    FrozenLatentNuisanceSpec,
    recommended_crossfit_spec,
)
from crychic.design import ContrastSpec, balanced_contrast
from crychic.resources import GeneNamespace, Species
from crychic.response import build_receiver_autonomous_program_resource


def _adata() -> AnnData:
    return AnnData(
        X=np.ones((1, 1), dtype=np.int64),
        obs=pd.DataFrame(
            {
                "sample_id": ["sample-1"],
                "subject_id": ["subject-1"],
                "condition": ["control"],
                "cell_type": ["Receiver"],
            },
            index=["cell-1"],
        ),
        var=pd.DataFrame(index=["G1"]),
    )


def _contrast() -> ContrastSpec:
    return balanced_contrast(
        ("treated",),
        ("control",),
        name="treated_vs_control",
    )


def _spec() -> CrossFitSpec:
    return CrossFitSpec(contrasts=(_contrast(),))


def test_recommended_builder_requires_explicit_contrasts_and_freezes_core_policy(
) -> None:
    spec = recommended_crossfit_spec(
        contrasts=(_contrast(),),
        root_seed=41,
    )

    assert len(spec.contrasts) == 1
    assert spec.training_spec.max_interactions is None
    assert spec.latent_nuisance_spec is not None
    assert spec.penalty_tuning_spec is not None
    assert spec.penalty_tuning_spec.root_seed == 41
    assert spec.outer_fold_partition_seed == 41

    with pytest.raises(ValueError, match="at least one ContrastSpec"):
        recommended_crossfit_spec(contrasts=())
    with pytest.raises(ValueError, match="max_interactions=None"):
        recommended_crossfit_spec(
            contrasts=(_contrast(),),
            training_spec=FoldTrainingSpec(max_interactions=1),
        )


def test_analyze_defaults_to_descriptive_crossfit_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Crychic(CrychicConfig(context_keys=("condition",)))
    adata = _adata()
    spec = _spec()
    marker: Any = object()
    calls: list[tuple[object, ...]] = []

    def descriptive(
        self: Crychic,
        supplied: AnnData,
        *,
        spec: CrossFitSpec,
        resource_bundle: object,
        target_prior: object,
        output_dir: str | Path | None,
    ) -> Any:
        calls.append(
            (self, supplied, spec, resource_bundle, target_prior, output_dir)
        )
        return marker

    def forbidden_legacy(*args: object, **kwargs: object) -> Any:
        raise AssertionError("default analysis must not call the legacy baseline")

    monkeypatch.setattr(Crychic, "fit_descriptive", descriptive)
    monkeypatch.setattr(Crychic, "fit", forbidden_legacy)

    observed = model.analyze(adata, spec=spec)

    assert observed is marker
    assert calls == [(model, adata, spec, None, None, None)]
    with pytest.raises(ValueError, match="requires an explicit CrossFitSpec"):
        model.analyze(adata)
    with pytest.raises(TypeError, match="CrossFitSpec"):
        model.analyze(adata, spec=object())
    with pytest.raises(ValueError, match="profile must be one of"):
        model.analyze(adata, profile="unregistered", spec=spec)


def test_legacy_profile_is_explicit_and_rejects_crossfit_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Crychic(CrychicConfig(context_keys=("condition",)))
    adata = _adata()
    marker: Any = object()
    calls: list[tuple[object, ...]] = []

    def legacy(
        self: Crychic,
        supplied: AnnData,
        *,
        resource_bundle: object,
        target_prior: object,
        output_dir: str | Path | None,
    ) -> Any:
        calls.append((self, supplied, resource_bundle, target_prior, output_dir))
        return marker

    def forbidden_crossfit(*args: object, **kwargs: object) -> Any:
        raise AssertionError("legacy_v01 must not call descriptive cross-fit")

    monkeypatch.setattr(Crychic, "fit", legacy)
    monkeypatch.setattr(Crychic, "fit_descriptive", forbidden_crossfit)

    observed = model.analyze(adata, profile=AnalysisProfile.LEGACY_V01)

    assert observed is marker
    assert calls == [(model, adata, None, None, None)]
    with pytest.raises(ValueError, match="does not accept a CrossFitSpec"):
        model.analyze(
            adata,
            profile=AnalysisProfile.LEGACY_V01,
            spec=_spec(),
        )


def test_fit_descriptive_rejects_an_untuned_spec_before_running_crossfit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Crychic(CrychicConfig(context_keys=("condition",)))
    untuned = CrossFitSpec(
        contrasts=(_contrast(),),
        outer_fold_partition_seed=7,
        latent_nuisance_spec=FrozenLatentNuisanceSpec(),
    )

    def forbidden_crossfit(*args: object, **kwargs: object) -> Any:
        raise AssertionError("an invalid recommended policy must not run cross-fit")

    monkeypatch.setattr(Crychic, "fit_crossfit", forbidden_crossfit)

    with pytest.raises(ValueError, match="PenaltyTuningSpec"):
        model.fit_descriptive(_adata(), spec=untuned)


def test_fit_descriptive_requires_an_explicit_outer_partition_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Crychic(CrychicConfig(context_keys=("condition",)))
    missing_seed = replace(
        recommended_crossfit_spec(contrasts=(_contrast(),), root_seed=7),
        outer_fold_partition_seed=None,
    )

    def forbidden_crossfit(*args: object, **kwargs: object) -> Any:
        raise AssertionError("an invalid recommended policy must not run cross-fit")

    monkeypatch.setattr(Crychic, "fit_crossfit", forbidden_crossfit)

    with pytest.raises(ValueError, match="outer fold partition seed"):
        model.fit_descriptive(_adata(), spec=missing_seed)


def test_fit_descriptive_rejects_an_interaction_cap_before_crossfit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Crychic(CrychicConfig(context_keys=("condition",)))
    base = recommended_crossfit_spec(contrasts=(_contrast(),), root_seed=7)
    capped = replace(
        base,
        training_spec=replace(base.training_spec, max_interactions=1),
    )

    def forbidden_crossfit(*args: object, **kwargs: object) -> Any:
        raise AssertionError("an invalid recommended policy must not run cross-fit")

    monkeypatch.setattr(Crychic, "fit_crossfit", forbidden_crossfit)

    with pytest.raises(ValueError, match="max_interactions=None"):
        model.fit_descriptive(_adata(), spec=capped)


def test_fit_descriptive_requires_a_trusted_nuisance_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Crychic(CrychicConfig(context_keys=("condition",)))
    missing_nuisance = replace(
        recommended_crossfit_spec(contrasts=(_contrast(),), root_seed=7),
        latent_nuisance_spec=None,
    )

    def forbidden_crossfit(*args: object, **kwargs: object) -> Any:
        raise AssertionError("an invalid recommended policy must not run cross-fit")

    monkeypatch.setattr(Crychic, "fit_crossfit", forbidden_crossfit)

    with pytest.raises(ValueError, match="FrozenLatentNuisanceSpec or a manifest"):
        model.fit_descriptive(_adata(), spec=missing_nuisance)


def test_fit_descriptive_rejects_latent_plus_untrusted_static_nuisance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Crychic(CrychicConfig(context_keys=("condition",)))
    untrusted = build_receiver_autonomous_program_resource(
        np.asarray([[1.0]], dtype=np.float64),
        feature_ids=("G1",),
        program_ids=("generic_program",),
        resource_id="caller-declared-autonomous-program",
        version="1",
        manifest_digest="a" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    hybrid = replace(
        recommended_crossfit_spec(contrasts=(_contrast(),), root_seed=7),
        autonomous_program_resource=untrusted,
    )

    def forbidden_crossfit(*args: object, **kwargs: object) -> Any:
        raise AssertionError("an invalid recommended policy must not run cross-fit")

    monkeypatch.setattr(Crychic, "fit_crossfit", forbidden_crossfit)

    with pytest.raises(ValueError, match="untrusted static autonomous resource"):
        model.fit_descriptive(_adata(), spec=hybrid)
