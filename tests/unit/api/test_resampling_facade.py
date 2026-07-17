from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import crychic.api.facade as facade_module
from crychic.api import Crychic
from crychic.core import CrychicConfig, SeedLineage
from crychic.design import balanced_contrast
from crychic.resources import ResourceBundle, TargetPrior
from crychic.workflow import CrossFitSpec, RepeatedCrossFitSpec


def _adata() -> AnnData:
    adata = AnnData(
        X=np.asarray([[1]], dtype=np.int64),
        obs=pd.DataFrame(
            {
                "sample_id": ["s1"],
                "subject_id": ["p1"],
                "condition": ["control"],
                "cell_type": ["Sender"],
            },
            index=["cell-1"],
        ),
        var=pd.DataFrame(index=["G"]),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def test_resample_crossfit_forwards_bounded_parallel_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = object.__new__(ResourceBundle)
    prior = object.__new__(TargetPrior)
    config = CrychicConfig(context_keys=("condition",))
    model = Crychic(config, resource_bundle=bundle, target_prior=prior)
    adata = _adata()
    spec = CrossFitSpec(
        contrasts=(
            balanced_contrast(
                ("treated",),
                ("control",),
                name="treated_vs_control",
            ),
        ),
    )
    lineage = SeedLineage(20260717)
    expected: Any = object()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(*args: object, **kwargs: object) -> Any:
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(facade_module, "run_full_pipeline_resampling", run)

    observed = model.resample_crossfit(
        adata,
        spec=spec,
        n_bootstraps=4,
        n_permutations=3,
        strata_keys=("site",),
        immutable_covariates=("batch",),
        seed_lineage=lineage,
        retain_children=True,
        n_jobs=5,
    )

    assert observed is expected
    assert calls == [
        (
            (adata, config, bundle, prior),
            {
                "spec": spec,
                "n_bootstraps": 4,
                "n_permutations": 3,
                "strata_keys": ("site",),
                "immutable_covariates": ("batch",),
                "seed_lineage": lineage,
                "retain_children": True,
                "n_jobs": 5,
            },
        )
    ]


def test_repeated_crossfit_forwards_bounded_parallel_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = object.__new__(ResourceBundle)
    prior = object.__new__(TargetPrior)
    config = CrychicConfig(context_keys=("condition",))
    model = Crychic(config, resource_bundle=bundle, target_prior=prior)
    adata = _adata()
    base_spec = CrossFitSpec(
        contrasts=(
            balanced_contrast(
                ("treated",),
                ("control",),
                name="treated_vs_control",
            ),
        ),
    )
    spec = RepeatedCrossFitSpec(crossfit_spec=base_spec, n_repeats=3)
    expected: Any = object()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(*args: object, **kwargs: object) -> Any:
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(facade_module, "run_repeated_subject_crossfit", run)

    observed = model.fit_repeated_crossfit(adata, spec=spec, n_jobs=8)

    assert observed is expected
    assert calls == [
        (
            (adata, config, bundle, prior),
            {"spec": spec, "n_jobs": 8},
        )
    ]
