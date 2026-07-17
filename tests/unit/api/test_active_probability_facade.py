from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import crychic.api.facade as facade_module
from crychic.api import Crychic
from crychic.core import CrychicConfig, FeatureUnavailableError
from crychic.design import balanced_contrast
from crychic.inference import ActiveProbabilitySpec
from crychic.resampling import ActiveNullSpec
from crychic.resources import ResourceBundle, TargetPrior
from crychic.workflow import CrossFitSpec


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


def _spec() -> CrossFitSpec:
    return CrossFitSpec(
        contrasts=(
            balanced_contrast(
                ("treated",),
                ("control",),
                name="treated_vs_control",
            ),
        ),
    )


def _resources() -> tuple[ResourceBundle, TargetPrior]:
    return object.__new__(ResourceBundle), object.__new__(TargetPrior)


def test_fit_active_probability_delegates_exact_runtime_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, prior = _resources()
    config = CrychicConfig(context_keys=("condition",))
    model = Crychic(config, resource_bundle=bundle, target_prior=prior)
    adata = _adata()
    crossfit_spec = _spec()
    null_spec = ActiveNullSpec()
    probability_spec = ActiveProbabilitySpec()
    pipeline_result: Any = object()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(*args: object, **kwargs: object) -> Any:
        calls.append((args, kwargs))
        return pipeline_result

    monkeypatch.setattr(facade_module, "run_active_probability_pipeline", run)

    observed = model.fit_active_probability(
        adata,
        spec=crossfit_spec,
        contrast_id_or_name="treated_vs_control",
        n_plans=200,
        n_jobs=3,
        universe_name="treated-active-edges",
        active_null_spec=null_spec,
        active_probability_spec=probability_spec,
        retain_children=True,
    )

    assert observed is pipeline_result
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (adata, config, bundle, prior)
    assert kwargs == {
        "crossfit_spec": crossfit_spec,
        "contrast_id_or_name": "treated_vs_control",
        "n_plans": 200,
        "n_jobs": 3,
        "universe_name": "treated-active-edges",
        "active_null_spec": null_spec,
        "active_probability_spec": probability_spec,
        "calibration_gate": None,
        "retain_children": True,
    }


def test_fit_active_probability_persists_only_pipeline_owned_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, prior = _resources()
    model = Crychic(
        CrychicConfig(context_keys=("condition",)),
        resource_bundle=bundle,
        target_prior=prior,
    )
    collection = object()
    distribution = object()
    universe = object()
    probability_spec = object()
    pipeline_result: Any = SimpleNamespace(
        probability_collection=collection,
        null_rerun=SimpleNamespace(distribution=distribution),
        universe=universe,
        active_probability_spec=probability_spec,
        calibration_gate=None,
    )
    persisted_result: Any = object()
    writer_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        facade_module,
        "run_active_probability_pipeline",
        lambda *args, **kwargs: pipeline_result,
    )

    def write(*args: object, **kwargs: object) -> Any:
        writer_calls.append((args, kwargs))
        return persisted_result

    monkeypatch.setattr(facade_module, "write_active_probability_result", write)
    destination = tmp_path / "active-probability-result"

    observed = model.fit_active_probability(
        _adata(),
        spec=_spec(),
        contrast_id_or_name="treated_vs_control",
        n_plans=1,
        output_dir=destination,
    )

    assert observed is persisted_result
    assert writer_calls == [
        (
            (destination,),
            {
                "collection": collection,
                "distribution": distribution,
                "universe": universe,
                "spec": probability_spec,
                "calibration_gate": None,
                "calibration_result": None,
                "replay_registry": None,
                "n_jobs": 1,
            },
        )
    ]


def test_fit_active_probability_requires_both_resources_before_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ = _resources()
    called = False

    def run(*args: object, **kwargs: object) -> Any:
        nonlocal called
        called = True
        raise AssertionError("workflow must not run")

    monkeypatch.setattr(facade_module, "run_active_probability_pipeline", run)
    missing_bundle = Crychic(CrychicConfig(context_keys=("condition",)))
    with pytest.raises(FeatureUnavailableError) as bundle_error:
        missing_bundle.fit_active_probability(
            _adata(),
            spec=_spec(),
            contrast_id_or_name="treated_vs_control",
            n_plans=1,
        )
    assert bundle_error.value.details.code == "resource_bundle_missing"

    missing_prior = Crychic(
        CrychicConfig(context_keys=("condition",)),
        resource_bundle=bundle,
    )
    with pytest.raises(FeatureUnavailableError) as prior_error:
        missing_prior.fit_active_probability(
            _adata(),
            spec=_spec(),
            contrast_id_or_name="treated_vs_control",
            n_plans=1,
        )
    assert prior_error.value.details.code == "target_prior_missing"
    assert called is False
