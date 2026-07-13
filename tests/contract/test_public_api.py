import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import crychic
from crychic.core import FeatureUnavailableError


def _adata() -> AnnData:
    adata = AnnData(
        X=np.asarray([[1]], dtype=np.int64),
        obs=pd.DataFrame(
            {
                "sample_id": ["s1"],
                "subject_id": ["p1"],
                "condition": ["control"],
                "cell_type": ["A"],
            },
            index=["cell-1"],
        ),
        var=pd.DataFrame(index=["G"]),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def test_public_api_has_reviewed_v0_1_symbols() -> None:
    assert set(crychic.__all__) == {
        "AttributionSupportMethod",
        "BaselineArtifacts",
        "BaselineDryRunPlan",
        "CommunicationMode",
        "ContextGraph",
        "ContrastSpec",
        "CrossFitArtifacts",
        "CrossFitSpec",
        "Crychic",
        "CrychicConfig",
        "CrychicResult",
        "ExpressionTransform",
        "FoldTrainingSpec",
        "InputSchema",
        "ResourceBundle",
        "TargetPrior",
        "__version__",
        "input_schema_from_config",
        "load_cellchat_resource",
        "load_cellphonedb_resource",
        "load_nichenet_target_prior",
        "run_subject_crossfit",
        "validate_anndata",
    }
    assert (
        crychic.AttributionSupportMethod.RELATIVE_COEFFICIENT_V1.value
        == "gated_prior_attribution_v1"
    )
    assert (
        crychic.AttributionSupportMethod.GATED_RESPONSE_NORM_V2.value
        == "gated_response_norm_attribution_support_v2"
    )
    assert (
        crychic.AttributionSupportMethod.EXPLAINED_SHARE_V3.value
        == "explained_share_attribution_support_v3"
    )
    assert isinstance(crychic.__version__, str)
    assert not hasattr(crychic, "TopoCCC")


def test_v0_1_facade_preserves_config_and_validates_input() -> None:
    config = crychic.CrychicConfig(context_keys=["condition"])
    model = crychic.Crychic(config)

    assert model.config is config
    assert isinstance(model.input_schema, crychic.InputSchema)
    assert model.validate(_adata()).report.n_obs == 1
    with pytest.raises(TypeError, match="AnnData"):
        model.validate(object())


def test_v0_1_fit_requires_a_versioned_resource() -> None:
    model = crychic.Crychic(crychic.CrychicConfig(context_keys=["condition"]))

    with pytest.raises(FeatureUnavailableError) as fit_error:
        model.fit(_adata())
    assert fit_error.value.details.code == "resource_bundle_missing"


def test_facade_requires_typed_configuration() -> None:
    with pytest.raises(TypeError, match="CrychicConfig"):
        crychic.Crychic({})  # type: ignore[arg-type]
