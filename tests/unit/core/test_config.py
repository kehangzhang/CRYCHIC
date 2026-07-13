from dataclasses import FrozenInstanceError

import pytest

from crychic.core import (
    CommunicationMode,
    ConfigurationError,
    CrychicConfig,
)


def test_config_normalizes_and_round_trips() -> None:
    config = CrychicConfig(
        counts_layer="counts",
        context_keys=["treatment", "region"],
        covariates=["batch", "sex"],
        categorical_covariates=["batch"],
        design="~ batch + treatment * region",
        communication_modes=["state", CommunicationMode.ECOSYSTEM],
        random_seed=17,
    )

    assert config.context_keys == ("treatment", "region")
    assert config.covariates == ("batch", "sex")
    assert config.categorical_covariates == ("batch",)
    assert config.to_dict()["categorical_covariates"] == ["batch"]
    assert config.communication_modes == (
        CommunicationMode.STATE,
        CommunicationMode.ECOSYSTEM,
    )
    assert CrychicConfig.from_dict(config.to_dict()) == config
    assert CrychicConfig.from_json(config.to_json()) == config
    assert len(config.digest) == 64


def test_config_is_frozen() -> None:
    config = CrychicConfig(context_keys=["condition"])

    with pytest.raises(FrozenInstanceError):
        config.random_seed = 2  # type: ignore[misc]


def test_normalized_only_input_requires_explicit_transform() -> None:
    with pytest.raises(ConfigurationError, match="declared together"):
        CrychicConfig(
            counts_layer=None,
            context_keys=["condition"],
            expression_source="X",
        )

    config = CrychicConfig(
        counts_layer=None,
        context_keys=["condition"],
        expression_source="X",
        expression_transform="log1p_normalized",
    )

    assert config.normalized_only
    assert config.expression_transform == "log1p_normalized"


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "kwargs",
    [
        {"context_keys": []},
        {"context_keys": ["condition", "condition"]},
        {"context_keys": ["sample_id"]},
        {"context_keys": ["condition"], "covariates": ["condition"]},
        {
            "context_keys": ["condition"],
            "covariates": ["age"],
            "categorical_covariates": ["batch"],
        },
        {"context_keys": ["condition"], "communication_modes": []},
        {"context_keys": ["condition"], "random_seed": -1},
        {"context_keys": ["condition"], "random_seed": 1.5},
        {"context_keys": ["condition"], "design": 7},
    ],
)
def test_config_rejects_invalid_or_conflicting_fields(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ConfigurationError):
        CrychicConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_unknown_persisted_fields() -> None:
    value = CrychicConfig(context_keys=["condition"]).to_dict()
    value["surprise"] = True

    with pytest.raises(ConfigurationError, match="unknown fields"):
        CrychicConfig.from_dict(value)


def test_digest_is_independent_of_mapping_insertion_order() -> None:
    config = CrychicConfig(
        context_keys=["condition"],
        covariates=["batch"],
    )
    value = config.to_dict()
    reversed_value = dict(reversed(tuple(value.items())))

    assert CrychicConfig.from_dict(reversed_value).digest == config.digest


def test_categorical_covariate_declaration_is_persisted_and_hashed() -> None:
    categorical = CrychicConfig(
        context_keys=["condition"],
        covariates=["batch_code"],
        categorical_covariates=["batch_code"],
    )
    continuous = CrychicConfig(
        context_keys=["condition"],
        covariates=["batch_code"],
    )

    restored = CrychicConfig.from_json(categorical.to_json())

    assert restored.categorical_covariates == ("batch_code",)
    assert restored.to_dict()["categorical_covariates"] == ["batch_code"]
    assert restored.digest == categorical.digest
    assert categorical.digest != continuous.digest


def test_config_accepts_legacy_payload_without_categorical_covariates() -> None:
    value = CrychicConfig(context_keys=["condition"]).to_dict()
    del value["categorical_covariates"]

    restored = CrychicConfig.from_dict(value)

    assert restored.categorical_covariates == ()
