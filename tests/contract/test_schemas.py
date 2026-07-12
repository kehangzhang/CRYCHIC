import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from crychic.core import CrychicConfig, RunProvenance, SeedLineage

ROOT = Path(__file__).parents[2]


def _load_schema(name: str) -> dict[str, object]:
    with (ROOT / "schemas" / name).open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def test_config_schema_accepts_runtime_contract() -> None:
    schema = _load_schema("config.schema.json")
    Draft202012Validator.check_schema(schema)
    config = CrychicConfig(
        context_keys=["treatment", "region"],
        covariates=["batch"],
        design="~ batch + treatment * region",
    )

    Draft202012Validator(schema).validate(config.to_dict())


def test_config_schema_rejects_incomplete_normalized_input() -> None:
    schema = _load_schema("config.schema.json")
    value = CrychicConfig(context_keys=["condition"]).to_dict()
    value["counts_layer"] = None

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(value)


def test_provenance_schema_accepts_runtime_contract() -> None:
    schema = _load_schema("provenance.schema.json")
    Draft202012Validator.check_schema(schema)
    config = CrychicConfig(context_keys=["condition"])
    provenance = RunProvenance(
        package_version="0.0.0",
        git_commit="ccc6795",
        config_digest=config.digest,
        seed_lineage=SeedLineage(config.random_seed),
        created_at="2026-07-12T00:00:00Z",
    )

    Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    ).validate(provenance.to_dict())
