import pytest

from crychic.core import ContractError, canonical_json, stable_id


def test_stable_id_is_mapping_order_invariant() -> None:
    first = stable_id(
        "context",
        {"region": "core", "treatment": "treated"},
        schema_version="1",
    )
    second = stable_id(
        "context",
        {"treatment": "treated", "region": "core"},
        schema_version="1",
    )

    assert first == second
    assert first.startswith("context_")


def test_stable_id_changes_with_semantic_components() -> None:
    first = stable_id("fold", {"repeat": 1, "fold": 1})
    second = stable_id("fold", {"repeat": 1, "fold": 2})

    assert first != second


def test_stable_id_rejects_invalid_namespace() -> None:
    with pytest.raises(ContractError, match="lowercase ASCII"):
        stable_id("Context", {"condition": "control"})


def test_canonical_json_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"value": float("nan")})
