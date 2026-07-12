import pytest

from crychic.core import ContractError, SeedLineage


def test_named_seed_derivation_is_deterministic_and_order_independent() -> None:
    root = SeedLineage(20260712)

    fold = root.derive("crossfit", "repeat-1", "fold-2")
    unrelated = root.derive("bootstrap", "resample-9")
    fold_again = root.derive("crossfit", "repeat-1", "fold-2")

    assert unrelated.seed != fold.seed
    assert fold_again.seed == fold.seed
    assert fold_again.path == fold.path


def test_seed_lineage_round_trip_verifies_derived_seed() -> None:
    lineage = SeedLineage(91).derive("receiver", "B-cell")

    assert SeedLineage.from_dict(lineage.to_dict()) == lineage

    invalid = lineage.to_dict()
    invalid["derived_seed"] = lineage.seed + 1
    with pytest.raises(ContractError, match="does not match"):
        SeedLineage.from_dict(invalid)


def test_python_random_is_isolated_and_reproducible() -> None:
    lineage = SeedLineage(7).derive("test")

    assert lineage.python_random().random() == lineage.python_random().random()


def test_empty_seed_derivation_is_rejected() -> None:
    with pytest.raises(ContractError, match="At least one label"):
        SeedLineage(1).derive()


def test_non_integer_seed_is_rejected() -> None:
    with pytest.raises(ContractError, match="Root seed"):
        SeedLineage(1.5)  # type: ignore[arg-type]
