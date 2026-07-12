import pytest

from crychic.core import ContractError, SchemaVersion


def test_schema_version_round_trip_and_ordering() -> None:
    version = SchemaVersion.parse("0.1.2")

    assert str(version) == "0.1.2"
    assert version < SchemaVersion(0, 2, 0)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "value", ["1", "1.2", "1.x.0", "1.2.3.4"]
)
def test_schema_version_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ContractError):
        SchemaVersion.parse(value)
