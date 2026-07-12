from collections.abc import Mapping

from crychic.core import SupportsCanonicalDict


class ExampleContract:
    def to_dict(self) -> Mapping[str, object]:
        return {"value": 1}


def test_canonical_dict_protocol_is_runtime_checkable() -> None:
    assert isinstance(ExampleContract(), SupportsCanonicalDict)
