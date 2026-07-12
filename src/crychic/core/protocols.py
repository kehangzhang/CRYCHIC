"""Generic protocols shared by lightweight contracts."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsCanonicalDict(Protocol):
    """Protocol for objects that expose a canonicalizable mapping."""

    def to_dict(self) -> Mapping[str, object]:
        """Return the complete public state of the object."""
