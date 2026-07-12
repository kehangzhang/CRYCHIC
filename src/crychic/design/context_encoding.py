"""One canonical context representation shared by every producer."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Protocol

import numpy as np

from crychic.core import canonical_json, stable_id


class ContextValues(Protocol):
    """Minimum row interface needed by the canonical context encoder."""

    def __getitem__(self, key: str, /) -> object: ...


def plain_context_value(value: object) -> Hashable:
    """Return a scalar context value while preserving its identity."""

    result = value.item() if isinstance(value, np.generic) else value
    if not isinstance(result, Hashable):
        raise TypeError(f"context value must be hashable, got {type(result)!r}")
    return result


def canonical_context(
    row: ContextValues, context_keys: Sequence[str]
) -> tuple[tuple[str, Hashable], ...]:
    """Return sorted context items for graph matching."""

    return tuple((key, plain_context_value(row[key])) for key in sorted(context_keys))


def context_mapping(
    row: ContextValues, context_keys: Sequence[str]
) -> dict[str, Hashable]:
    """Return the canonical persisted context mapping."""

    return dict(canonical_context(row, context_keys))


def context_fields(row: ContextValues, context_keys: Sequence[str]) -> tuple[str, str]:
    """Return the stable context ID and canonical JSON mapping."""

    payload = context_mapping(row, context_keys)
    return stable_id("context", payload), canonical_json(payload)


def context_id(row: ContextValues, context_keys: Sequence[str]) -> str:
    """Return only the shared stable context ID."""

    return context_fields(row, context_keys)[0]


def node_context_mapping(
    node: Hashable, context_keys: Sequence[str]
) -> dict[str, Hashable]:
    """Normalize a scalar or canonical graph node for persistence."""

    keys = tuple(context_keys)
    if isinstance(node, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 for item in node
    ):
        payload = {str(key): plain_context_value(value) for key, value in node}
        if set(payload) == set(keys):
            return dict(sorted(payload.items()))
    if len(keys) == 1:
        return {keys[0]: plain_context_value(node)}
    raise ValueError("multi-factor context nodes must be canonical (key, value) tuples")


def node_context_fields(node: Hashable, context_keys: Sequence[str]) -> tuple[str, str]:
    """Return stable fields for a graph context node."""

    payload = node_context_mapping(node, context_keys)
    return stable_id("context", payload), canonical_json(payload)


__all__ = [
    "ContextValues",
    "canonical_context",
    "context_fields",
    "context_id",
    "context_mapping",
    "node_context_fields",
    "node_context_mapping",
    "plain_context_value",
]
