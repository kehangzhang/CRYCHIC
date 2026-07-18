"""Operation-scoped validation memoization for producer-owned artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_VALIDATION_CACHE: ContextVar[dict[int, object] | None] = ContextVar(
    "crychic_producer_validation_cache",
    default=None,
)


@contextmanager
def validation_scope() -> Iterator[None]:
    """Deduplicate immutable-artifact validation within one trusted operation."""

    active_cache = _VALIDATION_CACHE.get()
    if active_cache is not None:
        yield
        return
    token = _VALIDATION_CACHE.set({})
    try:
        yield
    finally:
        _VALIDATION_CACHE.reset(token)


@contextmanager
def isolated_validation_scope() -> Iterator[None]:
    """Start a fresh cache, including after a process inherits parent context."""

    token = _VALIDATION_CACHE.set({})
    try:
        yield
    finally:
        _VALIDATION_CACHE.reset(token)


def validation_is_cached(artifact: object) -> bool:
    """Return whether this exact object was validated in the active scope."""

    cache = _VALIDATION_CACHE.get()
    return cache is not None and cache.get(id(artifact)) is artifact


def record_validation(artifact: object) -> None:
    """Retain a strong reference to one validated object for the active scope."""

    cache = _VALIDATION_CACHE.get()
    if cache is not None:
        cache[id(artifact)] = artifact
