from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from crychic.core._validation import (
    isolated_validation_scope,
    record_validation,
    validation_is_cached,
    validation_scope,
)


def test_validation_scope_is_nested_and_resets_after_exit() -> None:
    artifact = object()

    assert not validation_is_cached(artifact)
    with validation_scope():
        record_validation(artifact)
        assert validation_is_cached(artifact)
        with validation_scope():
            assert validation_is_cached(artifact)
    assert not validation_is_cached(artifact)


def test_validation_scope_resets_after_exception() -> None:
    artifact = object()

    with pytest.raises(RuntimeError, match="stop"):
        with validation_scope():
            record_validation(artifact)
            raise RuntimeError("stop")
    assert not validation_is_cached(artifact)


def test_validation_scope_is_context_local_across_threads() -> None:
    artifact = object()

    with validation_scope():
        record_validation(artifact)
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(validation_is_cached, artifact).result() is False
        assert validation_is_cached(artifact)


def test_isolated_validation_scope_does_not_reuse_parent_cache() -> None:
    artifact = object()

    with validation_scope():
        record_validation(artifact)
        assert validation_is_cached(artifact)
        with isolated_validation_scope():
            assert not validation_is_cached(artifact)
            record_validation(artifact)
            assert validation_is_cached(artifact)
        assert validation_is_cached(artifact)
