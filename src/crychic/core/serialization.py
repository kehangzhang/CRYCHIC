"""Canonical JSON serialization used for hashes and stable identifiers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def to_json_value(value: object) -> JSONValue:
    """Convert supported values into an unambiguous JSON-compatible tree."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical JSON does not permit non-finite floats")
        return value
    if isinstance(value, Enum):
        return to_json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Canonical JSON mappings require string keys")
            result[key] = to_json_value(item)
        return result
    if isinstance(value, Set):
        converted = [to_json_value(item) for item in value]
        return sorted(converted, key=_encoded_json_value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_value(item) for item in value]
    raise TypeError(f"Unsupported canonical JSON value: {type(value).__name__}")


def _encoded_json_value(value: JSONValue) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json(value: object) -> str:
    """Serialize a value deterministically using canonical project settings."""

    return _encoded_json_value(to_json_value(value))


def canonical_digest(value: object) -> str:
    """Return a full SHA-256 digest of a canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()
