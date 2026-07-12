"""Stable identifiers that do not depend on Python hash randomization."""

from __future__ import annotations

import re

from .errors import ContractError
from .serialization import canonical_digest

_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def stable_id(
    kind: str,
    components: object,
    *,
    schema_version: str = "1",
    digest_length: int = 32,
) -> str:
    """Build a deterministic namespaced ID from canonical components."""

    if not isinstance(kind, str) or _KIND_PATTERN.fullmatch(kind) is None:
        raise ContractError(
            "Stable ID kind must use lowercase ASCII letters, digits, and underscores",
            code="invalid_id_kind",
            field="kind",
            remediation="Use a value such as context or interaction_family",
        )
    if not isinstance(schema_version, str) or not schema_version:
        raise ContractError(
            "Stable ID schema version cannot be empty",
            code="invalid_id_schema_version",
            field="schema_version",
            remediation="Provide the owning ID schema version",
        )
    if not 16 <= digest_length <= 64:
        raise ContractError(
            "Stable ID digest length must be between 16 and 64 hex characters",
            code="invalid_id_digest_length",
            field="digest_length",
            remediation="Use the default 32-character digest",
        )
    payload = {
        "components": components,
        "kind": kind,
        "schema_version": schema_version,
    }
    return f"{kind}_{canonical_digest(payload)[:digest_length]}"
