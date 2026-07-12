"""Deterministic, explicitly named random-seed lineage."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import ContractError
from .serialization import canonical_digest

_MAX_SEED = 2**63 - 1


@dataclass(frozen=True, slots=True)
class SeedLineage:
    """A root seed and deterministic path that is independent of call order."""

    root_seed: int
    path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.root_seed, bool)
            or not isinstance(self.root_seed, int)
            or not 0 <= self.root_seed <= _MAX_SEED
        ):
            raise ContractError(
                f"Root seed must be an integer between 0 and {_MAX_SEED}",
                code="invalid_seed",
                field="root_seed",
                remediation="Provide a fixed non-negative 63-bit integer",
            )
        path = tuple(self.path)
        if any(
            not isinstance(label, str) or not label or label != label.strip()
            for label in path
        ):
            raise ContractError(
                (
                    "Seed lineage labels must be non-empty strings without outer "
                    "whitespace"
                ),
                code="invalid_seed_path",
                field="path",
                remediation="Use stable stage, fold, and resample labels",
            )
        object.__setattr__(self, "path", path)

    @property
    def seed(self) -> int:
        """Return the deterministic integer represented by this lineage."""

        if not self.path:
            return self.root_seed
        digest = canonical_digest(
            {"path": self.path, "root_seed": self.root_seed, "version": 1}
        )
        return int(digest[:16], 16) & _MAX_SEED

    def derive(self, *labels: str) -> SeedLineage:
        """Create a named child lineage without consuming mutable RNG state."""

        if not labels:
            raise ContractError(
                "At least one label is required to derive a seed",
                code="empty_seed_derivation",
                field="path",
                remediation="Name the stage, fold, or resample",
            )
        return SeedLineage(self.root_seed, self.path + tuple(labels))

    def python_random(self) -> random.Random:
        """Create an isolated standard-library random generator."""

        return random.Random(self.seed)

    def to_dict(self) -> dict[str, object]:
        """Return the persisted seed-lineage representation."""

        return {
            "root_seed": self.root_seed,
            "path": list(self.path),
            "derived_seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SeedLineage:
        """Restore and verify a persisted seed lineage."""

        root_seed = value.get("root_seed")
        path = value.get("path", ())
        if not isinstance(root_seed, int) or isinstance(root_seed, bool):
            raise ContractError(
                "Persisted seed lineage has an invalid root seed",
                code="invalid_seed",
                field="root_seed",
                remediation="Regenerate the seed manifest from a valid run",
            )
        if (
            not isinstance(path, Sequence)
            or isinstance(path, str)
            or not all(isinstance(label, str) for label in path)
        ):
            raise ContractError(
                "Persisted seed lineage has an invalid path",
                code="invalid_seed_path",
                field="path",
                remediation="Regenerate the seed manifest from a valid run",
            )
        lineage = cls(root_seed=root_seed, path=tuple(path))
        persisted_seed = value.get("derived_seed")
        if persisted_seed is not None and persisted_seed != lineage.seed:
            raise ContractError(
                "Persisted derived seed does not match its lineage",
                code="seed_lineage_mismatch",
                field="derived_seed",
                remediation="Reject the artifact and regenerate it from its root seed",
            )
        return lineage
