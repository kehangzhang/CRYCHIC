# ADR-003: Build, version, and lock strategy

- Status: Accepted
- Date: 2026-07-12
- Implementation status: Implemented

## Context

The package needs standards-compliant source and wheel builds across Python
3.11-3.13, one authoritative package version, and reproducible developer and
release dependency resolution without making the runtime depend on a project
manager.

## Decision

Use Hatchling as the PEP 517 build backend and `hatch-vcs` to derive versions
from Git tags. There is no manually maintained version constant; source-tree
imports fall back to `0.0.0` only when distribution metadata is unavailable.
Release tags use SemVer.

Use `uv` to resolve and commit a cross-platform `uv.lock` for development and
CI. The package remains installable with standard `pip`/PEP 517 tools. Release
or isolated benchmark environments may export pinned constraints from the
reviewed lock and must record the environment digest. Python support is
`>=3.11,<3.14`; adding 3.14 requires its own compatibility CI result.

## Alternatives considered

- Setuptools is mature but offers no advantage for this src-layout skeleton
  and requires more configuration for VCS versioning.
- A handwritten version creates two sources of truth.
- Unpinned optional dependencies make statistical and numerical reproduction
  fragile. Platform-specific frozen requirements are difficult to maintain as
  the sole source.

## Consequences

Building from an untagged checkout produces a development version. Runtime has
no dependency on Hatch or uv. Lock updates are reviewed changes and must note
numerical, schema, and benchmark effects when relevant.

## Statistical and compatibility impact

The build choice changes no estimand. Dependency updates can change numerical
behavior and therefore require regression or statistical calibration checks
proportional to the affected backend.

## Migration and rollback

PEP 517 metadata keeps a future backend migration possible without changing
the import package. Rollback pins the last reviewed lock and build backend;
released versions are never rewritten or recreated from a different lock.
