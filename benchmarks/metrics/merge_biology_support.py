"""Strictly merge versioned supportive-biology evidence tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

BIOLOGY_IDENTITY_COLUMNS = (
    "dataset",
    "observation_id",
    "method",
    "method_version",
    "analysis_track",
    "resource",
    "resource_version",
    "resource_mode",
    "score_semantics",
    "universe_id",
    "rank_scope",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source_list(path: str | Path) -> tuple[Path, ...]:
    """Resolve non-comment entries relative to a source-list file."""
    source_list = Path(path).resolve()
    sources: list[Path] = []
    for raw_line in source_list.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        sources.append(
            candidate.resolve()
            if candidate.is_absolute()
            else (source_list.parent / candidate).resolve()
        )
    return tuple(sources)


def merge_biology_support(
    sources: Sequence[str | Path], output_path: str | Path
) -> pd.DataFrame:
    """Merge schema-identical evidence tables and reject identity conflicts."""
    resolved = tuple(Path(source).resolve() for source in sources)
    if not resolved:
        raise ValueError("at least one biology-support source is required")

    tables: list[pd.DataFrame] = []
    expected_columns: tuple[str, ...] | None = None
    for source in resolved:
        if not source.is_file():
            raise FileNotFoundError(source)
        table = pd.read_csv(source, sep="\t")
        if "rank_scope" not in table:
            insertion = (
                table.columns.get_loc("evidence_class")
                if "evidence_class" in table
                else len(table.columns)
            )
            table.insert(insertion, "rank_scope", "global_common_functional")
        columns = tuple(map(str, table.columns))
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(
                "biology-support schemas must match exactly, including column order: "
                f"{source}"
            )
        missing = set(BIOLOGY_IDENTITY_COLUMNS).difference(columns)
        if missing:
            raise ValueError(
                f"biology-support source is missing identity columns: {sorted(missing)}"
            )
        if "source_biology_file" not in columns:
            raise ValueError("biology-support sources must retain source_biology_file")
        if table.loc[:, list(BIOLOGY_IDENTITY_COLUMNS)].isna().any().any():
            raise ValueError("biology-support identity columns must not be missing")
        tables.append(table)

    merged = pd.concat(tables, ignore_index=True, sort=False).drop_duplicates(
        ignore_index=True
    )
    duplicate = merged.duplicated(list(BIOLOGY_IDENTITY_COLUMNS), keep=False)
    if duplicate.any():
        conflicts = merged.loc[duplicate, list(BIOLOGY_IDENTITY_COLUMNS)]
        examples = conflicts.drop_duplicates().head(5).to_dict(orient="records")
        raise ValueError(
            "biology-support identity conflicts remain after exact-row deduplication: "
            f"{examples}"
        )

    merged = merged.sort_values(
        list(BIOLOGY_IDENTITY_COLUMNS), kind="stable", ignore_index=True
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        merged.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_tsv", type=Path)
    parser.add_argument("sources", nargs="*", type=Path)
    parser.add_argument("--source-list", action="append", type=Path, default=[])
    return parser


def main() -> None:
    args = _parser().parse_args()
    sources = [Path(source).resolve() for source in args.sources]
    for source_list in args.source_list:
        sources.extend(read_source_list(source_list))
    merged = merge_biology_support(sources, args.output_tsv)
    output = args.output_tsv.resolve()
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "rows": len(merged),
                "sha256": _sha256(output),
                "sources": [source.as_posix() for source in sources],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
