"""Immutable facade and lazy queries over a persisted result directory."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pandas as pd

from crychic.core import (
    CrychicConfig,
    RunProvenance,
    canonical_digest,
    canonical_json,
)
from crychic.scoring.contracts import (
    ScoringCollectionDocument,
    ScoringCollectionManifest,
)

from ._schema import (
    RESULT_SCHEMA_VERSION,
    TABLE_NAMES,
    edge_evidence_contract,
    load_schema_document,
    scoring_collections_contract,
    table_contract,
    validate_scoring_collection_links,
    validate_table,
)
from .bootstrap_support import (
    BootstrapSupportDocument,
    bootstrap_support_contract,
    validate_bootstrap_support_links,
    validate_bootstrap_support_registry,
)
from .errors import IncompleteResultError, ResultValidationError
from .persistence import (
    CONFIG_FILENAME,
    MANIFEST_FILENAME,
    PROVENANCE_FILENAME,
    STATUS_FILENAME,
    read_json,
    sha256_file,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_parquet(*args: object, **kwargs: object) -> pd.DataFrame:
    """Call pandas without version-specific PyArrow passthrough keywords."""

    reader = cast(Callable[..., pd.DataFrame], pd.read_parquet)
    return reader(*args, **kwargs)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _fail(message: str, *, code: str, field_name: str) -> None:
    raise ResultValidationError(
        message,
        code=code,
        field=field_name,
        remediation="Reject the artifact and regenerate it from a complete run",
    )


def _validate_manifest(value: Mapping[str, Any]) -> None:
    required = {
        "result_schema_version",
        "run_id",
        "status",
        "mode",
        "created_at",
        "code_version",
        "config_digest",
        "workflow_parameters",
        "workflow_digest",
        "provenance_digest",
        "input_digest",
        "resource_digests",
        "tables",
        "stages",
        "warnings",
    }
    optional = {"extensions"}
    if not required.issubset(value) or set(value).difference(required) - optional:
        _fail(
            "Run manifest fields do not match the v0.1 schema",
            code="invalid_run_manifest",
            field_name="run_manifest",
        )
    if value["result_schema_version"] != RESULT_SCHEMA_VERSION:
        _fail(
            "Result schema version is not supported",
            code="unsupported_result_schema",
            field_name="result_schema_version",
        )
    if value["status"] != "complete" or value["mode"] != "exploratory":
        _fail(
            "v0.1 run manifest must be complete and exploratory",
            code="invalid_run_manifest",
            field_name="status",
        )
    for name in ("run_id", "created_at", "code_version"):
        if not isinstance(value[name], str) or not value[name]:
            _fail(
                f"Run manifest {name!r} must be non-empty",
                code="invalid_run_manifest",
                field_name=name,
            )
    try:
        created_at = datetime.fromisoformat(value["created_at"].replace("Z", "+00:00"))
    except ValueError:
        _fail(
            "Run manifest created_at must be an ISO-8601 timestamp",
            code="invalid_run_manifest",
            field_name="created_at",
        )
    if created_at.tzinfo is None:
        _fail(
            "Run manifest created_at must include a timezone",
            code="invalid_run_manifest",
            field_name="created_at",
        )
    for name in ("config_digest", "workflow_digest", "provenance_digest"):
        if not isinstance(value[name], str) or _SHA256.fullmatch(value[name]) is None:
            _fail(
                f"Run manifest {name!r} must be a SHA-256 digest",
                code="invalid_run_manifest",
                field_name=name,
            )
    workflow_parameters = value["workflow_parameters"]
    if not isinstance(workflow_parameters, Mapping):
        _fail(
            "Run manifest workflow_parameters must be an object",
            code="invalid_run_manifest",
            field_name="workflow_parameters",
        )
    try:
        workflow_digest = canonical_digest(workflow_parameters)
    except (TypeError, ValueError):
        _fail(
            "Run manifest workflow_parameters are not canonicalizable",
            code="invalid_run_manifest",
            field_name="workflow_parameters",
        )
    if workflow_digest != value["workflow_digest"]:
        _fail(
            "Run manifest workflow digest does not match its parameters",
            code="result_digest_mismatch",
            field_name="workflow_digest",
        )
    input_digest = value["input_digest"]
    if input_digest is not None and (
        not isinstance(input_digest, str) or _SHA256.fullmatch(input_digest) is None
    ):
        _fail(
            "Run manifest input_digest is invalid",
            code="invalid_run_manifest",
            field_name="input_digest",
        )
    resources = value["resource_digests"]
    if not isinstance(resources, Mapping) or any(
        not isinstance(key, str)
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        for key, digest in resources.items()
    ):
        _fail(
            "Run manifest resource digests are invalid",
            code="invalid_run_manifest",
            field_name="resource_digests",
        )
    tables = value["tables"]
    if not isinstance(tables, Mapping) or set(tables) != set(TABLE_NAMES):
        _fail(
            "Run manifest table set is incomplete",
            code="invalid_run_manifest",
            field_name="tables",
        )
    for table_name in TABLE_NAMES:
        record = tables[table_name]
        contract = table_contract(table_name)
        if not isinstance(record, Mapping) or set(record) != {
            "filename",
            "rows",
            "sha256",
            "schema",
        }:
            _fail(
                f"Run manifest table record {table_name!r} is invalid",
                code="invalid_run_manifest",
                field_name=table_name,
            )
        if (
            record["filename"] != contract.filename
            or record["schema"] != contract.schema_filename
            or not isinstance(record["rows"], int)
            or isinstance(record["rows"], bool)
            or record["rows"] < 0
            or not isinstance(record["sha256"], str)
            or _SHA256.fullmatch(record["sha256"]) is None
        ):
            _fail(
                f"Run manifest table record {table_name!r} is incompatible",
                code="invalid_run_manifest",
                field_name=table_name,
            )
    if "extensions" in value:
        extensions = value["extensions"]
        edge_contract = edge_evidence_contract()
        collection_contract = scoring_collections_contract()
        support_contract = bootstrap_support_contract()
        allowed_extensions = {
            edge_contract.name,
            collection_contract.name,
            support_contract.name,
        }
        if (
            not isinstance(extensions, Mapping)
            or not extensions
            or not set(extensions).issubset(allowed_extensions)
        ):
            _fail(
                "Run manifest result extensions are invalid",
                code="invalid_run_manifest",
                field_name="extensions",
            )
        if edge_contract.name in extensions:
            extension = extensions[edge_contract.name]
            expected_fields = {
                "extension_schema_version",
                "filename",
                "rows",
                "sha256",
                "schema",
                "linked_tables",
            }
            if not isinstance(extension, Mapping) or set(extension) != expected_fields:
                _fail(
                    "Run manifest edge-evidence extension record is invalid",
                    code="invalid_run_manifest",
                    field_name=edge_contract.name,
                )
            linked_tables = extension["linked_tables"]
            if (
                extension["extension_schema_version"]
                != edge_contract.extension_schema_version
                or extension["filename"] != edge_contract.filename
                or extension["schema"] != edge_contract.schema_filename
                or not isinstance(extension["rows"], int)
                or isinstance(extension["rows"], bool)
                or extension["rows"] < 0
                or not isinstance(extension["sha256"], str)
                or _SHA256.fullmatch(extension["sha256"]) is None
                or not isinstance(linked_tables, Mapping)
                or set(linked_tables) != set(edge_contract.linked_tables)
            ):
                _fail(
                    "Run manifest edge-evidence extension record is incompatible",
                    code="invalid_run_manifest",
                    field_name=edge_contract.name,
                )
            for linked_table in edge_contract.linked_tables:
                if linked_tables[linked_table] != tables[linked_table]["sha256"]:
                    _fail(
                        "Edge-evidence extension linkage does not match its table",
                        code="result_digest_mismatch",
                        field_name=edge_contract.name,
                    )
        if collection_contract.name in extensions:
            extension = extensions[collection_contract.name]
            expected_fields = {
                "extension_schema_version",
                "filename",
                "collections",
                "sha256",
                "schema",
                "linked_tables",
            }
            if not isinstance(extension, Mapping) or set(extension) != expected_fields:
                _fail(
                    "Run manifest scoring-collection extension record is invalid",
                    code="invalid_run_manifest",
                    field_name=collection_contract.name,
                )
            try:
                collection_contract = scoring_collections_contract(
                    str(extension["extension_schema_version"])
                )
            except ResultValidationError:
                _fail(
                    "Run manifest scoring-collection extension version is unsupported",
                    code="unsupported_result_extension",
                    field_name="extension_schema_version",
                )
            linked_tables = extension["linked_tables"]
            if (
                extension["extension_schema_version"]
                != collection_contract.extension_schema_version
                or extension["filename"] != collection_contract.filename
                or extension["schema"] != collection_contract.schema_filename
                or not isinstance(extension["collections"], int)
                or isinstance(extension["collections"], bool)
                or extension["collections"] <= 0
                or not isinstance(extension["sha256"], str)
                or _SHA256.fullmatch(extension["sha256"]) is None
                or not isinstance(linked_tables, Mapping)
                or set(linked_tables) != set(collection_contract.linked_tables)
            ):
                _fail(
                    "Run manifest scoring-collection extension is incompatible",
                    code="invalid_run_manifest",
                    field_name=collection_contract.name,
                )
            for linked_table in collection_contract.linked_tables:
                if linked_tables[linked_table] != tables[linked_table]["sha256"]:
                    _fail(
                        "Scoring-collection linkage does not match its table",
                        code="result_digest_mismatch",
                        field_name=collection_contract.name,
                    )
        if support_contract.name in extensions:
            extension = extensions[support_contract.name]
            expected_fields = {
                "extension_schema_version",
                "specificity_support",
                "selection_frequency",
                "registry",
                "linked_tables",
            }
            if not isinstance(extension, Mapping) or set(extension) != expected_fields:
                _fail(
                    "Run manifest bootstrap-support extension record is invalid",
                    code="invalid_run_manifest",
                    field_name=support_contract.name,
                )
            linked_tables = extension["linked_tables"]
            if (
                extension["extension_schema_version"]
                != support_contract.extension_schema_version
                or not isinstance(linked_tables, Mapping)
                or set(linked_tables) != set(support_contract.linked_tables)
            ):
                _fail(
                    "Run manifest bootstrap-support extension is incompatible",
                    code="invalid_run_manifest",
                    field_name=support_contract.name,
                )
            artifacts = (
                ("specificity_support", support_contract.specificity, "rows"),
                ("selection_frequency", support_contract.selection, "rows"),
            )
            for artifact_name, artifact_contract, count_field in artifacts:
                artifact = extension[artifact_name]
                if (
                    not isinstance(artifact, Mapping)
                    or set(artifact) != {"filename", count_field, "sha256", "schema"}
                    or artifact["filename"] != artifact_contract.filename
                    or artifact["schema"] != artifact_contract.schema_filename
                    or not isinstance(artifact[count_field], int)
                    or isinstance(artifact[count_field], bool)
                    or artifact[count_field] <= 0
                    or not isinstance(artifact["sha256"], str)
                    or _SHA256.fullmatch(artifact["sha256"]) is None
                ):
                    _fail(
                        "Bootstrap-support table manifest is incompatible",
                        code="invalid_run_manifest",
                        field_name=artifact_name,
                    )
            registry = extension["registry"]
            if (
                not isinstance(registry, Mapping)
                or set(registry) != {"filename", "records", "sha256", "schema"}
                or registry["filename"] != support_contract.registry_filename
                or registry["schema"] != support_contract.registry_schema_filename
                or not isinstance(registry["records"], int)
                or isinstance(registry["records"], bool)
                or registry["records"] < 2
                or not isinstance(registry["sha256"], str)
                or _SHA256.fullmatch(registry["sha256"]) is None
            ):
                _fail(
                    "Bootstrap-support registry manifest is incompatible",
                    code="invalid_run_manifest",
                    field_name="registry",
                )
            for linked_table in support_contract.linked_tables:
                if linked_tables[linked_table] != tables[linked_table]["sha256"]:
                    _fail(
                        "Bootstrap-support linkage does not match differential",
                        code="result_digest_mismatch",
                        field_name=support_contract.name,
                    )
    if not isinstance(value["stages"], list) or not isinstance(value["warnings"], list):
        _fail(
            "Run manifest stages and warnings must be arrays",
            code="invalid_run_manifest",
            field_name="stages",
        )
    for stage in value["stages"]:
        if not isinstance(stage, Mapping) or set(stage) != {
            "name",
            "status",
            "reason_code",
        }:
            _fail(
                "Run manifest contains an invalid stage record",
                code="invalid_run_manifest",
                field_name="stages",
            )
        if (
            not isinstance(stage["name"], str)
            or not stage["name"]
            or stage["status"] not in {"complete", "skipped", "failed", "not_estimable"}
            or (
                stage["reason_code"] is not None
                and (
                    not isinstance(stage["reason_code"], str)
                    or not stage["reason_code"]
                )
            )
            or (stage["status"] != "complete" and stage["reason_code"] is None)
        ):
            _fail(
                "Run manifest contains an incompatible stage record",
                code="invalid_run_manifest",
                field_name="stages",
            )
    if any(
        not isinstance(warning, str) or not warning for warning in value["warnings"]
    ):
        _fail(
            "Run manifest warnings must be non-empty strings",
            code="invalid_run_manifest",
            field_name="warnings",
        )


def _context_filter(context: Mapping[str, object] | str) -> tuple[str, str]:
    if isinstance(context, str):
        if not context:
            raise ValueError("context ID cannot be empty")
        return "context_id", context
    if not isinstance(context, Mapping) or not context:
        raise TypeError("context must be a non-empty mapping or stable context ID")
    return "context_json", canonical_json(dict(context))


@dataclass(frozen=True, slots=True)
class CrychicResult:
    """Immutable handle that reads validated result tables on demand."""

    path: Path
    _manifest: Mapping[str, Any] = field(repr=False)
    _config: Mapping[str, Any] = field(repr=False)
    _provenance: Mapping[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> CrychicResult:
        """Load a complete result and reject interrupted or corrupted artifacts."""

        root = Path(path)
        try:
            status = read_json(root / STATUS_FILENAME)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ResultValidationError(
                "Result status marker is missing or corrupted",
                code="invalid_result_status",
                field="status",
                remediation="Reject the artifact and regenerate it from a complete run",
            ) from exc
        if status.get("status") != "complete":
            raise IncompleteResultError(
                "Incomplete result directories cannot be loaded as successful",
                code="incomplete_result",
                field="status",
                remediation="Resume the run or write a new complete result directory",
            )
        if status.get("result_schema_version") != RESULT_SCHEMA_VERSION:
            _fail(
                "Result status uses an unsupported schema version",
                code="unsupported_result_schema",
                field_name="result_schema_version",
            )
        try:
            config_value = read_json(root / CONFIG_FILENAME)
            provenance_value = read_json(root / PROVENANCE_FILENAME)
            manifest = read_json(root / MANIFEST_FILENAME)
            config = CrychicConfig.from_dict(config_value)
            provenance = RunProvenance.from_dict(provenance_value)
        except Exception as exc:
            raise ResultValidationError(
                "Result metadata is missing, corrupted, or incompatible",
                code="invalid_result_metadata",
                field="metadata",
                remediation="Reject the artifact and regenerate it from a complete run",
            ) from exc
        _validate_manifest(manifest)
        if config.digest != manifest["config_digest"]:
            _fail(
                "Config digest does not match the run manifest",
                code="result_digest_mismatch",
                field_name="config_digest",
            )
        if provenance.digest != manifest["provenance_digest"]:
            _fail(
                "Provenance digest does not match the run manifest",
                code="result_digest_mismatch",
                field_name="provenance_digest",
            )
        if provenance.config_digest != config.digest:
            _fail(
                "Provenance config digest does not match config",
                code="result_digest_mismatch",
                field_name="config_digest",
            )
        if provenance.result_schema_version != RESULT_SCHEMA_VERSION:
            _fail(
                "Provenance result schema version is incompatible",
                code="unsupported_result_schema",
                field_name="result_schema_version",
            )

        table_records = manifest["tables"]
        for table_name in TABLE_NAMES:
            contract = table_contract(table_name)
            record = table_records[table_name]
            table_path = root / contract.filename
            try:
                digest = sha256_file(table_path)
            except Exception as exc:
                raise ResultValidationError(
                    f"Result table {table_name!r} is missing or corrupted",
                    code="corrupted_result_table",
                    field=table_name,
                    remediation="Reject the artifact and regenerate it",
                ) from exc
            if digest != record["sha256"]:
                _fail(
                    f"Result table {table_name!r} does not match its manifest",
                    code="result_digest_mismatch",
                    field_name=table_name,
                )
            try:
                frame = _read_parquet(
                    table_path,
                    engine="pyarrow",
                )
            except Exception as exc:
                raise ResultValidationError(
                    f"Result table {table_name!r} is missing or corrupted",
                    code="corrupted_result_table",
                    field=table_name,
                    remediation="Reject the artifact and regenerate it",
                ) from exc
            if len(frame) != record["rows"]:
                _fail(
                    f"Result table {table_name!r} does not match its manifest",
                    code="result_digest_mismatch",
                    field_name=table_name,
                )
            validate_table(table_name, frame)

        extensions = manifest.get("extensions", {})
        if edge_evidence_contract().name in extensions:
            extension_contract = edge_evidence_contract()
            extension_record = extensions[extension_contract.name]
            extension_path = root / extension_contract.filename
            try:
                digest = sha256_file(extension_path)
                from pyarrow.parquet import ParquetFile  # type: ignore[import-untyped]

                parquet = ParquetFile(extension_path)
                row_count = parquet.metadata.num_rows
                columns = tuple(parquet.schema_arrow.names)
            except Exception as exc:
                raise ResultValidationError(
                    "Edge-evidence result extension is missing or corrupted",
                    code="corrupted_result_extension",
                    field=extension_contract.name,
                    remediation="Reject the artifact and regenerate it",
                ) from exc
            if (
                digest != extension_record["sha256"]
                or row_count != extension_record["rows"]
                or columns != extension_contract.columns
            ):
                _fail(
                    "Edge-evidence result extension does not match its manifest",
                    code="result_digest_mismatch",
                    field_name=extension_contract.name,
                )
        if scoring_collections_contract().name in extensions:
            collection_name = scoring_collections_contract().name
            collection_record = extensions[collection_name]
            collection_contract = scoring_collections_contract(
                str(collection_record["extension_schema_version"])
            )
            collection_path = root / collection_contract.filename
            try:
                collection_digest = sha256_file(collection_path)
                collection_value = read_json(collection_path)
                collection_document = ScoringCollectionDocument.from_dict(
                    collection_value
                )
            except Exception as exc:
                raise ResultValidationError(
                    "Scoring-collection extension is missing or corrupted",
                    code="corrupted_result_extension",
                    field=collection_contract.name,
                    remediation="Reject the artifact and regenerate it",
                ) from exc
            if (
                collection_digest != collection_record["sha256"]
                or len(collection_document.collections)
                != collection_record["collections"]
            ):
                _fail(
                    "Scoring-collection extension does not match its manifest",
                    code="result_digest_mismatch",
                    field_name=collection_contract.name,
                )
            try:
                source_scores = _read_parquet(
                    root / table_contract("sample_scores").filename,
                    columns=[
                        "subject_id",
                        "sample_id",
                        "context_id",
                        "design_row_id",
                        "edge_id",
                        "scoring_functional_id",
                        "repeat_id",
                        "fold_id",
                        "mode",
                    ],
                    engine="pyarrow",
                )
                source_interactions = _read_parquet(
                    root / table_contract("interactions").filename,
                    columns=["contrast", "sender", "receiver", "interaction_id"],
                    engine="pyarrow",
                )
                validate_scoring_collection_links(
                    collection_document,
                    source_scores,
                    source_interactions,
                )
            except ResultValidationError:
                raise
            except Exception as exc:
                raise ResultValidationError(
                    "Scoring-collection table linkage cannot be validated",
                    code="corrupted_result_extension",
                    field=collection_contract.name,
                    remediation="Reject the artifact and regenerate it",
                ) from exc
        support_contract = bootstrap_support_contract()
        if support_contract.name in extensions:
            support_record = extensions[support_contract.name]
            specificity_record = support_record["specificity_support"]
            selection_record = support_record["selection_frequency"]
            registry_record = support_record["registry"]
            specificity_path = root / support_contract.specificity.filename
            selection_path = root / support_contract.selection.filename
            registry_path = root / support_contract.registry_filename
            try:
                specificity_digest = sha256_file(specificity_path)
                selection_digest = sha256_file(selection_path)
                registry_digest = sha256_file(registry_path)
                specificity = _read_parquet(specificity_path, engine="pyarrow")
                selection = _read_parquet(selection_path, engine="pyarrow")
                registry = validate_bootstrap_support_registry(
                    read_json(registry_path)
                )
                document = BootstrapSupportDocument(
                    specificity_support=specificity,
                    selection_frequency=selection,
                    registry=registry,
                )
            except ResultValidationError:
                raise
            except Exception as exc:
                raise ResultValidationError(
                    "Bootstrap-support extension is missing or corrupted",
                    code="corrupted_result_extension",
                    field=support_contract.name,
                    remediation="Reject the artifact and regenerate it",
                ) from exc
            if (
                specificity_digest != specificity_record["sha256"]
                or selection_digest != selection_record["sha256"]
                or registry_digest != registry_record["sha256"]
                or len(specificity) != specificity_record["rows"]
                or len(selection) != selection_record["rows"]
                or (
                    len(specificity) + len(selection)
                    != registry_record["records"]
                )
            ):
                _fail(
                    "Bootstrap-support extension does not match its manifest",
                    code="result_digest_mismatch",
                    field_name=support_contract.name,
                )
            try:
                differential = _read_parquet(
                    root / table_contract("differential").filename,
                    columns=[
                        "hypothesis_level",
                        "hypothesis_id",
                        "contrast",
                        "mode",
                        "view",
                    ],
                    engine="pyarrow",
                )
                validate_bootstrap_support_links(document, differential)
            except ResultValidationError:
                raise
            except Exception as exc:
                raise ResultValidationError(
                    "Bootstrap-support differential linkage cannot be validated",
                    code="corrupted_result_extension",
                    field=support_contract.name,
                    remediation="Reject the artifact and regenerate it",
                ) from exc

        return cls(
            path=root.resolve(),
            _manifest=_freeze(manifest),
            _config=_freeze(config_value),
            _provenance=_freeze(provenance_value),
        )

    @property
    def manifest(self) -> Mapping[str, Any]:
        """Return immutable run-manifest metadata."""

        return self._manifest

    @property
    def config(self) -> Mapping[str, Any]:
        """Return immutable persisted configuration metadata."""

        return self._config

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Return immutable persisted provenance metadata."""

        return self._provenance

    @property
    def has_edge_evidence(self) -> bool:
        """Whether this result includes the optional edge-evidence extension."""

        extensions = self._manifest.get("extensions", {})
        return edge_evidence_contract().name in extensions

    @property
    def has_scoring_collections(self) -> bool:
        """Whether this result declares receiver-partitioned functionals."""

        extensions = self._manifest.get("extensions", {})
        return scoring_collections_contract().name in extensions

    @property
    def has_bootstrap_support(self) -> bool:
        """Whether authenticated specificity and selection support are present."""

        extensions = self._manifest.get("extensions", {})
        return bootstrap_support_contract().name in extensions

    def read_bootstrap_support_registry(self) -> Mapping[str, Any]:
        """Read and revalidate the exact bootstrap-support lineage registry."""

        if not self.has_bootstrap_support:
            raise KeyError("bootstrap_support")
        contract = bootstrap_support_contract()
        registry = validate_bootstrap_support_registry(
            read_json(self.path / contract.registry_filename)
        )
        return cast(Mapping[str, Any], _freeze(registry))

    def _read_bootstrap_support_table(
        self,
        *,
        specificity: bool,
        filters: Mapping[str, object] | None,
        columns: Sequence[str] | None,
    ) -> pd.DataFrame:
        if not self.has_bootstrap_support:
            raise KeyError("bootstrap_support")
        support = bootstrap_support_contract()
        contract = support.specificity if specificity else support.selection
        requested_columns = None if columns is None else list(columns)
        if requested_columns is not None:
            unknown_columns = set(requested_columns).difference(contract.columns)
            if unknown_columns:
                raise KeyError(sorted(unknown_columns)[0])
        parquet_filters: list[tuple[str, str, object]] | None = None
        if filters:
            unknown_filters = set(filters).difference(contract.columns)
            if unknown_filters:
                raise KeyError(sorted(unknown_filters)[0])
            parquet_filters = [
                (column, "==", value) for column, value in filters.items()
            ]
        return _read_parquet(
            self.path / contract.filename,
            columns=requested_columns,
            filters=parquet_filters,
            engine="pyarrow",
        )

    def read_specificity_support(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read authenticated specificity support with Parquet pushdown."""

        return self._read_bootstrap_support_table(
            specificity=True,
            filters=filters,
            columns=columns,
        )

    def read_selection_frequency(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read equal-bootstrap family selection frequency."""

        return self._read_bootstrap_support_table(
            specificity=False,
            filters=filters,
            columns=columns,
        )

    def read_scoring_collections(self) -> tuple[ScoringCollectionManifest, ...]:
        """Read validated receiver-child scoring collection manifests."""

        if not self.has_scoring_collections:
            raise KeyError("scoring_collections")
        contract = scoring_collections_contract()
        value = read_json(self.path / contract.filename)
        document = ScoringCollectionDocument.from_dict(value)
        return tuple(document.collections)

    def read_edge_evidence(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read the optional evidence ledger, with Parquet filter pushdown."""

        if not self.has_edge_evidence:
            raise KeyError("edge_evidence")
        contract = edge_evidence_contract()
        requested_columns = None if columns is None else list(columns)
        if requested_columns is not None:
            unknown_columns = set(requested_columns).difference(contract.columns)
            if unknown_columns:
                raise KeyError(sorted(unknown_columns)[0])
        parquet_filters: list[tuple[str, str, object]] | None = None
        if filters:
            unknown_filters = set(filters).difference(contract.columns)
            if unknown_filters:
                raise KeyError(sorted(unknown_filters)[0])
            parquet_filters = [
                (column, "==", value) for column, value in filters.items()
            ]
        return _read_parquet(
            self.path / contract.filename,
            columns=requested_columns,
            filters=parquet_filters,
            engine="pyarrow",
        )

    def read_table(
        self,
        name: str,
        *,
        filters: Mapping[str, object] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read one table, pushing equality filters into the Parquet backend."""

        contract = table_contract(name)
        requested_columns = None if columns is None else list(columns)
        if requested_columns is not None:
            unknown_columns = set(requested_columns).difference(contract.columns)
            if unknown_columns:
                raise KeyError(sorted(unknown_columns)[0])
        parquet_filters: list[tuple[str, str, object]] | None = None
        if filters:
            unknown_filters = set(filters).difference(contract.columns)
            if unknown_filters:
                raise KeyError(sorted(unknown_filters)[0])
            parquet_filters = [
                (column, "==", value) for column, value in filters.items()
            ]
        frame = _read_parquet(
            self.path / contract.filename,
            columns=requested_columns,
            filters=parquet_filters,
            engine="pyarrow",
        )
        return frame

    def rank_interactions(
        self,
        *,
        context: Mapping[str, object] | str,
        receiver: str,
        contrast: str,
        mode: str = "state",
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Rank one persisted estimand without recomputing a statistic."""

        if not receiver or not contrast:
            raise ValueError("receiver and contrast must be non-empty")
        if mode not in {"state", "ecosystem"}:
            raise ValueError("mode must be 'state' or 'ecosystem'")
        if top_n is not None and (
            isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0
        ):
            raise ValueError("top_n must be a positive integer or None")
        context_column, context_value = _context_filter(context)
        frame = self.read_table(
            "interactions",
            filters={
                context_column: context_value,
                "receiver": receiver,
                "contrast": contrast,
                "mode": mode,
            },
        )
        score_column = (
            "comm_strength" if frame["comm_strength"].notna().any() else "availability"
        )
        ranked = frame.sort_values(
            [score_column, "interaction_id", "sender"],
            ascending=[False, True, True],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
        return ranked if top_n is None else ranked.head(top_n).copy()

    def get_signature(
        self,
        *,
        context: Mapping[str, object] | str,
        sender: str,
        receiver: str,
        interaction: str,
    ) -> pd.DataFrame:
        """Return a persisted sender-LR-receiver signature selection."""

        if not sender or not receiver or not interaction:
            raise ValueError("sender, receiver, and interaction must be non-empty")
        context_column, context_value = _context_filter(context)
        frame = self.read_table(
            "signatures",
            filters={
                context_column: context_value,
                "sender_id": sender,
                "receiver": receiver,
                "interaction_id": interaction,
            },
        )
        return frame.sort_values(
            ["stability", "gene"],
            ascending=[False, True],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)


def run_manifest_schema() -> dict[str, Any]:
    """Expose the normative run-manifest schema for contract validation."""

    return dict(load_schema_document("run_manifest.schema.json"))
