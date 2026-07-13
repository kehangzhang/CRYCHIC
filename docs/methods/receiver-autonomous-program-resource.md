# Receiver-Autonomous Program Resource

CRYCHIC loads a registered receiver-autonomous nuisance basis only from a
code-reviewed, checksum-pinned static resource. The loader does not accept
`AnnData`, expression matrices, sample metadata, fold IDs, responses, scores, or
fitted programs. This keeps resource construction independent of every analysis
cohort and split.

## Static TSV Contract

The payload is a UTF-8, tab-separated feature-by-program matrix. Its first
column must be named `feature_id`; every remaining header is a unique program
ID. Feature IDs must also be unique. All weights are finite signed decimal
numbers, and every program must have at least one nonzero weight.

```text
feature_id\tcell_cycle\tstress
CDK1\t1.0\t0.0
FOS\t0.0\t0.8
JUN\t0.0\t-0.3
```

The loader canonicalizes both axes lexically before constructing the immutable
artifact. Axis order therefore does not change the matrix digest, while any
change to the raw payload still changes the manifest digest and artifact ID.

## Manifest Contract

Use the ordinary `ResourceManifest` fields and pin the exact payload SHA-256 and
byte size. A registered autonomous manifest additionally requires:

- `adapter_version` equal to
  `crychic-receiver-autonomous-feature-program-tsv-v1`;
- exactly one payload with role
  `receiver_autonomous_feature_by_program_matrix_v1`;
- a `.tsv` payload path relative to the supplied database root;
- exact `resource_id`, `version`, `species`, `gene_namespace`, and `license`
  values recorded in the code review registry;
- a registry scope of either `synthetic_benchmark_only` or
  `biological_reference`.

```json
{
  "resource_id": "receiver_autonomous_human_2026",
  "version": "2026.1",
  "species": "human",
  "gene_namespace": "HGNC symbol",
  "source_url": "https://example.org/reviewed-resource",
  "license": "CC-BY-4.0",
  "citation": "The reviewed resource citation.",
  "retrieved_at": "2026-07-14",
  "adapter_version": "crychic-receiver-autonomous-feature-program-tsv-v1",
  "payloads": [
    {
      "path": "receiver_autonomous/2026.1/programs.tsv",
      "sha256": "<64 lowercase hexadecimal characters>",
      "bytes": 12345,
      "role": "receiver_autonomous_feature_by_program_matrix_v1"
    }
  ],
  "transformation_log": [
    "Programs were declared without access to analysis expression data."
  ]
}
```

Reviewing a new resource requires a code change to the immutable registry in
`src/crychic/resources/autonomous_registry.py`. Each registration binds the
registration ID, canonical manifest digest, resource/version/species/namespace,
expected license, adapter, review scope, payload path/role/SHA-256/byte size, and
canonical matrix digest. The loader does not accept caller-supplied expected
digests or metadata, so computing a digest from an unreviewed manifest cannot
promote it to registered status.

## Loading

```python
from pathlib import Path

from crychic.response import load_receiver_autonomous_program_resource

fixture_root = Path(
    "benchmarks/fixtures/synthetic_receiver_autonomous_program"
).resolve()
programs = load_receiver_autonomous_program_resource(
    fixture_root,
    manifest_path=fixture_root / "manifest.json",
    registration_id="crychic.synthetic_receiver_autonomous_program.v1",
)
assert programs.verification_status == "manifest_verified_static_trusted_v1"
assert programs.is_manifest_verified_trusted
assert programs.review_scope == "synthetic_benchmark_only"
assert not programs.is_biological_reference_trusted
```

The loader first resolves the registration and verifies every registered
manifest and payload field. Payload paths may not contain symlinks and their
resolved location must remain below `database_root`. It then uses
`ResourceManifest.verify`, reads the exact payload bytes, hashes them again
before parsing, validates the TSV schema and registered matrix digest, and
constructs a byte-backed read-only matrix. The artifact ID binds the full
manifest digest, canonical feature/program axes, matrix digest, resource
release, species, namespace, verification status, registration ID, review scope,
and expected license. In-memory mutation of those fields is detected by the
artifact integrity check.

`build_receiver_autonomous_program_resource` remains available for controlled
diagnostics, but it always emits `caller_declared_static_unverified`; supplying
a manifest-like digest to that function cannot promote the result to trusted.

The code registry is the certification root for the supported public API, not a
cryptographic sandbox against arbitrary code execution. A caller able to modify
private module state, monkeypatch functions, or rewrite the installed package in
the same Python process is outside this trust boundary. Registry and artifact
checks prevent accidental promotion and unsupported public-API inputs; process
isolation, signed packages, and deployment controls are required against a
malicious in-process actor.

## Current Resource Availability

The repository contains one registered deterministic fixture with review scope
`synthetic_benchmark_only`. It is suitable only for pipeline tests and synthetic
benchmarks and is not a biological reference. The workspace database audit on
2026-07-14 found CellChat, CellPhoneDB, and NicheNet assets but no reviewed
biological receiver-autonomous feature-by-program matrix.
CellPhoneDB receptor-to-TF mappings and NicheNet ligand-target weights have
different biological semantics and must not be relabeled as autonomous
nuisance programs. A real resource therefore needs independent biological
curation, licensing review, a frozen TSV export, and a reviewed manifest before
official use.

The synthetic fixture may satisfy the algorithmic incremental-child lineage
gate in synthetic benchmarks because that gate checks
`is_manifest_verified_trusted`. This does not grant biological-reference
status. Biological analyses must additionally require
`is_biological_reference_trusted`; no such biological autonomous-program
resource is currently registered.
