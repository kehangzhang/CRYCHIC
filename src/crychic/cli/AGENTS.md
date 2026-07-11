# CLI Module

## Owns

- Command parsing, config-file loading, validate/dry-run/fit/export commands,
  structured logs, exit codes, and user-facing error presentation.

## Dependencies

- May import `api`, `workflow`, and `results` only through stable public
  contracts. It must contain no statistical or resource-parsing logic.

## Rules

- Commands are non-interactive by default and support explicit output paths,
  seeds, offline mode, and resumable runs.
- Validate configuration and input estimability before expensive execution.
- Logs include run ID and stage but must not expose patient-level values.
- Exit codes distinguish usage, validation, resource, convergence, and internal
  failures.
- `--help` and config examples are generated from the same typed config schema.

## Required Tests

- Help/config snapshots, dry-run, exit codes, invalid input, interrupted resume,
  offline resource failure, and a tiny CLI end-to-end run.
