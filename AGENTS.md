# CRYCHIC Repository Instructions

## Scope

These instructions apply to the whole repository. A nested `AGENTS.md` adds
module-specific constraints and takes precedence within its directory.

## Project Identity

- Project and distribution name: `CRYCHIC`.
- Python import package: `crychic`.
- Do not expose the provisional name `TopoCCC` or `topoccc` in public APIs.
- The normative implementation roadmap is `DEVELOPMENT_PLAN.md`.
- The source concept document is `../prompt.md`; some equations in that file
  were damaged during Markdown conversion, so recover and review equations in
  a versioned method specification before coding them.

## Non-Negotiable Statistical Rules

1. The inferential unit is the biological subject, represented by
   `subject_id`. Cells are observations used for aggregation, not independent
   replicates.
2. Keep `sample_id` and `subject_id` distinct. Repeated contexts from one
   subject stay together during splitting and bootstrap. Permutation follows a
   design-specific exchangeability map: between-subject factors permute subject
   blocks within strata, while paired within-subject factors use only valid
   restricted swaps or sign flips.
3. Structural absence of a cell type is missingness, not zero expression.
4. Inferential and exploratory modes are separate contracts. Never emit formal
   p-values, q-values, or calibrated probabilities when a design is not
   estimable.
5. Penalized coefficients support selection, denoising, and attribution. They
   must not be used directly to construct ordinary Wald p-values.
6. Formal point effects use an identical, contrast-level scoring functional on
   subject-level out-of-fold samples. Until a valid analytic variance is
   derived, uncertainty and calibration must rerun the complete learned
   pipeline under repeated cross-fitting and subject-level resampling.
7. Every data-dependent operation in cross-fitting belongs inside the training
   fold, including filtering, scaling, gating, LR clustering, tuning, and
   sender coupling.
8. Keep communication strength, active posterior probability, bootstrap
   specificity support, selection frequency, differential effect, and q-value
   as separate quantities and table grains.
9. Preserve observed, explained, and residual receiver responses. Never force
   all receiver biology to be explained by known communication resources.

## Architecture Rules

- Dependencies flow from foundational modules toward orchestration and user
  interfaces. Numerical modules must not import `workflow`, `api`, `cli`, or
  `visualization`.
- Exchange cross-module data through typed public contracts owned by the
  upstream producer. `core` owns only genuinely shared primitives; never import
  another module's private implementation.
- Keep the three graph meanings separate: context topology in `design`,
  molecular signaling topology in package `resources`, and output
  communication hypergraphs in `network`.
- `workflow` orchestrates stages but does not own statistical formulas.
- `visualization` consumes saved result contracts and never recomputes tests.
- Optional heavy backends must be lazily imported and exposed through typed
  protocols. The base installation must remain usable without them.

## Engineering Standards

- Use a `src` layout, typed public APIs, deterministic seed handling, sparse
  operations where practical, and explicit convergence diagnostics.
- Do not mutate user-owned `AnnData` objects in place unless the API explicitly
  documents and tests that behavior.
- All persisted artifacts carry schema version, package version, configuration
  digest, input checksum, resource versions, seed lineage, and fold identity.
- Add focused unit and contract tests with each change. Statistical behavior
  requires simulation or negative-control tests proportional to its risk.
- Error messages must identify the invalid field, affected analysis unit, and
  remediation. Never silently change a requested model or contrast.
- Use English identifiers and API documentation. User guides may be bilingual.

## Data and Resource Safety

- Never commit patient data, identifiable metadata, downloaded knowledge-base
  payloads, large matrices, benchmark raw output, credentials, or machine-local
  absolute paths.
- Track only tiny redistributable synthetic fixtures and resource manifests
  containing version, URL, checksum, license, citation, and retrieval date.
- A checksum or license mismatch is a hard resource-loading failure.

## Git Workflow

- Work on a topic branch; do not commit directly to protected `main`.
- Use Conventional Commits and keep commits logically reviewable.
- Do not rewrite or discard unrelated user changes.
- Run the relevant lint, type, unit, contract, and statistical checks before a
  commit. Record any check that could not be run.
- Release tags follow SemVer and are created only from commits that pass the
  release quality gates in `DEVELOPMENT_PLAN.md`.
