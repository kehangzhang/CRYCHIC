# ADR-006: Contrast-common sender functional

- Status: Accepted
- Date: 2026-07-13
- Implementation status: Candidate train/apply stage; partial, not certified OOF

## Context

The v0.1 sender assignment aggregates availability and prevalence separately
inside each context. That is useful as an exploratory descriptive table, but it
does not provide one sender function shared by all contexts in a differential
contrast. Sender evidence also contains ligand availability, so it must not
define the primary sender-unresolved LR score or be interpreted as a causal
source.

## Decision

The producer-owned fold training path fits a separate
`ContrastCommonSenderFunctional` from training availability only. It freezes:

- the union of candidate senders for each receiver and interaction across the
  training context set;
- one subject-pooled prevalence prior per candidate;
- the prevalence threshold, minimum training support and softmax temperature;
- training subject, context and interaction-filter provenance;
- a stable functional ID derived from the complete manifest.

For a held-out sample, application uses only that sample's ligand availability:

```text
raw_sender_evidence = local_ligand_availability * training_prevalence_prior
assignment_weight = frozen_temperature_softmax(raw_sender_evidence)
```

Held-out senders outside the frozen candidate universe are ignored. A missing
local ligand or non-estimable training prior cannot be refitted from held-out
data. When every candidate lacks evidence, every weight and entropy remains NA;
there is no uniform fallback.

Sender resolution is a secondary conservation operation. It multiplies the
sender-unresolved LR strength by normalized assignment weights and verifies
that resolved strengths sum back to the unchanged sender-unresolved value.

## Consequences

The candidate functional is common across contexts and insensitive to held-out
context relabeling or test-only sender poisons. It remains non-causal and
`partial_not_oof` because the response, downstream, family and scoring stages
are not yet connected. The legacy `assign_senders` API and default baseline
score are unchanged to preserve v0.1 reproducibility.
