# Signed Track-B evaluator contract

`benchmarks.metrics.signed_track_b` implements the preregisterable metric
contract for signed target-program recovery. It is an evaluator primitive, not
evidence that the current public workflow or the existing Track-B proxy has
passed this endpoint.

Each registered target program is represented by exactly two non-negative,
higher-is-stronger channels:

- `increased_activation_compatible` is the forward-contrast channel;
- `reduced_activation_compatible` is the explicit reverse-contrast channel.

These names deliberately do not mean molecular activation and inhibition. A
non-negative ligand-target prior cannot distinguish an actively repressing path
from reduced activation-compatible contribution. Molecular sign claims require
the future signed signaling-network contract.

For a program whose registered direction is forward, its forward channel is a
positive and its reverse channel is a negative. The mapping is reversed for a
registered reverse program. Both channels of a neutral program are negatives.
This expanded binary universe makes a high wrong-direction score a false
positive rather than silently relabeling it as recovery.

The evaluator computes tie-aware average precision separately for every
registered `seed x scenario_cell_id x receiver`. It then averages all
registered seeds within a cell and gives every registered scenario/receiver
cell equal weight in `signed_target_program_macro_auprc`. Every scenario for a
receiver must retain the exact same frozen program universe; neither program
count nor truth direction can be used to shrink the candidates. A missing
frozen program, missing direction channel,
failed fit, non-estimable fit, absent expected seed, or single-class truth cell
propagates `not_estimable`; no row is converted to score zero and no incomplete
cell is dynamically dropped.

Predictions must carry the SHA256 digest of the canonical registered truth
table. Reusing a `truth_set_id` after changing a direction, scenario, receiver,
or program therefore fails before metric computation. Outputs retain this
digest and the evaluator schema version.

AUPRC is accepted only for synthetic or perturbation target-program truth. The
outputs explicitly set sender, receptor-specific LR-edge, and native NicheNet
claims to false. The evaluator emits no p-value, q-value, FDR, confidence
interval, communication probability, or real-data accuracy claim.

Completion of the Track-B recommendation still requires a frozen multi-cell,
multi-seed truth manifest, independently held-out predictions from both forward
and reverse channels, and a persisted evaluation run. Existing
`nichenet_prior_activity` proxy artifacts do not meet that input contract and
must not be relabeled or converted into native NicheNet results.
