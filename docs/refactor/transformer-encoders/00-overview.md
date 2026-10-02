# Selectable encoders for a controlled materials benchmark

The user approved the architecture/evaluation proposal in
`experiments/transformer_encoder_20261002/README.md` on 2026-10-02 and requested implementation
and one completed test round. This plan makes that approved scope executable.

Decision: implement the encoder choices and their workflow integration as one coherent local
change. Training, prediction and inverse must construct the same architecture from the same
configuration. Keep the existing MLP defaults and existing Python Transformer defaults.
Feature-specific and grouped tokenization share a modern attention backbone; a token-wise MLP
provides the no-attention control. CLS, mean and concatenation pooling are compared directly. An explicit common
reconstruction decoder removes a confounder between architectures.

The user subsequently required PR review/merge, versioning and image deployment before RIKYU
training. The ordered PR scopes are:

1. **01 — selectable feature-token encoders**: production code, tests, documentation and the
   0.5.0 version bump. Review and squash-merge before any later implementation. The merge
   publishes the new RIKYU ARM image through the existing container workflow.
2. **02 — reproducible transfer benchmark**: experiment-owned preprocessing, paired configurations,
   runners and analysis. Implement only after PR 01 is confirmed merged, then review and merge
   before submitting jobs. Execute the installed package from the immutable published image;
   never override it with a source checkout or `PYTHONPATH`.

Remote PR creation, review handling and merge are authorized by that instruction. Scientific
results are recorded separately from implementation checks; no GPU training has started.

Alternatives considered: directly enlarging the old Transformer leaves the tokenizer question
unanswered; replacing KMD with element tokens changes the descriptor/inverse contract at the
same time. Fixed 94-element tokens remain a subsequent research direction. The first round runs
the six proposed controls plus paired grouped-Transformer mean/concat variants, with three
paired seeds. Its primary outcomes are transfer to held-out scalar/function tasks and transfer
scaling with source task counts 1, 3 and 7, at two target-label budgets. Source fitting metrics are
diagnostics, not the selection criterion. The source composition universe is anchored by a task
with labels on all source compositions. Real-environment smoke and GPU packing calibration
precede the fleet.

Non-goals: JEPA, task-specific query heads, removal of the shared latent interface, new distributed
training, unrelated activation changes, or changing the default MLP behavior.

Acceptance and implementation boundaries are in the numbered plan files.
