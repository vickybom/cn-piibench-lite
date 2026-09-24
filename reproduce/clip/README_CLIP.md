# Stage 3: clipping control + replication - Colab bundle

Two experiments Section 6.5.2 identified as its own open questions.

## Part CLIP (about 2.6 h) - run this first

Section 6.5.2: both leakage channels were exactly zero across a 6.6x change in
the noise multiplier. A quantity that does not move when its supposed cause
moves 6.6-fold is probably not driven by that cause. The candidate is
per-example gradient clipping, which DP-SGD applies identically at every budget.

A single "no noise" run would not settle it, because the private and undefended
arms differ in more than noise: lora_finetune vs dp_sgd_finetune, dropout 0.0 vs
0.05, max_len 384 vs 320, scheduled optimizer vs constant-rate AdamW.

So both cells go through dp_sgd_finetune, differing only in the bound:

  clip_on    clip_norm 1.0   noise 0    clipping active
  clip_off   clip_norm 1e9   noise 0    clipping inert

  clip_off leaks, clip_on clean -> clipping is the mechanism
  both clean                    -> neither; something else in that code path
  both leak                     -> contradicts the invariance; reconcile first

NEITHER CELL HAS A PRIVACY GUARANTEE. They are mechanistic controls, not
candidate defenses. The JSON records this in a `guarantee` field.

## Part REPLICATE (about 4.8 h)

Utility was not monotone in epsilon and each budget was trained once, so budget
is confounded with run. Retraining the undefended condition moved held-out
competence by 0.10 against a 0.20 spread across budgets.

Three seeded runs at eps 0.50 and three at eps 1.00 - the pair whose inversion
was largest (0.391 vs 0.211). Reports within-budget spread against
between-budget difference.

Replicates audit a stratified 25% sample of the matrix (2,940 of 11,760, equal
share of every template) rather than all of it: every DP condition in Section
6.5.2 returned exactly zero over the full matrix, so the sample confirms the
zero. Held-out competence, the quantity that varies, is measured in full on all
900 probes.

## Runtime

Part CLIP about 2.6 h, Part REPLICATE about 4.8 h. Checkpointed per condition;
they can be run in separate sessions.

## Send back

The whole clip_results.zip - summaries and per-query records together.
