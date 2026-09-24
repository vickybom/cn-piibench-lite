# Stage 4: noise threshold - Colab bundle

## The gap

  nm 0.000 -> 45 verbatim hits, recitation 0.270   (S6.5.3 clipping control)
  nm 0.098 -> 0 hits, recitation 0.000             (S6.5.2, loosest budget swept)

Every budget an operator would consider sits above 0.098, so the whole
transition lies in an interval no experiment has entered.

## Why it matters

At 560 lots the accountant prices the ladder:

  nm 0.010 -> eps 81.6    nm 0.040 -> eps 11.5
  nm 0.020 -> eps 28.0    nm 0.070 -> eps  5.9

If the threshold sits low, the smallest sufficient noise carries an epsilon in
the tens - not a guarantee anyone would report. That sharpens Section 6.5.4: the
operator choosing eps 0.5 is buying guarantee strength, and this says how much of
that buys observable protection.

## Design

Recitation leads (0.270 vs base 0.000, ~70x stronger per query than the matrix
rate of 0.0038 and 10x cheaper); a stratified 25% of the matrix confirms.

Two seeds per rung. Section 6.5.3 is what happens when a boundary is read off a
single run - it is where the eps 0.50 utility claim died. The readout refuses to
quote a threshold the seeds disagree about.

The nm=0.000 rung re-runs the zero-noise control at a fresh seed. If it does not
reproduce 45 hits / 0.270, the ladder has not reproduced its own endpoint and
nothing above it should be believed.

## Runtime

Seed 1 about 4 h, seed 2 about 4 h. Checkpointed per rung; separate sessions are
fine.

## Send back

The whole threshold_results.zip - summary and per-query records together.
