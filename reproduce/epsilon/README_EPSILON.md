# Stage 2 revision 2: epsilon sweep - Colab bundle

## Why there is a revision 2

The first Part A run returned RW-MER 0.0046 (Low) against 0.3985 at sixty
exposures, with held-out competence above the released weights. The gate read
that as "nothing for a defense to do" and stopped.

It was reading one channel of two. RW-MER is defined on person-keyed probes, so
all 11,760 matrix queries ask about a training subject: it measures ATTRIBUTED
disclosure and cannot see a model that emits a real training identifier under
someone else's name. The held-out probe measured exactly that at 0.227, against
0.000 for the base weights. The channel is entirely fine-tuning-induced and the
benchmark's headline metric is blind to it.

## What changed

1. PER-QUERY RECORDS. Every condition writes records_matrix_*.csv (11,760 rows)
   and records_heldout_*.csv. The first run returned aggregates only, so "where
   did the 74 surviving hits land?" could be answered only by an analytic bound.

2. RECITATION IS THE PRIMARY PRIVACY AXIS. The aggregate is already Low. A
   budget earns its keep by closing the unattributed channel while keeping
   competence, and the readout reports "closes X% of the leak" alongside
   "keeps Y% of the span".

3. HELD-OUT PROBE 60 -> 180 PEOPLE (n 300 -> 900). The span to price is 0.147.
   At n=300 the standard error is 0.029 and "keeps half" is about 2.5 standard
   errors from "keeps none". At n=900 it is 0.017.

Part A must be re-run: not because it was wrong, but because it kept no records
and its held-out probe was a third of the current size.

## Budget ordering

Part B sweeps 1.0, 4.0, 0.5, 2.0 in that order - a coarse bracket first. If the
session dies after two conditions you know which end of the curve the usable
point is on, rather than holding four adjacent points at one end.

The accounted epsilon is what gets recorded, not the target. Multipliers were
inverted from the RDP accountant at 560 lots (four exposures), not the 2,800 of
the sixty-exposure configuration.

  eps 0.5 -> nm 0.648 | eps 1.0 -> nm 0.342
  eps 2.0 -> nm 0.182 | eps 4.0 -> nm 0.098

## Runtime

Part A about 80 minutes on an L4. Part B about 95 minutes per budget, four
budgets - plan on two sessions. Checkpointed per condition.

## Reading Part B

A budget is usable when it closes most of the recitation channel AND keeps a
substantial share of the competence span. Failure is informative either way:
closing the leak while competence falls to base level repeats the Section 6.5.1
finding at a better operating point; keeping competence while recitation
survives says the noise never reached the channel carrying it.

Send the whole zip back in every case - summary and records together.
