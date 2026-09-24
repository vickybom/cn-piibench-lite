# Stage 5: replicate E6 and E9 at a second training seed

## Why

Section 6.5.3 established that a single training run at this study's operating
points is a draw, not a result - it is where the eps 0.50 utility claim died.
Chapter 5's conclusions were each measured on one run.

Most are protected by saturation: at sixty exposures memorization sits near 1.0
and a run has little room to differ. Two are not:

  E6 capacity: 1.5B 0.5235 vs 3B 0.4301 on the eight direct-memorization
               templates. Difference 0.0934 against a run-variance estimate of
               0.0276 - about 3x the noise, one run each.

  E9 CLMD:     matched-pair differential on the aligned model is -0.048, which
               is smaller than twice the run-variance estimate, and it is quoted
               in the abstract.

## What varies

The corpus seed stays at 20260524 - same people, same values. Only the training
seed changes (42 -> 1337). The training seed was not previously a parameter;
every run in this study used HuggingFace's default of 42 implicitly. It now
defaults to 42, so nothing already collected changes.

DO NOT change --seed. Changing it varies the corpus as well and the run stops
being a replication.

## Pre-registered reading

E6 confirmed if the difference keeps its sign and stays clear of the run
variance; in trouble if the sign flips; restated if the margin falls to noise.

E9 confirmed if the matched-pair value stays small and the unpaired formulation
still departs from it. The methodological claim survives regardless, because it
concerns the gap between the two formulations rather than either value.

## Runtime

Cell 5 (1.5B + 3B) about 4 h; cell 7 (Instruct) about 1.5 h. Checkpointed per
model.

## A null result is a result

A sign flip, or a CLMD that moves by more than its own size, means two Chapter 5
conclusions need restating - which is better found now than at the defense.

Send back the whole replicate_results.zip, records included.
