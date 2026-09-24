# Stage 1: generalizable-baseline search - Colab bundle

## Why

Section 6.5.1 found that at sixty exposures per record, no condition acquired
transferable schema competence. The undefended model's apparent 0.883 format
validity was 90% recitation of training subjects; its novel-and-valid rate was
0.089, below the released model's 0.236.

Without a baseline that has utility, no defense can be priced in utility terms.
This run searches for a fine-tuning configuration whose product generalises.

## Design

Two levers crossed: exposure (2/4/6/8/12 per record) and corpus diversity (fixed
phrasing vs four paraphrases per field). Document counts are identical between
the two corpus arms, so surface diversity is the only difference.

Three quantities are measured at every cell and never conflated: memorization on
training people, novel-and-valid on held-out people, and recitation on held-out
people. The held-out corpus uses a different seed and the script asserts it is
disjoint from training in both values and names.

## How to run

1. Upload `piibench_generalization.zip` to the `piibench` folder in Google Drive.
2. Open `RUN_ME_GENERALIZATION.ipynb` in Colab (File -> Upload notebook).
3. Runtime -> Change runtime type -> L4 GPU.
4. Run the cells in order.

## Runtime

11 cells (base + 2 corpora x 5 exposure levels) at 40 records: about 45 minutes
on an L4. Every cell checkpoints, so a disconnect costs at most the cell in
flight.

## Reading the result

A cell qualifies as a usable baseline when novel_valid exceeds the base model's
rate significantly while memorization stays low.

- Some cell qualifies -> Stage 2, the epsilon sweep, can proceed against that
  configuration.
- No cell qualifies -> this corpus cannot teach the schema at any exposure. That
  is a reportable result about the experimental design, and a reason NOT to run
  the epsilon sweep.

Send the JSON back in either case.
