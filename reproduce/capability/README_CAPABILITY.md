# E22 - capability cost of each defense

## Run order

1. Upload `piibench_capability.zip` to `MyDrive/piibench/`
2. Open `RUN_ME_CAPABILITY.ipynb` in Colab, GPU runtime
3. Cells 1-5, then cell 6 (about 1.5 h), then cell 8 (about 4 h)
4. Send back `capability_results.zip` - it is a single JSON, a few hundred KB

Sessions A and B can be days apart. Finished conditions are never recomputed.

## What is measured

C-Eval, val split, 1,346 questions over 52 subjects, five-shot from the dev
split of each subject. Options are scored by the model's probability at the
answer position rather than by generating a letter, so a base model is not
penalised for failing to follow an instruction.

Recorded per condition: accuracy, the distribution over predicted letters, the
modal letter's share, a `degenerate` flag, per-subject accuracy, and perplexity
for continuity with Table 6.2.

## Why C-Eval and not CMMLU

CMMLU ships a loading script and datasets 4.0 no longer runs those. C-Eval is
served as parquet and is read directly from the Hub, so the dataset library is
not involved. This was checked against the Hub before the bundle was built.

## The number that matters

`dp_eps075` is the condition the abstract's claim rests on. If it scores near
chance, "eliminated leakage entirely at epsilon = 0.75" is true and misleading at
the same time, and the manuscript has to say both.
