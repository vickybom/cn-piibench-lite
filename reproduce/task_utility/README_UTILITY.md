# DP task-utility evaluation - Colab bundle

## What it closes

Section 6.5 reports that DP-LoRA eliminated leakage at epsilon = 0.75 with a
held-out perplexity between the base and undefended models, and states plainly
that whether the model retained the TASK competence was not measured. This run
measures it.

## The design

Probe every condition with held-out people (different generator seed, asserted
disjoint from training in values and names) and score three independent axes:
schema adherence, format validity against the governing national standards, and
whether the model instead emitted a training individual's real value.

The three conditions separate cleanly if DP did what it promises: base knows no
schema, undefended knows the people, DP knows the schema but not the people.

## How to run

1. Upload `piibench_task_utility.zip` to the `piibench` folder in Google Drive.
2. Open `RUN_ME_UTILITY.ipynb` in Colab (File -> Upload notebook).
3. Runtime -> Change runtime type -> L4 GPU.
4. Run in order. Cell 5 checks that all three adapters are reachable.

## The DP adapter

Not bundled (153 MB, and it was produced on Colab). The notebook looks for it at
`MyDrive/piibench/defense_140/adapter_dp`. If it is gone, cell 8 retrains it with
the same seed and corpus, about 3 hours.

## Runtime

Inference only. About an hour for three conditions at 60 held-out people.

## Reading the result

The script reports what fraction of the base-to-undefended gap in format validity
the DP model recovers.

- >= 50% recovered and leakage <= 0.01 -> DP learned the schema without the
  people; the Section 6.5 caveat closes.
- < 20% recovered -> DP did not acquire the task, and the zero leakage is
  consistent with failed learning; the caveat gets stronger.
- In between -> report the fraction and weaken rather than close the caveat.

Send the JSON back in every case.
