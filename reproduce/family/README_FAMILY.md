# E21 - second and third model families

## Run order

1. Upload `piibench_family.zip` to `MyDrive/piibench/`
2. Open `RUN_ME_FAMILY.ipynb` in Colab, GPU runtime (L4 preferred, T4 workable)
3. Run cells top to bottom
4. Send back `family_results.zip`

## What varies

Only the model family. Corpus seed 20260524, twelve templates, LoRA r=32,
30 epochs x 2 repeats, training seed 42, greedy decoding, same detector.

The C1/D1 assistant persona name is swapped per family, because telling InternLM
that it is Qwen2.5 is a false identity claim that can move refusal behaviour. The
request text stays byte-identical; cell 5 asserts this before any GPU work.

DO NOT change `--seed`. It varies the corpus as well and the run stops being a
family comparison.

## Reference values, recomputed inside the run

The main Qwen run's per-query records ship in `reference/`. The script scores
them with the same functions it uses on the new families, so the comparison
cannot be an artefact of two different derivations.

| Quantity | Qwen2.5-1.5B fine-tuned |
|---|---|
| aggregate RW-MER | 0.3359 (High) |
| direct memorization A+B | 0.5235 |
| matched-pair CLMD | -0.0444 (C1-D1 -0.0469, C2-D2 -0.0419) |
| anchored : unanchored | 0.6949 : 0.0092 = 75.5x |
| null floor, released weights | 0.0000 (Low) |

## Runtime

E21a Yi-1.5-6B about 4 h; E21b DeepSeek-7B about 5 h on an L4. Checkpointed
per model - a dropped session resumes from the saved adapter.

## What this decides in the manuscript

The adviser's rating of model-family generalizability is **Weak**, and the
title's "small and medium domestic large language models" is broader than three
conditions of one family support. E21a supplies the family control at matched
scale; E21b supplies a genuinely medium-scale model. Together they are what lets
the title stand with an operational definition rather than being narrowed to
"the evaluated Qwen2.5 models".
