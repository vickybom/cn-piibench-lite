# Defense evaluation at 140 records

This bundle re-runs the three privacy defenses of Chapter 6 at the same
140-record scale used in Chapter 5.

## What is inside

- `pii_auditor/` and the runner scripts -- the audit toolkit
- `adapter/` -- the LoRA adapter already fine-tuned on the 140-record corpus
  (Qwen2.5-1.5B, rank 32, 30 epochs, seed 20260524). Shipping it skips roughly
  two hours of fine-tuning, and guarantees the defense arms are measured against
  exactly the same fine-tuned model as the Chapter 5 results.
- `RUN_ME.ipynb` -- the notebook to open in Colab
- `requirements-gpu.txt`

## How to run

1. Upload this zip to a folder named `piibench` in your Google Drive.
2. Open `RUN_ME.ipynb` in Colab (File -> Upload notebook), or upload the notebook
   separately and run it.
3. Set the runtime to an L4 GPU.
4. Run the cells in order.

Results are written to `MyDrive/piibench/defense_140`, so they survive a
disconnected session. Every stage checkpoints; re-running the same cell resumes.

## Expected runtime (L4)

| Stage | Time |
|---|---|
| Fine-tuning | skipped (adapter bundled) |
| Undefended audit | ~45 min |
| Output filtering | free (re-scored from stored outputs) |
| Unlearning + audit | ~50 min |
| DP-LoRA + audit | ~3 h |

Part 1 (filtering + unlearning) is about 1.5 hours; Part 2 (DP) about 3 hours.
They can be run in separate sessions.
