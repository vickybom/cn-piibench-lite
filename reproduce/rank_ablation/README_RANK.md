# LoRA rank x exposure ablation - Colab bundle (run 2)

## Why there is a run 2

Run 1 varied LoRA rank alone at 30 epochs (60 exposures per record) and returned
0.995 at rank 4 and 1.000 at ranks 8, 16, 32 and 64. That is a ceiling, not a
finding: Section 5.6.1 of the dissertation had already established that
memorization saturates at 16 exposures, so every cell of run 1 sat deep inside
the saturated regime where the probe cannot discriminate.

Run 2 sweeps rank AGAINST exposure and places the exposure levels around the knee
of the curve, near 8 exposures, where the rate is about 0.60 and there is room for
rank to show an effect.

## How to run

1. Upload `piibench_rank_ablation.zip` to the `piibench` folder in Google Drive.
2. Open `RUN_ME_RANK.ipynb` in Colab (File -> Upload notebook).
3. Runtime -> Change runtime type -> L4 GPU.
4. Run the cells in order. Cell 6 tells you whether the grid is informative.

## Runtime

25 cells (5 ranks x 5 exposure levels) at 40 records: about 75 minutes on an L4.

## Where things are written

- Checkpoint JSON and figures -> MyDrive/piibench/rank_grid_40/
- Per-cell adapters -> local /content/ scratch (disposable)

The JSON is rewritten after every cell, so a disconnect costs at most the cell in
flight, and re-running the same command skips completed cells.

## Reading the result

The script prints a rank x exposure table and labels each row ceiling, floor or
INFORMATIVE. Only the informative rows carry evidence.

- Rate rises with rank in those rows -> the adapter-capacity explanation of
  Section 5.6 is supported.
- Rows are flat -> the explanation is refuted, and Section 5.6 must withdraw the
  mechanism and report the capacity question as open.

Send the JSON back in either case.
