#!/usr/bin/env python
"""Build the Colab bundle for the LoRA rank x exposure ablation (run 2).

Run 1 varied rank alone at 30 epochs and hit a ceiling: 0.995 at rank 4, 1.000
everywhere else. Section 5.6.1 had already shown saturation at 16 exposures and
run 1 used 60, so no cell could discriminate. This bundle sweeps rank against
exposure with the operating point moved to the knee of the exposure curve.

Code only, no adapters: every cell trains its own from the released weights.
"""
from __future__ import annotations
import json, zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dist" / "piibench_rank_ablation.zip"

CODE = ["pii_auditor", "ablation_rank.py", "ablation_epochs.py",
        "requirements-gpu.txt"]

DRIVE = "/content/drive/MyDrive/piibench"

NOTEBOOK = {"cells": [], "metadata": {"accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def md(t):
    return {"cell_type": "markdown", "metadata": {},
            "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}


INTRO = """# LoRA Rank x Exposure Ablation (run 2)

**Run 1 was uninformative, and this run fixes it.** Varying rank alone at 30 epochs
returned 0.995 at rank 4 and 1.000 at every larger rank - a ceiling. The exposure
ablation in Section 5.6.1 had already shown that memorization saturates at 16
exposures per record, and run 1 used 60, so every cell sat deep in the saturated
regime and the probe had no room to discriminate. That is a measurement problem,
not a negative result.

This run moves the operating point to where the outcome can still move, and sweeps
**rank against exposure** rather than rank alone.

| Reading | What it means for the thesis |
|---|---|
| Rate rises with rank in the unsaturated rows | Adapter-capacity explanation **confirmed**; 5.6 upgrades from hypothesis to evidence |
| Rows flat wherever they are not saturated | Explanation **refuted**; 5.6 withdraws the mechanism and reports the capacity question as open |
| Saturation threshold shifts right as rank falls | The sharper confirmation - a smaller adapter needs more exposures to hold the same corpus |

Send the JSON back either way. A refutation rewrites the chapter; it is not a wasted run.

---

**Before starting:** Runtime -> Change runtime type -> **L4 GPU**.

Every grid cell checkpoints to Drive, so a disconnect costs at most the cell in flight."""

CELL5 = """## 5 - Run the grid: 5 ranks x 5 exposure levels

40 records; exposures 2 / 4 / 6 / 8 / 12 per record. These bracket the knee of the
exposure curve, which sits near 8 exposures where the rate is about 0.60.

**About 75 minutes on an L4.** Twenty-five cells, but each trains far fewer epochs
than run 1, so the total is close to what run 1 cost - and this time it carries
information.

Adapters go to local scratch; only the checkpoint JSON and the figure reach Drive."""

CELL6_CODE = r"""import json, os
from IPython.display import Image, display
rows = json.load(open(f'{DRIVE}/rank_grid_40/ablation_rank.json'))
ranks = sorted({r['rank'] for r in rows})
print('  exposures | ' + ' '.join(f'r={r:<4}' for r in ranks))
informative = []
for ex in sorted({r['exposures'] for r in rows}):
    cells = {r['rank']: r['mem_rate'] for r in rows if r['exposures'] == ex}
    line = ' '.join(f"{cells.get(r, float('nan')):<6.3f}" for r in ranks)
    vals = list(cells.values())
    if min(vals) >= 0.98:
        note = '  <- ceiling'
    elif max(vals) <= 0.02:
        note = '  <- floor'
    else:
        informative.append((ex, cells))
        note = f'  <- INFORMATIVE (spread {max(vals) - min(vals):+.3f})'
    print(f'  {ex:>9} | {line}{note}')
print()
if not informative:
    print('  No informative row. Lower --epochs-list further and re-run cell 5;')
    print('  completed cells are skipped, so nothing is recomputed.')
else:
    rise = sum(1 for ex, c in informative if c[max(c)] - c[min(c)] > 0.10)
    print(f'  {len(informative)} informative row(s), {rise} rising with rank.')
    if rise:
        print('  RISING -> supports the adapter-capacity explanation of section 5.6.')
    else:
        print('  FLAT   -> refutes it; section 5.6 gets rewritten. Still a result.')
fig = f'{DRIVE}/rank_grid_40/fig_ablation_rank.png'
if os.path.exists(fig):
    display(Image(fig))"""

CELL8_CODE = r"""import shutil, os, glob
os.makedirs('/content/send', exist_ok=True)
for d in ['rank_grid_40', 'rank_ablation_40', 'rank_ablation_140']:
    src = f'{DRIVE}/{d}'
    if not os.path.isdir(src):
        continue
    dst = f'/content/send/{d}'
    os.makedirs(dst, exist_ok=True)
    for pat in ['ablation_rank.json', 'fig_ablation_rank.png', 'fig_ablation_rank.pdf']:
        for f in glob.glob(f'{src}/{pat}'):
            shutil.copy(f, dst)
shutil.make_archive(f'{DRIVE}/rank_grid_results', 'zip', '/content/send')
sz = os.path.getsize(f'{DRIVE}/rank_grid_results.zip')
print('wrote', f'{DRIVE}/rank_grid_results.zip', round(sz / 1024, 1), 'KB')
print('Download it from Drive and send it back.')"""

RUN_GRID = (
    "!python ablation_rank.py \\\n"
    "  --model Qwen/Qwen2.5-1.5B \\\n"
    "  --persons 40 \\\n"
    "  --ranks 4 8 16 32 64 \\\n"
    "  --epochs-list {eps} \\\n"
    "  --out-dir $DRIVE/rank_grid_40 \\\n"
    "  --adapter-dir /content/adapters_grid"
)

NOTEBOOK["cells"] = [
    md(INTRO),

    md("## 1 - Confirm the GPU"),
    code("!nvidia-smi"),

    md("## 2 - Mount Google Drive"),
    code("from google.colab import drive\n"
         "drive.mount('/content/drive')\n"
         "import os\n"
         f"DRIVE = '{DRIVE}'\n"
         "os.makedirs(DRIVE, exist_ok=True)\n"
         "print('results will be written to:', DRIVE)"),

    md("## 3 - Unpack the bundle\n\n"
       "Upload `piibench_rank_ablation.zip` into the **`piibench` folder** of your "
       "Drive first (it replaces the run-1 zip if that is still there), then run "
       "this cell."),
    code("import zipfile, os\n"
         "src = f'{DRIVE}/piibench_rank_ablation.zip'\n"
         "assert os.path.exists(src), f'not found: {src} - upload the zip to Drive first'\n"
         "zipfile.ZipFile(src).extractall('/content/work')\n"
         "os.chdir('/content/work')\n"
         "print(sorted(os.listdir('.')))"),

    md("## 4 - Install dependencies\n(about 2 minutes)"),
    code("!pip install -q transformers accelerate bitsandbytes peft datasets pandas matplotlib"),

    md(CELL5),
    code(RUN_GRID.format(eps="1 2 3 4 6")),

    md("## 6 - Read the grid\n\n"
       "Cell 5 prints this too; this re-prints it without recomputing anything and "
       "shows the figure."),
    code(CELL6_CODE),

    md("## 7 - Only if cell 6 found no informative row\n\n"
       "Adds one higher exposure level. Completed cells are skipped, so nothing "
       "already measured is recomputed."),
    code(RUN_GRID.format(eps="1 2 3 4 6 8")),

    md("## 8 - Collect the results\n\nA few hundred kilobytes: JSON and figures only."),
    code(CELL8_CODE),

    md("---\n"
       "### If the session drops\n"
       "Re-run cells 2 -> 3 -> 4, then cell 5. You should see "
       "`resuming: N grid cells already measured`.\n\n"
       "### If you hit CUDA out of memory\n"
       "Rank 64 is the heaviest. Drop it: `--ranks 4 8 16 32`. Four ranks still show "
       "the shape.\n\n"
       "### What good output looks like\n"
       "Each cell prints `[rank R, E exposures] fine-tuning ...` then "
       "`-> verbatim memorization rate = 0.xxxx [checkpointed]`. You want rates "
       "spread across the range, not all 0.000 and not all 1.000."),
]

README = """# LoRA rank x exposure ablation - Colab bundle (run 2)

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
"""


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for item in CODE:
            p = ROOT / item
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.suffix == ".pyc" or "__pycache__" in f.parts:
                        continue
                    z.write(f, f.relative_to(ROOT)); n += 1
            elif p.exists():
                z.write(p, p.name); n += 1
            else:
                print(f"  WARNING: {item} not found")
        z.writestr("RUN_ME_RANK.ipynb", json.dumps(NOTEBOOK, ensure_ascii=False, indent=1))
        z.writestr("README_RANK.md", README)
    print(f"wrote {OUT}")
    print(f"  code files : {n}")
    print(f"  size       : {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
