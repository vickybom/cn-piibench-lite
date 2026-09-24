#!/usr/bin/env python
"""Build a single self-contained bundle for the 140-record defense evaluation.

The bundle carries the toolkit, the already-trained 140-record LoRA adapter (so
the two-hour fine-tuning step is skipped), and a notebook with the exact
commands. Upload it to Google Drive once and it runs from any machine.
"""
from __future__ import annotations
import json, os, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dist" / "piibench_defense140.zip"
ADAPTER = ROOT / "results_main_15b_140" / "adapter"

CODE = ["pii_auditor", "defense_eval.py", "analyze_records.py", "rerun_main.py",
        "rerun_unlearning.py", "unlearn_sweep.py", "make_figures.py",
        "requirements-gpu.txt"]

NOTEBOOK = {
 "cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
 "kernelspec": {"display_name": "Python 3", "name": "python3"},
 "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}

def md(t): return {"cell_type": "markdown", "metadata": {},
                   "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": t.splitlines(keepends=True)}

NOTEBOOK["cells"] = [
 md("# Defense Evaluation at 140 Records\n\n"
    "Re-runs the three privacy defenses at the 140-record scale so that Chapter 6 "
    "matches Chapter 5. The 140-record LoRA adapter ships inside this bundle, so the "
    "two-hour fine-tuning step is **skipped**.\n\n"
    "**Before starting:** Runtime → Change runtime type → **L4 GPU** (T4 also works, "
    "roughly twice as slow).\n\n"
    "Everything is written to Google Drive, and every stage checkpoints, so a "
    "disconnect costs only the stage in flight — just re-run the same cell."),

 md("## 1 · Confirm the GPU"),
 code("!nvidia-smi"),

 md("## 2 · Mount Google Drive"),
 code("from google.colab import drive\n"
      "drive.mount('/content/drive')\n"
      "import os\n"
      "DRIVE = '/content/drive/MyDrive/piibench'\n"
      "os.makedirs(DRIVE, exist_ok=True)\n"
      "print('results will be written to:', DRIVE)"),

 md("## 3 · Unpack the bundle\n"
    "Upload `piibench_defense140.zip` to the **`piibench` folder** in your Drive "
    "first (drag it in from the Drive web page). This cell unpacks it into the "
    "session."),
 code("import zipfile, os\n"
      "src = f'{DRIVE}/piibench_defense140.zip'\n"
      "assert os.path.exists(src), f'not found: {src} -- upload the zip to Drive first'\n"
      "zipfile.ZipFile(src).extractall('/content/work')\n"
      "os.chdir('/content/work')\n"
      "print(sorted(os.listdir('.')))\n"
      "print('adapter present:', os.path.exists('adapter/adapter_config.json'))"),

 md("## 4 · Install dependencies\n(~2 minutes.)"),
 code("!pip install -q transformers accelerate bitsandbytes peft datasets pandas"),

 md("## 5 · Part 1 — output filtering and unlearning\n"
    "Reuses the bundled adapter, so no fine-tuning. Skips the base arm, which was "
    "already measured at 0.000 for this corpus.\n\n"
    "**About 1.5 hours on an L4.**"),
 code("!python defense_eval.py --model Qwen/Qwen2.5-1.5B --persons 140 \\\n"
      "  --adapter adapter --skip-dp --skip-base \\\n"
      "  --out-dir $DRIVE/defense_140"),

 md("## 6 · Part 2 — differentially private fine-tuning\n"
    "Same output directory, so Part 1 is detected and skipped; only DP-LoRA runs.\n\n"
    "**About 3 hours on an L4** — DP-SGD computes one backward pass per example in "
    "order to clip per-example gradients, which is inherently slow.\n\n"
    "If you would rather stop after Part 1, skip this cell and tell me; the DP figure "
    "from the 40-record run can be reported with a scope note instead."),
 code("!python defense_eval.py --model Qwen/Qwen2.5-1.5B --persons 140 \\\n"
      "  --adapter adapter --skip-base \\\n"
      "  --out-dir $DRIVE/defense_140"),

 md("## 7 · Offline analysis (seconds, no GPU)"),
 code("!python analyze_records.py $DRIVE/defense_140/records.csv \\\n"
      "  --out $DRIVE/defense_140/analysis"),

 md("## 8 · Collect the results\n"
    "The files are already in Drive. This cell just makes a small archive of the "
    "parts to send back (it excludes the multi-hundred-megabyte adapters)."),
 code("import shutil, os, glob\n"
      "src = f'{DRIVE}/defense_140'\n"
      "os.makedirs('/content/send', exist_ok=True)\n"
      "for pat in ['results.json','summary.csv','per_category.csv','mer_long.csv',\n"
      "            'records.csv','rec_*.csv']:\n"
      "    for f in glob.glob(f'{src}/{pat}'):\n"
      "        shutil.copy(f, '/content/send/')\n"
      "if os.path.isdir(f'{src}/analysis'):\n"
      "    shutil.copytree(f'{src}/analysis', '/content/send/analysis', dirs_exist_ok=True)\n"
      "shutil.make_archive(f'{DRIVE}/defense140_results', 'zip', '/content/send')\n"
      "print('wrote', f'{DRIVE}/defense140_results.zip',\n"
      "      round(os.path.getsize(f'{DRIVE}/defense140_results.zip')/1e6, 1), 'MB')"),

 md("---\n"
    "### If the session drops\n"
    "Re-run cells 2 → 3 → 4, then the cell you were on. Completed work is detected "
    "and skipped; you should see lines such as `resuming: N queries already on disk` "
    "or `already complete`.\n\n"
    "### If you hit CUDA out of memory\n"
    "Add `--batch-size 12` to the command.\n\n"
    "### Watch for\n"
    "`[C4]` should print `[unlearn] NNN LoRA tensors enabled for gradient ascent`, and "
    "`[C5]` prints a per-epoch progress line with an ETA."),
]

README = """# Defense evaluation at 140 records

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
"""

def main():
    assert (ADAPTER / "adapter_config.json").exists(), f"adapter not found at {ADAPTER}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_code = n_adapter = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for item in CODE:
            p = ROOT / item
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.suffix == ".pyc" or "__pycache__" in f.parts:
                        continue
                    z.write(f, f.relative_to(ROOT)); n_code += 1
            elif p.exists():
                z.write(p, p.name); n_code += 1
        # adapter, excluding the training checkpoint directory
        for f in ADAPTER.rglob("*"):
            if not f.is_file() or "_train" in f.parts:
                continue
            z.write(f, Path("adapter") / f.relative_to(ADAPTER)); n_adapter += 1
        z.writestr("RUN_ME.ipynb", json.dumps(NOTEBOOK, ensure_ascii=False, indent=1))
        z.writestr("README_DEFENSE140.md", README)
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}")
    print(f"  code files    : {n_code}")
    print(f"  adapter files : {n_adapter}")
    print(f"  size          : {mb:.1f} MB")


if __name__ == "__main__":
    main()
