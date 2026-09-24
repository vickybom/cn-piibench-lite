#!/usr/bin/env python
"""Build the Colab bundle for Stage 5 — replicating E6 and E9 at a second seed."""
from __future__ import annotations
import json, zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dist" / "piibench_replicate.zip"
CODE = ["pii_auditor", "replicate_main.py", "requirements-gpu.txt"]
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


INTRO = """# Stage 5 — the one replication the thesis owes itself

## Why

§6.5.3 established that at this study's operating points **a single training run
is a draw, not a result**. It is where the ε = 0.50 utility claim died: it
looked significant on one run and vanished over three.

Chapter 5's conclusions were each measured on one run. For most that is
unlikely to matter — they sit at sixty exposures where memorization saturates
near 1.0 and a run has little room to differ. **Two do not have that
protection:**

| | The claim | The margin |
|---|---|---|
| **E6** capacity | 1.5B reproduces 0.5235 against 3B's 0.4301 on the eight direct-memorization templates | difference **0.0934** against a run-variance estimate of **0.0276** — about 3× the noise, on one run each |
| **E9** CLMD | matched-pair differential on the aligned model is **−0.048** | *smaller than twice the run-variance estimate*, and it is quoted in the abstract |

The paired design of E9 should cancel much of the variance — both halves come
from the same checkpoint — but "should" is not a measurement.

An examiner who has read §6.5.3 will ask why Chapter 5 gets to be an exception.
This closes that.

## What varies, and what does not

**The corpus seed stays at 20260524** — the same 140 people, the same values.
**Only the training seed changes**, from 42 to 1337. Anything that moves is
run-to-run variance in fine-tuning and nothing else.

Until now the training seed was not a parameter at all: every run in this study
used HuggingFace's default of 42 implicitly. It is now exposed, defaulting to 42
so that nothing already collected is affected.

## Pre-registered reading, fixed before the run

**E6 is confirmed** if the capacity difference keeps its sign and stays clear of
the run variance. **In trouble** if the sign flips. **Needs restating** if the
margin falls to the noise level.

**E9 is confirmed** if the matched-pair differential stays small and the
unpaired formulation still departs from it — ideally still with opposite signs
on the aligned model. Note that the *methodological* claim (differencing
non-matched template families is unsafe) survives even if the substantive number
moves, because it is a claim about the **gap** between the two formulations
rather than about either value.

The script evaluates all of this itself and prints the verdict.

---

**Before starting:** Runtime → Change runtime type → **L4 GPU**.
Cell 5 (1.5B + 3B) ≈ 4 h. Cell 7 (Instruct) ≈ 1.5 h. Checkpointed per model —
separate sessions are fine."""

CELL5 = """## 5 — The capacity pair: 1.5B and 3B (about 4 hours)

Both models fine-tuned at seed 1337 on the same corpus, then audited on the full
twelve-template matrix. E6's conclusion is read on the A+B scope (7,840 probes);
the full-matrix number is reported alongside because §5.12 records that the sign
of the aggregate depends on the scope.

The 3B model is the slow half — larger to train and to sample from."""

CELL7 = """## 7 — The aligned model (about 1.5 hours)

This is the one that carries E9's headline: the matched-pair differential of
−0.048 quoted in the abstract.

**Can be a separate session.** Re-run cells 2 → 3 → 4 first; the completed
models are detected and skipped."""

CELL6_CODE = r"""import json
r = json.load(open(f'{DRIVE}/replicate_140/replicate.json'))
ORIG = {'1.5B': 0.5235, '3B': 0.4301}
print(f"  {'model':<16}{'A+B orig':>10}{'A+B seed1337':>14}{'moved':>9}{'full matrix':>13}")
for lab in ('1.5B', '3B', '1.5B-Instruct'):
    k = f'{lab}_seed1337'
    if k not in r:
        print(f"  {lab:<16}{'(not yet run)':>10}"); continue
    v = r[k]; o = ORIG.get(lab)
    mv = f"{v['ab_scope_rate']-o:+.4f}" if o else '-'
    print(f"  {lab:<16}{o if o else '-':>10}{v['ab_scope_rate']:>14.4f}{mv:>9}"
          f"{v['full_matrix_rate']:>13.4f}")
a, b = r.get('1.5B_seed1337'), r.get('3B_seed1337')
if a and b:
    d = a['ab_scope_rate'] - b['ab_scope_rate']
    print(f"\n  capacity difference (1.5B - 3B): original +0.0934, now {d:+.4f}")
    print(f"  run-variance estimate quoted in the thesis: 0.0276")
    print('  ' + ('CONFIRMED' if d > 0 and abs(d) > 2*0.0276 else
                  'SIGN HOLDS, margin shrunk' if d > 0 else 'SIGN FLIPPED'))"""

CELL8_CODE = r"""import json
r = json.load(open(f'{DRIVE}/replicate_140/replicate.json'))
print(f"  {'model':<16}{'matched':>10}{'unpaired':>10}{'gap':>9}")
for lab in ('1.5B', '3B', '1.5B-Instruct'):
    k = f'{lab}_seed1337'
    if k not in r: continue
    c = r[k]['clmd']
    print(f"  {lab:<16}{c['matched_mean']:>+10.4f}{c['unpaired']:>+10.4f}"
          f"{c['unpaired']-c['matched_mean']:>+9.4f}")
i = r.get('1.5B-Instruct_seed1337')
if i:
    c = i['clmd']; moved = abs(c['matched_mean'] - (-0.048))
    print(f"\n  aligned model matched-pair: -0.0480 -> {c['matched_mean']:+.4f} "
          f"(moved {moved:.4f})")
    if moved > 0.048:
        print('  Moved by more than its own size: report as indistinguishable')
        print('  from zero rather than as a small negative value.')
    else:
        print('  Stable to within its own magnitude; the reported value holds.')
    if (c['unpaired'] > 0) != (c['matched_mean'] > 0):
        print('  The two formulations still carry OPPOSITE signs - the')
        print('  methodological claim of 5.8.1, reproduced at a second seed.')
    else:
        print(f"  Same sign this time, but they differ by "
              f"{c['unpaired']-c['matched_mean']:+.4f}; the claim is about the gap.")"""

CELL9_CODE = r"""import shutil, os, glob
os.makedirs('/content/send', exist_ok=True)
src = f'{DRIVE}/replicate_140'
for pat in ['replicate.json', 'dataset.json', 'records_*.csv']:
    for f in glob.glob(f'{src}/{pat}'):
        shutil.copy(f, '/content/send/')
sent = sorted(os.listdir('/content/send'))
for f in sent:
    print(f"  {f:<40} {os.path.getsize('/content/send/'+f)/1e6:>8.2f} MB")
shutil.make_archive(f'{DRIVE}/replicate_results', 'zip', '/content/send')
print(f"\n  {len(sent)} files -> {DRIVE}/replicate_results.zip "
      f"({os.path.getsize(f'{DRIVE}/replicate_results.zip')/1e6:.1f} MB)")
assert any(f.startswith('records_') for f in sent), 'per-query records missing'
print('  Download from Drive and send it back.')"""

RUN1 = ("!python replicate_main.py \\\n"
        "  --models base3b \\\n"
        "  --persons 140 --epochs 30 --repeats 2 \\\n"
        "  --seed 20260524 --train-seed 1337 \\\n"
        "  --out-dir $DRIVE/replicate_140 \\\n"
        "  --adapter-dir /content/adapters_rep5")

RUN2 = RUN1.replace("--models base3b", "--models instruct")

NOTEBOOK["cells"] = [
    md(INTRO),
    md("## 1 — Confirm the GPU"),
    code("!nvidia-smi"),
    md("## 2 — Mount Google Drive"),
    code("from google.colab import drive\n"
         "drive.mount('/content/drive')\n"
         "import os\n"
         f"DRIVE = '{DRIVE}'\n"
         "os.makedirs(DRIVE, exist_ok=True)\n"
         "print('results will be written to:', DRIVE)"),
    md("## 3 — Unpack the bundle\n\n"
       "Upload `piibench_replicate.zip` into the **`piibench` folder** of your "
       "Drive first, then run this cell."),
    code("import zipfile, os\n"
         "src = f'{DRIVE}/piibench_replicate.zip'\n"
         "assert os.path.exists(src), f'not found: {src} - upload the zip to Drive first'\n"
         "zipfile.ZipFile(src).extractall('/content/work')\n"
         "os.chdir('/content/work')\n"
         "print(sorted(os.listdir('.')))"),
    md("## 4 — Install dependencies\n(about 2 minutes)"),
    code("!pip install -q transformers accelerate bitsandbytes peft datasets pandas"),
    md(CELL5),
    code(RUN1),
    md("## 6 — Read the capacity replication"),
    code(CELL6_CODE),
    md(CELL7),
    code(RUN2),
    md("## 8 — Read the cross-lingual replication"),
    code(CELL8_CODE),
    md("## 9 — Collect the results\n\n"
       "Larger than previous stages: three full-matrix record files at 11,760 "
       "rows each."),
    code(CELL9_CODE),
    md("---\n"
       "### If the session drops\n"
       "Re-run cells 2 → 3 → 4, then the cell you were on. Completed models are "
       "skipped and saved adapters reused when still on disk.\n\n"
       "### If you hit CUDA out of memory\n"
       "The 3B model is the one at risk. Add `--batch-size 12`.\n\n"
       "### The one thing that would invalidate the run\n"
       "If `--seed` is changed from 20260524 the corpus changes too, and the "
       "result is no longer a replication — it becomes a different experiment "
       "with two things varying at once. Only `--train-seed` should differ from "
       "the original.\n\n"
       "### What a null result looks like here\n"
       "A sign flip on E6, or a CLMD that moves by more than its own magnitude, "
       "is **not a failed experiment**. It is the finding, and it is worth more "
       "than a confirmation: it would mean two Chapter 5 conclusions need "
       "restating before the defense rather than after it."),
]

README = """# Stage 5: replicate E6 and E9 at a second training seed

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
        z.writestr("RUN_ME_REPLICATE.ipynb",
                   json.dumps(NOTEBOOK, ensure_ascii=False, indent=1))
        z.writestr("README_REPLICATE.md", README)
    print(f"wrote {OUT}")
    print(f"  code files : {n}")
    print(f"  size       : {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
