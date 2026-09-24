#!/usr/bin/env python
"""Build the Colab bundle for Stage 1 — the generalizable-baseline search.

Code only, no adapters: every cell trains its own from the released weights.
"""
from __future__ import annotations
import json, zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dist" / "piibench_generalization.zip"
CODE = ["pii_auditor", "generalization_baseline.py", "task_utility.py",
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


INTRO = """# Stage 1 — find a fine-tuning regime that GENERALISES

**Why this run exists.** Section 6.5.1 measured task competence on held-out people
and found that no condition had any. The undefended model reached 0.883 format
validity, but 286 of its 318 well-formed answers were the actual records of
training subjects: its rate of producing a value that is both well formed and
*novel* was 0.089, **below the released model's 0.236**. At sixty exposures per
record the fine-tuning produced recall, not a transferable schema.

That leaves every defense unpriceable. You cannot say what a defense costs in
utility when the undefended baseline has no utility to lose. So before the
privacy budget of DP-SGD can be swept meaningfully, a fine-tuning configuration
has to exist whose product actually generalises. This run looks for one.

**Two levers, crossed.**

| Lever | Why |
|---|---|
| **Exposure** (2–12 per record) | §5.6.1 puts the memorization threshold near eight; the main experiments used sixty. Below it the model may abstract a schema without storing individuals. |
| **Corpus diversity** (fixed vs varied phrasing) | The default corpus gives each person eight sentences in fixed phrasing, so memorising eight strings is cheaper than abstracting a schema. The varied corpus states the same facts through four paraphrases per field. **Document counts are identical between the two arms** — only surface diversity differs. |

**Three quantities at every cell, never conflated:**

| Measure | On whom | Question |
|---|---|---|
| `memorization` | training people | did it store the individuals? |
| `novel_valid` | held-out people | did it learn the schema? (well formed AND not any training subject's value) |
| `recitation` | held-out people | asked about a stranger, did it answer with a training subject's record? |

The third quantity is the one whose absence made the first reading of §6.5.1
wrong. It is reported at every cell here rather than derived afterwards.

**Success criterion, fixed before the run:** a cell is a usable baseline when
`novel_valid` exceeds the base model's rate significantly while `memorization`
stays low. If no cell qualifies, the honest conclusion is that this corpus cannot
teach the schema at any exposure — a reportable result about the experimental
design, and a reason **not** to proceed to the epsilon sweep.

---

**Before starting:** Runtime -> Change runtime type -> **L4 GPU**. About 45 minutes.
Every cell checkpoints to Drive."""

CELL5 = """## 5 - Run the grid

Two corpora x five exposure levels, plus a one-off base-model reference.
40 person records, 60 held-out people, LoRA rank 32 throughout.

**About 45 minutes on an L4.** Adapters go to local scratch; only the checkpoint
JSON reaches Drive."""

CELL6_CODE = r"""import json, math
from statistics import NormalDist
rows = json.load(open(f'{DRIVE}/generalization_40/generalization.json'))
b = next(r for r in rows if r['corpus'] == 'base')
bn, bnn = b['novel_valid'], b['n_heldout']

def z2(p1, p2, n1, n2):
    pp = (p1*n1 + p2*n2) / (n1+n2)
    se = math.sqrt(pp*(1-pp)*(1/n1 + 1/n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1-p2)/se
    return z, 2*NormalDist().cdf(-abs(z))

print(f'base novel_valid = {bn:.3f}  (n={bnn})')
print()
print(f"  {'corpus':<8} {'exp':>4} {'memorised':>10} {'novel':>8} {'recited':>9}  verdict")
winners = []
for r in rows:
    if r['corpus'] == 'base':
        continue
    z, pv = z2(r['novel_valid'], bn, r['n_heldout'], bnn)
    gain = r['novel_valid'] - bn
    if gain > 0 and pv < 0.05:
        v = f'GENERALISES +{gain:.3f} p={pv:.3f}'
        winners.append((r, gain, pv))
    elif r['memorization'] >= 0.5:
        v = 'memorises'
    else:
        v = 'neither'
    print(f"  {r['corpus']:<8} {r['exposures']:>4} {r['memorization']:>10.3f} "
          f"{r['novel_valid']:>8.3f} {r['recitation']:>9.3f}  {v}")
print()
if winners:
    winners.sort(key=lambda x: -x[1])
    r, gain, pv = winners[0]
    print(f"BASELINE FOUND: {r['corpus']} corpus, {r['exposures']} exposures")
    print(f"  novel {r['novel_valid']:.3f} vs base {bn:.3f} (+{gain:.3f}, p={pv:.4f})")
    print(f"  memorization {r['memorization']:.3f}")
    print('  -> Stage 2 (epsilon sweep) can proceed against this configuration.')
else:
    print('NO CELL GENERALISES.')
    print('  Every configuration either memorises or learns nothing transferable.')
    print('  -> Do NOT run the epsilon sweep. Report the negative result: the')
    print('     utility cost of the defenses cannot be priced on this corpus.')"""

CELL7_CODE = r"""import shutil, os, glob
os.makedirs('/content/send', exist_ok=True)
src = f'{DRIVE}/generalization_40'
for pat in ['generalization.json']:
    for f in glob.glob(f'{src}/{pat}'):
        shutil.copy(f, '/content/send/')
shutil.make_archive(f'{DRIVE}/generalization_results', 'zip', '/content/send')
sz = os.path.getsize(f'{DRIVE}/generalization_results.zip')
print('wrote', f'{DRIVE}/generalization_results.zip', round(sz/1024, 1), 'KB')
print('Download it from Drive and send it back.')"""

RUN = ("!python generalization_baseline.py \\\n"
       "  --model Qwen/Qwen2.5-1.5B \\\n"
       "  --persons 40 \\\n"
       "  --epochs-list 1 2 3 4 6 \\\n"
       "  --corpora fixed varied \\\n"
       "  --rank 32 \\\n"
       "  --out-dir $DRIVE/generalization_40 \\\n"
       "  --adapter-dir /content/adapters_gen")

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
       "Upload `piibench_generalization.zip` into the **`piibench` folder** of your "
       "Drive first, then run this cell."),
    code("import zipfile, os\n"
         "src = f'{DRIVE}/piibench_generalization.zip'\n"
         "assert os.path.exists(src), f'not found: {src} - upload the zip to Drive first'\n"
         "zipfile.ZipFile(src).extractall('/content/work')\n"
         "os.chdir('/content/work')\n"
         "print(sorted(os.listdir('.')))"),
    md("## 4 - Install dependencies\n(about 2 minutes)"),
    code("!pip install -q transformers accelerate bitsandbytes peft datasets pandas"),
    md(CELL5),
    code(RUN),
    md("## 6 - Read the grid\n\n"
       "Cell 5 prints this too; this re-prints it without recomputing anything."),
    code(CELL6_CODE),
    md("## 7 - Collect the results\n\nA few kilobytes of JSON."),
    code(CELL7_CODE),
    md("---\n"
       "### If the session drops\n"
       "Re-run cells 2 -> 3 -> 4, then cell 5. You should see "
       "`resuming: N cells already measured`.\n\n"
       "### If you hit CUDA out of memory\n"
       "Nothing here is heavy — rank 32 at 40 records. If it happens anyway, reduce "
       "the probe batch by editing `probe(..., batch_size=16)` in "
       "`generalization_baseline.py`.\n\n"
       "### What good output looks like\n"
       "Each cell prints `[corpus, N exposures] M docs, fine-tuning ...` then "
       "`memorization X | novel_valid Y | recitation Z`. What you are hoping to see "
       "is a row where `novel_valid` is clearly above the base value while "
       "`memorization` is still small. A row where `recitation` is large and "
       "`novel_valid` is small is the failure mode §6.5.1 found."),
]

README = """# Stage 1: generalizable-baseline search - Colab bundle

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
        z.writestr("RUN_ME_GENERALIZATION.ipynb",
                   json.dumps(NOTEBOOK, ensure_ascii=False, indent=1))
        z.writestr("README_GENERALIZATION.md", README)
    print(f"wrote {OUT}")
    print(f"  code files : {n}")
    print(f"  size       : {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
