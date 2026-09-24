#!/usr/bin/env python
"""Build the Colab bundle for Stage 4 — locating the noise threshold."""
from __future__ import annotations
import json, zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dist" / "piibench_threshold.zip"
CODE = ["pii_auditor", "noise_threshold.py", "clip_vs_noise.py", "epsilon_sweep.py",
        "generalization_baseline.py", "task_utility.py", "requirements-gpu.txt"]
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


INTRO = """# Stage 4 — locate the noise threshold, and price the guarantee it carries

## The gap this closes

| noise multiplier | leakage | where measured |
|---|---|---|
| **0.000** | 45 verbatim hits, recitation **0.270** | §6.5.3 clipping control |
| **0.098** | 0 hits, recitation **0.000** | §6.5.2, the loosest budget swept |

Every budget an operator would consider sits above 0.098. **The entire
transition lies inside an interval no experiment has entered.** Locating it
gives the smallest perturbation that suffices — the quantity §6.5.4 argues an
operator actually needs, because within the swept range the choice of epsilon
changes the strength of the guarantee and nothing that can be measured.

## Why the answer is likely to be uncomfortable

At the 560 lots of this configuration the accountant prices the ladder like this:

| nm | 0.000 | 0.010 | 0.020 | 0.040 | 0.070 | 0.098 |
|---|---|---|---|---|---|---|
| **ε** | unbounded | **81.6** | **28.0** | **11.5** | **5.9** | 3.99 |

If the empirical threshold sits at a small multiplier, the guarantee attached to
the smallest sufficient noise is **weak to the point of vacuity** — an epsilon in
the tens or hundreds is not a privacy statement anyone would report.

That would *sharpen* §6.5.4, not soften it. The operator who chooses ε = 0.5 is
paying for guarantee strength; this experiment says how much of that payment
buys observable protection. The expected answer is none of it.

## Two instruments

**Recitation leads.** On the full matrix the signal is 45 hits in 11,760 probes
— a rate of 0.0038. On the held-out probe it is 0.270 against a base rate of
exactly zero. That is roughly seventy times stronger per query and ten times
cheaper to collect. A stratified quarter of the matrix confirms it; at nm = 0
that quarter should return about eleven hits, so returning zero would be a
one-in-fifty-thousand event. The sample is not the weak link.

## Two seeds, because §6.5.3 happened

§6.5.3 established that a single run at this operating point is a draw, not a
result — it is where the ε = 0.50 utility claim died. **Every rung here is run at
two seeds, and the readout refuses to quote a threshold the seeds disagree
about.** Running one seed and reading a boundary off it would be repeating the
exact mistake the dissertation now devotes a section to.

The **nm = 0.000 rung is not padding**: it re-runs the zero-noise control at a
fresh seed, so the ladder has to reproduce a known endpoint (45 hits, 0.270)
before its interior is believed.

---

**Before starting:** Runtime → Change runtime type → **L4 GPU**.
Seed 1 ≈ 4 h (cell 5), seed 2 ≈ 4 h (cell 7). Checkpointed per rung — run them
in separate sessions if you like."""

CELL5 = """## 5 — Seed 1: the first ladder (about 4 hours)

Five rungs: nm 0.000, 0.010, 0.020, 0.040, 0.070. Each is trained, then audited
on a stratified quarter of the matrix plus all 900 held-out probes.

The readout at the end will say **PROVISIONAL** — that is correct and expected.
One seed cannot settle a boundary. Read it to see roughly where the transition
is, then run cell 7."""

CELL7 = """## 7 — Seed 2: confirm it (about 4 hours)

Same five rungs, different training seed. This is the cell that turns a
provisional bracket into a reportable one.

**Can be a separate session.** Re-run cells 2 → 3 → 4 first; completed rungs are
detected and skipped."""

CELL6_CODE = r"""import json, math
r = json.load(open(f'{DRIVE}/threshold_140/threshold.json'))
rungs = {}
for k, v in r.items():
    rungs.setdefault(v['noise_multiplier'], {})[v['run']] = v
print(f"  {'nm':>7} {'eps':>11} {'seed':>5} {'hits':>10} {'recite':>9} {'novel':>8}  verdict")
for nm in sorted(rungs):
    for run in sorted(rungs[nm]):
        v = rungs[nm][run]
        e = 'unbounded' if v.get('epsilon') is None else f"{v['epsilon']:.2f}"
        leaks = v['recitation'] > 0.02 or v['raw_hits'] > 0
        print(f"  {nm:>7.3f} {e:>11} {run:>5} "
              f"{str(v['raw_hits'])+'/'+str(v['n_probes']):>10} "
              f"{v['recitation']:>9.4f} {v['novel_valid']:>8.4f}  {'LEAKS' if leaks else 'clean'}")
print(f"\n  {'0.000':>7} {'unbounded':>11} {'ref':>5} {'45/11760':>10} {0.27:>9.4f} {0.4078:>8.4f}  LEAKS  (6.5.3)")
print(f"  {'0.098':>7} {'3.99':>11} {'ref':>5} {'0/11760':>10} {0.0:>9.4f} {0.4144:>8.4f}  clean  (6.5.2)")
done = [nm for nm in rungs if len(rungs[nm]) >= 2]
print(f"\n  rungs with two seeds: {len(done)}/{len(rungs)}")
if len(done) < len(rungs):
    print('  PROVISIONAL - run cell 7 before quoting any threshold.')"""

CELL8_CODE = r"""import json, math
r = json.load(open(f'{DRIVE}/threshold_140/threshold.json'))
rungs = {}
for k, v in r.items():
    rungs.setdefault(v['noise_multiplier'], {})[v['run']] = v
def leaks(v): return v['recitation'] > 0.02 or v['raw_hits'] > 0
disagree = [nm for nm in rungs if len(rungs[nm]) >= 2
            and len({leaks(x) for x in rungs[nm].values()}) > 1]
complete = [nm for nm in rungs if len(rungs[nm]) >= 2 and nm not in disagree]
if disagree:
    print(f'  SEEDS DISAGREE at nm {sorted(disagree)} - those rungs are draws, not')
    print('  boundaries. Add seeds there before quoting anything.')
leaky = [nm for nm in complete if any(leaks(x) for x in rungs[nm].values())]
clean = [nm for nm in complete if nm not in leaky]
if leaky and clean:
    lo, hi = max(leaky), min(clean)
    e = rungs[hi][min(rungs[hi])].get('epsilon')
    print(f'  THRESHOLD BRACKETED: leaks at nm {lo:.3f}, clean by nm {hi:.3f}, both seeds.')
    print(f'  Smallest sufficient noise carries epsilon {e} at delta 1e-5.')
    if e and e > 10:
        print()
        print('  That is not a reportable privacy guarantee. So the protection an')
        print('  operator can actually observe is bought at a budget nobody would')
        print('  publish, and every tighter budget buys guarantee strength alone.')
        print('  This is the Section 6.5.4 argument with a number attached.')
elif complete and not leaky:
    print('  Every rung clean - transition is below this ladder. Check the nm=0.000')
    print('  rung reproduced the control (45 hits / 0.270); if not, suspect the run.')
elif complete and not clean:
    print('  Every rung leaks - transition is above this ladder, between the top')
    print('  rung and nm 0.098. Extend upward.')"""

CELL9_CODE = r"""import shutil, os, glob
os.makedirs('/content/send', exist_ok=True)
src = f'{DRIVE}/threshold_140'
for pat in ['threshold.json', 'dataset.json', 'records_*.csv']:
    for f in glob.glob(f'{src}/{pat}'):
        shutil.copy(f, '/content/send/')
sent = sorted(os.listdir('/content/send'))
for f in sent:
    print(f"  {f:<40} {os.path.getsize('/content/send/'+f)/1024:>9.1f} KB")
shutil.make_archive(f'{DRIVE}/threshold_results', 'zip', '/content/send')
print(f"\n  {len(sent)} files -> {DRIVE}/threshold_results.zip "
      f"({os.path.getsize(f'{DRIVE}/threshold_results.zip')/1e6:.1f} MB)")
assert any(f.startswith('records_matrix') for f in sent), 'per-query records missing'
print('  Download from Drive and send it back.')"""

RUN1 = ("!python noise_threshold.py \\\n"
        "  --model Qwen/Qwen2.5-1.5B \\\n"
        "  --persons 140 --epochs 2 --repeats 2 \\\n"
        "  --heldout-persons 180 \\\n"
        "  --multipliers 0.0 0.010 0.020 0.040 0.070 \\\n"
        "  --runs 1 --matrix-fraction 0.25 \\\n"
        "  --out-dir $DRIVE/threshold_140 \\\n"
        "  --adapter-dir /content/adapters_thr")

RUN2 = RUN1.replace("--runs 1", "--runs 2")

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
       "Upload `piibench_threshold.zip` into the **`piibench` folder** of your "
       "Drive first, then run this cell."),
    code("import zipfile, os\n"
         "src = f'{DRIVE}/piibench_threshold.zip'\n"
         "assert os.path.exists(src), f'not found: {src} - upload the zip to Drive first'\n"
         "zipfile.ZipFile(src).extractall('/content/work')\n"
         "os.chdir('/content/work')\n"
         "print(sorted(os.listdir('.')))"),
    md("## 4 — Install dependencies\n(about 2 minutes)"),
    code("!pip install -q transformers accelerate bitsandbytes peft datasets pandas"),
    md(CELL5),
    code(RUN1),
    md("## 6 — Read the provisional ladder"),
    code(CELL6_CODE),
    md(CELL7),
    code(RUN2),
    md("## 8 — Read the bracketed threshold\n\n"
       "This is the cell whose output goes into the paper."),
    code(CELL8_CODE),
    md("## 9 — Collect the results"),
    code(CELL9_CODE),
    md("---\n"
       "### If the session drops\n"
       "Re-run cells 2 → 3 → 4, then the cell you were on. Completed rungs are "
       "skipped and saved adapters reused when still on disk.\n\n"
       "### If you hit CUDA out of memory\n"
       "Add `--batch-size 12`.\n\n"
       "### The one result that would invalidate the run\n"
       "If the **nm = 0.000** rung comes back clean, the ladder has failed to "
       "reproduce the §6.5.3 control (45 hits, recitation 0.270) and nothing "
       "above it can be trusted. Check that rung first, before reading the "
       "interior of the ladder.\n\n"
       "### If the transition is not inside the ladder\n"
       "Everything leaking means it sits between the top rung and 0.098 — rerun "
       "with `--multipliers 0.080 0.090`. Everything clean above nm 0 means it "
       "sits below 0.010 — rerun with `--multipliers 0.002 0.005`."),
]

README = """# Stage 4: noise threshold - Colab bundle

## The gap

  nm 0.000 -> 45 verbatim hits, recitation 0.270   (S6.5.3 clipping control)
  nm 0.098 -> 0 hits, recitation 0.000             (S6.5.2, loosest budget swept)

Every budget an operator would consider sits above 0.098, so the whole
transition lies in an interval no experiment has entered.

## Why it matters

At 560 lots the accountant prices the ladder:

  nm 0.010 -> eps 81.6    nm 0.040 -> eps 11.5
  nm 0.020 -> eps 28.0    nm 0.070 -> eps  5.9

If the threshold sits low, the smallest sufficient noise carries an epsilon in
the tens - not a guarantee anyone would report. That sharpens Section 6.5.4: the
operator choosing eps 0.5 is buying guarantee strength, and this says how much of
that buys observable protection.

## Design

Recitation leads (0.270 vs base 0.000, ~70x stronger per query than the matrix
rate of 0.0038 and 10x cheaper); a stratified 25% of the matrix confirms.

Two seeds per rung. Section 6.5.3 is what happens when a boundary is read off a
single run - it is where the eps 0.50 utility claim died. The readout refuses to
quote a threshold the seeds disagree about.

The nm=0.000 rung re-runs the zero-noise control at a fresh seed. If it does not
reproduce 45 hits / 0.270, the ladder has not reproduced its own endpoint and
nothing above it should be believed.

## Runtime

Seed 1 about 4 h, seed 2 about 4 h. Checkpointed per rung; separate sessions are
fine.

## Send back

The whole threshold_results.zip - summary and per-query records together.
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
        z.writestr("RUN_ME_THRESHOLD.ipynb",
                   json.dumps(NOTEBOOK, ensure_ascii=False, indent=1))
        z.writestr("README_THRESHOLD.md", README)
    print(f"wrote {OUT}")
    print(f"  code files : {n}")
    print(f"  size       : {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
