#!/usr/bin/env python
"""Build the Colab bundle for Stage 3 — the clipping control and the replication.

Two independent experiments in one notebook. The clipping control runs first
because it is shorter and the more consequential of the two.
"""
from __future__ import annotations
import json, zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dist" / "piibench_clip.zip"
CODE = ["pii_auditor", "clip_vs_noise.py", "epsilon_sweep.py",
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


INTRO = """# Stage 3 — the two things Section 6.5.2 could not settle

Section 6.5.2 found that at the generalization operating point **every** privacy
budget closed both leakage channels completely, and that at two of four budgets
the defended model ended up *better* than the released weights at the task. Two
questions were left explicitly open, and both are now written into the
dissertation as limitations. This notebook closes them.

| | Question | Cost |
|---|---|---|
| **Part CLIP** | Is the protection coming from the noise, or from the clipping? | ≈ 2.6 h |
| **Part REPLICATE** | Is the spread across budgets real, or is it run-to-run noise? | ≈ 4.8 h |

**Run Part CLIP first.** It is shorter and it can change the mechanistic reading
of the whole chapter. If you only finish one session, finish that one.

---

## Part CLIP — why a single "no noise" run would not have been enough

Both channels were exactly zero across a **6.6-fold** change in the noise
multiplier, including the loosest budget where very little noise is added. A
quantity that does not move when its supposed cause moves 6.6-fold is probably
not being driven by that cause. The component DP-SGD applies identically at
every budget is **per-example gradient clipping**.

The obvious control is "clipping, no noise". On its own that would not settle
it, because the private and undefended arms of Section 6.5.2 **differ in more
than noise**:

| | undefended arm | private arm |
|---|---|---|
| code path | `lora_finetune` | `dp_sgd_finetune` |
| LoRA dropout | 0.0 | 0.05 |
| max_len | 384 | 320 |
| optimizer | HuggingFace Trainer, scheduled | constant-rate AdamW |

Any one of those could suppress memorization by itself. So this part runs **two
cells through the same function**, differing only in the clipping bound:

| cell | clip_norm | noise | meaning |
|---|---|---|---|
| `clip_on` | 1.0 | 0 | clipping active, no noise at all |
| `clip_off` | 1e9 | 0 | identical path, clipping inert |

**How to read it**

- `clip_off` leaks, `clip_on` clean → **clipping is the mechanism.** The
  protection in Section 6.5.2 does not depend on the budget, so the budget can
  be chosen for its guarantee rather than for its effect.
- both clean → **neither clipping nor noise.** Something else in that code path
  suppresses memorization, and Section 6.5.2's mechanistic paragraph has to be
  replaced rather than refined.
- both leak → contradicts the invariance that motivated the control; reconcile
  before reporting either.

Neither cell carries a privacy guarantee. They are **mechanistic controls, not
candidate defenses**, and the script says so in its own output.

---

## Part REPLICATE — how big is the run-to-run noise, really?

Utility was not monotone in epsilon and each budget was trained once, so budget
is confounded with run. The confound is known to be large: retraining the
*undefended* condition moved held-out competence by 0.10 on an identical probe
set, against a 0.20 spread across four budgets.

This part trains **3 seeded runs at each of ε = 0.50 and ε = 1.00** — the pair
whose inversion was largest, 0.391 against 0.211 — and puts the within-budget
spread next to the between-budget difference.

The replicates audit a **stratified 25 % sample** of the twelve-template matrix
rather than all 11,760 probes: every DP condition in Section 6.5.2 returned
exactly zero over the full matrix, so the sample is there to confirm that zero,
while the quantity that actually varies between runs — held-out competence — is
measured in full on all 900 held-out probes.

---

**Before starting:** Runtime → Change runtime type → **L4 GPU**. Everything
checkpoints per condition; re-running a cell resumes where it stopped."""

CELL5 = """## 5 — Part CLIP (about 2.6 hours)

Two cells, same code path, differing only in whether the per-example gradient
norm is bounded. Each is trained and then audited on the full twelve-template
matrix plus the 900-probe held-out set, so the numbers land on the same
instrument as Table 6.8.

The script prints a VERDICT block naming which of the three outcomes occurred."""

CELL7 = """## 7 — Part REPLICATE (about 4.8 hours)

Six trainings: three seeded runs at ε = 0.50 and three at ε = 1.00.

**This can be run in a separate session.** If the runtime drops, re-run cells
2 → 3 → 4 and then this cell; completed runs are detected and skipped."""

CELL6_CODE = r"""import json
r = json.load(open(f'{DRIVE}/clip_140/clip.json'))
on, off = r['clip_on'], r['clip_off']
print(f"  {'cell':<12} {'clip_norm':>10} {'matrix hits':>13} {'RW-MER':>8} {'recite':>8} {'novel':>8}")
for k in ('clip_on','clip_off'):
    x = r[k]
    print(f"  {k:<12} {x['clip_norm']:>10.0e} {str(x['raw_hits'])+'/'+str(x['n_probes']):>13} "
          f"{x['rwmer']:>8.4f} {x['recitation']:>8.4f} {x['novel_valid']:>8.4f}")
print(f"  {'-'*62}")
print(f"  {'undefended':<12} {'n/a':>10} {'38/11760':>13} {0.0022:>8.4f} {0.1289:>8.4f} {0.5278:>8.4f}")
print(f"  {'DP (all 4)':<12} {'1e+00':>10} {'0/11760':>13} {0.0:>8.4f} {0.0:>8.4f} {'.21-.41':>8}")
print()
ol = off['raw_hits'] > 0 or off['recitation'] > 0.02
cl = on['raw_hits'] > 0 or on['recitation'] > 0.02
if ol and not cl:
    print('  CLIPPING is the mechanism. The budget can be chosen for its')
    print('  guarantee rather than for its effect.')
elif not ol and not cl:
    print('  NEITHER clipping nor noise. Something else in the DP code path')
    print('  (dropout 0.05 / max_len 320 / constant-rate AdamW) suppresses it.')
    print('  Next control: vary those three one at a time.')
else:
    print('  Unexpected pattern - see the VERDICT block printed by cell 5.')
print()
print('  where any surviving hits sit:')
for k in ('clip_on','clip_off'):
    cats = {c: e['hits'] for c, e in r[k]['matrix']['by_category'].items() if e['hits']}
    print(f"    {k:<10} {cats if cats else 'none'}")"""

CELL8_CODE = r"""import json, statistics as st
r = json.load(open(f'{DRIVE}/clip_140/replicate.json'))
groups = {}
for k, v in r.items():
    if '_run' in k:
        groups.setdefault(k.split('_run')[0], []).append(v)
print(f"  {'budget':<12} {'runs':>5} {'mean':>9} {'sd':>8} {'min':>8} {'max':>8} {'range':>8}  hits")
stats = {}
for g in sorted(groups):
    vs = sorted(x['novel_valid'] for x in groups[g])
    sd = st.stdev(vs) if len(vs) > 1 else 0.0
    stats[g] = (st.mean(vs), vs)
    h = sum(x['raw_hits'] for x in groups[g])
    print(f"  {g:<12} {len(vs):>5} {st.mean(vs):>9.4f} {sd:>8.4f} {vs[0]:>8.4f} "
          f"{vs[-1]:>8.4f} {vs[-1]-vs[0]:>8.4f}  {h}")
print()
if len(stats) >= 2:
    gs = sorted(stats, key=lambda g: stats[g][0])
    between = stats[gs[-1]][0] - stats[gs[0]][0]
    within = max(v[1][-1] - v[1][0] for v in stats.values())
    print(f"  between-budget difference of means : {between:.4f}")
    print(f"  largest within-budget range        : {within:.4f}")
    print(f"  the single-run difference reported in Table 6.8 was 0.1800")
    print()
    if within >= abs(between):
        print('  Run spread within one budget is as large as the gap between')
        print('  budgets. Table 6.8 ordering is noise; Section 6.5.2 was right to')
        print('  refuse to read it as a curve, and can now say so with evidence.')
    else:
        print('  The between-budget gap survives the within-budget spread. There is')
        print('  a real budget effect, and the non-monotone ordering needs an')
        print('  explanation rather than a variance argument.')"""

CELL9_CODE = r"""import shutil, os, glob
os.makedirs('/content/send', exist_ok=True)
src = f'{DRIVE}/clip_140'
for pat in ['clip.json', 'replicate.json', 'dataset.json', 'records_*.csv']:
    for f in glob.glob(f'{src}/{pat}'):
        shutil.copy(f, '/content/send/')
sent = sorted(os.listdir('/content/send'))
for f in sent:
    print(f"  {f:<42} {os.path.getsize('/content/send/'+f)/1024:>9.1f} KB")
shutil.make_archive(f'{DRIVE}/clip_results', 'zip', '/content/send')
sz = os.path.getsize(f'{DRIVE}/clip_results.zip')
print(f"\n  {len(sent)} files -> {DRIVE}/clip_results.zip  ({sz/1e6:.1f} MB)")
assert any(f.startswith('records_matrix_clip') for f in sent), \
    'clip control records missing - run cell 5 first'
print('  Download from Drive and send it back.')"""

PART_CLIP = ("!python clip_vs_noise.py --part clip \\\n"
             "  --model Qwen/Qwen2.5-1.5B \\\n"
             "  --persons 140 --epochs 2 --repeats 2 \\\n"
             "  --heldout-persons 180 \\\n"
             "  --out-dir $DRIVE/clip_140 \\\n"
             "  --adapter-dir /content/adapters_clip")

PART_REP = ("!python clip_vs_noise.py --part replicate \\\n"
            "  --model Qwen/Qwen2.5-1.5B \\\n"
            "  --persons 140 --epochs 2 --repeats 2 \\\n"
            "  --heldout-persons 180 \\\n"
            "  --budgets 0.5 1.0 --multipliers 0.648 0.342 \\\n"
            "  --runs 3 --matrix-fraction 0.25 \\\n"
            "  --out-dir $DRIVE/clip_140 \\\n"
            "  --adapter-dir /content/adapters_rep")

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
       "Upload `piibench_clip.zip` into the **`piibench` folder** of your Drive "
       "first, then run this cell."),
    code("import zipfile, os\n"
         "src = f'{DRIVE}/piibench_clip.zip'\n"
         "assert os.path.exists(src), f'not found: {src} - upload the zip to Drive first'\n"
         "zipfile.ZipFile(src).extractall('/content/work')\n"
         "os.chdir('/content/work')\n"
         "print(sorted(os.listdir('.')))"),
    md("## 4 — Install dependencies\n(about 2 minutes)"),
    code("!pip install -q transformers accelerate bitsandbytes peft datasets pandas"),
    md(CELL5),
    code(PART_CLIP),
    md("## 6 — Read the clipping control"),
    code(CELL6_CODE),
    md(CELL7),
    code(PART_REP),
    md("## 8 — Read the replication"),
    code(CELL8_CODE),
    md("## 9 — Collect the results"),
    code(CELL9_CODE),
    md("---\n"
       "### If the session drops\n"
       "Re-run cells 2 → 3 → 4, then the cell you were on. Completed conditions "
       "are skipped, and saved adapters are reused when they are still on disk.\n\n"
       "### If you hit CUDA out of memory\n"
       "Add `--batch-size 12`.\n\n"
       "### What good output looks like\n"
       "Part CLIP prints two `[clip]` blocks, each ending in `-> RW-MER … hits … "
       "recitation … novel …`, then a VERDICT block. Part REPLICATE prints six "
       "`[rep]` blocks then a table of means and ranges.\n\n"
       "### A note on what these runs are\n"
       "The two clipping cells have **no privacy guarantee** — the noise "
       "multiplier is zero, so epsilon is unbounded. They exist to identify the "
       "mechanism, not to be deployed, and the summary JSON records that in a "
       "`guarantee` field."),
]

README = """# Stage 3: clipping control + replication - Colab bundle

Two experiments Section 6.5.2 identified as its own open questions.

## Part CLIP (about 2.6 h) - run this first

Section 6.5.2: both leakage channels were exactly zero across a 6.6x change in
the noise multiplier. A quantity that does not move when its supposed cause
moves 6.6-fold is probably not driven by that cause. The candidate is
per-example gradient clipping, which DP-SGD applies identically at every budget.

A single "no noise" run would not settle it, because the private and undefended
arms differ in more than noise: lora_finetune vs dp_sgd_finetune, dropout 0.0 vs
0.05, max_len 384 vs 320, scheduled optimizer vs constant-rate AdamW.

So both cells go through dp_sgd_finetune, differing only in the bound:

  clip_on    clip_norm 1.0   noise 0    clipping active
  clip_off   clip_norm 1e9   noise 0    clipping inert

  clip_off leaks, clip_on clean -> clipping is the mechanism
  both clean                    -> neither; something else in that code path
  both leak                     -> contradicts the invariance; reconcile first

NEITHER CELL HAS A PRIVACY GUARANTEE. They are mechanistic controls, not
candidate defenses. The JSON records this in a `guarantee` field.

## Part REPLICATE (about 4.8 h)

Utility was not monotone in epsilon and each budget was trained once, so budget
is confounded with run. Retraining the undefended condition moved held-out
competence by 0.10 against a 0.20 spread across budgets.

Three seeded runs at eps 0.50 and three at eps 1.00 - the pair whose inversion
was largest (0.391 vs 0.211). Reports within-budget spread against
between-budget difference.

Replicates audit a stratified 25% sample of the matrix (2,940 of 11,760, equal
share of every template) rather than all of it: every DP condition in Section
6.5.2 returned exactly zero over the full matrix, so the sample confirms the
zero. Held-out competence, the quantity that varies, is measured in full on all
900 probes.

## Runtime

Part CLIP about 2.6 h, Part REPLICATE about 4.8 h. Checkpointed per condition;
they can be run in separate sessions.

## Send back

The whole clip_results.zip - summaries and per-query records together.
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
        z.writestr("RUN_ME_CLIP.ipynb",
                   json.dumps(NOTEBOOK, ensure_ascii=False, indent=1))
        z.writestr("README_CLIP.md", README)
    print(f"wrote {OUT}")
    print(f"  code files : {n}")
    print(f"  size       : {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
