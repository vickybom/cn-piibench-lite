#!/usr/bin/env python
"""Build the Colab bundle for E22 - capability cost of each defense."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dist" / "piibench_capability.zip"
CODE = ["pii_auditor", "capability_eval.py", "test_dp_resume.py",
        "relabel_condition.py", "requirements-gpu.txt"]
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


INTRO = """# E22 - what each defense costs on a recognised Chinese benchmark

## Why

Two items in the adviser's review are the same objection:

**O.** Perplexity on generic Chinese text is not enough to show that a defense
preserves useful capability. A proper utility evaluation should include at least
one recognised Chinese benchmark, such as C-Eval.

**Q.** The abstract says DP-SGD *"eliminated leakage entirely at epsilon =
0.75"* without saying whether that model can still do anything. **A model that
learns nothing cannot leak.** Section 6.5.1 already had to be rewritten once for
exactly that reason, and Section 6.5.5 records a second instance.

So every defense condition is scored on C-Eval, and no privacy number is
reported without the capability number beside it.

## Conditions

| | What it is | Expected |
|---|---|---|
| `base` | released weights | the reference ceiling |
| `undefended` | the checkpoint every leakage figure is measured on | close to base |
| `unlearned` | 60 steps of gradient ascent (Section 6.4) | perplexity was 35,550 - this one should be wrecked |
| `dp_eps075` | the abstract's epsilon = 0.75, noise 1.0 over 2,800 steps | **the number the abstract depends on** |
| `dp_plateau` | noise 0.02, the competence-preserving setting of Section 6.5.5 | should hold up |

Output filtering is not a condition here. It rewrites generated text and never
touches the weights, so its C-Eval score is the undefended model's by
construction - worth stating in the write-up, not worth GPU time.

## Two decisions that were checked before this was written

**C-Eval, not CMMLU.** CMMLU is still distributed as a loading script, and
datasets 4.0 no longer executes those. C-Eval is served as parquet, one file per
subject, so it is read directly and the dataset library is not in the path at
all. Verified against the Hub: 52 subjects, 1,346 questions in the val split.
The test split is larger but its labels are withheld.

**Likelihood scoring, not generation.** Asking a base model to emit the letter
"B" measures instruction-following as much as knowledge, and Section 5.8.4 of
the thesis is about exactly that confound arising elsewhere in this study. Each
option is scored by the model's own probability at the answer position: no
instruction-following, one forward pass per question, and far faster.

## The trap this run is built to avoid

25% is chance on a four-way question. A model that always answers "A" also
scores about 25%. The distribution over predicted letters is recorded with every
accuracy, and a condition that collapses onto one letter is flagged rather than
reported as a number - the same guard that caught DeepSeek in E21.
"""

CELL_CHECK = """## 5 - Check the environment and the benchmark, before any GPU time

Downloads the 52 subject files and parses them. Fails in a minute if something
is wrong, rather than after the first defense has been trained.
"""

CELL_A = """## 6 - Session A: base, undefended, unlearned (about 1.5 h)

`base` needs no training. `undefended` is a normal LoRA run at the study's
settings. `unlearned` starts from that adapter, so run them in this order.

Everything is written to Drive as it completes and the script skips conditions
already present in `capability.json`, so a dropped session resumes where it
stopped.
"""

RUN_A = """!python capability_eval.py \\
    --conditions base,undefended,unlearned \\
    --out $DRIVE/results_capability"""

CELL_RESUME = """## 7b - Ten seconds: prove the checkpoint actually resumes

Before spending eight hours on a run whose safety net has never been tested.

A checkpoint that restores the weights but forgets the optimizer moments or the
RNG streams still looks like it works - it just continues on a different
trajectory, and nothing downstream reveals that the model which finishes is not
the model the run was supposed to produce. This trains a toy module on the CPU
twice, once straight through and once interrupted and resumed, and compares the
LoRA tensors. It also checks that resuming under a different noise multiplier is
refused, and that behaviour is unchanged when no checkpoint directory is asked
for.

Expected last line: `RESUME IS EXACT - safe to rely on`.
"""

CELL_B = """## 8 - Session B: the two differentially private conditions (about 8 h each)

DP-SGD clips gradients per example, which means one backward pass per example
rather than per batch: 2,240 texts over 30 epochs is 67,200 of them. Measured on
an L4, one condition took 7.8 hours. The earlier estimate of four hours for both
was wrong, and shortening the run is not an option - epsilon is fixed by the
noise multiplier, the sampling rate and the number of steps composed, so fewer
epochs is a different guarantee, not the same result sooner.

`dp_eps075` is the one the abstract's claim rests on.

**Checkpointed after every epoch, to Drive.** The first attempt at this cell
lost 7.8 hours to a dropped connection at epoch 29 of 30, because the adapter
was written only once training returned. Now the LoRA weights, the optimizer
moments, the epoch counter and both RNG streams are saved to
`results_capability/ckpt_<condition>/` at the end of each epoch, and rerunning
this cell picks up from the last completed one. An interruption costs one epoch,
about sixteen minutes.

Resuming does not alter the privacy accounting: the same 30 epochs of steps are
composed either way. The checkpoint records the mechanism it was written under
and refuses to resume under a different one, so a checkpoint cannot quietly
acquire an epsilon it never had.

Safe to run in a separate session: cell 6's results are already on Drive and
will not be recomputed.
"""

RUN_B = """!python capability_eval.py \\
    --conditions dp_eps075,dp_plateau \\
    --out $DRIVE/results_capability"""

CELL_D = """## 10 - Cheap alternative: score a DP adapter that already exists

Differentially private training is slow because it computes per-sample
gradients: 2,240 texts over 30 epochs is 67,200 of them, and one condition took
close to eight hours on an L4. If that budget is not available, an adapter
trained in an earlier round can be scored directly - loading and 1,346 forward
passes, about twenty minutes, no training at all.

The pilot DP-LoRA checkpoint is the useful one. Its training parameters are
recorded in `results_defense_final/results.json` under `meta.dp`:

| | pilot adapter | `dp_eps075` |
|---|---|---|
| noise multiplier | 1.0 | 1.0 |
| clip norm | 1.0 | 1.0 |
| steps | 800 (10 epochs) | 2,800 (30 epochs) |
| corpus | 40 records | 140 records |
| epsilon | 1.48 | 0.75 |

**This is a matched-noise proxy, not a bound.** An earlier draft of this cell
argued that epsilon = 1.48 is the weaker guarantee, so the model must be the
less damaged one, so a score at chance would bound the epsilon = 0.75 model
from above. That inference does not hold. Both runs use *the same* noise
multiplier and the same clip norm; the epsilon differs because the 40-record
pilot has a higher per-step sampling rate, and sampling rate is a fact about the
accounting, not about how much noise reaches the weights. Ordering by epsilon
therefore says nothing about ordering by capability.

What the pilot does give is a genuine measurement of what sigma = 1.0 DP-LoRA
does to this exact model, rank and target set:

* at chance on C-Eval - strong evidence that sigma = 1.0 costs the capability,
  and `dp_eps075` takes 3.5 times as many noisy steps at the same per-step
  noise, so expecting it to be at least as damaged is reasonable. Report it as
  evidence, and let the full run supply the number the abstract cites;
* still capable - the concern is not settled either way, and only the full run
  will say.

Either way this is worth twenty minutes, and neither answer replaces cell 8.

Upload the adapter folder to `MyDrive/piibench/adapter_dp_pilot/` first. Extract
the zip and upload the **files**; unzipping and then uploading the folder into a
folder of the same name leaves them one level too deep. The next cell says which
you have, and the runner descends one level on its own if it can.
"""

CHECK_PILOT = """import os
p = f'{DRIVE}/adapter_dp_pilot'

# The mount caches its directory listing. Anything uploaded to Drive after the
# mount was made is simply invisible to this runtime until it is remounted -
# the folder is plainly there in the browser and absent here, which reads like
# a wrong path and is not one.
if not os.path.isdir(p):
    print('not visible yet - remounting, in case the upload came after the mount')
    from google.colab import drive
    drive.flush_and_unmount()
    drive.mount('/content/drive')

if not os.path.isdir(p):
    print(f'still not there: {p}')
    up = os.path.dirname(p)
    if os.path.isdir(up):
        print(f'  {up} contains: {sorted(os.listdir(up))}')
else:
    for root, dirs, files in os.walk(p):
        d = os.path.relpath(root, p)
        for f in sorted(files):
            print(f'  {os.path.getsize(os.path.join(root, f))/1e6:8.2f} MB  '
                  f'{f if d == "." else os.path.join(d, f)}')
    hits = [r for r, _, fs in os.walk(p) if 'adapter_config.json' in fs]
    print()
    if not hits:
        print('  no adapter_config.json anywhere below - upload the extracted '
              'files, not the zip')
    else:
        w = [f for f in os.listdir(hits[0]) if f.endswith('.safetensors')]
        state = w if w else 'MISSING - the 148 MB file has not finished uploading'
        print(f'  adapter found in {hits[0]}')
        print(f'  weights: {state}')"""

RUN_D = """!python capability_eval.py \\
    --conditions '' \\
    --extra dp_pilot_eps148=$DRIVE/adapter_dp_pilot \\
    --out $DRIVE/results_capability"""

CELL_LOSS = """## 11 - The measurement that decides what a high score means

No training, no retraining: it loads each adapter already on Drive and measures
its loss on the fine-tuning corpus. A few minutes per condition.

**It writes `corpus_loss.json`, not `capability.json`.** Safe to run beside
cell 8, including in a second session, because the two never write the same
file - the trainer rewrites `capability.json` every time a condition finishes,
and a second process holding an older copy would erase whatever landed in
between. Conditions not yet scored are simply skipped, so run it again after
cell 8 to pick up the DP ones.

**A second Google account is a different Drive.** `capability.json` lives in the
account that ran cell 6. To reach it from another account, share the `piibench`
folder from the first (Editor, so this cell can write its output there), then in
the second account's Drive use *Add shortcut to Drive* so it appears under
`My Drive`, and mount again. Without that, this cell finds an empty results
folder and says so.

Prefer to run the whole pass on one machine when you can. The comparison that
matters is between conditions, and conditions measured on the same GPU in the
same pass are the cleanest version of it.

**Running this on its own.** In the session that just finished cell 8, nothing
else is needed - run this cell. In a fresh session, run **2** (mount), **3**
(unpack), **4** (install), then this one; about two minutes of setup. Cells 5
to 10 can all be skipped: this pass trains nothing, scores no questions, and
does not fetch C-Eval at all. It reads `capability.json` for the list of
conditions and their accuracies, loads each adapter in turn, and stops.

**Why it is needed.** The pilot DP adapter came back at C-Eval 0.6166 against
the base model's 0.6568, with perplexity 16.95 against 16.72 - almost untouched,
and far above the undefended model's 0.4547. Read carelessly that says the
defense preserved capability. It equally says the adapter never learned
anything, and the pilot round's own numbers point that way: DP-LoRA scored
RW-MER 0.0000 and CLMD 0.0000, digit for digit the base model's row, while the
undefended model scored 0.4340 and -0.2437.

The mechanism makes it plausible. With clip 1.0 and lot size 8 the summed
clipped gradient has norm at most 8, while Gaussian noise at sigma = 1.0 over
36,929,536 LoRA parameters has expected norm sigma x sqrt(d) = 6,077 - about
**760 parts noise to 1 part signal**. At the `dp_plateau` setting of sigma =
0.02 the same ratio is 15 to 1.

C-Eval cannot separate "protected while learning" from "never learned". Corpus
loss can, and it is the difference between a defense result and a null result:

* **level with base** - the adapter never fitted the records. Its zero leakage
  is uninformative, and the abstract's sentence needs to say so.
* **far below base**, near the undefended model - it did learn the records and
  the leakage really was suppressed. That is a genuine defense result.
"""

RUN_LOSS = """!python capability_eval.py --corpus-loss \\
    --extra dp_pilot_eps148=$DRIVE/adapter_dp_pilot \\
    --out $DRIVE/results_capability"""

READ = """import json
d = json.load(open(f'{DRIVE}/results_capability/capability.json', encoding='utf-8'))
print(f"model {d['model']}   {d['n_questions']} questions   {d['shots']}-shot")
print('environment:', d['environment'])
print()
print(f"{'condition':14s}{'C-Eval':>9}{'vs chance':>11}{'modal letter':>14}{'ppl':>10}")
print('-' * 60)
for k, s in d['conditions'].items():
    print(f"{k:14s}{s['accuracy']:>9.4f}{s['accuracy']-0.25:>+11.4f}"
          f"{s['modal_letter_share']:>13.1%}{str(s['perplexity']):>10}"
          + ('   DEGENERATE' if s['degenerate'] else ''))
print()
for k, s in d['conditions'].items():
    print(f"  {k:14s} letters {s['letter_distribution']}")"""

COLLECT = """import os, zipfile, glob

src = f'{DRIVE}/results_capability'
out = f'{DRIVE}/capability_results.zip'

# The adapters live here too and come to several hundred megabytes. They stay on
# Drive so a rerun reuses them; there is no reason to send them back.
want = [f for f in (os.path.join(src, 'capability.json'),
                    os.path.join(src, 'corpus_loss.json')) if os.path.exists(f)]
assert want, f'nothing to collect in {src} - did the runs write there?'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in want:
        z.write(f, os.path.basename(f))
        print(f'  {os.path.getsize(f)/1e6:7.3f} MB  {os.path.basename(f)}')

import json
d = json.load(open(want[0], encoding='utf-8'))
missing = [c for c in ('base', 'undefended', 'unlearned', 'dp_eps075', 'dp_plateau')
           if c not in d['conditions']]
print('\\nconditions completed:', sorted(d['conditions']))
if missing:
    print('  !! not finished:', missing)
print(f'\\nwrote {out}  ({os.path.getsize(out)/1e3:.0f} KB)')
print('send this one back')"""

INVENTORY = """import os, json, glob

res = f'{DRIVE}/results_capability'
if not os.path.isdir(res):
    print(f'{res} does not exist yet - nothing has run')
else:
    j = f'{res}/capability.json'
    if os.path.exists(j):
        d = json.load(open(j, encoding='utf-8'))
        print('scored already :', sorted(d['conditions']))
    else:
        print('scored already : nothing, capability.json is not there')

    print()
    for p in sorted(glob.glob(f'{res}/adapter_*')):
        n = os.path.join(p, 'adapter_model.safetensors')
        sz = os.path.getsize(n) / 1e6 if os.path.exists(n) else 0
        state = f'{sz:.0f} MB, complete' if sz else 'INCOMPLETE - no weights file'
        print(f'  adapter  {os.path.basename(p):24s} {state}')

    # A checkpoint is the difference between resuming an interrupted DP run and
    # paying for it a second time, so report exactly how far it got.
    for p in sorted(glob.glob(f'{res}/ckpt_*')):
        f = os.path.join(p, 'dp_state.pt')
        if os.path.exists(f):
            import torch
            st = torch.load(f, map_location='cpu', weights_only=False)
            print(f"  ckpt     {os.path.basename(p):24s} through epoch "
                  f"{st['epoch']+1}/{st['epochs']}, noise {st['noise_multiplier']} "
                  f"- rerunning cell 8 resumes from epoch {st['epoch']+2}")
        else:
            print(f'  ckpt     {os.path.basename(p):24s} empty')
    if not glob.glob(f'{res}/ckpt_*'):
        print('  ckpt     none - any DP run here predates checkpointing')"""

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
    md("## 2b - What is already on Drive\n\n"
       "Read this before starting anything. Conditions already in "
       "`capability.json` are skipped, an `adapter_*` folder is reused instead "
       "of retrained, and a `ckpt_*` folder means an interrupted DP run can be "
       "resumed from the epoch shown rather than started over."),
    code(INVENTORY),
    md("## 3 - Unpack the bundle\n\n"
       "Upload `piibench_capability.zip` into the **`piibench` folder** of your "
       "Drive first, then run this cell."),
    code("import zipfile, os\n"
         "src = f'{DRIVE}/piibench_capability.zip'\n"
         "assert os.path.exists(src), f'not found: {src} - upload the zip first'\n"
         "zipfile.ZipFile(src).extractall('/content/work')\n"
         "os.chdir('/content/work')\n"
         "print(sorted(os.listdir('.')))"),
    md("## 4 - Install dependencies\n\n"
       "`-U` is on **bitsandbytes only**. Applied to the whole list it upgrades "
       "transformers as well, and a condition trained under a different library "
       "version is not comparable with the ones before it - that is how E21b "
       "came to fail after E21a had already finished."),
    code("# -U on bitsandbytes ONLY - see the note above\n"
         '!pip install -q -U "bitsandbytes>=0.46.1"\n'
         "!pip install -q transformers accelerate peft datasets pandas pyarrow "
         "huggingface_hub\n"
         "import torch, transformers\n"
         "print('transformers', transformers.__version__, '| torch', torch.__version__)\n"
         "print('every condition must run under these same versions')"),
    md(CELL_CHECK),
    code("!python capability_eval.py --check-only --out $DRIVE/results_capability"),
    md(CELL_A),
    code(RUN_A),
    md("## 7 - Read what is finished so far"),
    code(READ),
    md(CELL_RESUME),
    code("!python test_dp_resume.py"),
    md(CELL_B),
    code(RUN_B),
    md("## 9 - Read all five conditions"),
    code(READ),
    md(CELL_D),
    code(CHECK_PILOT),
    code(RUN_D),
    md(CELL_LOSS),
    code(RUN_LOSS),
    md("## 12 - Collect the results"),
    code(COLLECT),
    md("---\n"
       "### If the session drops or the runtime restarts\n"
       "**Re-run sections 2 to 4 first.** A restart clears every pip install, "
       "and the first thing to notice is the 4-bit quantizer, which fails only "
       "after downloads have started. Then re-run the cell you were on: "
       "conditions already in `capability.json` are skipped and adapters already "
       "on Drive are reused.\n\n"
       "### If you hit CUDA out of memory\n"
       "Add `--batch-size 4`. The five-shot prompts are long, which is what "
       "consumes the memory here rather than the model.\n\n"
       "### If a condition comes out at 25%\n"
       "Look at `modal_letter_share` before reading anything into it. Chance is "
       "25% and a model that always answers 'A' also scores 25%; only the letter "
       "distribution separates the two. The script flags the second case as "
       "`DEGENERATE`, and a degenerate condition is a training failure to report "
       "as such, not a capability measurement.\n\n"
       "### What would change the manuscript\n"
       "If `dp_eps075` lands near chance, the abstract's *eliminated leakage "
       "entirely at epsilon = 0.75* has to be reported together with the fact "
       "that the model cannot do the benchmark - which is the adviser's point Q, "
       "and better found here than at the defense.")
]

README = """# E22 - capability cost of each defense

## Run order

1. Upload `piibench_capability.zip` to `MyDrive/piibench/`
2. Open `RUN_ME_CAPABILITY.ipynb` in Colab, GPU runtime
3. Cells 1-5, then cell 6 (about 1.5 h), then cell 8 (about 4 h)
4. Send back `capability_results.zip` - it is a single JSON, a few hundred KB

Sessions A and B can be days apart. Finished conditions are never recomputed.

## What is measured

C-Eval, val split, 1,346 questions over 52 subjects, five-shot from the dev
split of each subject. Options are scored by the model's probability at the
answer position rather than by generating a letter, so a base model is not
penalised for failing to follow an instruction.

Recorded per condition: accuracy, the distribution over predicted letters, the
modal letter's share, a `degenerate` flag, per-subject accuracy, and perplexity
for continuity with Table 6.2.

## Why C-Eval and not CMMLU

CMMLU ships a loading script and datasets 4.0 no longer runs those. C-Eval is
served as parquet and is read directly from the Hub, so the dataset library is
not involved. This was checked against the Hub before the bundle was built.

## The number that matters

`dp_eps075` is the condition the abstract's claim rests on. If it scores near
chance, "eliminated leakage entirely at epsilon = 0.75" is true and misleading at
the same time, and the manuscript has to say both.
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
                    z.write(f, f.relative_to(ROOT))
                    n += 1
            elif p.exists():
                z.write(p, p.name)
                n += 1
            else:
                print(f"  WARNING: {item} not found")
        z.writestr("RUN_ME_CAPABILITY.ipynb",
                   json.dumps(NOTEBOOK, ensure_ascii=False, indent=1))
        z.writestr("README_CAPABILITY.md", README)
    print(f"wrote {OUT}")
    print(f"  code files : {n}")
    print(f"  size       : {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
