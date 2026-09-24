#!/usr/bin/env python
"""Build the Colab bundle for E21 - the second and third model families."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dist" / "piibench_family.zip"
CODE = ["pii_auditor", "second_family.py", "requirements-gpu.txt"]
# The Qwen comparison is recomputed inside the run from these, by the same
# functions that score the new families, so a difference in the output cannot be
# a difference in how two numbers were derived.
REFERENCE = {
    "results_main_15b_140/records_finetuned.csv": "reference/qwen15b_records_finetuned.csv",
    "results_main_15b_140/records_base.csv": "reference/qwen15b_records_base.csv",
    "results_main_15b_140/dataset.json": "reference/qwen15b_dataset.json",
}
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


INTRO = """# E21 - does any of this hold outside Qwen?

## Why

The adviser's comment-3 review rates **model-family generalizability: Weak**,
and the reasoning is hard to argue with. Qwen2.5-1.5B, Qwen2.5-3B and
Qwen2.5-1.5B-Instruct are three conditions of *one family*. Nothing measured on
them licenses a claim about "domestic Chinese LLMs" as a class. He separately
questions whether 3B counts as "medium" at all.

The review names the families the conclusions cannot reach: *DeepSeek, Baichuan,
ChatGLM, InternLM, Yi*. Two of those five are Llama-architecture and natively
supported by transformers, which is what this bundle runs:

| | Model | What it decides |
|---|---|---|
| **E21a** | `01-ai/Yi-1.5-6B` | A second family, at a genuinely **medium** scale. This is the condition that decides whether the word "medium" in the title has evidence behind it. |
| **E21b** | `deepseek-ai/deepseek-llm-7b-base` | A **third** family, also at medium scale. Two independent families agreeing is a much harder result to wave away than one. |

### Why not a matched-scale 1.8B control

InternLM2-1.8B was the first choice, because holding scale fixed separates a
*family* effect from a *scale* effect. It cost four failed runs. That repository
ships its own modelling code, pinned to the transformers of early 2024, and on
the current library it fails in a new place each time one is patched:
`rope_scaling` at load, the KV-cache API at generation, then the attention mask
inside the decoder. Every remaining Chinese family at that scale - InternLM,
ChatGLM, Baichuan, MiniCPM - ships remote code the same way.

The trade is worth taking. Family diversity is what the review actually asks
for, and two families beat one control. Section 12 keeps InternLM2 available for
anyone who wants the matched-scale condition and is willing to pin the library.

## What is held fixed

Corpus seed 20260524 - the same 140 people with the same values. Same twelve
templates, same LoRA rank 32 / alpha 64, same 30 epochs x 2 repeats (sixty
exposures), same training seed 42, same greedy decoding, same detector.
**The model family is the only thing that varies.**

## The one exception, and why it matters

Templates C1 and D1 carry a system persona that names the assistant:
*"You are Qwen2.5..."* / *"你是通义千问2.5..."*.

Telling Yi or DeepSeek that it is Qwen asserts a false identity. That can move
refusal behaviour, and a cross-family difference produced that way would be a
prompt artefact rather than a property of the model. The name is therefore
swapped per family. Everything else in those templates - persona wording,
request structure, the value-only instruction - stays **byte-identical**, which
is what keeps C1/D1 a matched pair.

## The Qwen comparison is recomputed, not quoted

The stored per-query records of the main Qwen run travel inside this bundle. The
reference numbers are recomputed from them by the same functions that score the
new families, so any difference in the output is a difference in the models.

## Pre-registered reading, fixed before the run

| Quantity | Qwen2.5-1.5B | Confirmed if | In trouble if |
|---|---|---|---|
| Null floor (released weights) | 0.0000, Low | both new families also return zero | anything above zero - the corpus is not novel to that model, and the null-floor argument needs restating for it |
| Aggregate RW-MER (fine-tuned) | 0.3359, **High** | both reach the High band | a family resists - every leakage claim then narrows to the families that leaked |
| Direct memorization, A+B | 0.5235 | comparable magnitude | far lower - memorization is not family-general at these exposures |
| Matched-pair CLMD | C1-D1 -0.0469, C2-D2 -0.0419, mean **-0.0444** | both stay small and non-positive | a clear **positive** differential - the cross-lingual conclusion is then a Qwen finding and must say so |
| Anchored : unanchored | 0.6949 : 0.0092 = **75.5x** | ordering survives on both | it collapses - entity anchoring is not the general mechanism claimed |

**A null result is a result here.** A family that does not leak, or that
reverses the cross-lingual sign, is worth more than a confirmation: it bounds
the claim before the panel does.
"""

CELL_CHECK = """## 6 - Verify both model ids before spending GPU time

Cheap, and it fails in seconds rather than twenty minutes into a load. If either
line says FAIL, fix the id before going on.
"""

RUN_CHECK = """!python second_family.py --model 01-ai/Yi-1.5-6B --check-only
!python second_family.py --model deepseek-ai/deepseek-llm-7b-base --check-only"""

CELL_A = """## 7 - E21a: Yi-1.5-6B (about 4 h)

Llama-architecture and natively supported, so no remote code, the study's own
LoRA target names apply, and nothing needs patching. On a T4 rather than an L4,
add `--batch-size 12`.
"""

RUN_A = """# --out is on Drive, so records, metrics and the adapter all survive a
# dropped runtime. Nothing that matters is written to /content.
!python second_family.py \\
    --model 01-ai/Yi-1.5-6B \\
    --tag yi-1.5-6b \\
    --out $DRIVE/results_family"""

CELL_B = """## 9 - E21b: DeepSeek-LLM-7B, retry at a lower learning rate (about 3.5 h)

The first attempt at the study's default `lr=3e-4` **collapsed**: 155 distinct
outputs over 11,760 probes, one of them 42.9% of the total, the same 69-character
string whatever it was asked. Healthy runs sit near 1.3%. It scored 0.0000 on
every template, and that zero is a training failure, not a privacy result - a
model that cannot retrieve anything cannot leak anything.

Three changes:

| | Why |
|---|---|
| `--lr 1e-4` | 3e-4 over sixty exposures is too aggressive for 7B in 4-bit. 1e-4 is the usual QLoRA starting point at that size. |
| `--tag deepseek-7b-lr1e4` | A new tag. With the old one the script would find the collapsed adapter on Drive and reuse it, reproducing the failure exactly. |
| `--skip-base` | The released-weight audit is already done and returned 0 of 11,760. Saves about ninety minutes. |

The script now puts a few hundred probes through **before** the full audit and
stops if the model is still collapsed, so a second failure costs minutes rather
than three hours.
"""

RUN_B = """!python second_family.py \\
    --model deepseek-ai/deepseek-llm-7b-base \\
    --tag deepseek-7b-lr1e4 \\
    --lr 1e-4 --skip-base \\
    --out $DRIVE/results_family --batch-size 12"""

CELL_C = """## 12 - Optional: InternLM2-1.8B, the matched-scale control

Only worth attempting if you want a second family at the *same* 1.5-1.8B scale
as the baseline, which would separate a family effect from a scale effect.

InternLM2 ships its own modelling code, pinned to the transformers of early
2024. On the current library it fails in a new place each time one is patched -
`rope_scaling` at load, the KV-cache API at generation, then the attention mask
inside the decoder. `pii_auditor/compat.py` carries the first two repairs; the
third is inside the attention and cannot be reached from outside.

The way through is to give that code the library it was written against:

```
!pip install -q "transformers==4.44.2" "accelerate==0.33.0" "peft==0.12.0"
```

Restart the runtime, re-run sections 2-4, then this cell. **Reinstall current
transformers before running anything else**, and record in the ledger which
library version produced which condition - it is a reproducibility fact, not an
implementation detail.
"""

RUN_C = """!python second_family.py \\
    --model internlm/internlm2-1_8b \\
    --tag internlm2-1.8b \\
    --out $DRIVE/results_family"""

READ = """import json, glob
for f in sorted(glob.glob(f'{DRIVE}/results_family/family_*.json')):
    d = json.load(open(f, encoding='utf-8'))
    ft, ref = d.get('finetuned'), d.get('reference_qwen15b')
    print('=' * 74)
    print(f"{d['model']}   persona {d['persona']['en']!r}")
    if 'base' in d:
        print(f"  null floor (released weights) : {d['base']['aggregate_rwmer']}"
              f"  [{d['base']['risk_band']}]")
    if ft:
        print(f"  aggregate RW-MER (fine-tuned) : {ft['aggregate_rwmer']}"
              f"  [{ft['risk_band']}]"
              + (f"   vs Qwen {ref['aggregate_rwmer']} [{ref['risk_band']}]" if ref else ''))
        print(f"  direct memorization A+B       : {ft['direct_memorization_AB']}"
              + (f"   vs Qwen {ref['direct_memorization_AB']}" if ref else ''))
        print(f"  matched-pair CLMD             : {ft['matched_clmd']['mean']}"
              f"   pairs {ft['matched_clmd']['pairs']}"
              + (f"   vs Qwen {ref['matched_clmd']['mean']}" if ref else ''))
        print(f"  anchored : unanchored         : {ft['anchored']} : {ft['unanchored']}"
              f"  = {ft['anchor_ratio']}x"
              + (f"   vs Qwen {ref['anchor_ratio']}x" if ref else ''))"""

COLLECT = """import os, zipfile, glob

src = f'{DRIVE}/results_family'
out = f'{DRIVE}/family_results.zip'

# Records and metrics only. The two adapters live in this folder as well and
# come to roughly half a gigabyte between them; they stay on Drive so a rerun
# can reuse them, but there is no reason to send them back.
want = []
for pat in ('family_*.json', 'records_*.csv', 'dataset.json'):
    want += glob.glob(os.path.join(src, pat))
assert want, f'nothing to collect in {src} - did the runs write there?'

with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(want):
        z.write(f, os.path.basename(f))
        print(f'  {os.path.getsize(f)/1e6:7.2f} MB  {os.path.basename(f)}')

done = {os.path.basename(f).replace('family_', '').replace('.json', '')
        for f in glob.glob(os.path.join(src, 'family_*.json'))}
print('\\nfamilies completed:', sorted(done) or 'NONE')
for tag in ('yi-1.5-6b', 'deepseek-7b-lr1e4'):
    if tag not in done:
        print(f'  !! {tag} has no family_*.json - that run did not finish')

print(f'\\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')
print('send this one back')"""

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
       "Upload `piibench_family.zip` into the **`piibench` folder** of your "
       "Drive first, then run this cell."),
    code("import zipfile, os\n"
         "src = f'{DRIVE}/piibench_family.zip'\n"
         "assert os.path.exists(src), f'not found: {src} - upload the zip to Drive first'\n"
         "zipfile.ZipFile(src).extractall('/content/work')\n"
         "os.chdir('/content/work')\n"
         "print(sorted(os.listdir('.')))"),
    md("## 4 - Install dependencies\n(about 2 minutes)"),
    code("# -U on bitsandbytes ONLY. Applied to the whole list it upgrades\n"
         "# transformers as well, and a condition trained under a different\n"
         "# library version is not comparable with the ones before it.\n"
         '!pip install -q -U "bitsandbytes>=0.46.1"\n'
         "!pip install -q transformers accelerate peft datasets pandas "
         "huggingface_hub einops sentencepiece\n"
         "import torch, transformers\n"
         "print('transformers', transformers.__version__, '| torch', torch.__version__)\n"
         "print('every condition must run under these same versions')"),
    md("## 5 - Sanity-check the persona swap\n\n"
       "Confirms that only the assistant name changes and the request text is "
       "identical across families. If the user message differs, the matched-pair "
       "property is broken and the CLMD comparison is not valid."),
    code("import sys; sys.path.insert(0, '.')\n"
         "from pii_auditor import m2_prompts as M\n"
         "from pii_auditor.m1_generator import generate_persons\n"
         "p = generate_persons(140, seed=20260524)[0].as_record()\n"
         "seen = {}\n"
         "for fam in ('qwen', 'internlm', 'yi'):\n"
         "    M.set_assistant_name(fam)\n"
         "    s, u = M._render_C_parts('C1', p, 'national_id')\n"
         "    print(f'{fam:9s} {s}')\n"
         "    seen[fam] = u\n"
         "M.set_assistant_name('qwen')\n"
         "assert len(set(seen.values())) == 1, 'user message differs across families'\n"
         "print('\\nuser message identical across all three families: OK')"),
    md(CELL_CHECK),
    code(RUN_CHECK),
    md("## 6b - Which of these repositories ship their own model code\n\n"
       "A repository carrying `auto_map` in its config supplies its own "
       "modelling code, pinned to the transformers of its release date. That is "
       "what cost this study four failed InternLM2 runs. Both models run here "
       "should report **natively supported**; if either says REMOTE CODE, expect "
       "trouble and read section 12 first."),
    code("import json\n"
         "from huggingface_hub import hf_hub_download\n"
         "for m in ('01-ai/Yi-1.5-6B', 'deepseek-ai/deepseek-llm-7b-base',\n"
         "          'internlm/internlm2-1_8b'):\n"
         "    cfg = json.load(open(hf_hub_download(m, 'config.json'), encoding='utf-8'))\n"
         "    print(f\"{m:38s} model_type={cfg.get('model_type','?'):10s} \"\n"
         "          f\"{'REMOTE CODE' if 'auto_map' in cfg else 'native'}\")"),
    md(CELL_A),
    code(RUN_A),
    md("## 8 - Read E21a"),
    code(READ),
    md(CELL_B),
    code(RUN_B),
    md("## 10 - Read both families side by side"),
    code(READ),
    md("## 11 - Collect the results"),
    code(COLLECT),
    md(CELL_C),
    code(RUN_C),
    md("---\n"
       "### If a library turns out to be the wrong version\n"
       "Restart the runtime, which restores Colab's preinstalled packages, then "
       "run section 4 again **exactly as written**. Do not add `-U` to the "
       "second line: upgrading transformers mid-experiment is how E21b came to "
       "fail on `TrainingArguments` after E21a had already finished under the "
       "older one, and a condition trained under a different library is not "
       "comparable with the ones before it. Each run records its library "
       "versions in `family_*.json`; if two conditions disagree, either rerun "
       "the odd one out or state the difference in the write-up.\n\n"
       "### If the session drops or the runtime restarts\n"
       "**Re-run sections 2 → 3 → 4 before anything else.** A restart clears "
       "every pip install, and the first thing to notice is the 4-bit quantizer: "
       "it fails with *requires bitsandbytes>=0.46.1* only after the config and "
       "tokenizer have downloaded. The runner now checks the environment before "
       "it touches the network and stops in two seconds instead.\n\n"
       "A finished adapter on Drive is reused rather than retrained, and the "
       "base-model audit can be skipped with `--skip-base` once collected.\n\n"
       "### If you hit CUDA out of memory\n"
       "DeepSeek-7B is the one at risk on a T4. Add `--batch-size 8`. If it still "
       "fails during fine-tuning, `--batch-size 8`.\n\n"
       "### If InternLM2 fails to load\n"
       "It needs `trust_remote_code`, which the script sets for you. If the error "
       "is *target modules not found*, the `--lora-targets auto` default did not "
       "apply - pass it explicitly.\n\n"
       "`compat.py` repairs the two places that file has already been caught "
       "out: the `rope_scaling` key at load time, and "
       "`prepare_inputs_for_generation`, whose cache handling reads "
       "`get_max_length()` and gets a shape *tuple* back on a current "
       "transformers. If cached generation still fails, the backend says so and "
       "finishes the run with `use_cache=False` - correct, just slower.\n\n"
       "If it raises some *other* old-API error from inside "
       "`modeling_internlm2.py`, the escape hatch is to give it the library it "
       "was written against:\n\n"
       "```\n!pip install -q 'transformers<4.45'\n```\n\n"
       "Restart the runtime after that, re-run sections 2-4, and run E21a alone. "
       "Then reinstall current transformers before E21b, because Yi is fine on "
       "it. Note in the ledger which library version produced which condition.\n\n"
       "### The one thing that would invalidate the run\n"
       "Changing `--seed` from 20260524. That changes the corpus as well as the "
       "model, and the result stops being a family comparison.\n\n"
       "### What a null result looks like here\n"
       "A family that does not leak after sixty exposures, or one whose "
       "matched-pair CLMD comes out clearly positive, is **not a failed "
       "experiment**. It is the finding, and it is what tells the thesis how far "
       "its claims actually reach.")
]

README = """# E21 - second and third model families

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
"""


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = ref = 0
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
        for src, dst in REFERENCE.items():
            p = ROOT / src
            if p.exists():
                z.write(p, dst)
                ref += 1
            else:
                print(f"  WARNING: reference {src} not found")
        z.writestr("RUN_ME_FAMILY.ipynb",
                   json.dumps(NOTEBOOK, ensure_ascii=False, indent=1))
        z.writestr("README_FAMILY.md", README)
    print(f"wrote {OUT}")
    print(f"  code files      : {n}")
    print(f"  reference files : {ref}")
    print(f"  size            : {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
