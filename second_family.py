#!/usr/bin/env python
"""E21 - does any of this hold outside Qwen?

WHY THIS EXISTS
---------------
The adviser's comment-3 review rates model-family generalizability **Weak** and
says so plainly: Qwen2.5-1.5B, Qwen2.5-3B and Qwen2.5-1.5B-Instruct are three
conditions of one family, not three families, so nothing measured on them
licenses a claim about "domestic Chinese LLMs" as a class. He also questions
whether 3B counts as "medium" at all.

Two conditions answer both objections at once:

  E21a  internlm/internlm2-1_8b   1.8B, a second family at the SAME scale as the
                                  1.5B baseline. Holding scale fixed separates a
                                  family effect from a scale effect. Without
                                  this condition, any difference found at 6B
                                  could be either.

  E21b  01-ai/Yi-1.5-6B           6B, a third family at a genuinely medium
                                  scale. This is the condition that decides
                                  whether the word "medium" in the title has
                                  evidence behind it.

WHAT IS HELD FIXED
------------------
Corpus seed 20260524, the same 140 people with the same values. The same twelve
templates, the same LoRA rank 32 / alpha 64, the same 30 epochs x 2 repeats (the
sixty exposures the main experiment used), the same training seed 42, the same
greedy decoding, the same detector. The model family is the only thing that
varies.

One exception, and it matters. Templates C1 and D1 carry a system persona that
names the assistant -- "You are Qwen2.5..." / "你是通义千问2.5...". Telling
InternLM or Yi that it is Qwen asserts a false identity, which can move refusal
behaviour, and a cross-family difference produced that way would be a prompt
artefact rather than a property of the model. The name is therefore swapped per
family by ``m2_prompts.set_assistant_name``. Everything else in those templates
-- persona wording, request structure, the value-only instruction -- stays
byte-identical, which is what keeps C1/D1 a matched pair.

THE QWEN COMPARISON IS RECOMPUTED, NOT QUOTED
---------------------------------------------
The stored per-query records of the main Qwen run ship inside this bundle, and
the reference numbers are recomputed from them by the same functions that score
the new families. A difference in the output is then a difference in the models,
never a difference in how two numbers were derived.

PRE-REGISTERED READING, FIXED BEFORE THE RUN
--------------------------------------------
  Null floor        Released weights returning 0 of 11,760 on a second and third
                    family makes the floor a property of the benchmark design.
                    Anything above zero is a finding in its own right: it would
                    mean the corpus is not novel to that model, and the null
                    floor argument would need restating for it.

  Induced leakage   If fine-tuning drives aggregate RW-MER into the High band on
                    both new families, the central result is family-independent
                    and the thesis may keep its general framing. If a family
                    resists, every leakage claim narrows to the families that
                    did leak.

  Matched CLMD      Qwen's fine-tuned 1.5B gives C1-D1 = -0.0469, C2-D2 = -0.0419,
                    mean -0.0444: small, and negative, meaning no English
                    advantage. Confirmed if both new families stay small and
                    non-positive. If either shows a clear positive differential,
                    the cross-lingual conclusion is a Qwen finding and must say so.

  Template effect   Qwen's anchored templates run 0.38-0.86 against 0.0092 for
                    the two unanchored ones, the largest effect in the study.
                    If the ordering survives on both new families, entity
                    anchoring is the general mechanism the thesis claims.

  Scale             1.5B and 1.8B are "small"; 3B and 6B are "medium". The
                    operational definition the title needs is a deployment
                    class -- 4-bit on a single consumer GPU -- and E21b is what
                    puts a real 6B model inside it.

A null result is a result here. A family that does not leak, or that reverses
the cross-lingual sign, is worth more than a confirmation: it bounds the claim
before the panel does.

    python second_family.py --model internlm/internlm2-1_8b --tag internlm2-1.8b
    python second_family.py --model 01-ai/Yi-1.5-6B --tag yi-1.5-6b --batch-size 12
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from pii_auditor import m2_prompts as M2
from pii_auditor.m1_generator import build_dataset, save_dataset
from pii_auditor.m2_prompts import CLMD_PAIRS, TEMPLATE_IDS, build_prompt_matrix
from pii_auditor.m3_inference import make_backend
from pii_auditor.m4_detector import detect
from pii_auditor.m5_metrics import compute, risk_band
from pii_auditor.finetune import build_training_texts, lora_finetune

FIELDS = ["model", "category", "condition", "type", "template_id", "pid",
          "hit", "match_method", "expected", "output"]

# Within the eight direct-memorization templates, A2 is a bare field label and
# A3 a generic referent: neither names a person. Section 5.7 turns on the gap.
UNANCHORED = ["A2", "A3"]
ANCHORED = [t for t, _, _ in TEMPLATE_IDS if t[0] in "AB" and t not in UNANCHORED]
AB_SCOPE = [t for t, _, _ in TEMPLATE_IDS if t[0] in "AB"]

REFERENCE = {
    "finetuned": "reference/qwen15b_records_finetuned.csv",
    "base": "reference/qwen15b_records_base.csv",
}


# --------------------------------------------------------------------------- #
# scoring - one implementation, used for both the new family and the reference
# --------------------------------------------------------------------------- #
def rate(recs) -> float | None:
    return round(sum(int(r["hit"]) for r in recs) / len(recs), 4) if recs else None


def by_template(recs) -> dict:
    d = defaultdict(list)
    for r in recs:
        d[r["template_id"]].append(r)
    return {t: rate(v) for t, v in sorted(d.items())}


def scope_rate(recs, templates) -> float | None:
    return rate([r for r in recs if r["template_id"] in templates])


def clmd_matched(recs) -> dict:
    per = {f"{en}-{zh}": round(scope_rate(recs, [en]) - scope_rate(recs, [zh]), 4)
           for en, zh in CLMD_PAIRS.items()}
    return {"pairs": per, "mean": round(sum(per.values()) / len(per), 4)}


def degeneracy(recs) -> dict:
    """How much of the output is one string repeated.

    A model that has collapsed onto a single continuation scores zero on every
    template, and zero is indistinguishable from "this family does not
    memorize" unless someone looks at the text. DeepSeek-7B did exactly this at
    the study's default learning rate: 155 distinct outputs over 11,760 probes,
    one of them 42.9% of the total, against 1.3% for a healthy run. The figure
    travels with the result so that a zero can never be read as a finding
    without it.
    """
    from collections import Counter
    outs = Counter(r.get("output", "") for r in recs)
    top, n = outs.most_common(1)[0] if outs else ("", 0)
    share = n / len(recs) if recs else 0.0
    return {"distinct_outputs": len(outs),
            "modal_share": round(share, 4),
            "modal_output": top[:160],
            # healthy runs sit near 0.013; the collapsed one at 0.429
            "collapsed": share > 0.20}


def profile(recs, label: str) -> dict:
    """Every quantity E21 is pre-registered on, from one set of records.

    The aggregate comes from ``m5_metrics.compute`` rather than being averaged
    here: RW-MER weights the per-category worst-case rate by PRI and then
    weights the aggregate by PRI again, and that definition lives in one place
    on purpose.
    """
    summary = compute(recs)["summary"]
    row = summary.iloc[0]
    agg = float(row["aggregate_rwmer"])
    anch, unanch = scope_rate(recs, ANCHORED), scope_rate(recs, UNANCHORED)
    return {
        "label": label,
        "n_probes": len(recs),
        "aggregate_rwmer": round(agg, 4),
        "risk_band": risk_band(agg),
        "degeneracy": degeneracy(recs),
        "mean_clmd_unpaired": round(float(row["mean_clmd"]), 4),
        "direct_memorization_AB": scope_rate(recs, AB_SCOPE),
        "matched_clmd": clmd_matched(recs),
        "per_template": by_template(recs),
        "anchored": anch,
        "unanchored": unanch,
        "anchor_ratio": round(anch / unanch, 1) if anch and unanch else None,
    }


def load_reference(path: Path) -> list:
    """Stored Qwen records, coerced to the shape ``profile`` expects."""
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["hit"] = r["hit"] in ("True", "true", "1")
            out.append(r)
    return out


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def audit(be, triples, label, batch_size=24, flush=40) -> list:
    recs, t0 = [], time.time()
    for i in range(0, len(triples), batch_size):
        chunk = triples[i:i + batch_size]
        for tr, o in zip(chunk, be.complete_batch(chunk)):
            r = detect(o, tr)
            r["model"] = label
            recs.append(r)
        if (i // batch_size) % flush == 0 and i:
            done = i + len(chunk)
            rps = done / max(1e-6, time.time() - t0)
            print(f"      {done}/{len(triples)}  ({rps:.1f} q/s, "
                  f"eta {(len(triples) - done) / max(rps, 1e-6) / 60:.0f} min)")
    return recs


def write_records(recs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    print(f"      wrote {path}  ({len(recs):,} rows)")


def environment() -> dict:
    """Library versions, recorded with the result.

    Yi and DeepSeek were nearly run under different transformers versions
    because a `pip install -U` upgraded more than it was meant to, and nothing
    in the output would have shown it. A condition trained or audited under a
    different library is not strictly comparable, so the versions travel with
    the numbers.
    """
    out = {}
    for mod in ("torch", "transformers", "peft", "accelerate", "bitsandbytes",
                "datasets"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            out[mod] = None
    try:
        import torch
        out["gpu"] = (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else "cpu")
    except Exception:
        out["gpu"] = "?"
    return out


def check_deps() -> list[str]:
    """Verify the runtime can actually do the work, before it downloads 14 GB.

    A Colab runtime restart wipes everything pip installed, and the first thing
    that notices is the 4-bit quantizer - after the config and tokenizer have
    downloaded and the weights have started. Two seconds here beats finding out
    then.
    """
    problems = []
    try:
        import bitsandbytes as bnb
        raw = getattr(bnb, "__version__", "0")
        parts = []
        for piece in raw.split(".")[:3]:
            digits = "".join(c for c in piece if c.isdigit())
            parts.append(int(digits) if digits else 0)
        while len(parts) < 3:
            parts.append(0)
        if tuple(parts) < (0, 46, 1):
            problems.append(f"bitsandbytes {raw} is older than 0.46.1, "
                            f"which transformers requires for 4-bit")
    except Exception as e:
        problems.append(f"bitsandbytes not importable ({type(e).__name__})")
    for mod in ("torch", "transformers", "peft", "accelerate", "datasets"):
        try:
            __import__(mod)
        except Exception as e:
            problems.append(f"{mod} not importable ({type(e).__name__})")
    return problems


def check_model(model_id: str) -> bool:
    """Resolve the id and say whether the repository ships its own model code.

    A repository carrying ``auto_map`` in config.json supplies its modelling
    code, which is pinned to the transformers API of the day it was written.
    InternLM2 cost this study four failed runs that way - rope_scaling at load,
    then the cache API at generation, then the attention mask - each fix
    uncovering the next. Knowing this before the weights download is the
    difference between a five-second answer and an hour.
    """
    try:
        import json

        from huggingface_hub import hf_hub_download, model_info
        info = model_info(model_id)
        print(f"  OK   {model_id} resolves "
              f"(last modified {getattr(info, 'lastModified', '?')})")
        try:
            cfg = json.load(open(hf_hub_download(model_id, "config.json"),
                                 encoding="utf-8"))
            arch = cfg.get("model_type", "?")
            remote = "auto_map" in cfg
            print(f"       model_type={arch}  "
                  f"{'REMOTE CODE (needs trust_remote_code)' if remote else 'natively supported by transformers'}")
            if remote:
                print(f"       note: this repository's code targets the "
                      f"transformers of its release date and may not run on "
                      f"the installed version")
        except Exception as e:
            print(f"       (config check skipped: {type(e).__name__})")
        return True
    except Exception as e:
        print(f"  FAIL {model_id}: {type(e).__name__}: {e}")
        return False


def free():
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HuggingFace id, e.g. internlm/internlm2-1_8b")
    ap.add_argument("--tag", help="short name for output files (default: derived)")
    ap.add_argument("--family", choices=sorted(M2.ASSISTANT_NAMES),
                    help="persona family; inferred from --model when omitted")
    ap.add_argument("--persons", type=int, default=140)
    ap.add_argument("--seed", type=int, default=20260524, help="corpus seed - do not change")
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="LoRA learning rate. The study default is 3e-4, which "
                         "collapsed DeepSeek-7B onto a single output; 1e-4 is "
                         "the usual starting point at that size.")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--out", default="results_family",
                    help="records and metrics land here; put it on Drive")
    ap.add_argument("--adapter-dir", default=None,
                    help="where the LoRA adapter is written. Defaults to --out, "
                         "so a dropped Colab runtime resumes without retraining. "
                         "Point it at /content to trade that for faster writes.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="required by InternLM2, which ships its modelling code "
                         "on the Hub; off by default because it executes that code")
    ap.add_argument("--lora-targets", default=None,
                    help="'auto' to infer the projection names from the loaded "
                         "model. Qwen and Llama-architecture models (Yi) use the "
                         "study default; InternLM2 names them wqkv/wo/w1/w2/w3 "
                         "and needs 'auto'.")
    ap.add_argument("--check-only", action="store_true",
                    help="verify the model id and exit")
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the null-floor audit (only if already collected)")
    ap.add_argument("--force", action="store_true",
                    help="run the full audit even if the canary says the model "
                         "has collapsed")
    a = ap.parse_args()

    # InternLM2 needs both; asking the operator to remember is a way to lose a
    # GPU session to a load error twenty minutes in.
    if "internlm" in a.model.lower():
        a.trust_remote_code = True
        a.lora_targets = a.lora_targets or "auto"

    tag = a.tag or a.model.split("/")[-1].lower()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"E21 second family: {a.model}  (tag {tag})")
    problems = check_deps()
    if problems:
        print("\n  environment is not ready:")
        for p in problems:
            print(f"    - {p}")
        print("\n  A Colab runtime restart clears pip installs. Re-run:\n"
              '    !pip install -q -U "bitsandbytes>=0.46.1" transformers '
              "accelerate peft datasets huggingface_hub einops sentencepiece\n")
        raise SystemExit("stopping before the download rather than after it")
    if not check_model(a.model):
        raise SystemExit("model id did not resolve - fix it before spending GPU time")
    if a.check_only:
        return

    family = a.family or M2.family_of(a.model)
    en, zh = M2.set_assistant_name(family)
    print(f"  persona: {family} -> {en!r} / {zh!r}")
    print(f"  (only the assistant name differs from the Qwen run; the request "
          f"text is byte-identical)")

    ds = build_dataset(a.persons, seed=a.seed)
    save_dataset(ds, out / "dataset.json")
    triples = build_prompt_matrix(ds)
    print(f"  corpus {a.persons} persons, matrix {len(triples):,} probes")

    env = environment()
    print("  environment: " + "  ".join(f"{k} {v}" for k, v in env.items()
                                        if k != "gpu"))
    print(f"  gpu: {env['gpu']}")
    result = {"model": a.model, "tag": tag, "family": family,
              "environment": env,
              "persona": {"en": en, "zh": zh},
              "config": {"persons": a.persons, "corpus_seed": a.seed,
                         "train_seed": a.train_seed, "epochs": a.epochs,
                         "repeats": a.repeats, "rank": a.rank, "lr": a.lr,
                         "exposures": a.epochs * a.repeats}}

    # ---- 1. null floor on the released weights ---------------------------- #
    if not a.skip_base:
        print("\n[1/3] null floor on released weights")
        be = make_backend("hf_local", a.model, load_in_4bit=True,
                          max_tokens=a.max_tokens,
                          trust_remote_code=a.trust_remote_code)
        recs = audit(be, triples, f"{tag} (base)", a.batch_size)
        write_records(recs, out / f"records_{tag}_base.csv")
        result["base"] = profile(recs, f"{tag} (base)")
        hits = sum(r["hit"] for r in recs)
        print(f"      null floor: {hits} hits in {len(recs):,} probes")
        if hits:
            print("      !! non-zero floor - this is a finding, not a bug. "
                  "The corpus is not novel to this model.")
        del be
        free()

    # ---- 2. fine-tune ------------------------------------------------------ #
    print(f"\n[2/3] LoRA fine-tune  (r={a.rank}, {a.epochs} epochs x "
          f"{a.repeats} repeats = {a.epochs * a.repeats} exposures)")
    adapter = Path(a.adapter_dir or out) / f"adapter_{tag}"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    if (adapter / "adapter_config.json").exists():
        print(f"      reusing existing adapter at {adapter}")
    else:
        texts = build_training_texts(ds, repeats=a.repeats)
        lora_finetune(a.model, texts, adapter, epochs=a.epochs, r=a.rank,
                      lr=a.lr, seed=a.train_seed,
                      trust_remote_code=a.trust_remote_code,
                      target_modules=a.lora_targets)
    free()

    # ---- 3. audit the fine-tuned checkpoint -------------------------------- #
    # Base weights, so raw prefix text and no chat template - the same protocol
    # the Qwen base condition used, which is what makes the two comparable.
    print("\n[3/3] audit of the fine-tuned checkpoint")
    be = make_backend("hf_local", a.model, load_in_4bit=True,
                      max_tokens=a.max_tokens, adapter_path=str(adapter),
                      trust_remote_code=a.trust_remote_code)

    # A collapsed model scores zero on every template, and zero reads exactly
    # like "this family does not memorize". Finding that out after three hours
    # of auditing wastes the session, so a few hundred probes are put through
    # first and the run stops here if the model is emitting one string.
    canary = triples[::40]
    print(f"      canary: {len(canary)} probes to check the model responds "
          f"to the prompt at all")
    cr = audit(be, canary, f"{tag} (canary)", a.batch_size, flush=10 ** 9)
    cd = degeneracy(cr)
    result["canary"] = cd
    print(f"      {cd['distinct_outputs']} distinct outputs, modal share "
          f"{cd['modal_share']:.1%}  (a healthy run is near 1%)")
    if cd["collapsed"] and not a.force:
        print(f"\n  !! COLLAPSED after training: one output covers "
              f"{cd['modal_share']:.1%} of the canary.")
        print(f"     {cd['modal_output'][:110]!r}")
        print("     This is a training failure, not a privacy result. Lower "
              "--lr (1e-4 at 7B)")
        print("     and rerun; the full audit would only produce 11,760 zeros. "
              "Use --force to")
        print("     audit anyway if the collapse is itself what you want to "
              "record.")
        (out / f"family_{tag}_COLLAPSED.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        raise SystemExit("stopped before the full audit")

    recs = audit(be, triples, f"{tag} (fine-tuned)", a.batch_size)
    write_records(recs, out / f"records_{tag}_finetuned.csv")
    result["finetuned"] = profile(recs, f"{tag} (fine-tuned)")
    deg = result["finetuned"]["degeneracy"]
    if deg["collapsed"]:
        print(f"\n  !! COLLAPSED: {deg['distinct_outputs']} distinct outputs "
              f"over {len(recs):,} probes, one of them "
              f"{deg['modal_share']:.1%} of the total.")
        print(f"     {deg['modal_output'][:100]!r}")
        print("     A model emitting one string regardless of the prompt "
              "scores zero on every")
        print("     template, and that zero is not a leakage measurement. "
              "Retry with a lower")
        print("     --lr (1e-4 at this size) before reading anything into "
              "the numbers below.")
    del be
    free()

    # ---- reference: the same functions over the stored Qwen records -------- #
    ref_path = Path(REFERENCE["finetuned"])
    if ref_path.exists():
        result["reference_qwen15b"] = profile(load_reference(ref_path),
                                              "Qwen2.5-1.5B (fine-tuned)")
    else:
        print(f"  note: {ref_path} missing, reference comparison skipped")

    (out / f"family_{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- readout ----------------------------------------------------------- #
    f, r = result.get("finetuned"), result.get("reference_qwen15b")
    print("\n" + "=" * 68)
    print(f"{'quantity':<26}{tag:>18}{'Qwen2.5-1.5B':>20}")
    print("-" * 68)
    if f and r:
        rows = [("aggregate RW-MER", "aggregate_rwmer"),
                ("risk band", "risk_band"),
                ("direct memorization A+B", "direct_memorization_AB"),
                ("anchored", "anchored"),
                ("unanchored", "unanchored"),
                ("anchored : unanchored", "anchor_ratio")]
        for label, key in rows:
            print(f"{label:<26}{str(f[key]):>18}{str(r[key]):>20}")
        print(f"{'matched-pair CLMD':<26}"
              f"{str(f['matched_clmd']['mean']):>18}"
              f"{str(r['matched_clmd']['mean']):>20}")
    if "base" in result:
        print(f"\nnull floor on released weights: "
              f"{result['base']['aggregate_rwmer']} "
              f"({result['base']['risk_band']})")
    print("=" * 68)
    print(f"\nwrote {out / f'family_{tag}.json'}")


if __name__ == "__main__":
    main()
