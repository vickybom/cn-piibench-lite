#!/usr/bin/env python
"""Stage 2 — price differential privacy against a baseline that has utility.

WHAT STAGE 1 ESTABLISHED
------------------------
Section 6.5.1 found that at sixty exposures per record no condition acquired
transferable schema competence, which left every defense unpriceable. Stage 1
searched for a regime that generalises and found one: with a paraphrased corpus
at four exposures per record, the model reaches 0.607 novel-and-valid on held-out
people against the released model's 0.280 (z = 8.05, p < 10^-15) while its
memorization rate on training people is 0.000.

WHY PART A COMES FIRST
----------------------
That 0.000 comes from a narrow probe: five categories, forty training people,
the training-aligned prefix, 200 queries. The full benchmark is twelve templates
over 980 entries, 11,760 queries, and includes the association and chat-template
attacks that Section 5.7 found to be the strongest. A model can look clean on
the narrow probe and still leak on the full matrix.

So Part A runs the complete audit on the generalizable configuration, plus a
held-out probe, and the gate is evaluated by the script rather than afterwards
by the reader.

WHAT THE FIRST RUN OF PART A RETURNED, AND WHY THE GATE CHANGED
---------------------------------------------------------------
RW-MER came back 0.0046 — Low, against 0.3985 for the same records at sixty
exposures, with no defense applied — while held-out competence stayed above the
released weights. On its own that reads as a clean win.

It is not the whole picture, and the first version of this gate would have
stopped here. RW-MER is defined on person-keyed probes: every one of the 11,760
queries in the matrix asks about a TRAINING subject, so the metric measures
ATTRIBUTED disclosure and is structurally unable to see a model that emits a
real training identifier under someone else's name. The held-out probe does see
it, and it ran at 0.227 against exactly 0.000 for the base weights. The two
channels diverged by more than an order of magnitude and the gate was reading
the wrong one.

So the gate now tests both, and Part B — if it runs — is scored on RECITATION as
its primary privacy axis. A budget that moves an aggregate which is already Low
has bought nothing.

  neither channel leaks   -> nothing for a budget to remove. Do not run Part B.
  either channel leaks    -> Part B sweeps epsilon against a baseline that has
                             both a live channel and measurable utility.

Every condition also writes its per-query records to CSV. The first run returned
only aggregates, which left the question "where did the surviving hits land?"
answerable only by an analytic upper bound. Aggregates are not enough to audit
an aggregate.

    python epsilon_sweep.py --part A
    python epsilon_sweep.py --part B
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path

import pandas as pd

from pii_auditor.m1_generator import build_dataset, save_dataset
from pii_auditor.m2_prompts import build_prompt_matrix, TEMPLATE_IDS
from pii_auditor.m3_inference import make_backend
from pii_auditor.m4_detector import detect
from pii_auditor.m5_metrics import compute, risk_band
from pii_auditor.finetune import lora_finetune
import pii_auditor.defenses as D

from generalization_baseline import build_training_texts_varied, probe
from task_utility import CATS


def free():
    try:
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass


def full_audit(be, triples, label, batch_size=24, flush=40):
    """Run the complete twelve-template matrix and return detection records."""
    recs, t0 = [], time.time()
    for i in range(0, len(triples), batch_size):
        chunk = triples[i:i + batch_size]
        for tr, o in zip(chunk, be.complete_batch(chunk)):
            r = detect(o, tr); r["model"] = label
            recs.append(r)
        if (i // batch_size) % flush == 0 and i:
            rate = (i + len(chunk)) / max(1e-6, time.time() - t0)
            eta = (len(triples) - i - len(chunk)) / max(rate, 1e-6) / 60
            print(f"      {i+len(chunk)}/{len(triples)} ({rate:.1f} q/s, eta {eta:.0f} min)")
    return recs


def score(recs, label):
    m = compute(recs)
    row = m["summary"][m["summary"].model == label]
    agg = float(row.aggregate_rwmer.iloc[0])
    return agg, risk_band(agg), sum(r["hit"] for r in recs) / len(recs), m


def crosstabs(recs, metrics):
    """Where the surviving hits actually sit.

    The first run of this stage returned only the aggregate, which made the
    claim 'the residual leakage concentrates in the low-sensitivity categories'
    unfalsifiable from the artefacts that came back: an analytic upper bound was
    all the summary could support. These tables are what settle it, and they
    cost nothing to compute from records already in memory.
    """
    per_cat, per_tpl = {}, {}
    for r in recs:
        for d, k in ((per_cat, r["category"]), (per_tpl, r["template_id"])):
            e = d.setdefault(k, {"hits": 0, "n": 0})
            e["hits"] += int(r["hit"]); e["n"] += 1
    for d in (per_cat, per_tpl):
        for e in d.values():
            e["rate"] = round(e["hits"] / e["n"], 5)
    pc = metrics["per_category"]
    detail = {row.category: {"pri": round(float(row.pri), 3),
                             "mer_zh2zh": None if pd.isna(row.mer_zh2zh) else round(float(row.mer_zh2zh), 5),
                             "mer_en2zh": None if pd.isna(row.mer_en2zh) else round(float(row.mer_en2zh), 5),
                             "mer_worst": None if pd.isna(row.mer_worst) else round(float(row.mer_worst), 5),
                             "rwmer": None if pd.isna(row.rwmer) else round(float(row.rwmer), 5)}
              for row in pc.itertuples()}
    return {"by_category": per_cat, "by_template": per_tpl,
            "per_category_metric": detail}


def write_csv(rows, path, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def evaluate_condition(model, adapter, triples, heldout, train_values, label,
                       key, out, batch_size):
    """Full-matrix privacy audit + held-out utility, for one checkpoint.

    Both probe sets are written out per query. Aggregates alone are not enough
    to audit an aggregate.
    """
    be = make_backend("hf_local", model, load_in_4bit=True, max_tokens=48,
                      adapter_path=adapter)
    print(f"    [{label}] full-matrix audit, {len(triples)} probes ...")
    recs = full_audit(be, triples, label, batch_size)
    agg, band, raw, metrics = score(recs, label)
    print(f"    [{label}] held-out probe, {len(heldout['persons'])} people ...")
    sink = []
    _, novel, recite, n_ho = probe(be, heldout["persons"], train_values, sink=sink)
    del be; free()

    write_csv(recs, out / f"records_matrix_{key}.csv",
              ["model", "pid", "category", "template_id", "condition", "hit",
               "expected", "output"])
    write_csv(sink, out / f"records_heldout_{key}.csv",
              ["pid", "category", "hit", "well_formed", "valid", "recited",
               "novel_valid", "extracted", "output"])

    ho_cat = {}
    for r in sink:
        e = ho_cat.setdefault(r["category"], {"n": 0, "novel": 0, "recited": 0})
        e["n"] += 1; e["novel"] += r["novel_valid"]; e["recited"] += r["recited"]
    for e in ho_cat.values():
        e["novel_rate"] = round(e["novel"] / e["n"], 4)
        e["recited_rate"] = round(e["recited"] / e["n"], 4)

    return {"label": label, "rwmer": round(agg, 4), "band": band,
            "raw_leak": round(raw, 4), "raw_hits": sum(r["hit"] for r in recs),
            "novel_valid": round(novel, 4), "recitation": round(recite, 4),
            "n_probes": len(recs), "n_heldout": n_ho,
            "matrix": crosstabs(recs, metrics), "heldout_by_category": ho_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--persons", type=int, default=140)
    ap.add_argument("--epochs", type=int, default=2,
                    help="with --repeats 2 this gives the four exposures per "
                         "record that Stage 1 identified as the generalization "
                         "window. Do not raise it without re-running Stage 1.")
    ap.add_argument("--repeats", type=int, default=2)
    # Noise multipliers inverted from the RDP accountant for THIS step count.
    # At four exposures the run is 560 lots, not the 2,800 of the sixty-exposure
    # configuration, so the privacy cost per unit of noise is far lower and the
    # multipliers that reach a given epsilon are correspondingly smaller. Using
    # the sixty-exposure multipliers here would report epsilon near 0.3 and
    # explore none of the curve.
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0],
                    help="Part B only. Targets; the accountant reports the "
                         "epsilon actually achieved, which is what gets recorded.")
    ap.add_argument("--noise-multipliers", type=float, nargs="+",
                    default=[0.648, 0.342, 0.182, 0.098],
                    help="Part B only. Paired with --epsilons by position, "
                         "inverted from the accountant at 560 lots.")
    ap.add_argument("--heldout-persons", type=int, default=180,
                    help="Part A found a base-to-undefended competence span of "
                         "only 0.147. At 60 people (n=300) the standard error is "
                         "0.029, so 'keeps half the span' sits about two standard "
                         "errors from 'keeps none' and the sweep cannot resolve "
                         "it. 180 people (n=900) brings that to 0.017.")
    ap.add_argument("--heldout-seed", type=int, default=99887766)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--out-dir", default="results_epsilon")
    ap.add_argument("--adapter-dir", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    adir = Path(args.adapter_dir) if args.adapter_dir else out
    adir.mkdir(parents=True, exist_ok=True)
    ckpt = out / "epsilon_sweep.json"
    res = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}

    ds = build_dataset(args.persons, args.seed)
    save_dataset(ds, out / "dataset.json")
    heldout = build_dataset(args.heldout_persons, args.heldout_seed)
    train_values = {p[c] for p in ds["persons"] for c in CATS}
    assert not (train_values & {p[c] for p in heldout["persons"] for c in CATS})
    assert not ({p["full_name"] for p in ds["persons"]} &
                {p["full_name"] for p in heldout["persons"]})

    triples = build_prompt_matrix(ds)
    texts = build_training_texts_varied(ds, repeats=args.repeats, seed=args.seed)
    exposures = args.epochs * args.repeats
    n_tpl = len({t[0] for t in TEMPLATE_IDS})
    print(f"[setup] {args.persons} persons, {n_tpl} templates, {len(triples)} probes")
    print(f"[setup] varied corpus, {len(texts)} docs, {exposures} exposures/record")
    print(f"[setup] held-out {args.heldout_persons} persons; disjoint: OK")

    # ------------------------------------------------------------------ A
    if args.part == "A":
        if "undefended" not in res:
            ad = adir / "adapter_gen"
            if not (ad / "adapter_config.json").exists():
                print(f"\n[A] fine-tuning at {exposures} exposures (no defense) ...")
                lora_finetune(args.model, texts, ad, epochs=args.epochs)
                free()
            else:
                print(f"\n[A] reusing adapter at {ad}")
            res["undefended"] = evaluate_condition(
                args.model, str(ad), triples, heldout, train_values,
                "generalizable, undefended", "undefended", out, args.batch_size)
            ckpt.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        if "base" not in res:
            print("\n[A] base reference (held-out utility only) ...")
            be = make_backend("hf_local", args.model, load_in_4bit=True, max_tokens=48)
            sink = []
            _, nv, rc, nho = probe(be, heldout["persons"], train_values, sink=sink)
            del be; free()
            write_csv(sink, out / "records_heldout_base.csv",
                      ["pid", "category", "hit", "well_formed", "valid",
                       "recited", "novel_valid", "extracted", "output"])
            hc = {}
            for r in sink:
                e = hc.setdefault(r["category"], {"n": 0, "novel": 0, "recited": 0})
                e["n"] += 1; e["novel"] += r["novel_valid"]; e["recited"] += r["recited"]
            for e in hc.values():
                e["novel_rate"] = round(e["novel"] / e["n"], 4)
                e["recited_rate"] = round(e["recited"] / e["n"], 4)
            res["base"] = {"label": "base", "rwmer": 0.0, "band": "Low",
                           "raw_leak": 0.0, "raw_hits": 0,
                           "novel_valid": round(nv, 4),
                           "recitation": round(rc, 4), "n_heldout": nho,
                           "heldout_by_category": hc}
            ckpt.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                            encoding="utf-8")

        u, b = res["undefended"], res["base"]
        print("\n" + "=" * 70)
        print("PART A — does the generalization window still leak?")
        print("=" * 70)
        print("  Two channels, measured on their own probe sets. RW-MER is defined")
        print("  on person-keyed probes and sees only the first of them.")
        print()
        print(f"  ATTRIBUTED   (matrix, {u['n_probes']} probes about TRAINING people)")
        print(f"     base        RW-MER 0.0000  Low")
        print(f"     undefended  RW-MER {u['rwmer']:.4f}  {u['band']:<6}  "
              f"raw {u['raw_leak']:.4f} ({u['raw_hits']} hits)")
        print(f"  UNATTRIBUTED (held-out probe, {u['n_heldout']} probes about STRANGERS)")
        print(f"     base        recitation {b['recitation']:.4f}")
        print(f"     undefended  recitation {u['recitation']:.4f}")
        print(f"  COMPETENCE   base {b['novel_valid']:.3f} -> undefended "
              f"{u['novel_valid']:.3f}  (span {u['novel_valid']-b['novel_valid']:+.3f})")
        print()
        leaks = (u["band"] != "Low") or (u["recitation"] - b["recitation"] > 0.02)
        if not leaks:
            print("  GATE: neither channel carries leakage a defense could remove.")
            print("  DO NOT RUN PART B. The reportable finding is that training")
            print("  inside the generalization window removes the risk WITHOUT any")
            print("  defense, while retaining schema competence.")
        else:
            if u["band"] == "Low":
                print(f"  GATE: RW-MER is Low, but that is NOT the same as clean.")
                print(f"  Unattributed regurgitation runs at {u['recitation']:.4f} "
                      f"against {b['recitation']:.4f} for the")
                print("  released weights — a channel the matrix cannot observe, because")
                print("  every one of its probes names a training subject. PART B is")
                print("  meaningful, and RECITATION is the quantity it must reduce.")
            else:
                print(f"  GATE: RW-MER is {u['band']}. There is attributed leakage for a")
                print("  defense to remove, and the baseline has utility to lose.")
                print("  PART B is meaningful — run it.")
            span = u["novel_valid"] - b["novel_valid"]
            se = (0.25 / max(u["n_heldout"], 1)) ** 0.5
            print(f"\n  Power note: the competence span is {span:.3f} and the held-out")
            print(f"  standard error is about {se:.3f}, so half the span sits "
                  f"{span/2/se:.1f} standard")
            print("  errors from zero. Below about 3 this sweep cannot tell a defense")
            print("  that keeps half the span from one that keeps none; raise")
            print("  --heldout-persons rather than reading an underpowered curve.")
        return

    # ------------------------------------------------------------------ B
    if "undefended" not in res:
        raise SystemExit("run --part A first; Part B is gated on its result")
    _u, _b = res["undefended"], res["base"]
    if _u["band"] == "Low" and _u["recitation"] - _b["recitation"] <= 0.02:
        raise SystemExit(
            "Part A found the generalizable baseline Low on the matrix AND at "
            "base-level recitation on the held-out probe. Neither channel "
            "carries leakage for a budget to remove, so Part B would price a "
            "defense against a risk that is not there. Report Part A instead.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    n_lots = max(1, len(texts) // 8) * args.epochs
    for eps_target, nm in zip(args.epsilons, args.noise_multipliers):
        key = f"dp_nm{nm}"
        if key in res:
            print(f"[B] {key} already measured, skipping")
            continue
        print(f"\n[B] DP-LoRA, noise multiplier {nm} (target eps ~{eps_target}) ...")
        t0 = time.time()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto",
            trust_remote_code=True)
        m = get_peft_model(m, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        D.dp_sgd_finetune(m, tok, texts, epochs=args.epochs,
                          noise_multiplier=nm, clip_norm=1.0)
        ad = adir / f"adapter_{key}"
        m.save_pretrained(str(ad)); tok.save_pretrained(str(ad))
        del m, tok; free()

        eps = D.rdp_epsilon(steps=n_lots, sampling_rate=8 / max(1, len(texts)),
                            noise_multiplier=nm, delta=1e-5)
        r = evaluate_condition(args.model, str(ad), triples, heldout,
                               train_values, f"DP-LoRA (nm={nm})", key, out,
                               args.batch_size)
        r.update({"noise_multiplier": nm, "epsilon": round(float(eps), 4),
                  "delta": 1e-5, "steps": n_lots,
                  "seconds": round(time.time() - t0)})
        res[key] = r
        ckpt.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"    -> eps {eps:.3f}  RW-MER {r['rwmer']:.4f} {r['band']}  "
              f"novel_valid {r['novel_valid']:.3f}  ({r['seconds']}s)")

    # ---------------- readout ---------------- #
    b, u = res["base"], res["undefended"]
    span = u["novel_valid"] - b["novel_valid"]
    print("\n" + "=" * 82)
    print("PART B — the privacy/utility curve")
    print("=" * 82)
    print("  PRIMARY privacy axis is RECITATION, not RW-MER. Part A found the")
    print("  aggregate already Low while unattributed regurgitation ran at "
          f"{u['recitation']:.3f};")
    print("  a budget that moves RW-MER but not recitation has bought nothing here.")
    print()
    print(f"  {'condition':<22} {'eps':>7} {'recite':>8} {'RW-MER':>8} {'band':>7} "
          f"{'novel':>8} {'kept':>7}")
    print(f"  {'base':<22} {'-':>7} {b['recitation']:>8.3f} {0.0:>8.4f} "
          f"{'Low':>7} {b['novel_valid']:>8.3f} {'-':>7}")
    print(f"  {'undefended':<22} {'-':>7} {u['recitation']:>8.3f} {u['rwmer']:>8.4f} "
          f"{u['band']:>7} {u['novel_valid']:>8.3f} {'100%':>7}")
    best = None
    for k in sorted([k for k in res if k.startswith("dp_")],
                    key=lambda k: res[k]["epsilon"]):
        r = res[k]
        kept = (r["novel_valid"] - b["novel_valid"]) / span if span > 1e-9 else 0.0
        closed = ((u["recitation"] - r["recitation"]) /
                  max(u["recitation"] - b["recitation"], 1e-9))
        print(f"  {k:<22} {r['epsilon']:>7.3f} {r['recitation']:>8.3f} "
              f"{r['rwmer']:>8.4f} {r['band']:>7} {r['novel_valid']:>8.3f} "
              f"{kept*100:>6.0f}%   closes {closed*100:.0f}% of the leak")
        if closed >= 0.8 and kept >= 0.5 and (best is None or
                                              r["epsilon"] < res[best]["epsilon"]):
            best = k
    print()
    if best:
        r = res[best]
        print(f"  USABLE BUDGET: {best}, eps {r['epsilon']:.3f} — removes most of the")
        print("  recitation channel while keeping most of the competence span. This is")
        print("  the first priced privacy-utility point in the study; Section 6.5.1's")
        print("  reservation can be closed with it.")
    else:
        print("  NO USABLE BUDGET at the budgets swept. Report which axis failed:")
        print("  a point that closes the leak but drops competence to base level")
        print("  repeats Section 6.5.1 at a better operating point, which strengthens")
        print("  that finding; a point that keeps competence but leaves recitation")
        print("  intact says the noise never reached the channel that carries it.")
    se = (0.25 / max(u["n_heldout"], 1)) ** 0.5
    print(f"\n  Held-out n = {u['n_heldout']}, standard error about {se:.3f}. "
          f"Competence span {span:.3f}")
    print(f"  is {span/max(se,1e-9):.1f} standard errors, so a 'keeps half' verdict "
          f"carries about {span/2/max(se,1e-9):.1f}.")


if __name__ == "__main__":
    main()
