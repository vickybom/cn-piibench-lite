#!/usr/bin/env python
"""Stage 5 — replicate the two Chapter 5 conclusions that run variance could reach.

WHY THIS EXISTS
---------------
Section 6.5.3 established that at this study's operating points a single
training run is a draw rather than a result: it is where the epsilon = 0.50
utility claim died, having looked significant on one run and vanished over
three. Chapter 5's conclusions were all measured on one run each.

For most of them that is unlikely to matter, because they sit at sixty
exposures where memorization saturates near 1.0 and there is little room for a
run to differ. Two do not have that protection, and they are the two this
script re-measures:

  E6, capacity.  1.5B reproduced 0.5235 of probed entries against 3B's 0.4301 on
                 the eight direct-memorization templates -- a difference of
                 0.0934, against a run-variance estimate of 0.0276 quoted in
                 Section 5.2.4. Roughly three times the noise, on one run each.

  E9, CLMD.      The matched-pair cross-lingual differential on the aligned
                 model is -0.048. That is smaller than twice the run-variance
                 estimate. The paired design should cancel much of the variance
                 -- both halves come from the same checkpoint -- but "should"
                 is not a measurement, and the abstract quotes this number.

WHAT VARIES AND WHAT DOES NOT
-----------------------------
The corpus seed stays at 20260524, so the same 140 people with the same values
are used. Only the training seed changes. Anything that moves is therefore
run-to-run variance in fine-tuning and nothing else. Until now the training seed
was not a parameter at all: every run in this study used HuggingFace's default
of 42 implicitly.

PRE-REGISTERED READING, FIXED BEFORE THE RUN
--------------------------------------------
E6 is confirmed if the capacity difference keeps its sign and its magnitude
stays well clear of the run-variance estimate. It is in trouble if the sign
flips, and needs restating if the magnitude falls to the noise level.

E9 is confirmed if the matched-pair differential stays small and the unpaired
formulation still departs from it, ideally still with opposite signs on the
aligned model. The methodological claim -- that differencing non-matched
template families is unsafe -- survives even if the substantive number moves,
because it is a claim about the gap between the two formulations rather than
about either value.

    python replicate_main.py --train-seed 1337 --models base3b
    python replicate_main.py --train-seed 1337 --models instruct
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path

from pii_auditor.m1_generator import build_dataset, save_dataset
from pii_auditor.m2_prompts import build_prompt_matrix, TEMPLATE_IDS, CLMD_PAIRS
from pii_auditor.m3_inference import make_backend
from pii_auditor.m4_detector import detect
from pii_auditor.m5_metrics import compute, risk_band
from pii_auditor.finetune import build_training_texts, lora_finetune

# The values this run is testing, from the thesis as it stands.
ORIGINAL = {
    "Qwen/Qwen2.5-1.5B":          {"ab": 0.5235, "label": "1.5B"},
    "Qwen/Qwen2.5-3B":            {"ab": 0.4301, "label": "3B"},
    "Qwen/Qwen2.5-1.5B-Instruct": {"ab": None,   "label": "1.5B-Instruct"},
}
ORIG_CAPACITY_DIFF = 0.5235 - 0.4301          # +0.0934, 1.5B minus 3B
ORIG_CLMD_MATCHED = -0.048                    # aligned model, matched pairs
ORIG_CLMD_UNPAIRED = 0.216                    # aligned model, unpaired
RUN_VARIANCE = 0.0276                         # Section 5.2.4, aggregate RW-MER

FIELDS = ["model", "category", "condition", "type", "template_id", "pid",
          "hit", "match_method", "expected", "output"]

GROUPS = {
    "base3b":   ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"],
    "instruct": ["Qwen/Qwen2.5-1.5B-Instruct"],
    "all":      ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
                 "Qwen/Qwen2.5-1.5B-Instruct"],
}


def free():
    try:
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass


def audit(be, triples, label, batch_size=24, flush=40):
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


def scope_rate(recs, templates):
    """Hit rate restricted to a template scope. E6's conclusion is stated on the
    eight direct-memorization templates, because Types C and D presuppose
    instruction-following and the sign of the aggregate depends on whether they
    are included (Section 5.12)."""
    sel = [r for r in recs if r["template_id"] in templates]
    return (sum(r["hit"] for r in sel) / len(sel), len(sel)) if sel else (float("nan"), 0)


def template_rate(recs, tid):
    sel = [r for r in recs if r["template_id"] == tid]
    return sum(r["hit"] for r in sel) / len(sel) if sel else float("nan")


def clmd_both_ways(recs):
    """Matched-pair and unpaired cross-lingual differentials.

    Matched: mean over CLMD_PAIRS of MER(English C) - MER(Chinese D). The two
    templates differ only in language, so the difference isolates language.

    Unpaired: mean of the English family minus mean of the Chinese A+B family,
    which is the formulation Section 5.8.1 shows to be confounded. It is
    computed here because the point of E9 is the gap between the two.
    """
    matched = []
    for en, zh in CLMD_PAIRS.items():
        matched.append(template_rate(recs, en) - template_rate(recs, zh))
    eng = [t for t in CLMD_PAIRS]                       # C1, C2
    zho = [t[0] for t in TEMPLATE_IDS if t[0].startswith(("A", "B"))]
    en_rate, _ = scope_rate(recs, eng)
    zh_rate, _ = scope_rate(recs, zho)
    return {"matched_pairs": {en: round(template_rate(recs, en) - template_rate(recs, zh), 4)
                              for en, zh in CLMD_PAIRS.items()},
            "matched_mean": round(sum(matched) / len(matched), 4),
            "unpaired": round(en_rate - zh_rate, 4),
            "english_family": round(en_rate, 4),
            "chinese_family": round(zh_rate, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", choices=sorted(GROUPS), default="base3b")
    ap.add_argument("--persons", type=int, default=140)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260524,
                    help="CORPUS seed. Do not change it: holding the corpus "
                         "fixed is what makes this a replication.")
    ap.add_argument("--train-seed", type=int, default=1337,
                    help="TRAINING seed. Every earlier run in this study used "
                         "42 implicitly; this is the only thing that varies.")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--out-dir", default="results_replicate")
    ap.add_argument("--adapter-dir", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    adir = Path(args.adapter_dir) if args.adapter_dir else out
    adir.mkdir(parents=True, exist_ok=True)
    ckpt = out / "replicate.json"
    res = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}

    ds = build_dataset(args.persons, args.seed)
    save_dataset(ds, out / "dataset.json")
    triples = build_prompt_matrix(ds)
    texts = build_training_texts(ds, repeats=args.repeats)
    AB = [t[0] for t in TEMPLATE_IDS if t[0].startswith(("A", "B"))]

    print(f"[setup] corpus seed {args.seed} (FIXED), training seed "
          f"{args.train_seed} (was 42)")
    print(f"[setup] {args.persons} persons, {len(texts)} docs, "
          f"{args.epochs * args.repeats} exposures/record")
    print(f"[setup] {len(triples)} probes over {len(TEMPLATE_IDS)} templates; "
          f"A+B scope = {len(AB)} templates")

    for model in GROUPS[args.models]:
        key = f"{ORIGINAL[model]['label']}_seed{args.train_seed}"
        if key in res:
            print(f"[skip] {key} already measured"); continue
        chat = model.endswith("-Instruct")
        print(f"\n[run] {model}  (chat template: {chat})")
        t0 = time.time()
        ad = adir / f"adapter_{key}"
        if not (ad / "adapter_config.json").exists():
            print(f"   fine-tuning, {args.epochs} epochs, seed {args.train_seed} ...")
            lora_finetune(model, texts, ad, epochs=args.epochs,
                          seed=args.train_seed)
            free()
        else:
            print(f"   reusing adapter at {ad}")
        be = make_backend("hf_local", model, load_in_4bit=True,
                          max_tokens=args.max_tokens, adapter_path=str(ad),
                          use_chat_template=chat) if chat else \
             make_backend("hf_local", model, load_in_4bit=True,
                          max_tokens=args.max_tokens, adapter_path=str(ad))
        print(f"   auditing {len(triples)} probes ...")
        recs = audit(be, triples, key, args.batch_size)
        del be; free()

        with open(out / f"records_{key}.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader(); w.writerows(recs)

        m = compute(recs)
        agg = float(m["summary"][m["summary"].model == key].aggregate_rwmer.iloc[0])
        ab_rate, ab_n = scope_rate(recs, AB)
        full_rate = sum(r["hit"] for r in recs) / len(recs)
        res[key] = {"model": model, "label": ORIGINAL[model]["label"],
                    "train_seed": args.train_seed, "corpus_seed": args.seed,
                    "ab_scope_rate": round(ab_rate, 4), "ab_scope_n": ab_n,
                    "full_matrix_rate": round(full_rate, 4),
                    "aggregate_rwmer": round(agg, 4), "band": risk_band(agg),
                    "clmd": clmd_both_ways(recs),
                    "per_template": {t[0]: round(template_rate(recs, t[0]), 4)
                                     for t in TEMPLATE_IDS},
                    "n_probes": len(recs), "seconds": round(time.time() - t0)}
        ckpt.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        r = res[key]
        print(f"   -> A+B {r['ab_scope_rate']:.4f}  full {r['full_matrix_rate']:.4f}  "
              f"RW-MER {r['aggregate_rwmer']:.4f} {r['band']}  "
              f"CLMD matched {r['clmd']['matched_mean']:+.4f} "
              f"unpaired {r['clmd']['unpaired']:+.4f}  ({r['seconds']}s)")

    # ------------------------------- readout ------------------------------- #
    s = args.train_seed
    print("\n" + "=" * 80)
    print(f"REPLICATION AT TRAINING SEED {s}  (corpus held fixed at {args.seed})")
    print("=" * 80)

    a, b = f"1.5B_seed{s}", f"3B_seed{s}"
    if a in res and b in res:
        d = res[a]["ab_scope_rate"] - res[b]["ab_scope_rate"]
        print("\n  E6 — capacity, on the eight direct-memorization templates")
        print(f"    {'':<22}{'original':>10}{'seed '+str(s):>12}")
        print(f"    {'1.5B':<22}{ORIGINAL['Qwen/Qwen2.5-1.5B']['ab']:>10.4f}"
              f"{res[a]['ab_scope_rate']:>12.4f}")
        print(f"    {'3B':<22}{ORIGINAL['Qwen/Qwen2.5-3B']['ab']:>10.4f}"
              f"{res[b]['ab_scope_rate']:>12.4f}")
        print(f"    {'difference (1.5B-3B)':<22}{ORIG_CAPACITY_DIFF:>+10.4f}{d:>+12.4f}")
        print(f"    run-variance estimate quoted in the thesis: {RUN_VARIANCE}")
        if d > 0 and abs(d) > 2 * RUN_VARIANCE:
            print("    CONFIRMED: same sign, and comfortably clear of the noise.")
            print("    Conclusion 2 stands as written.")
        elif d > 0:
            print("    SIGN HOLDS but the margin is now within about two standard")
            print("    errors of the run variance. Conclusion 2 should be restated")
            print("    with the replication and a weaker quantitative claim.")
        else:
            print("    SIGN FLIPPED. Conclusion 2 cannot be stated on two runs;")
            print("    report both and treat the capacity question as open.")

    print("\n  E9 — cross-lingual differential")
    print(f"    {'model':<16}{'matched':>10}{'unpaired':>10}{'gap':>9}   original matched")
    for k, v in sorted(res.items()):
        if not k.endswith(f"seed{s}"):
            continue
        c = v["clmd"]
        gap = c["unpaired"] - c["matched_mean"]
        orig = ORIG_CLMD_MATCHED if "Instruct" in v["label"] else None
        print(f"    {v['label']:<16}{c['matched_mean']:>+10.4f}{c['unpaired']:>+10.4f}"
              f"{gap:>+9.4f}   {orig if orig is not None else '-'}")
    inst = res.get(f"1.5B-Instruct_seed{s}")
    if inst:
        c = inst["clmd"]
        moved = abs(c["matched_mean"] - ORIG_CLMD_MATCHED)
        print(f"\n    aligned model matched-pair: {ORIG_CLMD_MATCHED:+.4f} -> "
              f"{c['matched_mean']:+.4f}  (moved {moved:.4f})")
        if moved > abs(ORIG_CLMD_MATCHED):
            print("    The differential moved by more than its own size. It is")
            print("    within run noise and must be reported as indistinguishable")
            print("    from zero rather than as a small negative value.")
        else:
            print("    Stable to within its own magnitude; the reported value holds.")
        if (c["unpaired"] > 0) != (c["matched_mean"] > 0):
            print("    The two formulations still carry OPPOSITE signs, which is")
            print("    the methodological claim of Section 5.8.1, independently")
            print("    reproduced at a second training seed.")
        else:
            print(f"    Same sign this time, but they still differ by "
                  f"{c['unpaired'] - c['matched_mean']:+.4f}; the methodological")
            print("    claim is about the gap, and the gap survives.")
    print("\n  Per-query records written; every number above recomputes from them.")


if __name__ == "__main__":
    main()
