#!/usr/bin/env python
"""Stage 3 — the two measurements Section 6.5.2 could not make.

Section 6.5.2 reported that at the generalization operating point every privacy
budget closed both leakage channels completely, and that at two of four budgets
the defended model was better than the released weights at the task. Two things
it could not settle are the subject of this script.

PART CLIP — is it the noise, or the clipping?
---------------------------------------------
Both leakage channels were exactly zero across a 6.6-fold change in the noise
multiplier, including the loosest budget where little noise is added. A quantity
that does not move when the supposed cause moves 6.6-fold is probably not being
driven by that cause. The component DP-SGD applies identically at every budget
is per-example gradient clipping.

The obvious control is "clipping, no noise". On its own that is not enough,
because the differentially private arm and the undefended arm of Section 6.5.2
differ in more than noise: the undefended arm goes through lora_finetune, with
LoRA dropout 0.0, max_len 384 and the HuggingFace optimizer schedule, while the
private arm goes through dp_sgd_finetune, with dropout 0.05, max_len 320 and a
constant-rate AdamW. Any of those could suppress memorization on its own.

So this part runs TWO cells, both through dp_sgd_finetune with everything
identical except the clipping bound:

  clip_on   clip_norm 1.0,  noise 0   -> clipping active, no noise, eps unbounded
  clip_off  clip_norm 1e9,  noise 0   -> same code path, clipping inert

  clip_off leaks, clip_on does not  -> clipping is the mechanism.
  both clean                        -> neither clipping nor noise: something
                                       else in the DP code path suppresses
                                       memorization, and Section 6.5.2's
                                       mechanistic reading needs replacing.
  both leak                         -> noise was doing the work after all,
                                       which would contradict the invariance
                                       that motivated this control.

Neither cell carries a privacy guarantee. They are mechanistic controls, not
candidate defenses, and the script labels them that way in its output.

PART REPLICATE — is the spread across budgets real?
---------------------------------------------------
Utility was not monotone in epsilon, and each budget was trained once, so budget
is confounded with training run. The confound is not small: retraining the
undefended condition moved held-out competence by 0.10 on an identical probe
set, against a spread of 0.20 across four budgets.

This part trains several seeded runs at each of two budgets -- 0.50 and 1.00,
the pair whose inversion was largest -- and reports the within-budget spread
next to the between-budget difference. If the within-budget spread covers the
between-budget difference, the ordering in Table 6.8 is noise and must be
reported as such.

To keep the cost bearable the replicates skip the full twelve-template audit and
run a stratified sample of it instead. Every DP condition in Section 6.5.2
returned exactly zero hits over 11,760 probes; the quantity that actually varies
between runs is held-out competence, which the 900-probe held-out set measures
directly. The sample is there to confirm that zero, not to re-measure it.

    python clip_vs_noise.py --part clip
    python clip_vs_noise.py --part replicate --runs 3
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

from pii_auditor.m1_generator import build_dataset, save_dataset
from pii_auditor.m2_prompts import build_prompt_matrix, TEMPLATE_IDS
import pii_auditor.defenses as D

from epsilon_sweep import evaluate_condition, free, write_csv
from generalization_baseline import build_training_texts_varied
from task_utility import CATS


def dp_train(model_name, texts, epochs, clip_norm, noise_multiplier, seed,
             adapter_dir, batch_size=8):
    """Train through the DP-SGD code path. Every cell in this script uses this
    function, so the only differences between cells are its arguments."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        trust_remote_code=True)
    m = get_peft_model(m, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    D.dp_sgd_finetune(m, tok, texts, epochs=epochs, clip_norm=clip_norm,
                      noise_multiplier=noise_multiplier, batch_size=batch_size)
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(adapter_dir)); tok.save_pretrained(str(adapter_dir))
    del m, tok; free()
    return adapter_dir


def stratified(triples, fraction, seed=0):
    """Take the same fraction of every template, so the sample keeps the
    template balance that Section 5.7 showed dominates extraction."""
    import random
    rng = random.Random(seed)
    by_t = {}
    for t in triples:
        by_t.setdefault(t["template_id"], []).append(t)
    out = []
    for tid in sorted(by_t):
        g = by_t[tid][:]
        rng.shuffle(g)
        out.extend(g[:max(1, int(len(g) * fraction))])
    return out


def mean_sd(xs):
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    return mu, (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["clip", "replicate"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--persons", type=int, default=140)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--heldout-persons", type=int, default=180)
    ap.add_argument("--heldout-seed", type=int, default=99887766)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--runs", type=int, default=3,
                    help="replicate only: seeded runs per budget")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.5, 1.0],
                    help="replicate only: target epsilons, paired with "
                         "--multipliers by position")
    ap.add_argument("--multipliers", type=float, nargs="+", default=[0.648, 0.342])
    ap.add_argument("--matrix-fraction", type=float, default=0.25,
                    help="replicate only: stratified share of the twelve-template "
                         "matrix to audit. Every DP condition in Section 6.5.2 "
                         "returned zero over the full matrix; this confirms the "
                         "zero rather than re-measuring it.")
    ap.add_argument("--out-dir", default="results_clip")
    ap.add_argument("--adapter-dir", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    adir = Path(args.adapter_dir) if args.adapter_dir else out
    adir.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{args.part}.json"
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
    print(f"[setup] {args.persons} persons, {len(triples)} matrix probes, "
          f"{len({t[0] for t in TEMPLATE_IDS})} templates")
    print(f"[setup] varied corpus, {len(texts)} docs, {exposures} exposures/record")
    print(f"[setup] held-out {args.heldout_persons} people, "
          f"{args.heldout_persons * 5} probes; disjoint: OK")

    def save():
        ckpt.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # ================================================================== clip
    if args.part == "clip":
        CELLS = [("clip_on",  1.0,  "clipping active, no noise"),
                 ("clip_off", 1e9,  "same code path, clipping inert")]
        for key, cn, desc in CELLS:
            if key in res:
                print(f"[clip] {key} already measured, skipping"); continue
            print(f"\n[clip] {key}: clip_norm={cn:g}, noise=0 — {desc}")
            t0 = time.time()
            ad = adir / f"adapter_{key}"
            if not (ad / "adapter_config.json").exists():
                dp_train(args.model, texts, args.epochs, cn, 0.0, args.seed, ad)
            else:
                print(f"       reusing adapter at {ad}")
            r = evaluate_condition(args.model, str(ad), triples, heldout,
                                   train_values, f"{key} (control)", key, out,
                                   args.batch_size)
            r.update({"clip_norm": cn, "noise_multiplier": 0.0,
                      "epsilon": None, "guarantee": "none — mechanistic control",
                      "seconds": round(time.time() - t0)})
            res[key] = r; save()
            print(f"    -> RW-MER {r['rwmer']:.4f} {r['band']}  "
                  f"hits {r['raw_hits']}/{r['n_probes']}  "
                  f"recitation {r['recitation']:.4f}  novel {r['novel_valid']:.4f}")

        on, off = res["clip_on"], res["clip_off"]
        print("\n" + "=" * 76)
        print("PART CLIP — is it the clipping, or the noise, or neither?")
        print("=" * 76)
        print("  Neither cell carries a privacy guarantee. Both are controls.\n")
        print(f"  {'cell':<10} {'clip_norm':>10} {'hits':>12} {'RW-MER':>8} "
              f"{'recite':>8} {'novel':>8}")
        for k in ("clip_on", "clip_off"):
            r = res[k]
            print(f"  {k:<10} {r['clip_norm']:>10.0e} "
                  f"{str(r['raw_hits'])+'/'+str(r['n_probes']):>12} "
                  f"{r['rwmer']:>8.4f} {r['recitation']:>8.4f} {r['novel_valid']:>8.4f}")
        print(f"\n  reference, Section 6.5.2 at this operating point:")
        print(f"  {'undefended':<10} {'(no DP path)':>10} {'38/11,760':>12} "
              f"{0.0022:>8.4f} {0.1289:>8.4f} {0.5278:>8.4f}")
        print(f"  {'DP arms':<10} {'1.0':>10} {'0/11,760':>12} "
              f"{0.0:>8.4f} {0.0:>8.4f} {'0.21-0.41':>8}")
        print()
        off_leaks = off["raw_hits"] > 0 or off["recitation"] > 0.02
        on_leaks = on["raw_hits"] > 0 or on["recitation"] > 0.02
        if off_leaks and not on_leaks:
            print("  VERDICT: CLIPPING is the mechanism. With the bound removed the")
            print("  same code path leaks; with it restored, and no noise at all,")
            print("  leakage is gone. Section 6.5.2's hypothesis is supported, and")
            print("  the practical reading changes: the protection observed there")
            print("  does not depend on the privacy budget, so the budget can be")
            print("  chosen for its guarantee rather than for its effect.")
        elif not off_leaks and not on_leaks:
            print("  VERDICT: NEITHER clipping nor noise. Both cells are clean, so")
            print("  something else in this code path — dropout 0.05, max_len 320,")
            print("  or the constant-rate AdamW in place of the scheduled optimizer")
            print("  — suppresses memorization on its own. Section 6.5.2's")
            print("  mechanistic paragraph must be replaced, not merely refined,")
            print("  and the next control is to vary those three one at a time.")
        elif off_leaks and on_leaks:
            print("  VERDICT: clipping alone does NOT suppress it; some noise is")
            print("  needed. That contradicts the invariance across budgets that")
            print("  motivated this control, and the two observations should be")
            print("  reconciled before either is reported.")
        else:
            print("  VERDICT: unclipped is clean while clipped leaks. That ordering")
            print("  is not explicable by the stated mechanism; suspect the run")
            print("  rather than the theory and repeat both cells.")
        return

    # ============================================================= replicate
    frac = args.matrix_fraction
    sample = stratified(triples, frac, seed=args.seed)
    print(f"[setup] replicate audit uses {len(sample)}/{len(triples)} probes "
          f"({frac:.0%}, stratified by template)")

    for eps, nm in zip(args.budgets, args.multipliers):
        acct = D.rdp_epsilon(steps=max(1, len(texts) // 8) * args.epochs,
                             sampling_rate=8 / max(1, len(texts)),
                             noise_multiplier=nm, delta=1e-5)
        for run in range(1, args.runs + 1):
            key = f"eps{eps}_run{run}"
            if key in res:
                print(f"[rep] {key} already measured, skipping"); continue
            print(f"\n[rep] eps~{eps} (nm={nm}, accounted {acct:.3f}) run {run}/{args.runs}")
            t0 = time.time()
            ad = adir / f"adapter_{key}"
            if not (ad / "adapter_config.json").exists():
                dp_train(args.model, texts, args.epochs, 1.0, nm,
                         args.seed + 1000 * run, ad)
            r = evaluate_condition(args.model, str(ad), sample, heldout,
                                   train_values, f"DP eps{eps} run{run}", key,
                                   out, args.batch_size)
            r.update({"epsilon": round(float(acct), 4), "noise_multiplier": nm,
                      "run": run, "train_seed": args.seed + 1000 * run,
                      "matrix_fraction": frac, "seconds": round(time.time() - t0)})
            res[key] = r; save()
            print(f"    -> hits {r['raw_hits']}/{r['n_probes']}  "
                  f"recitation {r['recitation']:.4f}  novel {r['novel_valid']:.4f}  "
                  f"({r['seconds']}s)")

    print("\n" + "=" * 76)
    print("PART REPLICATE — is the spread across budgets bigger than run noise?")
    print("=" * 76)
    groups = {}
    for k, v in res.items():
        if "_run" in k:
            groups.setdefault(k.split("_run")[0], []).append(v)
    print(f"  {'budget':<12} {'runs':>5} {'novel: mean':>12} {'sd':>8} "
          f"{'min':>8} {'max':>8}   hits")
    stats = {}
    for g in sorted(groups):
        vs = sorted(x["novel_valid"] for x in groups[g])
        mu, sd = mean_sd(vs)
        stats[g] = (mu, sd, vs)
        h = sum(x["raw_hits"] for x in groups[g])
        print(f"  {g:<12} {len(vs):>5} {mu:>12.4f} {sd:>8.4f} {vs[0]:>8.4f} "
              f"{vs[-1]:>8.4f}   {h} total")
    print()
    if len(stats) >= 2:
        gs = sorted(stats, key=lambda g: stats[g][0])
        lo, hi = gs[0], gs[-1]
        between = stats[hi][0] - stats[lo][0]
        within = max(stats[g][2][-1] - stats[g][2][0] for g in stats)
        print(f"  between-budget difference of means : {between:.4f}")
        print(f"  largest within-budget range        : {within:.4f}")
        print(f"  Section 6.5.2 reported these budgets as differing by 0.1800")
        print()
        if within >= abs(between):
            print("  VERDICT: run-to-run spread within a single budget is as large as")
            print("  the difference between budgets. The ordering in Table 6.8 is not")
            print("  evidence about epsilon, and Section 6.5.2's refusal to read it as")
            print("  a curve was correct. Report the budgets as a set of points that")
            print("  all close the leakage, with utility varying by run.")
        else:
            print("  VERDICT: the between-budget difference survives the within-budget")
            print("  spread. There is a real effect of the budget on utility, and the")
            print("  non-monotone ordering needs an explanation rather than a")
            print("  variance argument. Report both, and say which budgets separate.")
    save()


if __name__ == "__main__":
    main()
