#!/usr/bin/env python
"""Stage 4 — locate the noise threshold, and price the guarantee it carries.

WHAT SECTIONS 6.5.3 AND 6.5.4 LEFT OPEN
---------------------------------------
Leakage at this operating point stands at 45 verbatim reproductions and 0.270
recitation with no noise at all, and at exactly zero by a noise multiplier of
0.098 -- the smallest the epsilon sweep evaluated. Every budget an operator
would consider sits above that. The entire transition therefore lies inside an
interval no experiment has entered.

Locating it gives the smallest perturbation that suffices, which is the quantity
Section 6.5.4 argues an operator actually needs, because within the swept range
the choice of epsilon changes the strength of the guarantee and nothing that can
be measured.

WHY THE ANSWER IS LIKELY TO BE UNCOMFORTABLE
--------------------------------------------
The accountant, at the 560 lots of this configuration, prices the rungs of this
ladder like this:

    nm 0.000 -> epsilon unbounded          nm 0.020 -> epsilon 27.98
    nm 0.010 -> epsilon 81.56              nm 0.040 -> epsilon 11.50
                                           nm 0.070 -> epsilon  5.90

If the empirical threshold turns out to sit at a small multiplier, the guarantee
attached to the smallest sufficient noise is weak to the point of vacuity -- an
epsilon in the tens or hundreds is not a meaningful privacy statement. That
would sharpen Section 6.5.4 rather than soften it: the operator who chooses
epsilon 0.5 is paying for a guarantee, and this experiment says how much of that
payment buys observable protection. The expected answer is none of it.

TWO INSTRUMENTS, AND WHY RECITATION IS THE PRIMARY ONE
------------------------------------------------------
On the full matrix the signal is 45 hits in 11,760 probes, a rate of 0.0038. On
the held-out probe the signal is 0.270 against a base rate of exactly zero. The
second is roughly seventy times stronger per query and ten times cheaper to
collect, so it leads, and a stratified quarter of the matrix confirms it. At
nm = 0 that quarter should return about eleven hits; returning zero instead
would be a one-in-fifty-thousand event, so the sample is not the weak link.

SEEDS
-----
Section 6.5.3 established that a single run at this operating point is a draw
rather than a result. Every rung is therefore run at two seeds, and the readout
refuses to report a threshold that the two seeds disagree about.

    python noise_threshold.py --runs 1     # first ladder, about 4 hours
    python noise_threshold.py --runs 2     # completes it, resumes automatically
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

from pii_auditor.m1_generator import build_dataset, save_dataset
from pii_auditor.m2_prompts import build_prompt_matrix
import pii_auditor.defenses as D

from epsilon_sweep import evaluate_condition, free
from clip_vs_noise import dp_train, stratified
from generalization_baseline import build_training_texts_varied
from task_utility import CATS

LEAK_EPS = 0.02          # recitation at or below this counts as "no leakage"


def accounted(nm, steps, q, delta=1e-5):
    if nm <= 0:
        return float("inf")
    return float(D.rdp_epsilon(steps=steps, sampling_rate=q,
                               noise_multiplier=nm, delta=delta))


def fmt_eps(e):
    return "unbounded" if math.isinf(e) else f"{e:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--persons", type=int, default=140)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--heldout-persons", type=int, default=180)
    ap.add_argument("--heldout-seed", type=int, default=99887766)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--multipliers", type=float, nargs="+",
                    default=[0.0, 0.010, 0.020, 0.040, 0.070],
                    help="the 0.0 rung is not padding: it re-runs the "
                         "zero-noise control at a fresh seed, so the ladder has "
                         "to reproduce a known endpoint before its interior is "
                         "believed.")
    ap.add_argument("--runs", type=int, default=1,
                    help="which seed to run. Run 1 first for a complete "
                         "provisional ladder, then 2 to test it.")
    ap.add_argument("--matrix-fraction", type=float, default=0.25)
    ap.add_argument("--out-dir", default="results_threshold")
    ap.add_argument("--adapter-dir", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    adir = Path(args.adapter_dir) if args.adapter_dir else out
    adir.mkdir(parents=True, exist_ok=True)
    ckpt = out / "threshold.json"
    res = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}

    ds = build_dataset(args.persons, args.seed)
    save_dataset(ds, out / "dataset.json")
    heldout = build_dataset(args.heldout_persons, args.heldout_seed)
    train_values = {p[c] for p in ds["persons"] for c in CATS}
    assert not (train_values & {p[c] for p in heldout["persons"] for c in CATS})
    assert not ({p["full_name"] for p in ds["persons"]} &
                {p["full_name"] for p in heldout["persons"]})

    triples = build_prompt_matrix(ds)
    sample = stratified(triples, args.matrix_fraction, seed=args.seed)
    texts = build_training_texts_varied(ds, repeats=args.repeats, seed=args.seed)
    n_lots = max(1, len(texts) // 8) * args.epochs
    q = 8 / max(1, len(texts))

    print(f"[setup] {args.persons} persons, {len(texts)} docs, "
          f"{args.epochs * args.repeats} exposures/record, {n_lots} lots")
    print(f"[setup] audit {len(sample)}/{len(triples)} matrix probes "
          f"({args.matrix_fraction:.0%}, stratified) + "
          f"{args.heldout_persons * 5} held-out probes")
    print(f"[setup] ladder, seed {args.runs}:")
    for nm in args.multipliers:
        print(f"           nm {nm:<6} -> epsilon {fmt_eps(accounted(nm, n_lots, q))}")

    for nm in args.multipliers:
        key = f"nm{nm}_run{args.runs}"
        if key in res:
            print(f"[rung] {key} already measured, skipping"); continue
        eps = accounted(nm, n_lots, q)
        print(f"\n[rung] noise multiplier {nm} (epsilon {fmt_eps(eps)}), "
              f"seed {args.runs}")
        t0 = time.time()
        ad = adir / f"adapter_{key}"
        if not (ad / "adapter_config.json").exists():
            dp_train(args.model, texts, args.epochs, 1.0, nm,
                     args.seed + 1000 * args.runs, ad)
        r = evaluate_condition(args.model, str(ad), sample, heldout,
                               train_values, f"nm={nm} run{args.runs}", key,
                               out, args.batch_size)
        r.update({"noise_multiplier": nm, "run": args.runs,
                  "epsilon": None if math.isinf(eps) else round(eps, 3),
                  "train_seed": args.seed + 1000 * args.runs,
                  "matrix_fraction": args.matrix_fraction,
                  "guarantee": "none" if nm <= 0 else "epsilon-delta",
                  "seconds": round(time.time() - t0)})
        res[key] = r
        ckpt.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"    -> hits {r['raw_hits']}/{r['n_probes']}  "
              f"recitation {r['recitation']:.4f}  novel {r['novel_valid']:.4f}  "
              f"({r['seconds']}s)")

    # ---------------------------- readout ---------------------------- #
    rungs = {}
    for k, v in res.items():
        rungs.setdefault(v["noise_multiplier"], {})[v["run"]] = v
    order = sorted(rungs)
    print("\n" + "=" * 80)
    print("NOISE THRESHOLD LADDER")
    print("=" * 80)
    print("  Recitation leads; the matrix quarter confirms. 'leaks' means "
          f"recitation > {LEAK_EPS}")
    print(f"\n  {'nm':>7} {'epsilon':>11} {'seed':>5} {'hits':>8} {'recite':>9} "
          f"{'novel':>8}  verdict")
    disagree = []
    for nm in order:
        eps = accounted(nm, n_lots, q)
        verdicts = set()
        for run in sorted(rungs[nm]):
            v = rungs[nm][run]
            leaks = v["recitation"] > LEAK_EPS or v["raw_hits"] > 0
            verdicts.add(leaks)
            print(f"  {nm:>7.3f} {fmt_eps(eps):>11} {run:>5} "
                  f"{str(v['raw_hits'])+'/'+str(v['n_probes']):>8} "
                  f"{v['recitation']:>9.4f} {v['novel_valid']:>8.4f}  "
                  f"{'LEAKS' if leaks else 'clean'}")
        if len(verdicts) > 1:
            disagree.append(nm)
    print(f"\n  reference  {'unbounded':>11} {'-':>5} {'45/11760':>8} "
          f"{0.2700:>9.4f} {0.4078:>8.4f}  LEAKS   (§6.5.3, full matrix)")
    print(f"  reference  {'3.99':>11} {'-':>5} {'0/11760':>8} "
          f"{0.0:>9.4f} {0.4144:>8.4f}  clean   (§6.5.2, full matrix)")

    complete = [nm for nm in order if len(rungs[nm]) >= 2]
    print()
    if disagree:
        print(f"  SEEDS DISAGREE at nm {disagree}. That is the Section 6.5.3")
        print("  lesson repeating: at those rungs the outcome is a draw, not a")
        print("  threshold. Report the disagreement and add seeds there before")
        print("  quoting any boundary.")
    if not complete:
        print("  PROVISIONAL: only one seed per rung so far. Run --runs 2 before")
        print("  quoting a threshold; Section 6.5.3 is what happens otherwise.")
    agreed = [nm for nm in complete if nm not in disagree]
    leaky = [nm for nm in agreed
             if any(rungs[nm][r]["recitation"] > LEAK_EPS or rungs[nm][r]["raw_hits"] > 0
                    for r in rungs[nm])]
    clean = [nm for nm in agreed if nm not in leaky]
    if agreed and leaky and clean:
        lo, hi = max(leaky), min(clean)
        e_hi = accounted(hi, n_lots, q)
        print(f"  THRESHOLD BRACKETED: leakage survives at nm {lo:.3f} and is gone")
        print(f"  by nm {hi:.3f}, on both seeds.")
        print(f"\n  The smallest sufficient noise carries epsilon {fmt_eps(e_hi)}"
              f" at delta 1e-5.")
        if not math.isinf(e_hi) and e_hi > 10:
            print("  That is not a meaningful privacy guarantee. The empirical")
            print("  protection an operator can observe is therefore purchased at a")
            print("  budget nobody would report, and every tighter budget buys")
            print("  guarantee strength alone. This is the Section 6.5.4 argument")
            print("  with a number attached to it.")
        else:
            print("  That is a defensible budget, so the smallest sufficient noise")
            print("  and a reportable guarantee coincide here — which would qualify")
            print("  the Section 6.5.4 argument rather than sharpen it.")
    elif agreed and not leaky:
        print(f"  EVERY rung is clean, including nm {min(agreed):.3f}. The transition")
        print("  is below the bottom of this ladder. If the nm 0.0 rung is also")
        print("  clean the ladder has failed to reproduce the Section 6.5.3 control")
        print("  and the run should be treated as suspect before anything else.")
    elif agreed and not clean:
        print("  EVERY rung leaks. The transition is above the top of this ladder,")
        print("  between the highest rung here and nm 0.098. Extend upward.")


if __name__ == "__main__":
    main()
