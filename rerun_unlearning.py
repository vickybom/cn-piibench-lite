#!/usr/bin/env python
"""Re-measure the unlearning arm only, with a retain set disjoint from the
capability probe.

The first defense evaluation descended on the same eight sentences that were
used to measure perplexity, so the reported utility figure for the unlearning
condition reflected memorization of the probe rather than capability retention.
This script repeats that arm alone -- reusing the saved LoRA adapter, so no
re-fine-tuning is needed -- with RETAIN_TEXTS and UTILITY_TEXTS disjoint, and
measures perplexity on the same held-out probe used for every other condition
so the numbers remain comparable.

    python rerun_unlearning.py --adapter results_defense/adapter
"""
from __future__ import annotations

import argparse, json, csv, time
from pathlib import Path

from pii_auditor.m1_generator import build_dataset
from pii_auditor.m2_prompts import build_prompt_matrix
from pii_auditor.m3_inference import make_backend
from pii_auditor.m4_detector import detect
from pii_auditor.m5_metrics import compute
from pii_auditor.finetune import build_training_texts
from pii_auditor import defenses as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--adapter", default="results_defense/adapter")
    ap.add_argument("--persons", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--unlearn-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--out-dir", default="results_unlearn_fixed")
    args = ap.parse_args()

    assert set(D.RETAIN_TEXTS).isdisjoint(D.UTILITY_TEXTS), \
        "retain set and capability probe must be disjoint"

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    short = args.model.rstrip("/").split("/")[-1]

    ds = build_dataset(args.persons, args.seed)
    triples = build_prompt_matrix(ds)
    texts = build_training_texts(ds, repeats=args.repeats)
    print(f"[setup] {len(triples)} probes | {len(texts)} forget docs | "
          f"retain={len(D.RETAIN_TEXTS)} probe={len(D.UTILITY_TEXTS)} (disjoint)")

    if not (Path(args.adapter) / "adapter_config.json").exists():
        raise SystemExit(f"adapter not found at {args.adapter}; "
                         f"run defense_eval.py first or pass --adapter")

    # reference: undefended perplexity on the same held-out probe
    be = make_backend("hf_local", args.model, load_in_4bit=True,
                      max_tokens=args.max_tokens, adapter_path=args.adapter)
    ppl_undef = D.perplexity(be.lm, be.tok)
    print(f"[ref] undefended perplexity on held-out probe = {ppl_undef:.2f}")

    print(f"[unlearn] gradient ascent, {args.unlearn_steps} steps ...")
    D.unlearn_gradient_ascent(be.lm, be.tok, forget_texts=texts,
                              retain_texts=D.RETAIN_TEXTS,
                              steps=args.unlearn_steps)
    ppl_unl = D.perplexity(be.lm, be.tok)
    print(f"[unlearn] perplexity on held-out probe = {ppl_unl:.2f}")

    label = f"{short} (+ unlearning)"
    recs, t0 = [], time.time()
    for i in range(0, len(triples), args.batch_size):
        chunk = triples[i:i + args.batch_size]
        for tr, o in zip(chunk, be.complete_batch(chunk)):
            r = detect(o, tr); r["model"] = label; r["prompt"] = tr["prompt"]
            recs.append(r)
        if (i // args.batch_size) % 25 == 0 and i:
            print(f"    {i+len(chunk)}/{len(triples)} "
                  f"({(i+len(chunk))/max(1e-6,time.time()-t0):.1f} q/s)")
    hits = sum(r["hit"] for r in recs)
    print(f"[unlearn] {hits}/{len(recs)} hits")

    metrics = compute(recs)
    (out / "summary.json").write_text(json.dumps({
        "model": label,
        "aggregate_rwmer": float(metrics["summary"].iloc[0]["aggregate_rwmer"]),
        "risk_band": str(metrics["summary"].iloc[0]["risk_band"]),
        "raw_leak_rate": hits / len(recs),
        "perplexity_heldout_probe": ppl_unl,
        "perplexity_undefended_reference": ppl_undef,
        "unlearn_steps": args.unlearn_steps,
        "retain_probe_disjoint": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["per_category"].to_csv(out / "per_category.csv", index=False,
                                   encoding="utf-8-sig")
    with open(out / "records.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model","category","condition","type",
                                          "template_id","pid","hit","match_method",
                                          "expected","output"])
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    print("\n=== Unlearning (corrected utility measurement) ===")
    print(metrics["summary"].to_string(index=False))
    print(f"  perplexity: undefended {ppl_undef:.2f} -> unlearned {ppl_unl:.2f}")
    print("\nArtifacts in", out)


if __name__ == "__main__":
    main()
