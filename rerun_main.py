#!/usr/bin/env python
"""Scaled re-run of the main memorization experiment, with matched-pair CLMD.

Addresses two examiner-facing weaknesses of the first run:

  (1) Scale. The reported corpus was 1,000 entries but the experiments used 40
      person records. This script runs the primary model at a configurable and
      much larger record count (default 140 records = 20 per category), so the
      headline result and the earlier 40-record result can be compared as a
      sensitivity analysis.

  (2) CLMD confound. The original differential compared the mean of the Chinese
      families (Type A + B, eight templates) against the mean of the English
      family (Type C, two templates), so language was confounded with template
      design. Type D templates -- literal Chinese renderings of C1 and C2 --
      are now probed as well, giving a matched-pair differential
      MER(C1) - MER(D1) and MER(C2) - MER(D2) that isolates language alone.

The run is RESUMABLE: the adapter and the per-condition record files are written
as they complete, and re-invoking the script skips whatever is already on disk.
A Colab disconnect therefore costs only the stage in flight.

    python rerun_main.py --model Qwen/Qwen2.5-1.5B --persons 140 --epochs 30
"""
from __future__ import annotations

import argparse, csv, json, os, time
from pathlib import Path

from pii_auditor.m1_generator import build_dataset, save_dataset
from pii_auditor.m2_prompts import build_prompt_matrix, TEMPLATE_IDS, CLMD_PAIRS
from pii_auditor.m3_inference import make_backend
from pii_auditor.m4_detector import detect
from pii_auditor.m5_metrics import compute
from pii_auditor.m6_report import write_reports
from pii_auditor.dashboard import build_from_file
from pii_auditor.pipeline import _collect_sample_io
from pii_auditor.finetune import build_training_texts, lora_finetune

FIELDS = ["model", "category", "condition", "type", "template_id", "pid",
          "hit", "match_method", "expected", "output"]


def free():
    try:
        import torch, gc; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass


def load_records(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["hit"] = str(r["hit"]).lower() in ("true", "1")
        r["pid"] = int(r["pid"])
    return rows


def save_records(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def audit(be, triples, label, out_csv: Path, batch_size=24, flush_every=40):
    """Audit with incremental checkpointing; resumes from a partial file."""
    done = load_records(out_csv)
    seen = {(r["pid"], r["category"], r["template_id"]) for r in done}
    if seen:
        print(f"  [{label}] resuming: {len(done)} queries already on disk")
    todo = [t for t in triples
            if (t["pid"], t["category"], t["template_id"]) not in seen]
    if not todo:
        print(f"  [{label}] already complete ({len(done)} queries)")
        return done, []
    recs, store, t0 = list(done), {}, time.time()
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        for tr, o in zip(chunk, be.complete_batch(chunk)):
            r = detect(o, tr); r["model"] = label; r["prompt"] = tr["prompt"]
            recs.append(r)
            _collect_sample_io(store, r, label)
        if (i // batch_size) % flush_every == 0 and i:
            save_records(out_csv, recs)
            rate = (i + len(chunk)) / max(1e-6, time.time() - t0)
            eta = (len(todo) - i - len(chunk)) / max(rate, 1e-6) / 60
            print(f"    {i+len(chunk)}/{len(todo)} ({rate:.1f} q/s, eta {eta:.0f} min)")
    save_records(out_csv, recs)
    hits = sum(r["hit"] for r in recs)
    print(f"  [{label}] done: {hits}/{len(recs)} hits ({time.time()-t0:.0f}s)")
    return recs, [x for b in store.values() for x in b]


def paired_clmd(recs, label):
    """MER(C) - MER(D) per matched template pair and per category."""
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0])
    for r in recs:
        if r["model"] != label:
            continue
        acc[(r["template_id"], r["category"])][1] += 1
        if r["hit"]:
            acc[(r["template_id"], r["category"])][0] += 1

    def mer(tid, cat):
        h, n = acc.get((tid, cat), (0, 0))
        return (h / n if n else None), n

    out = []
    cats = sorted({c for (_, c) in acc})
    for en, zh in CLMD_PAIRS.items():
        for c in cats:
            me, ne = mer(en, c)
            mz, nz = mer(zh, c)
            if me is None or mz is None:
                continue
            out.append({"pair": f"{en}-{zh}", "category": c,
                        "mer_en": round(me, 4), "n_en": ne,
                        "mer_zh": round(mz, 4), "n_zh": nz,
                        "paired_clmd": round(me - mz, 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--persons", type=int, default=140,
                    help="person records; 140 = 20 per category")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--chat", action="store_true",
                    help="audit through the chat template (use for -Instruct)")
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the base-model arm (already known to be ~0)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=20260524)
    args = ap.parse_args()

    short = args.model.rstrip("/").split("/")[-1]
    out = Path(args.out_dir or f"results_main_{short}_{args.persons}")
    out.mkdir(parents=True, exist_ok=True)
    chat_kw = {"use_chat_template": True} if args.chat else {}

    ds = build_dataset(args.persons, args.seed)
    save_dataset(ds, out / "dataset.json")
    triples = build_prompt_matrix(ds)
    texts = build_training_texts(ds, repeats=args.repeats)
    n_tpl = len({t[0] for t in TEMPLATE_IDS})
    print(f"[setup] {args.persons} persons x 7 categories x {n_tpl} templates "
          f"= {len(triples)} probes/condition | {len(texts)} training docs")
    print(f"[setup] entries across categories = {args.persons * 7}")

    # ---- fine-tune (resumable via the saved adapter) ---------------------- #
    adapter = out / "adapter"
    if (adapter / "adapter_config.json").exists():
        print(f"[1] reusing adapter at {adapter}")
    else:
        print(f"[1] fine-tuning {args.model}: {args.epochs} epochs x "
              f"{args.repeats} repeats = {args.epochs*args.repeats} exposures/record")
        lora_finetune(args.model, texts, adapter, epochs=args.epochs)
        free()

    all_recs, all_io = [], []

    # ---- fine-tuned arm --------------------------------------------------- #
    L_FT = f"{short} (fine-tuned)"
    ft_csv = out / "records_finetuned.csv"
    if len(load_records(ft_csv)) < len(triples):
        print("[2] auditing FINE-TUNED model ...")
        be = make_backend("hf_local", args.model, load_in_4bit=True,
                          max_tokens=args.max_tokens,
                          adapter_path=str(adapter), **chat_kw)
        r, io_ = audit(be, triples, L_FT, ft_csv, args.batch_size)
        del be; free()
    else:
        print("[2] fine-tuned arm already complete")
        r, io_ = load_records(ft_csv), []
    all_recs += r; all_io += io_

    # ---- base arm --------------------------------------------------------- #
    if not args.skip_base:
        L_B = f"{short} (base)"
        b_csv = out / "records_base.csv"
        if len(load_records(b_csv)) < len(triples):
            print("[3] auditing BASE model ...")
            be = make_backend("hf_local", args.model, load_in_4bit=True,
                              max_tokens=args.max_tokens, **chat_kw)
            r, io_ = audit(be, triples, L_B, b_csv, args.batch_size)
            del be; free()
        else:
            print("[3] base arm already complete")
            r, io_ = load_records(b_csv), []
        all_recs += r; all_io += io_

    # ---- metrics ---------------------------------------------------------- #
    metrics = compute(all_recs)
    pairs = paired_clmd(all_recs, L_FT)
    meta = {"backend": "hf_local (scaled main run)", "is_simulated": False,
            "models": sorted({r["model"] for r in all_recs}),
            "n_entries": args.persons * 7, "n_persons": args.persons,
            "n_templates": n_tpl, "conditions": ["zh2zh", "en2zh"],
            "seed": args.seed, "total_queries": len(all_recs),
            "finetune": {"epochs": args.epochs, "repeats": args.repeats,
                         "exposures_per_record": args.epochs * args.repeats},
            "chat_protocol": bool(args.chat),
            "paired_clmd": pairs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    paths = write_reports(metrics, meta, all_io, out)
    build_from_file(paths["results_json"], out / "dashboard.html")
    (out / "paired_clmd.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Model summary ===")
    print(metrics["summary"].to_string(index=False))
    print("\n=== Matched-pair CLMD (isolates language; same template both sides) ===")
    print(f"{'pair':>8} {'category':<20} {'MER_EN':>8} {'MER_ZH':>8} {'CLMD':>8}")
    for p in pairs:
        print(f"{p['pair']:>8} {p['category']:<20} {p['mer_en']:>8.3f} "
              f"{p['mer_zh']:>8.3f} {p['paired_clmd']:>+8.3f}")
    if pairs:
        m = sum(p["paired_clmd"] for p in pairs) / len(pairs)
        print(f"{'':>8} {'MEAN':<20} {'':>8} {'':>8} {m:>+8.3f}")
    print("\nArtifacts in", out)


if __name__ == "__main__":
    main()
