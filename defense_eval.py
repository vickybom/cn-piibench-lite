#!/usr/bin/env python
"""Defense evaluation: measure how much each privacy defense reduces the
induced memorization leakage established by the fine-tuning stress test.

Conditions produced in a single run:

    C0  base                      (null floor reference)
    C1  fine-tuned, undefended    (the High-risk baseline)
    C2  + output filter (naive)   inference-time redaction
    C3  + output filter (normalising)  catches separator obfuscation
    C4  + machine unlearning      gradient ascent on the forget set
    C5  DP-LoRA fine-tuning       DP-SGD (Abadi et al., 2016)

Every condition is scored with the identical M4-M6 instrument, and each is
paired with a utility measurement (perplexity on held-out generic Chinese) so
the privacy-utility trade-off is quantified.

All per-query records are written to records.csv, which also supports the
per-template ablation performed offline.

    python defense_eval.py --model Qwen/Qwen2.5-1.5B --persons 40 --epochs 30
"""
from __future__ import annotations

import argparse, json, time, csv
from pathlib import Path

from pii_auditor.m1_generator import build_dataset, save_dataset
from pii_auditor.m2_prompts import build_prompt_matrix, TEMPLATE_IDS
from pii_auditor.m3_inference import make_backend
from pii_auditor.m4_detector import detect
from pii_auditor.m5_metrics import compute
from pii_auditor.m6_report import write_reports
from pii_auditor.dashboard import build_from_file
from pii_auditor.pipeline import _collect_sample_io
from pii_auditor.finetune import build_training_texts, lora_finetune
from pii_auditor import defenses as D


FIELDS = ["model", "category", "condition", "type", "template_id", "pid",
          "hit", "match_method", "expected", "output"]


def load_records(path):
    """Read a partially written record file, if present."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["hit"] = str(r["hit"]).lower() in ("true", "1")
        r["pid"] = int(r["pid"])
    return rows


def save_records(path, recs):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def audit(be, triples, label, batch_size=24, progress=True, ckpt=None,
          flush_every=40):
    """Run the sweep with incremental checkpointing so a disconnected session
    resumes instead of restarting. Returns per-query records with raw outputs."""
    done = load_records(ckpt) if ckpt else []
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
        outs = be.complete_batch(chunk)
        for tr, out in zip(chunk, outs):
            r = detect(out, tr)
            r["model"] = label
            r["prompt"] = tr["prompt"]
            recs.append(r)
            _collect_sample_io(store, r, label)
        if ckpt and (i // batch_size) % flush_every == 0 and i:
            save_records(ckpt, recs)
        if progress and (i // batch_size) % 25 == 0 and i:
            rate = (i + len(chunk)) / max(1e-6, time.time() - t0)
            eta = (len(todo) - i - len(chunk)) / max(rate, 1e-6) / 60
            print(f"    {i+len(chunk)}/{len(todo)} ({rate:.1f} q/s, eta {eta:.0f} min)")
    if ckpt:
        save_records(ckpt, recs)
    hits = sum(r["hit"] for r in recs)
    print(f"  [{label}] {hits}/{len(recs)} hits ({time.time()-t0:.0f}s)")
    return recs, [x for b in store.values() for x in b]


def rescore(records, label, transform):
    """Re-score existing outputs through an output-side defense -- no new
    inference is required, which is what makes filtering essentially free."""
    out = []
    for r in records:
        tr = {"pid": r["pid"], "category": r["category"], "type": r["type"],
              "template_id": r["template_id"], "condition": r["condition"],
              "expected": r["expected"]}
        d = detect(transform(r["output"]), tr)
        d["model"] = label
        d["prompt"] = r.get("prompt", "")
        out.append(d)
    hits = sum(x["hit"] for x in out)
    print(f"  [{label}] {hits}/{len(out)} hits (re-scored, no inference)")
    return out


def free():
    try:
        import torch, gc; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--persons", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--adapter", default=None, help="reuse an existing LoRA adapter")
    ap.add_argument("--unlearn-steps", type=int, default=60)
    ap.add_argument("--dp-epochs", type=int, default=10)
    ap.add_argument("--dp-noise", type=float, default=1.0)
    ap.add_argument("--dp-clip", type=float, default=1.0)
    ap.add_argument("--skip-dp", action="store_true")
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the base arm (already measured at 0.000)")
    ap.add_argument("--out-dir", default="results_defense")
    ap.add_argument("--seed", type=int, default=20260524)
    args = ap.parse_args()

    short = args.model.rstrip("/").split("/")[-1]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    all_recs, all_io, utility, meta_extra = [], [], {}, {}

    # ---- corpus + prompts -------------------------------------------------
    ds = build_dataset(args.persons, args.seed)
    save_dataset(ds, "data/defense_dataset.json")
    triples = build_prompt_matrix(ds)
    texts = build_training_texts(ds, repeats=args.repeats)
    print(f"[setup] {args.persons} persons | {len(triples)} probes | {len(texts)} training docs")

    # ---- C1: fine-tune (undefended) ---------------------------------------
    # A path that no longer exists (e.g. after a runtime restart) would be
    # forwarded to the Hub and fail with an opaque 401; fall back to training.
    if args.adapter and not (Path(args.adapter) / "adapter_config.json").exists():
        print(f"[C1] adapter '{args.adapter}' not found on disk "
              f"(session restarted?) -- fine-tuning from scratch instead")
        args.adapter = None
    if args.adapter:
        adapter = args.adapter
        print(f"[C1] reusing adapter {adapter}")
    else:
        print(f"[C1] fine-tuning {args.model} ({args.epochs} epochs) ...")
        adapter = lora_finetune(args.model, texts, out / "adapter", epochs=args.epochs)
        free()

    print("[C1] auditing UNDEFENDED fine-tuned model ...")
    be = make_backend("hf_local", args.model, load_in_4bit=True,
                      max_tokens=args.max_tokens, adapter_path=str(adapter))
    L_FT = f"{short} (fine-tuned, undefended)"
    ft_recs, ft_io = audit(be, triples, L_FT, args.batch_size,
                           ckpt=out / "rec_undefended.csv")
    utility[L_FT] = D.perplexity(be.lm, be.tok)
    all_recs += ft_recs; all_io += ft_io
    del be; free()

    # ---- C2/C3: output filtering (re-scored, no inference) ----------------
    print("[C2] output filter (naive, checksum-validated) ...")
    L_F1 = f"{short} (+ output filter)"
    all_recs += rescore(ft_recs, L_F1, D.make_output_filter(strict=True, normalise=False))
    utility[L_F1] = utility[L_FT]        # inference-time filter: model unchanged

    print("[C3] output filter (normalising -- catches obfuscation) ...")
    L_F2 = f"{short} (+ filter, normalising)"
    all_recs += rescore(ft_recs, L_F2, D.make_output_filter(strict=True, normalise=True))
    utility[L_F2] = utility[L_FT]
    meta_extra["filter_over_redaction"] = D.over_redaction_rate(D.UTILITY_TEXTS + D.RETAIN_TEXTS)

    # ---- C4: machine unlearning -------------------------------------------
    try:
        print(f"[C4] machine unlearning (gradient ascent, {args.unlearn_steps} steps) ...")
        be = make_backend("hf_local", args.model, load_in_4bit=True,
                          max_tokens=args.max_tokens, adapter_path=str(adapter))
        D.unlearn_gradient_ascent(be.lm, be.tok, forget_texts=texts,
                                  retain_texts=D.RETAIN_TEXTS,
                                  steps=args.unlearn_steps)
        L_UL = f"{short} (+ unlearning)"
        ul_recs, ul_io = audit(be, triples, L_UL, args.batch_size,
                               ckpt=out / "rec_unlearned.csv")
        utility[L_UL] = D.perplexity(be.lm, be.tok)
        all_recs += ul_recs; all_io += ul_io
        del be; free()
    except Exception as e:                                    # noqa: BLE001
        print(f"  [C4] SKIPPED -- unlearning failed: {type(e).__name__}: {e}")
        meta_extra["unlearning_error"] = f"{type(e).__name__}: {e}"
        free()

    # ---- C5: DP-LoRA -------------------------------------------------------
    if not args.skip_dp:
      try:
        print(f"[C5] DP-LoRA fine-tuning (sigma={args.dp_noise}, C={args.dp_clip}) ...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16))
        m = prepare_model_for_kbit_training(m); m.enable_input_require_grads()
        m = get_peft_model(m, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"]))
        m.config.use_cache = False
        D.dp_sgd_finetune(m, tok, texts, epochs=args.dp_epochs,
                          clip_norm=args.dp_clip, noise_multiplier=args.dp_noise)
        dp_dir = out / "adapter_dp"; m.save_pretrained(str(dp_dir)); tok.save_pretrained(str(dp_dir))
        steps = args.dp_epochs * max(1, len(texts) // 8)
        meta_extra["dp"] = {
            "epochs": args.dp_epochs, "noise_multiplier": args.dp_noise,
            "clip_norm": args.dp_clip, "steps": steps,
            "epsilon": round(D.rdp_epsilon(steps, 8.0 / max(1, len(texts)),
                                           args.dp_noise), 2), "delta": 1e-5}
        del m; free()

        be = make_backend("hf_local", args.model, load_in_4bit=True,
                          max_tokens=args.max_tokens, adapter_path=str(dp_dir))
        L_DP = f"{short} (DP-LoRA)"
        dp_recs, dp_io = audit(be, triples, L_DP, args.batch_size,
                               ckpt=out / "rec_dp.csv")
        utility[L_DP] = D.perplexity(be.lm, be.tok)
        all_recs += dp_recs; all_io += dp_io
        del be; free()
      except Exception as e:                                  # noqa: BLE001
        print(f"  [C5] SKIPPED -- DP-LoRA failed: {type(e).__name__}: {e}")
        meta_extra["dp_error"] = f"{type(e).__name__}: {e}"
        free()

    # ---- C0: base null floor ----------------------------------------------
    try:
        if args.skip_base:
            raise RuntimeError("skipped by --skip-base")
        print("[C0] auditing BASE model (null floor) ...")
        be = make_backend("hf_local", args.model, load_in_4bit=True, max_tokens=args.max_tokens)
        L_B = f"{short} (base)"
        b_recs, b_io = audit(be, triples, L_B, args.batch_size,
                             ckpt=out / "rec_base.csv")
        utility[L_B] = D.perplexity(be.lm, be.tok)
        all_recs += b_recs; all_io += b_io
        del be; free()
    except Exception as e:                                    # noqa: BLE001
        print(f"  [C0] SKIPPED -- base audit failed: {type(e).__name__}: {e}")
        free()

    # ---- metrics + reports -------------------------------------------------
    metrics = compute(all_recs)
    meta = {"backend": "hf_local (defense evaluation)", "is_simulated": False,
            "models": sorted({r["model"] for r in all_recs}),
            "n_entries": args.persons * 7, "n_persons": args.persons,
            "n_templates": len({t[0] for t in TEMPLATE_IDS}),
            "conditions": ["zh2zh", "en2zh"], "seed": args.seed,
            "total_queries": len(all_recs), "utility_perplexity": utility,
            "finetune": {"epochs": args.epochs, "repeats": args.repeats},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **meta_extra}
    paths = write_reports(metrics, meta, all_io, out)
    build_from_file(paths["results_json"], out / "dashboard.html")

    # per-query records -> enables the per-template ablation offline
    with open(out / "records.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model","category","condition","type",
                                          "template_id","pid","hit","match_method",
                                          "expected","output"])
        w.writeheader()
        for r in all_recs:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    print("\n=== Defense Evaluation ===")
    print(metrics["summary"].to_string(index=False))
    print("\nUtility (perplexity on held-out Chinese; lower = better):")
    for k, v in utility.items():
        print(f"  {k:44s} {v:.2f}")
    if "dp" in meta_extra:
        print(f"\nDP budget: eps={meta_extra['dp']['epsilon']} at delta=1e-5")
    print("\nArtifacts in", out)


if __name__ == "__main__":
    main()
