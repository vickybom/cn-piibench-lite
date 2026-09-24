#!/usr/bin/env python
"""One-command GPU run of the CN-PIIBench-Lite audit on BASE Qwen-2.5 weights.

Intended for a cloud or local GPU (Colab / AutoDL / CUDA workstation). Runs the
paper's primary models via the hf_local backend under 4-bit quantization and
writes the results + defense dashboard.

    python run_gpu.py                       # 1.5B + 3B, 300 entries (pilot)
    python run_gpu.py --full                # 1.5B + 3B + 7B, 1000 entries
    python run_gpu.py --models Qwen/Qwen2.5-1.5B --n-entries 1000

Base (not -Instruct) weights are used because the study measures memorization
of pre-trained weights via prefix / chat-template completion (Carlini et al. 2023a).
"""
from __future__ import annotations

import argparse
from pii_auditor.pipeline import run

DEFAULT_MODELS = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]
FULL_MODELS = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="HF repo ids of base weights")
    ap.add_argument("--n-entries", type=int, default=None)
    ap.add_argument("--full", action="store_true",
                    help="1.5B+3B+7B at 1000 entries (paper's standard config)")
    ap.add_argument("--no-4bit", action="store_true", help="load in fp16 instead of 4-bit")
    ap.add_argument("--out-dir", default="results_gpu")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="batched generation; lower it if you hit CUDA OOM")
    args = ap.parse_args()

    models = args.models or (FULL_MODELS if args.full else DEFAULT_MODELS)
    # n_entries = TOTAL synthetic entries across the seven categories, per the
    # proposal (1,000 entries ~= 140/category => 10,000 queries per model).
    n_entries = args.n_entries or (1000 if args.full else 350)

    out = run(
        models=models,
        backend="hf_local",
        n_entries=n_entries,
        out_dir=args.out_dir,
        data_path="data/cn_piibench_lite.json",
        backend_kwargs={"load_in_4bit": not args.no_4bit, "max_tokens": args.max_tokens},
        batch_size=args.batch_size,
    )
    print("\n=== Model Risk Summary ===")
    print(out["metrics"]["summary"].to_string(index=False))
    print("\nArtifacts:")
    for k, v in out["paths"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
