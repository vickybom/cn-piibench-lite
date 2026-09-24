#!/usr/bin/env python
"""PII-Auditor-CN-Lite command-line interface.

Examples
--------
  # Full simulated demonstration run across three model scales:
  python cli.py audit --backend demo \
      --models qwen2.5-1.5b qwen2.5-3b qwen2.5-7b --n-entries 200

  # Real API pilot on DashScope (needs $DASHSCOPE_API_KEY):
  python cli.py audit --backend dashscope \
      --models qwen2.5-1.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct \
      --n-entries 30 --max-queries-per-model 300

  # Just (re)generate the synthetic dataset:
  python cli.py generate --n-entries 1000 --seed 20260524
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pii_auditor.localenv import load_local_env
from pii_auditor.m1_generator import build_dataset, save_dataset
from pii_auditor.pipeline import run

load_local_env()   # pick up .env (e.g. DASHSCOPE_API_KEY) if present


def cmd_generate(args):
    ds = build_dataset(args.n_entries, args.seed)
    path = save_dataset(ds, args.data_path)
    print(f"Generated {ds['meta']['n_persons']} synthetic persons -> {path}")


def cmd_audit(args):
    bk = {}
    if args.base_url:
        bk["base_url"] = args.base_url
    if args.max_tokens:
        bk["max_tokens"] = args.max_tokens
    out = run(
        models=args.models,
        backend=args.backend,
        n_entries=args.n_entries,
        seed=args.seed,
        out_dir=args.out_dir,
        data_path=args.data_path,
        backend_kwargs=bk,
        max_queries_per_model=args.max_queries_per_model,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        categories=args.categories,
    )
    print("\n=== Model Risk Summary ===")
    print(out["metrics"]["summary"].to_string(index=False))
    print("\nArtifacts:")
    for k, v in out["paths"].items():
        print(f"  {k}: {v}")
    if out["meta"]["is_simulated"]:
        print("\n** NOTE: simulation backend — results are DEMONSTRATION only, "
              "not real model measurements. **")


def build_parser():
    p = argparse.ArgumentParser(prog="pii-auditor-cn-lite",
                                description="External black-box Chinese PII memorization auditor.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate the synthetic PII dataset (M1)")
    g.add_argument("--n-entries", type=int, default=1000)
    g.add_argument("--seed", type=int, default=20260524)
    g.add_argument("--data-path", default="data/cn_piibench_lite.json")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("audit", help="run the full M1-M6 audit pipeline")
    a.add_argument("--backend", default="demo",
                   choices=["demo", "dashscope", "openai_compat", "hf_api", "hf_local"])
    a.add_argument("--models", nargs="+",
                   default=["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"])
    a.add_argument("--n-entries", type=int, default=200)
    a.add_argument("--seed", type=int, default=20260524)
    a.add_argument("--out-dir", default="results")
    a.add_argument("--data-path", default="data/cn_piibench_lite.json")
    a.add_argument("--base-url", default=None, help="override OpenAI-compatible base URL")
    a.add_argument("--max-tokens", type=int, default=None)
    a.add_argument("--max-queries-per-model", type=int, default=None,
                   help="cap queries per model (useful for API pilots)")
    a.add_argument("--concurrency", type=int, default=1,
                   help="parallel API requests (ignored for demo backend)")
    a.add_argument("--batch-size", type=int, default=1,
                   help="batched generation for local GPU backends")
    a.add_argument("--categories", nargs="+", default=None)
    a.set_defaults(func=cmd_audit)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
