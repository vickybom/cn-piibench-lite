#!/usr/bin/env python
"""The aggregation rule is the unweighted mean of the per-category RW-MER.

It applied PRI a second time until 2026-08-18. That change moves every
aggregate in the dissertation, so it needs a test that fails if the old rule
ever comes back — from a revert, a merge, or a copy of the older function in
another script.

    python test_aggregate_rule.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pii_auditor import m5_metrics
from pii_auditor.pri import PRI, CATEGORY_ORDER

S = sum(PRI[c] for c in CATEGORY_ORDER)
N = len(CATEGORY_ORDER)


def synthetic(rates):
    """One record per (category, condition) cell at the requested rate."""
    recs = []
    for c, r in rates.items():
        for cond in ("zh2zh", "en2zh"):
            n = 100
            k = round(r * n) if cond == "zh2zh" else 0
            for i in range(n):
                recs.append({"model": "m", "category": c, "condition": cond,
                             "hit": i < k})
    return recs


def main():
    ok = []

    # A category set with deliberately uneven rates, so the two rules cannot
    # coincide by symmetry.
    rates = {c: v for c, v in zip(CATEGORY_ORDER,
                                  [0.9, 0.1, 0.7, 0.3, 0.6, 0.2, 0.5])}
    out = m5_metrics.compute(synthetic(rates))
    got = float(out["summary"].iloc[0]["aggregate_rwmer"])

    want_new = round(sum(PRI[c] * rates[c] for c in CATEGORY_ORDER) / N, 4)
    want_old = round(sum(PRI[c] ** 2 * rates[c] for c in CATEGORY_ORDER) / S, 4)
    ok.append(("aggregate is the unweighted mean of per-category RW-MER",
               abs(got - want_new) < 5e-5, f"{got} against {want_new}"))
    ok.append(("aggregate is NOT the PRI-squared form",
               abs(got - want_old) > 1e-4,
               f"old rule would give {want_old}"))

    # per-category values must be untouched by the change
    pc = out["per_category"].set_index("category")
    bad = [c for c in CATEGORY_ORDER
           if abs(float(pc.loc[c, "rwmer"]) - rates[c] * PRI[c]) > 1e-9]
    ok.append(("per-category RW-MER is still MER x PRI", not bad, str(bad)))

    # a zero condition must stay exactly zero under either rule
    zero = m5_metrics.compute(synthetic({c: 0.0 for c in CATEGORY_ORDER}))
    ok.append(("a null condition aggregates to exactly zero",
               float(zero["summary"].iloc[0]["aggregate_rwmer"]) == 0.0, ""))

    # the second implementation must agree with the engine
    src = (ROOT / "stats_recompute.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "rwmer")
    body = "\n".join(src.splitlines()[fn.lineno - 1:fn.end_lineno])
    ok.append(("stats_recompute does not re-weight by PRI",
               "num += pri * (worst * pri)" not in body,
               "it still squares the Index"))
    ok.append(("stats_recompute divides by the category count",
               "den += 1" in body, "it still sums PRI as the denominator"))

    width = max(len(n) for n, _, _ in ok)
    for name, good, detail in ok:
        print(f"  {'OK ' if good else 'BAD'} {name:<{width}}"
              + (f"   {detail}" if not good else ""))
    bad = [n for n, g, _ in ok if not g]
    print("\n" + ("AGGREGATION RULE IS OPTION A" if not bad
                  else f"{len(bad)} FAILED: {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
