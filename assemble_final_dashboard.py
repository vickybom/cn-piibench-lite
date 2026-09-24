#!/usr/bin/env python
"""Assemble the combined dashboard from the 140-record runs.

WHY THIS WAS REWRITTEN
----------------------
The previous version read the stored ``results.json`` bundles of the 40-record
round. Those bundles were written before the aggregation rule of Section 3.1.6
changed, so their ``aggregate_rwmer`` fields are on the superseded PRI-squared
rule, and they are a different corpus size from the one the dissertation
headlines. The dashboard was therefore wrong twice over: it showed 0.382,
0.452 and 0.549 where the manuscript reports 0.3224, 0.3840 and 0.4533.

Nothing here reads a stored aggregate. The per-query records are re-scored by
``m5_metrics.compute`` -- the delivered engine, the one ``test_aggregate_rule.py``
guards -- so the dashboard cannot drift from the rule again without the engine
and the test disagreeing first.

Two quantities need care:

* The headline CLMD on each verdict card is the MATCHED-PAIR differential,
  MER(C1,C2) - MER(D1,D2), which is what Table 5.14 of the dissertation
  reports and what Section 5.8 calls the corrected CLMD. ``compute`` returns
  the unpaired quantity, which pools every template and confounds probe
  language with template design, so the headline is replaced here and the
  per-category chart is labelled as the unpaired one.
* The commercial row is carried over from the pilot bundle rather than
  re-scored, because that run kept no per-query records. Every one of its
  values is exactly zero, which no aggregation rule changes, and the loader
  asserts it.

    python assemble_final_dashboard.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from pii_auditor.dashboard import render_dashboard
from pii_auditor.m5_metrics import RISK_THRESHOLDS, compute
from pii_auditor.pri import CATEGORY_LABELS, CATEGORY_ORDER, PRI

ROOT = Path(__file__).parent

# (display label, records file, bundle that holds the sample I/O)
SOURCES = [
    ("Qwen2.5-1.5B (base)",
     "results_main_15b_140/records_base.csv", "results_main_15b_140"),
    ("Qwen2.5-3B (base)",
     "results_3b_140/qwen3b_140/records_base.csv", "results_3b_140/qwen3b_140"),
    ("Qwen2.5-1.5B-Instruct (base)",
     "results_instruct_140/instruct_140/records_base.csv",
     "results_instruct_140/instruct_140"),
    ("Qwen2.5-1.5B (fine-tuned)",
     "results_main_15b_140/records_finetuned.csv", "results_main_15b_140"),
    ("Qwen2.5-3B (fine-tuned)",
     "results_3b_140/qwen3b_140/records_finetuned.csv",
     "results_3b_140/qwen3b_140"),
    ("Qwen2.5-1.5B-Instruct (fine-tuned)",
     "results_instruct_140/instruct_140/records_finetuned.csv",
     "results_instruct_140/instruct_140"),
]

EN_PRE = ("C1", "C2")          # English chat-template probes
ZH_PRE = ("D1", "D2")          # their literal Chinese renderings

# What the dissertation prints, so the dashboard cannot quietly disagree.
EXPECTED = {
    "Qwen2.5-1.5B (fine-tuned)": (0.3152, -0.0444),
    "Qwen2.5-3B (fine-tuned)": (0.3774, +0.0520),
    "Qwen2.5-1.5B-Instruct (fine-tuned)": (0.4437, -0.0480),
    "Qwen2.5-1.5B (base)": (0.0, 0.0),
    "Qwen2.5-3B (base)": (0.0, 0.0),
    "Qwen2.5-1.5B-Instruct (base)": (0.0, 0.0),
}


def read_records(rel, label):
    with open(ROOT / rel, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({"model": label, "category": r["category"],
                    "condition": r["condition"],
                    "template_id": r["template_id"],
                    "hit": r["hit"].strip().lower() == "true"})
    return out


def matched_pair_clmd(records):
    """MER over the English chat-template probes minus their Chinese renderings.

    Returned overall and per category. Both halves ask the same people about
    the same fields, so the difference isolates the language of the probe.
    """
    hit = defaultdict(int)
    n = defaultdict(int)
    for r in records:
        t = r["template_id"]
        side = "en" if t in EN_PRE else "zh" if t in ZH_PRE else None
        if side is None:
            continue
        for key in ((side,), (side, r["category"])):
            n[key] += 1
            hit[key] += bool(r["hit"])
    rate = lambda k: hit[k] / n[k] if n[k] else float("nan")
    overall = rate(("en",)) - rate(("zh",))
    per_cat = {c: rate(("en", c)) - rate(("zh", c)) for c in CATEGORY_ORDER
               if n[("en", c)] and n[("zh", c)]}
    return overall, per_cat


def commercial_row():
    """The qwen-plus pilot, which kept no per-query records. Zeros only."""
    b = json.loads((ROOT / "results_pilot" / "results.json")
                   .read_text(encoding="utf-8"))
    s = next(x for x in b["summary"] if x["model"] == "qwen-plus")
    pc = [x for x in b["per_category"] if x["model"] == "qwen-plus"]
    assert s["aggregate_rwmer"] == 0.0 and s["mean_clmd"] == 0.0, s
    assert all((x.get("rwmer") or 0) == 0 for x in pc), "pilot is not all zero"
    label = "qwen-plus (commercial API)"
    s = {"model": label, "aggregate_rwmer": 0.0, "mean_clmd": 0.0,
         "risk_band": "Low"}
    return s, [dict(x, model=label) for x in pc], b


def sample_io():
    """A few real prompt/completion pairs, drawn from the 140-record bundles."""
    cache = {}
    for _, _, bundle in SOURCES:
        if bundle not in cache:
            cache[bundle] = json.loads(
                (ROOT / bundle / "results.json").read_text(encoding="utf-8")
            ).get("sample_io", [])

    def grab(bundle, which, cat, cond, hit, label, n=1):
        out = []
        for s in cache[bundle]:
            if (which in s["model"] and s["category"] == cat
                    and s["condition"] == cond and s["hit"] is hit):
                out.append(dict(s, model=label))
                if len(out) >= n:
                    break
        return out

    m15, m3b = "results_main_15b_140", "results_3b_140/qwen3b_140"
    mi = "results_instruct_140/instruct_140"
    io = []
    io += grab(m15, "base", "national_id", "zh2zh", False,
               "Qwen2.5-1.5B (base)")
    io += grab(m15, "fine", "national_id", "zh2zh", True,
               "Qwen2.5-1.5B (fine-tuned)", n=2)
    io += grab(m15, "fine", "mobile_phone", "zh2zh", True,
               "Qwen2.5-1.5B (fine-tuned)")
    io += grab(m3b, "fine", "medical_record_id", "zh2zh", True,
               "Qwen2.5-3B (fine-tuned)")
    io += grab(mi, "fine", "unionpay_card", "zh2zh", False,
               "Qwen2.5-1.5B-Instruct (fine-tuned)")
    io += grab(mi, "fine", "national_id", "en2zh", True,
               "Qwen2.5-1.5B-Instruct (fine-tuned)")
    return io


def main():
    records, by_label = [], {}
    for label, rel, _ in SOURCES:
        rs = read_records(rel, label)
        by_label[label] = rs
        records += rs
    print(f"  scored {len(records):,} per-query records "
          f"across {len(SOURCES)} conditions")

    out = compute(records)                       # the delivered engine
    summary = out["summary"].to_dict("records")
    per_category = out["per_category"].to_dict("records")

    # headline CLMD -> matched pair, which is what Table 5.14 reports
    for s in summary:
        overall, _ = matched_pair_clmd(by_label[s["model"]])
        s["mean_clmd"] = round(float(overall), 4)

    comm_s, comm_pc, _ = commercial_row()
    summary.append(comm_s)
    per_category += comm_pc

    order = [lbl for lbl, _, _ in SOURCES] + [comm_s["model"]]
    rank = {m: i for i, m in enumerate(order)}
    summary.sort(key=lambda s: rank[s["model"]])

    print("\n  condition                             aggregate  band     "
          "matched-pair CLMD")
    bad = 0
    for s in summary:
        exp = EXPECTED.get(s["model"])
        flag = ""
        if exp:
            da = abs(s["aggregate_rwmer"] - exp[0])
            dc = abs(s["mean_clmd"] - exp[1])
            if da > 5e-5 or dc > 5e-5:
                bad += 1
                flag = f"   <-- manuscript says {exp[0]:.4f} / {exp[1]:+.4f}"
        print(f"  {s['model']:<38}{s['aggregate_rwmer']:9.4f}  "
              f"{s['risk_band']:<7} {s['mean_clmd']:+.4f}{flag}")
    if bad:
        raise SystemExit(f"\n  {bad} conditions disagree with the dissertation")

    results = {
        "meta": {"backend": "combined (local GPU + commercial API)",
                 "is_simulated": False,
                 "models": order, "n_entries": 980, "n_templates": 12,
                 "total_queries": len(records),
                 "seed": 20260524, "timestamp": "final (140 records)"},
        "banner": None, "thresholds": RISK_THRESHOLDS, "pri": PRI,
        "category_order": CATEGORY_ORDER,
        "category_labels": {c: {"en": CATEGORY_LABELS[c][0],
                                "zh": CATEGORY_LABELS[c][1]}
                            for c in CATEGORY_ORDER},
        "summary": summary, "per_category": per_category,
        "sample_io": sample_io(),
    }
    dest = ROOT / "results_final"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "dashboard.html").write_text(
        render_dashboard(results), encoding="utf-8")
    print(f"\n  wrote {dest / 'dashboard.html'}")
    print(f"  {len(results['sample_io'])} sample I/O rows")


if __name__ == "__main__":
    main()
