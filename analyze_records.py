#!/usr/bin/env python
"""Offline analyses over the per-query records.csv produced by a GPU run.

Produces the Chapter 4 ablations that need no further inference:
  * per-template / per-attack-type effectiveness (which attack works best)
  * per-condition breakdown by category
  * defense comparison table (privacy-utility trade-off)
  * figures for each

    python analyze_records.py results_defense/records.csv --out results_defense/analysis
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})

TYPE_NAME = {"A": "Type A\nprefix", "B": "Type B\nassociation", "C": "Type C\ncross-lingual"}
LAB = {"national_id":"National ID","social_credit_code":"Social Credit","unionpay_card":"UnionPay",
       "mobile_phone":"Mobile","medical_record_id":"Medical","soe_employee_id":"SOE ID",
       "full_name":"Name"}


def load(path):
    df = pd.read_csv(path)
    if df["hit"].dtype == object:
        df["hit"] = df["hit"].astype(str).str.lower().isin(["true", "1"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records_csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--undefended-contains", default="undefended",
                    help="substring identifying the undefended fine-tuned model")
    args = ap.parse_args()
    out = Path(args.out or Path(args.records_csv).parent / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.records_csv)
    print(f"loaded {len(df)} records | models: {df['model'].nunique()}")

    # ---- 1. per-template effectiveness (undefended model) ------------------
    tgt = df[df["model"].str.contains(args.undefended_contains, case=False, na=False)]
    if tgt.empty:
        tgt = df[df["model"] == df["model"].iloc[0]]
    tpl = (tgt.groupby(["type", "template_id"])["hit"].agg(["mean", "count"])
              .reset_index().rename(columns={"mean": "mer", "count": "n"}))
    tpl.to_csv(out / "per_template.csv", index=False, encoding="utf-8-sig")
    print("\n=== Per-template effectiveness ===")
    print(tpl.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {"A": "#2c6fbb", "B": "#6aa8ff", "C": "#e5484d"}
    ax.bar(tpl["template_id"], tpl["mer"], color=[colors.get(t, "#888") for t in tpl["type"]])
    ax.set_ylabel("MER"); ax.set_xlabel("Template")
    ax.set_title("Extraction effectiveness by adversarial template")
    handles = [plt.Rectangle((0,0),1,1,color=c) for c in colors.values()]
    ax.legend(handles, ["Type A (prefix)","Type B (association)","Type C (cross-lingual)"],
              frameon=False, fontsize=8)
    fig.tight_layout(); _save(fig, out, "fig_template_ablation")

    # ---- 2. per attack type x category ------------------------------------
    piv = tgt.pivot_table(index="category", columns="type", values="hit", aggfunc="mean")
    piv = piv.reindex([c for c in LAB if c in piv.index])
    piv.to_csv(out / "type_by_category.csv", encoding="utf-8-sig")
    print("\n=== Attack type x category (MER) ===")
    print(piv.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = range(len(piv)); w = 0.26
    for k, t in enumerate([c for c in ["A","B","C"] if c in piv.columns]):
        ax.bar([i + (k-1)*w for i in x], piv[t].values, w,
               label=TYPE_NAME[t].replace("\n"," "), color=colors[t])
    ax.set_xticks(list(x)); ax.set_xticklabels([LAB.get(i,i) for i in piv.index],
                                               rotation=30, ha="right")
    ax.set_ylabel("MER"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Attack-type effectiveness by PII category")
    fig.tight_layout(); _save(fig, out, "fig_type_by_category")

    # ---- 3. defense comparison --------------------------------------------
    dfc = df.groupby("model")["hit"].agg(["mean", "count"]).reset_index()
    dfc.columns = ["model", "raw_leak_rate", "n"]
    dfc = dfc.sort_values("raw_leak_rate", ascending=False)
    dfc.to_csv(out / "defense_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n=== Defense comparison (raw leak rate) ===")
    print(dfc.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.0, 0.55*len(dfc)+1.8))
    ax.barh(dfc["model"], dfc["raw_leak_rate"], color="#e5484d")
    for i, v in enumerate(dfc["raw_leak_rate"]):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("Raw verbatim leak rate")
    ax.set_title("Leakage by defense condition")
    fig.tight_layout(); _save(fig, out, "fig_defense_comparison")

    print("\nWrote analysis to", out)


def _save(fig, out, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("  wrote", name)


if __name__ == "__main__":
    main()
