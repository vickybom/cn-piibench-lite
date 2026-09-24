#!/usr/bin/env python
"""Expanded, publication-grade Chapter-4 figures with 95% Wilson CIs."""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "results_final" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
import pub_style  # publication rcParams; overrides anything set above
BAND = {"Low": "#1a9e5b", "Medium": "#d99a00", "High": "#e5484d"}
CATS = ["national_id","social_credit_code","unionpay_card","mobile_phone",
        "medical_record_id","soe_employee_id","full_name"]
LAB = {"national_id":"National ID","social_credit_code":"Social Credit","unionpay_card":"UnionPay",
       "mobile_phone":"Mobile","medical_record_id":"Medical","soe_employee_id":"SOE ID","full_name":"Name"}
PRI = {"national_id":0.943,"social_credit_code":0.793,"unionpay_card":0.771,"mobile_phone":0.729,
       "medical_record_id":0.707,"soe_employee_id":0.593,"full_name":0.521}

def load(p): return json.loads((ROOT/p/"results.json").read_text(encoding="utf-8"))
def pc(b, which): return {x["category"]: x for x in b["per_category"] if which in x["model"]}
def nmap(folder, which):
    import pandas as pd
    df = pd.read_csv(ROOT/folder/"mer_long.csv")
    df = df[df["model"].str.contains(which, regex=False)]
    return {(r["category"], r["condition"]): int(r["n"]) for _, r in df.iterrows()}

def wilson(p, n, z=1.96):
    if n == 0: return (0, 0)
    k = p*n; ph = k/n; d = 1+z*z/n
    c = (ph + z*z/(2*n))/d
    h = (z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n)))/d
    return max(0, c-h), min(1, c+h)

def save(fig, name):
    pub_style.finish(fig)
    fig.savefig(OUT/f"{name}.png"); fig.savefig(OUT/f"{name}.pdf")
    plt.close(fig); print("  wrote", name)

def main():
    ft15, ft3, fti = load("results_finetune_1.5b"), load("results_finetune_3b"), load("results_finetune_instruct")
    p15, p3, pi = pc(ft15,"fine"), pc(ft3,"fine"), pc(fti,"fine")
    n15, n3, ni = nmap("results_finetune_1.5b","fine)"), nmap("results_finetune_3b","fine)"), nmap("results_finetune_instruct","fine)")

    # ---- Fig A: per-category ZH MER with 95% CI (3 fine-tuned models) ---- #
    fig, ax = plt.subplots(figsize=(9, 4.4)); x = np.arange(len(CATS)); w = 0.26
    for k,(name,p,nm,col) in enumerate([("1.5B (base ft)",p15,n15,"#6aa8ff"),
                                        ("3B (base ft)",p3,n3,"#2c6fbb"),
                                        ("1.5B-Instruct ft",pi,ni,"#e5484d")]):
        ys=[p[c]["mer_zh2zh"] for c in CATS]
        errs=[[y-wilson(y,nm.get((c,"zh2zh"),320))[0] for y,c in zip(ys,CATS)],
              [wilson(y,nm.get((c,"zh2zh"),320))[1]-y for y,c in zip(ys,CATS)]]
        ax.bar(x+(k-1)*w, ys, w, yerr=errs, capsize=2, label=name, color=col,
               error_kw={"elinewidth":0.8,"alpha":0.7})
    ax.set_xticks(x); ax.set_xticklabels([LAB[c] for c in CATS], rotation=30, ha="right")
    ax.set_ylabel("MER (ZH→ZH)  ±95% CI"); ax.set_ylim(0,1); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Monolingual memorization by category (fine-tuned models)")
    fig.tight_layout(); save(fig, "fig_mer_ci")

    # ---- Fig B: cross-lingual (EN) extraction grows with scale ---- #
    fig, ax = plt.subplots(figsize=(9, 4.2)); w=0.38
    ax.bar(x-w/2, [p15[c]["mer_en2zh"] for c in CATS], w, label="Qwen2.5-1.5B (ft)", color="#9ec5ff")
    ax.bar(x+w/2, [p3[c]["mer_en2zh"] for c in CATS], w, label="Qwen2.5-3B (ft)", color="#2c6fbb")
    ax.set_xticks(x); ax.set_xticklabels([LAB[c] for c in CATS], rotation=30, ha="right")
    ax.set_ylabel("MER (EN→ZH)"); ax.set_ylim(0,1); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Cross-lingual extraction grows with model scale")
    fig.tight_layout(); save(fig, "fig_scaling_en")

    # ---- Fig C: risk landscape — MER vs PRI, bubble = RW-MER (Instruct ft) ---- #
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for c in CATS:
        r = pi[c]; mer = r["mer_worst"]; pri = PRI[c]; rw = r["rwmer"]
        col = BAND["High"] if rw>=0.2 else (BAND["Medium"] if rw>=0.05 else BAND["Low"])
        ax.scatter(pri, mer, s=1200*rw+40, color=col, alpha=0.65, edgecolor="k", linewidth=0.5)
        ax.annotate(LAB[c], (pri, mer), fontsize=8, xytext=(4,4), textcoords="offset points")
    ax.set_xlabel("PII Risk Index (PRI)"); ax.set_ylabel("Worst-case MER")
    ax.set_title("Risk landscape (Instruct ft): bubble size proportional to RW-MER")
    ax.grid(alpha=0.2); fig.tight_layout(); save(fig, "fig_risk_landscape")

    # ---- Fig D: aggregate scaling + alignment (RW-MER by condition, grouped) ---- #
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    conds=["1.5B ft","3B ft","1.5B-Instruct ft"]
    vals=[load(f)["summary"] for f in ["results_finetune_1.5b","results_finetune_3b","results_finetune_instruct"]]
    agg=[next(s for s in v if "fine" in s["model"])["aggregate_rwmer"] for v in vals]
    clm=[next(s for s in v if "fine" in s["model"])["mean_clmd"] for v in vals]
    xx=np.arange(3)
    ax.bar(xx-0.2, agg, 0.4, label="Aggregate RW-MER", color="#e5484d")
    ax.bar(xx+0.2, clm, 0.4, label="Mean CLMD", color="#2c6fbb")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(xx); ax.set_xticklabels(conds); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Severity rises with scale; CLMD sign flips with alignment")
    fig.tight_layout(); save(fig, "fig_scale_align")
    print("done")

if __name__ == "__main__":
    main()
