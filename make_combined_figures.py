#!/usr/bin/env python
"""Combined Chapter-4 figures across all measured conditions.

Reads the results.json bundles produced by the base pilot, the three
fine-tuning stress tests (1.5B / 3B / Instruct) and the commercial API demo,
and emits the comparison figures used in Chapter 4, plus a consolidated
data table (combined_summary.json) for the write-up.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "results_final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
import pub_style  # publication rcParams

plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})

BAND = {"Low": "#1a9e5b", "Medium": "#d99a00", "High": "#e5484d"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def per_cat(bundle, which="fine"):
    """category -> row dict for the model whose label contains `which`."""
    out = {}
    for x in bundle["per_category"]:
        if which in x["model"]:
            out[x["category"]] = x
    return out


CATS = ["national_id", "social_credit_code", "unionpay_card", "mobile_phone",
        "medical_record_id", "soe_employee_id", "full_name"]
LABELS = {"national_id": "National ID", "social_credit_code": "Social Credit",
          "unionpay_card": "UnionPay", "mobile_phone": "Mobile",
          "medical_record_id": "Medical", "soe_employee_id": "SOE ID",
          "full_name": "Name"}


def main():
    base = load(ROOT / "results_base_pilot" / "results.json")
    ft15 = load(ROOT / "results_finetune_1.5b" / "results.json")
    ft3 = load(ROOT / "results_finetune_3b" / "results.json")
    fti = load(ROOT / "results_finetune_instruct" / "results.json")
    demo = load(ROOT / "results_pilot" / "results.json")

    def agg(bundle, which):
        for s in bundle["summary"]:
            if which in s["model"]:
                return s
        return None

    # ---------- Fig 1: aggregate RW-MER by condition (headline) ---------- #
    conditions = [
        ("Qwen2.5-1.5B\n(base)", 0.0, "Low"),
        ("Qwen2.5-3B\n(base)", 0.0, "Low"),
        ("qwen-plus\n(commercial)", 0.0, "Low"),
        ("Qwen2.5-1.5B\n(fine-tuned)", agg(ft15, "fine")["aggregate_rwmer"], "High"),
        ("Qwen2.5-3B\n(fine-tuned)", agg(ft3, "fine")["aggregate_rwmer"], "High"),
        ("Qwen2.5-1.5B-Instruct\n(fine-tuned)", agg(fti, "fine")["aggregate_rwmer"], "High"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(conditions))
    vals = [c[1] for c in conditions]
    ax.bar(xs, vals, color=[BAND[c[2]] for c in conditions], width=0.62)
    ax.axhline(0.05, ls="--", color="#888", lw=1); ax.axhline(0.20, ls="--", color="#888", lw=1)
    ax.text(len(xs) - 0.4, 0.055, "Medium", color="#888", fontsize=8)
    ax.text(len(xs) - 0.4, 0.21, "High", color="#888", fontsize=8)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in conditions], fontsize=8)
    ax.set_ylabel("Aggregate RW-MER"); ax.set_ylim(0, 0.62)
    ax.set_title("PII leakage by condition: base ≈ 0 (Low) vs fine-tuned (High)")
    fig.tight_layout(); _save(fig, "fig4_leakage_by_condition")

    # ---------- Fig 2: MER heatmap for the three fine-tuned models ------- #
    conds = [("1.5B ft", per_cat(ft15, "fine")), ("3B ft", per_cat(ft3, "fine")),
             ("1.5B-Instruct ft", per_cat(fti, "fine"))]
    M = np.array([[conds[j][1][c]["mer_worst"] for c in CATS] for j in range(3)])
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels([LABELS[c] for c in CATS], rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels([c[0] for c in conds])
    for i in range(3):
        for j in range(len(CATS)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.6 else "black", fontsize=8)
    ax.set_title("Worst-case MER by category (fine-tuned models)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="MER")
    fig.tight_layout(); _save(fig, "fig4_mer_heatmap_finetuned")

    # ---------- Fig 3: CLMD by category across the three ft models ------- #
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(CATS)); w = 0.26
    series = [("Qwen2.5-1.5B (base ft)", per_cat(ft15, "fine"), "#6aa8ff"),
              ("Qwen2.5-3B (base ft)", per_cat(ft3, "fine"), "#2c6fbb"),
              ("Qwen2.5-1.5B-Instruct ft", per_cat(fti, "fine"), "#e5484d")]
    for k, (name, pc, col) in enumerate(series):
        ax.bar(x + (k - 1) * w, [pc[c]["clmd"] or 0 for c in CATS], w, label=name, color=col)
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CATS], rotation=30, ha="right")
    ax.set_ylabel("CLMD = MER(EN→ZH) − MER(ZH→ZH)")
    ax.set_title("Cross-lingual gap emerges with scale and alignment")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); _save(fig, "fig4_clmd_mechanism")

    # ---------- Fig 4: ZH vs EN for the Instruct model (refusal bypass) -- #
    pc = per_cat(fti, "fine")
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(CATS)); w = 0.38
    ax.bar(x - w / 2, [pc[c]["mer_zh2zh"] for c in CATS], w, label="ZH→ZH (Chinese)", color="#2c6fbb")
    ax.bar(x + w / 2, [pc[c]["mer_en2zh"] for c in CATS], w, label="EN→ZH (English)", color="#e5484d")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CATS], rotation=30, ha="right")
    ax.set_ylabel("MER"); ax.set_ylim(0, 1)
    ax.set_title("Qwen2.5-1.5B-Instruct (fine-tuned): English bypasses Chinese refusal")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); _save(fig, "fig4_instruct_zh_vs_en")

    # ---------- consolidated table ---------- #
    summary = {
        "conditions": {
            "base_1.5B": 0.0, "base_3B": 0.0,
            "commercial_qwen_plus": agg(demo, "qwen-plus")["aggregate_rwmer"],
            "commercial_qwen_turbo": agg(demo, "qwen-turbo")["aggregate_rwmer"],
            "ft_1.5B": agg(ft15, "fine")["aggregate_rwmer"],
            "ft_3B": agg(ft3, "fine")["aggregate_rwmer"],
            "ft_instruct": agg(fti, "fine")["aggregate_rwmer"],
        },
        "clmd": {
            "ft_1.5B": agg(ft15, "fine")["mean_clmd"],
            "ft_3B": agg(ft3, "fine")["mean_clmd"],
            "ft_instruct": agg(fti, "fine")["mean_clmd"],
        },
        "per_category": {
            "ft_1.5B": {c: per_cat(ft15, "fine")[c] for c in CATS},
            "ft_3B": {c: per_cat(ft3, "fine")[c] for c in CATS},
            "ft_instruct": {c: per_cat(fti, "fine")[c] for c in CATS},
        },
    }
    (ROOT / "results_final" / "combined_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote combined_summary.json and 4 figures to", OUT)


def _save(fig, name):
    pub_style.finish(fig)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("  wrote", name)


if __name__ == "__main__":
    main()
