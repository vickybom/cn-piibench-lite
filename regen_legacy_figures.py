#!/usr/bin/env python
"""Redraw the ten figures whose generating code is no longer on disk.

Each is rebuilt from the stored result JSON rather than from a remembered
number, and written under its original filename so the document can be updated
by substitution. Styling comes from pub_style, so all of them land at the
text-block width and above 600 dpi.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pub_style as PS

ROOT = Path(__file__).parent
F = ROOT / "results_final"
OUT = F / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CATS = ["national_id", "social_credit_code", "unionpay_card", "mobile_phone",
        "medical_record_id", "soe_employee_id", "full_name"]
LAB = {"national_id": "National ID", "social_credit_code": "Social credit",
       "unionpay_card": "UnionPay", "mobile_phone": "Mobile",
       "medical_record_id": "Medical rec.", "soe_employee_id": "SOE ID",
       "full_name": "Name"}


def save(fig, name):
    PS.finish(fig)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}")


def jl(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 5.3
def scaling_corrected():
    """Table 5.6 says the capacity difference is read on RW-MER, not on the raw
    rate; the two carry opposite signs on the full matrix, so the scope AND the
    metric both have to be stated. Section 5.6.4 then showed the run-variance
    band the chapter used to be six times too narrow for these models, which is
    the second band drawn here."""
    sm = jl("results_final/corrected_140.json")["scope_matrix"]
    OLD_BAND, MEASURED = 0.0276, 0.1585
    scopes = [("AB", "Type A+B\n(8 templates)"),
              ("t10", "Ten templates\n(A+B, C1, C2)"),
              ("t12", "Full twelve-template\nmatrix")]
    diffs = [sm["3b"][s + "_rwmer"] - sm["15b"][s + "_rwmer"] for s, _ in scopes]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(scopes))
    ax.axvspan(-MEASURED, MEASURED, color=PS.RED, alpha=0.10, lw=0, zorder=0)
    ax.axvspan(-OLD_BAND, OLD_BAND, color=PS.YELLOW, alpha=0.35, lw=0, zorder=1)
    ax.barh(y, diffs, 0.42, color=[PS.BLUE if d < 0 else PS.GREY for d in diffs],
            edgecolor="black", linewidth=0.4, zorder=3)
    for i, d in enumerate(diffs):
        ax.annotate(f"{d:+.4f}", (d, i),
                    xytext=(7 if d > 0 else -7, 0), textcoords="offset points",
                    va="center", ha="left" if d > 0 else "right", fontsize=7.5)
    ax.axvline(0, color="black", lw=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in scopes])
    ax.set_xlabel("Aggregate RW-MER, 3B $-$ 1.5B   (negative = 3B retains less)")
    ax.set_xlim(-0.20, 0.20)
    ax.set_ylim(-0.6, len(scopes) - 0.25)
    ax.grid(axis="y", visible=False)
    ax.text(OLD_BAND, len(scopes) - 0.42, "  run-variance band assumed\n"
            "  in Ch. 5 ($\\pm$0.028)", fontsize=7, color="0.3", va="center")
    ax.text(-MEASURED, -0.42, "measured spread for the 1.5B model\n"
            "across two training runs ($\\pm$0.159, §5.6.4)",
            fontsize=7, color=PS.RED, va="center", ha="left")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_scaling_corrected")


# ------------------------------------------------------------------ 5.4
def ablation_epochs():
    d = sorted(jl("results_final/ablation_epochs.json"), key=lambda r: r["exposures"])
    x = [r["exposures"] for r in d]
    y = [r["mem_rate"] for r in d]
    n = d[0]["n"]
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    ax.plot(x, y, color=PS.BLUE, marker="o", zorder=3)
    ax.axvspan(6, 10, color=PS.YELLOW, alpha=0.25, lw=0, zorder=0)
    ax.axhline(0.5, color=PS.GREY, ls=":", lw=1.0)
    ax.annotate("threshold near eight\nexposures per record", (8, 0.52),
                xytext=(11, 0.40), fontsize=7.5, color="0.25", linespacing=1.2,
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("Training exposures per record (log scale)")
    ax.set_ylabel("Verbatim memorization rate")
    ax.set_ylim(-0.05, 1.12)
    ax.set_title(f"n = {n} probes per point", fontsize=8.5)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_ablation_epochs")


# ------------------------------------------------------------------ 5.5
def ablation_rank():
    g = jl("results_final/rank_grid_40.json")
    ranks = sorted({r["rank"] for r in g})
    exps = sorted({r["epochs"] * 2 for r in g})
    grid = {(r["rank"], r["epochs"] * 2): r["mem_rate"] for r in g}
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    marks = ["o", "s", "^", "D", "v"]
    cols = [PS.GREY, PS.CYAN, PS.BLUE, PS.PURPLE, PS.RED]
    for e, ls, mk, c in zip(exps, styles, marks, cols):
        y = [grid.get((r, e), np.nan) for r in ranks]
        ax.plot(ranks, y, ls=ls, marker=mk, color=c, label=f"{e} exposures",
                zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("LoRA adapter rank (log scale)")
    ax.set_ylabel("Verbatim memorization rate")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.1)
    ax.annotate("the informative row:\n0.030 to 0.770 across rank",
                (64, 0.77), xytext=(9, 0.60), fontsize=7.5, color="0.25",
                linespacing=1.2,
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    fig.tight_layout(pad=0.4)
    save(fig, "fig_ablation_rank")


# ------------------------------------------------------------------ 5.8
def template_ablation():
    # exact rates from the records, not the bundle's four-decimal copies: the
    # bundle holds B3 as 0.8235 and a second rounding printed it as 0.824,
    # while 807 / 980 is 0.823 -- the label and Table 5.11 must agree with
    # the records at the precision they print
    pt = {}
    with open(ROOT / "results_main_15b_140" / "records_finetuned.csv",
              encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            k = pt.setdefault(r["template_id"], [0, 0])
            k[0] += r["hit"].strip().lower() == "true"
            k[1] += 1
    pt = {t: n / d for t, (n, d) in pt.items()}
    items = sorted(pt.items(), key=lambda kv: -kv[1])
    labs = [k for k, _ in items]
    vals = [v for _, v in items]
    anchored = {"A1", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "D1", "D2"}
    cols = [PS.BLUE if k in anchored else PS.RED for k in labs]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    b = ax.bar(range(len(labs)), vals, 0.68, color=cols,
               edgecolor="black", linewidth=0.4, zorder=3)
    for r, v in zip(b, vals):
        ax.annotate(f"{v:.3f}", (r.get_x() + r.get_width() / 2, v),
                    xytext=(0, 2), textcoords="offset points", ha="center",
                    fontsize=7)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_xlabel("Prompt template")
    ax.set_ylabel("Extraction rate")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PS.BLUE, edgecolor="black", lw=0.4,
                             label="names a specific person"),
                       Patch(facecolor=PS.RED, edgecolor="black", lw=0.4,
                             label="does not name anyone")],
              loc="upper right")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_template_ablation")


# ------------------------------------------------------------------ 5.9
def scale_sensitivity():
    big = jl("results_final/corrected_140.json")["15b"]["per_category"]
    small = jl("results_finetune_1.5b/results.json")
    pc40 = {}
    for row in small.get("per_category", []):
        if "fine-tuned" in str(row.get("model", "")).lower():
            pc40[row["category"]] = row.get("mer_zh2zh", row.get("mer_zh"))
    cats = [c for c in CATS if c in big and c in pc40]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.arange(len(cats))
    w = 0.36
    v40 = [pc40[c] for c in cats]
    v140 = [big[c]["mer_zh"] for c in cats]
    for j, (vals, lab, c, h) in enumerate([(v40, "40 records", PS.GREY, ""),
                                           (v140, "140 records", PS.BLUE, "//")]):
        ax.bar(x + (j - 0.5) * w, vals, w, label=lab, color=c,
               edgecolor="black", linewidth=0.4, hatch=h, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LAB[c] for c in cats], rotation=20, ha="right")
    # both sources are the A+B rate: corrected_140.json's mer_zh is n = 1,120
    # (140 x 8) and the 40-record bundle predates Type D (40 x 8 = 320)
    ax.set_ylabel("Direct monolingual baseline MER\n(Types A and B)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_scale_sensitivity")


# ------------------------------------------------------------------ 5.10
def clmd_confound():
    """Values are taken from Table 5.11 rather than recomputed. An earlier draft
    of this figure derived the unpaired quantity from per-template rates and got
    a different answer, because the table's definition is not the naive
    difference of template-family means. A figure must agree with the table it
    sits beside, so the table is the source."""
    rows = [("Qwen2.5-1.5B\n(base weights)", -0.3713, -0.0444),
            ("Qwen2.5-3B\n(base weights)",   -0.0929, +0.0520),
            ("Qwen2.5-1.5B-Instruct\n(aligned)", +0.2161, -0.0480)]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    y = np.arange(len(rows))
    h = 0.34
    for j, (idx, lab, c, hh) in enumerate(
            [(1, "Unpaired (template families differenced)", PS.RED, ""),
             (2, "Matched-pair (language isolated)", PS.BLUE, "//")]):
        vals = [r[idx] for r in rows]
        b = ax.barh(y + (j - 0.5) * h, vals, h, label=lab, color=c,
                    edgecolor="black", linewidth=0.4, hatch=hh, zorder=3)
        for r, v in zip(b, vals):
            ax.annotate(f"{v:+.4f}", (v, r.get_y() + r.get_height() / 2),
                        xytext=(6 if v > 0 else -6, 0),
                        textcoords="offset points", va="center",
                        ha="left" if v > 0 else "right", fontsize=7.2)
    ax.axvline(0, color="black", lw=0.9, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Cross-lingual differential (English $-$ Chinese)")
    ax.set_xlim(-0.52, 0.40)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.grid(axis="y", visible=False)
    ax.annotate("opposite signs on\nthe aligned model", (0.108, 2),
                xytext=(0.13, 1.35), fontsize=7.2, color="0.25",
                linespacing=1.2, ha="center",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    fig.tight_layout(pad=0.4)
    save(fig, "fig_clmd_confound")


# ------------------------------------------------------------------ 5.11
def paired_instruct():
    """The caption claims comparability 'in every category', so the figure has
    to be per category, not per template pair. ins.paired carries both matched
    pairs broken down by category; the two pairs are pooled here."""
    paired = jl("results_final/corrected_140.json")["ins"]["paired"]
    agg = {}
    for r in paired:
        e = agg.setdefault(r["category"], {"en": [], "zh": []})
        e["en"].append(r["mer_en"]); e["zh"].append(r["mer_zh"])
    cats = [c for c in CATS if c in agg]
    en = [float(np.mean(agg[c]["en"])) for c in cats]
    zh = [float(np.mean(agg[c]["zh"])) for c in cats]

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.arange(len(cats))
    w = 0.36
    for j, (vals, lab, c, h) in enumerate(
            [(en, "English chat-template (C1, C2)", PS.RED, ""),
             (zh, "Chinese chat-template (D1, D2)", PS.BLUE, "//")]):
        ax.bar(x + (j - 0.5) * w, vals, w, label=lab, color=c,
               edgecolor="black", linewidth=0.4, hatch=h, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LAB[c] for c in cats], rotation=20, ha="right")
    ax.set_ylabel("Extraction rate")
    ax.set_ylim(0, 1.24)
    ax.legend(loc="upper center", ncol=2, columnspacing=1.2)
    ax.grid(axis="x", visible=False)
    gaps = [a - b for a, b in zip(en, zh)]
    n_en_higher = sum(1 for g in gaps if g > 0.005)
    ax.text(0.02, 0.87, f"English exceeds Chinese in {n_en_higher} of "
            f"{len(gaps)} categories; gaps run "
            f"{min(gaps):+.3f} to {max(gaps):+.3f}",
            transform=ax.transAxes, fontsize=7.2, color="0.3")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_paired_instruct")


# ------------------------------------------------------------------ 6.1
def defense_privacy():
    d = jl("results_final/defense_140.json")
    arms = [("undefended", "Undefended"), ("filter", "Output filter"),
            ("unlearning", "Unlearning"), ("dp_lora", "DP-LoRA")]
    vals = [d[k]["rwmer"] for k, _ in arms]
    bands = [d[k]["band"] for k, _ in arms]
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    b = ax.bar(range(len(arms)), vals, 0.55,
               color=[PS.BAND[x] for x in bands],
               edgecolor="black", linewidth=0.4, zorder=3)
    for r, v, bd in zip(b, vals, bands):
        ax.annotate(f"{v:.4f}\n{bd}", (r.get_x() + r.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=7.5, linespacing=1.2)
    ax.axhline(0.05, color=PS.GREY, ls=":", lw=1.0)
    ax.axhline(0.20, color=PS.GREY, ls="--", lw=1.0)
    ax.text(3.42, 0.052, "Low / Medium", fontsize=7, color="0.35", ha="right")
    ax.text(3.42, 0.207, "Medium / High", fontsize=7, color="0.35", ha="right")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([l for _, l in arms])
    ax.set_ylabel("Aggregate RW-MER")
    ax.set_ylim(0, 0.42)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_defense_privacy")


# ------------------------------------------------------------------ 6.2
def defense_tradeoff():
    d = jl("results_final/defense_140.json")
    pts = [("Base (no fine-tune)", 0.0, d["base_reference_ppl"], PS.GREY, "o"),
           ("Undefended", d["undefended"]["rwmer"], d["undefended"]["ppl"], PS.RED, "s"),
           ("Output filter", d["filter"]["rwmer"], d["filter"]["ppl"], PS.GREEN, "^"),
           ("Unlearning", d["unlearning"]["rwmer"], d["unlearning"]["ppl"], PS.PURPLE, "v"),
           ("DP-LoRA", d["dp_lora"]["rwmer"], d["dp_lora"]["ppl"], PS.BLUE, "D")]
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    for lab, x, y, c, mk in pts:
        ax.scatter(x, y, s=46, color=c, marker=mk, edgecolor="black",
                   linewidth=0.5, zorder=3)
    ax.set_yscale("log")
    ax.axvspan(0, 0.05, color=PS.GREEN, alpha=0.10, lw=0, zorder=0)
    ax.annotate("unlearning leaves the model\nunusable: perplexity 3.6 x 10$^4$",
                (d["unlearning"]["rwmer"], d["unlearning"]["ppl"]),
                xytext=(0.19, 4.2e3), fontsize=7.4, color="0.25", linespacing=1.2,
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    ax.set_xlabel("Aggregate RW-MER (lower is safer)")
    ax.set_ylabel("Perplexity, held-out text (log)")
    ax.set_xlim(-0.03, 0.45)
    ax.set_ylim(9, 4e5)
    OFF = {"Base (no fine-tune)": (10, 3), "Undefended": (-9, -12),
           "Output filter": (9, 4), "Unlearning": (10, -2), "DP-LoRA": (-9, 6)}
    for lab, x, y, c, mk in pts:
        dx, dy = OFF[lab]
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.2, color=c,
                    ha="left" if dx > 0 else "right")
    ax.text(0.025, 2.2e5, "Low band", fontsize=7, color=PS.GREEN, ha="center")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_defense_tradeoff")


# ------------------------------------------------------------------ 6.3
def unlearn_frontier():
    d = sorted(jl("results_final/unlearn_sweep.json"), key=lambda r: r["steps"])
    x = [r["steps"] for r in d]
    leak = [r["leak_rate"] for r in d]
    ppl = [r["perplexity"] for r in d]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(x, leak, color=PS.RED, marker="o", label="Leakage rate", zorder=3)
    ax.set_xlabel("Gradient-ascent steps")
    ax.set_ylabel("Leakage rate", color=PS.RED)
    ax.set_ylim(-0.05, 1.1)
    ax.tick_params(axis="y", labelcolor=PS.RED)
    ax2 = ax.twinx()
    ax2.plot(x, ppl, color=PS.BLUE, ls="--", marker="s", label="Perplexity",
             zorder=3)
    ax2.set_yscale("log")
    ax2.set_ylabel("Perplexity (log)", color=PS.BLUE)
    ax2.tick_params(axis="y", labelcolor=PS.BLUE)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.annotate("leakage falls only when the\nmodel is already destroyed",
                (60, 0.05), xytext=(21, 0.45), fontsize=7.4, color="0.25",
                linespacing=1.2,
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_unlearn_frontier")


if __name__ == "__main__":
    print("redrawing the ten figures whose source was lost")
    for fn in (scaling_corrected, ablation_epochs, ablation_rank,
               template_ablation, scale_sensitivity, clmd_confound,
               paired_instruct, defense_privacy, defense_tradeoff,
               unlearn_frontier):
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\noutput in {OUT}")
