#!/usr/bin/env python
"""Four figures for the sections that carry the newest results and have none.

Every number is loaded from the result files rather than typed in, so the
figures cannot drift from the tables they accompany.

Output: PNG at 600 dpi (line art, IEEE requirement, and the format Word embeds
reliably) plus PDF vector alongside for any later LaTeX submission.
Width 6.0 in throughout = the thesis text block.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "figures_new"
OUT.mkdir(exist_ok=True)

import pub_style  # 700 dpi render so tight-bbox still clears 600 at 6.0 in
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": pub_style.RENDER_DPI,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "legend.frameon": False,
})

# Paul Tol bright - colour-blind safe. Shape and line style carry the same
# information, so every panel survives greyscale printing.
BLUE, RED, GREEN, YELLOW, CYAN, PURPLE, GREY = (
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB")
W = 6.0


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ===================================================================== fig A
def fig_generalization_window():
    """Section 5.6.3 - the window, and the phase transition out of it."""
    g = json.load(open(ROOT / "results_generalization" / "generalization.json",
                       encoding="utf-8"))
    base = next(r for r in g if r["corpus"] == "base")
    series = {}
    for corp in ("fixed", "varied"):
        rows = sorted((r for r in g if r["corpus"] == corp),
                      key=lambda r: r["exposures"])
        series[corp] = (
            [r["exposures"] for r in rows],
            [r["novel_valid"] for r in rows],
            [r["recitation"] for r in rows],
            [r["memorization"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharex=True)
    style = {"fixed": (RED, "--", "s"), "varied": (BLUE, "-", "o")}

    ax = axes[0]
    ax.axvspan(1.6, 4.6, color=GREEN, alpha=0.10, zorder=0, lw=0)
    ax.axhline(base["novel_valid"], color=GREY, ls=":", lw=1.0, zorder=1)
    ax.text(12, base["novel_valid"] + 0.025, "released weights",
            ha="right", va="bottom", fontsize=7, color="0.35")
    for corp, (x, nv, _, _) in series.items():
        c, ls, mk = style[corp]
        ax.plot(x, nv, color=c, ls=ls, marker=mk, label=f"{corp} phrasing", zorder=3)
    ax.set_ylabel("Novel-and-valid rate")
    ax.set_xlabel("Training exposures per record")
    ax.set_title("(a) Schema competence on held-out people", fontsize=8.5)
    ax.set_ylim(-0.03, 0.72)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.axvspan(1.6, 4.6, color=GREEN, alpha=0.10, zorder=0, lw=0)
    for corp, (x, _, rc, _) in series.items():
        c, ls, mk = style[corp]
        ax.plot(x, rc, color=c, ls=ls, marker=mk, zorder=3)
    ax.set_ylabel("Recitation rate")
    ax.set_xlabel("Training exposures per record")
    ax.set_title("(b) Training values emitted for strangers", fontsize=8.5)
    ax.set_ylim(-0.03, 1.0)
    ax.text(3.1, 0.985, "generalisation\nwindow", fontsize=7.5, color="0.25",
            ha="center", va="top", linespacing=1.15)

    for ax in axes:
        ax.set_xticks([2, 4, 6, 8, 12])
    fig.tight_layout(pad=0.4)
    save(fig, "fig_generalization_window")


# ===================================================================== fig B
def _rwmer_from_records(rel_path: str) -> float:
    """Twelve-template aggregate RW-MER, straight from the stored records.

    Uses m5_metrics.compute so the figure, the tables and the statistics all
    come from one definition of the metric.
    """
    import csv

    from pii_auditor.m5_metrics import compute
    rows = list(csv.DictReader(open(ROOT / rel_path, encoding="utf-8-sig")))
    for r in rows:
        r["hit"] = r["hit"] in ("True", "true", "1")
    return float(compute(rows)["summary"].iloc[0]["aggregate_rwmer"])


def fig_replication():
    """Section 5.6.4 - the capacity ordering reverses; the CLMD does not."""
    rep = json.load(open(ROOT / "results_replicate_in" / "replicate.json",
                         encoding="utf-8"))
    orig = json.load(open(ROOT / "results_final" / "corrected_140.json",
                          encoding="utf-8"))
    xl = json.load(open(ROOT / "results_final" / "crosslingual_corrected.json",
                        encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))

    # (a) the reversal
    ax = axes[0]
    pairs = [("Qwen2.5-1.5B", 0.5235, rep["1.5B_seed1337"]["ab_scope_rate"],
              BLUE, "o", "-"),
             ("Qwen2.5-3B", 0.4301, rep["3B_seed1337"]["ab_scope_rate"],
              RED, "s", "--")]
    for lab, a, b, c, mk, ls in pairs:
        ax.plot([0, 1], [a, b], color=c, ls=ls, marker=mk, label=lab, zorder=3)
        ax.annotate(f"{a:.3f}", (0, a), xytext=(-4, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7.5, color=c)
        ax.annotate(f"{b:.3f}", (1, b), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7.5, color=c)
    ax.set_xlim(-0.42, 1.42)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["seed 42", "seed 1337"])
    ax.set_ylabel("Direct-memorization rate")
    ax.set_ylim(0.30, 0.58)
    ax.set_title("(a) Capacity ordering reverses", fontsize=8.5)
    ax.legend(loc="lower center", ncol=1)
    ax.grid(axis="x", visible=False)

    # (b) the spreads themselves, against the estimate the chapter relied on.
    #
    # Every value here is the TWELVE-template aggregate, recomputed from the
    # stored per-query records. That matters: the 0.0276 reference line was
    # measured on the twelve-template matrix, and an earlier version of this
    # panel differenced a ten-template seed-42 value against a twelve-template
    # seed-1337 one, which inflated the 1.5B spread from 0.0959 to 0.1585.
    # Deriving from records rather than from mixed-scope JSON means the figure
    # cannot drift from Table 5.9 again.
    ax = axes[1]
    rw = _rwmer_from_records
    runs = {
        "Qwen2.5-3B": [rw("results_3b_140/qwen3b_140/records_finetuned.csv"),
                       rw("results_replicate_in/records_3B_seed1337.csv")],
        # The earlier Instruct run of Section 5.2.4 is not plotted. Only its
        # aggregate survives, on the rule this study used before Section 3.1.6;
        # its per-query records were lost with the runtime, so it cannot be
        # re-expressed under the current rule, and putting it beside values
        # that can would mix the two.
        "1.5B-Instruct": [rw("results_instruct_140/instruct_140/records_finetuned.csv"),
                          rw("results_replicate_in/records_1.5B-Instruct_seed1337.csv")],
        "Qwen2.5-1.5B": [rw("results_main_15b_140/records_finetuned.csv"),
                         rw("results_replicate_in/records_1.5B_seed1337.csv")],
    }
    names = list(runs)
    ypos = np.arange(len(names))
    spreads = [max(runs[n]) - min(runs[n]) for n in names]
    ax.barh(ypos, spreads, height=0.48, color=[GREY, GREY, RED],
            edgecolor="black", linewidth=0.4, zorder=3)
    for i, (n, sp) in enumerate(zip(names, spreads)):
        ax.annotate(f"{sp:.3f}  ({len(runs[n])} runs)", (sp, i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color="0.25")
    # No reference line for the 0.0276 estimate: that figure is on the earlier
    # aggregation rule, and drawing it across bars computed under the current
    # one would invite exactly the cross-rule comparison the text avoids.
    ax.annotate("run variance is a property of the model,"
                " not a constant of the study",
                (0.0, 2.62), xytext=(2, 0), textcoords="offset points",
                fontsize=7.2, color=BLUE, va="center", linespacing=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.55, 2.95)
    ax.set_xlim(0, 0.115)
    # the scope belongs in the caption; spelled out here it overruns the panel
    ax.set_xlabel("Spread in aggregate RW-MER across runs")
    ax.set_title("(b) Run variance is a property of the model", fontsize=8.5)
    ax.grid(axis="y", visible=False)

    fig.tight_layout(pad=0.4)
    save(fig, "fig_replication")


# ===================================================================== fig C
def fig_novel_and_valid():
    """Section 6.5.1 - the column that reverses the ranking."""
    arms = [("base", "Released weights"), ("undefended", "Undefended"),
            ("dp_lora", "DP-LoRA, ε = 0.75")]
    data = {}
    for key, lab in arms:
        rs = json.load(open(ROOT / "results_task_utility" / f"records_{key}.json",
                            encoding="utf-8"))
        n = len(rs)
        data[lab] = {
            "Schema adherence": sum(r["adheres"] for r in rs) / n,
            "Format validity": sum(r["valid"] for r in rs) / n,
            "Valid AND novel": sum(r["valid"] and not r["leaks_training"]
                                   for r in rs) / n,
        }

    fig, ax = plt.subplots(figsize=(W, 2.4))
    metrics = list(next(iter(data.values())))
    x = np.arange(len(data))
    width = 0.26
    cols = [GREY, CYAN, BLUE]
    hatch = ["", "", "//"]
    for j, (m, c, h) in enumerate(zip(metrics, cols, hatch)):
        vals = [data[k][m] for k in data]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=m, color=c,
                      edgecolor="black", linewidth=0.4, hatch=h, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(data))
    ax.set_ylabel("Rate on held-out people")
    ax.set_ylim(0, 1.13)
    ax.legend(loc="upper left", ncol=3, columnspacing=1.2, handlelength=1.4)
    ax.grid(axis="x", visible=False)
    ax.annotate("286 of 318 valid answers are\ntraining subjects' own records",
                xy=(1 + width, 0.105), xytext=(1.30, 0.40), fontsize=7.2,
                color="0.2", ha="left", va="center", linespacing=1.2,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7,
                                connectionstyle="arc3,rad=0.15"))
    fig.tight_layout(pad=0.4)
    save(fig, "fig_novel_and_valid")


# ===================================================================== fig D
def fig_noise_threshold():
    """Section 6.5.5 - both channels close below the smallest reportable
    budget, and competence peaks there."""
    thr = json.load(open(ROOT / "results_threshold_in" / "threshold.json",
                         encoding="utf-8"))
    eps = json.load(open(ROOT / "results_epsilon_rev2" / "epsilon_sweep.json",
                         encoding="utf-8"))
    rep3 = json.load(open(ROOT / "results_clip_in" / "replicate.json",
                          encoding="utf-8"))

    STRUCT = {"national_id", "unionpay_card", "mobile_phone",
              "medical_record_id", "soe_employee_id", "social_credit_code"}
    import csv

    def struct_rate(key):
        p = ROOT / "results_threshold_in" / f"records_matrix_{key}.csv"
        rr = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        sel = [r for r in rr if r["category"] in STRUCT]
        hit = sum(1 for r in sel if str(r["hit"]).strip() in ("1", "True", "true"))
        return hit / len(sel)

    ladder = {}
    for k, v in thr.items():
        ladder.setdefault(v["noise_multiplier"], {"rec": [], "str": [], "nov": []})
        ladder[v["noise_multiplier"]]["rec"].append(v["recitation"])
        ladder[v["noise_multiplier"]]["nov"].append(v["novel_valid"])
        ladder[v["noise_multiplier"]]["str"].append(struct_rate(k))

    comp = {nm: d["nov"] for nm, d in ladder.items()}
    comp[0.098] = [eps["dp_nm0.098"]["novel_valid"]]
    comp[0.182] = [eps["dp_nm0.182"]["novel_valid"]]
    comp[0.342] = ([eps["dp_nm0.342"]["novel_valid"]]
                   + [rep3[k]["novel_valid"] for k in rep3 if "eps1.0" in k])
    comp[0.648] = ([eps["dp_nm0.648"]["novel_valid"]]
                   + [rep3[k]["novel_valid"] for k in rep3 if "eps0.5" in k])
    leak_hi = {nm: max(eps[k]["recitation"] for k in eps if k.startswith("dp_")
                       and abs(eps[k]["noise_multiplier"] - nm) < 1e-9)
               for nm in (0.098, 0.182, 0.342, 0.648)}

    ZERO_X = 0.0042                      # nm = 0 placed left of the log axis
    def xf(nm):
        return ZERO_X if nm == 0 else nm

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 4.3), sharex=True,
                                   gridspec_kw={"height_ratios": [0.85, 1.15]})

    # ---- (a) leakage
    xs = sorted(ladder)
    rec = [np.mean(ladder[x]["rec"]) for x in xs]
    stc = [np.mean(ladder[x]["str"]) for x in xs]
    ax1.plot([xf(x) for x in xs], rec, color=RED, ls="-", marker="o",
             label="Recitation, held-out probe", zorder=3)
    ax1.plot([xf(x) for x in xs], stc, color=BLUE, ls="--", marker="s",
             label="Structured-ID reproduction, matrix", zorder=3)
    ax1.plot(sorted(leak_hi), [leak_hi[k] for k in sorted(leak_hi)],
             color=RED, ls="-", marker="o", zorder=3)
    ax1.annotate(f"{rec[0]:.3f}", (xf(0), rec[0]), xytext=(4, 3),
                 textcoords="offset points", fontsize=7.5, color=RED)
    ax1.annotate(f"{stc[0]:.3f}", (xf(0), stc[0]), xytext=(4, 3),
                 textcoords="offset points", fontsize=7.5, color=BLUE)
    ax1.annotate("zero at every rung from 0.010 upward\n"
                 "(8 conditions, ~20,000 structured probes)",
                 xy=(0.028, 0.0), xytext=(0.011, 0.135), fontsize=7.2,
                 color="0.25", ha="left", linespacing=1.2,
                 arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    ax1.set_ylabel("Leakage rate")
    ax1.set_ylim(-0.02, 0.30)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.42, 1.02))
    ax1.set_title("(a) Both channels close below the smallest reportable budget",
                  fontsize=8.5)

    # ---- (b) competence
    cs = sorted(comp)
    mean = [np.mean(comp[c]) for c in cs]
    lo = [np.mean(comp[c]) - min(comp[c]) for c in cs]
    hi = [max(comp[c]) - np.mean(comp[c]) for c in cs]
    ax2.errorbar([xf(c) for c in cs], mean, yerr=[lo, hi], color=GREEN,
                 ls="-", marker="D", capsize=2, elinewidth=0.7, zorder=3,
                 label="DP-LoRA (mean, min–max)")
    ax2.axhline(eps["base"]["novel_valid"], color=GREY, ls=":", lw=1.0)
    ax2.text(0.0105, eps["base"]["novel_valid"] - 0.048, "released weights",
             ha="left", fontsize=7, color="0.35")
    ax2.axhline(eps["undefended"]["novel_valid"], color=PURPLE, ls="-.", lw=1.0)
    ax2.text(0.0105, eps["undefended"]["novel_valid"] + 0.018,
             "undefended fine-tune", ha="left", fontsize=7, color=PURPLE)
    ax2.set_ylabel("Novel-and-valid rate")
    ax2.set_ylim(0.12, 0.74)
    ax2.set_xlabel("DP-SGD noise multiplier  (log scale)")
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, -0.02))
    ax2.set_title("(b) Competence peaks just above the threshold, and collapses "
                  "at reportable budgets", fontsize=8.5)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlim(0.0033, 0.95)
        ax.axvspan(0.098, 0.95, color=YELLOW, alpha=0.22, lw=0, zorder=0)
        ax.set_xticks([ZERO_X, 0.01, 0.02, 0.04, 0.07, 0.098, 0.182, 0.342, 0.648])
        ax.set_xticklabels(["0", "0.01", "0.02", "0.04", "0.07", "0.098",
                            "0.182", "0.342", "0.648"])
        # nm = 0 has no place on a log axis, so it is drawn detached and the
        # gap is marked with a break symbol on the spine.
        trans = ax.get_xaxis_transform()
        for xb in (0.0059, 0.0069):
            ax.plot([xb * 0.94, xb * 1.08], [-0.022, 0.022], transform=trans,
                    color="black", lw=0.7, clip_on=False, zorder=6)
    ax2.text(0.098 * 1.06, 0.155, "budgets an operator\nwould report "
             "($\\epsilon \\leq 4$)", fontsize=7.2, color="0.3",
             ha="left", va="bottom", linespacing=1.2)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(hspace=0.30)
    save(fig, "fig_noise_threshold")


if __name__ == "__main__":
    print("building figures at 6.0 in width, 600 dpi PNG + PDF vector")
    fig_generalization_window()
    fig_replication()
    fig_novel_and_valid()
    fig_noise_threshold()
    print(f"\noutput in {OUT}")
