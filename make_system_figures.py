#!/usr/bin/env python
"""Figures for the System Design and Implementation chapter."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "results_final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
import pub_style  # publication rcParams

BLUE, GREY, RED, GREEN = "#2c6fbb", "#5b6472", "#e5484d", "#1a9e5b"


def box(ax, x, y, w, h, text, fc, ec, fs=8.5, tc="white", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(ax, p1, p2, color="#333", style="-|>", lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def architecture():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")

    # zones
    ax.add_patch(FancyBboxPatch((0.15, 0.5), 3.3, 4.0, boxstyle="round,pad=0.02",
                                fc="#eef4fc", ec="#c3d6ee", lw=1))
    ax.text(1.8, 4.28, "BEFORE the model", fontsize=9, color=BLUE, fontweight="bold", ha="center")
    ax.add_patch(FancyBboxPatch((3.75, 0.5), 2.5, 4.0, boxstyle="round,pad=0.02",
                                fc="#f4f4f6", ec="#d5d8de", lw=1))
    ax.text(5.0, 4.28, "AROUND the model", fontsize=9, color=GREY, fontweight="bold", ha="center")
    ax.add_patch(FancyBboxPatch((6.55, 0.5), 3.3, 4.0, boxstyle="round,pad=0.02",
                                fc="#fdeeef", ec="#f2c9cc", lw=1))
    ax.text(8.2, 4.28, "AFTER the model", fontsize=9, color=RED, fontweight="bold", ha="center")

    box(ax, 0.45, 3.15, 2.7, 0.85, "M1  Synthetic PII Generator\n7 categories · checksums · seeded", BLUE, BLUE)
    box(ax, 0.45, 2.05, 2.7, 0.85, "M2  Prompt-Matrix Builder\nType A / B / C · ZH & EN", BLUE, BLUE)
    box(ax, 4.0, 2.6, 2.0, 1.4, "M3\nInference\nController\n\ngreedy · batched\nlocal / API", GREY, GREY)
    box(ax, 4.0, 1.0, 2.0, 1.05, "Target LLM\n(black box)\nweights untouched", "white", GREY, tc="#333")
    box(ax, 6.85, 3.15, 2.7, 0.85, "M4  Leakage Detector\nexact token + checksum", RED, RED)
    box(ax, 6.85, 2.05, 2.7, 0.85, "M5  Metric Engine\nMER · CLMD · RW-MER", RED, RED)
    box(ax, 6.85, 0.95, 2.7, 0.85, "M6  Risk Classifier / Report\nLow · Medium · High", RED, RED)

    # inputs: M1/M2 -> M3
    arrow(ax, (3.15, 3.55), (4.0, 3.5))
    arrow(ax, (3.15, 2.45), (4.0, 3.05))
    # M3 <-> target model
    arrow(ax, (4.8, 2.6), (4.8, 2.05), color=GREY)
    arrow(ax, (5.2, 2.05), (5.2, 2.6), color=GREY)
    # completions: M3 -> up and across into M4
    arrow(ax, (6.0, 3.6), (6.45, 3.6), color=GREY, style="-")
    arrow(ax, (6.45, 3.6), (6.85, 3.6))
    # analysis chain flows DOWNWARD: M4 -> M5 -> M6
    arrow(ax, (8.2, 3.15), (8.2, 2.9))
    arrow(ax, (8.2, 2.05), (8.2, 1.8))
    ax.text(3.35, 3.75, "prompts", fontsize=7.5, color="#333")
    ax.text(6.02, 3.75, "completions", fontsize=7.5, color="#333")
    ax.text(4.42, 2.28, "query", fontsize=7, color=GREY, rotation=90, va="center")
    ax.text(5.42, 2.28, "text", fontsize=7, color=GREY, rotation=90, va="center")

    ax.text(5.0, 0.2, "The toolkit never reads or modifies weights, gradients, or training data.",
            ha="center", fontsize=8, style="italic", color="#555")
    fig.tight_layout(); save(fig, "fig_sys_architecture")


def dataflow():
    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.4); ax.axis("off")
    stages = [("config\n+ seed", "#8895a7"), ("persons\n(M1)", BLUE), ("triples\n(M2)", BLUE),
              ("completions\n(M3)", GREY), ("records\n(M4)", RED),
              ("metrics\n(M5)", RED), ("report\n(M6)", GREEN)]
    w, gap = 1.15, 0.28
    x = 0.25
    for i, (label, c) in enumerate(stages):
        box(ax, x, 0.85, w, 0.8, label, c, c, fs=8)
        if i < len(stages) - 1:
            arrow(ax, (x + w, 1.25), (x + w + gap, 1.25))
        x += w + gap
    ax.text(5.0, 0.45, "every stage is a plain, serialisable artifact — "
            "so any stage can be re-run, cached, or audited independently",
            ha="center", fontsize=8, style="italic", color="#555")
    fig.tight_layout(); save(fig, "fig_sys_dataflow")


def save(fig, name):
    pub_style.finish(fig)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig); print("  wrote", name)


if __name__ == "__main__":
    architecture(); dataflow()
