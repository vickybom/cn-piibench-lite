#!/usr/bin/env python
"""The model-agnostic detection pipeline, requested by the panel chair.

Dr. Raga's 28 July comments ask for a schematic that makes three things explicit:
(1) the decoupling of the target LLM interface from the evaluation stage, (2) the
dual-track detection flow, and (3) where the known ground-truth entry enters.
The post-processing boundary is drawn as the vertical rule: nothing to its right
consults model weights, logits or internal state, which is what makes the
procedure model-agnostic rather than model-specific.

The name track is drawn as exact matching. The proposal specified a bounded
fuzzy match; Section 4.5 records that it produced false positives from common
characters embedded in ordinary words and was withdrawn, so the figure shows the
detector as it was actually run.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import pub_style as PS

OUT = Path(__file__).parent / "results_final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FILL = {"blue": "#E8EEF7", "red": "#FBE9EC", "green": "#E7F2EA",
        "amber": "#FCF3DF", "grey": "#EFF0F2"}
EDGE = {"blue": PS.BLUE, "red": PS.RED, "green": PS.GREEN,
        "amber": "#9C7A15", "grey": "#7A7F86"}


def box(ax, x, y, w, h, text, tone="blue", fs=7.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                fc=FILL[tone], ec=EDGE[tone], lw=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#1b1f24", linespacing=1.4,
            fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p1, p2, ls="-", color="#3a3f45", lw=1.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9,
                                 lw=lw, color=color, ls=ls, shrinkA=2, shrinkB=2,
                                 zorder=4))


def main():
    fig, ax = plt.subplots(figsize=(6.0, 3.30))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    # Fill the canvas. Left at the default subplot margins, bbox_inches="tight"
    # trims them away and the drawing lands at 4.9 in, which pushes the
    # effective resolution below 600 dpi once it is placed at column width.
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    # ---- zones -----------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.005, 0.10), 0.395, 0.80,
                                boxstyle="round,pad=0.004", fc="#F7F9FC",
                                ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((0.435, 0.06), 0.560, 0.84,
                                boxstyle="round,pad=0.004", fc="#F6FAF7",
                                ec="none", zorder=0))
    ax.text(0.20, 0.955, "Generation — black box", ha="center", fontsize=8.4,
            fontweight="bold", color=PS.BLUE)
    ax.text(0.715, 0.955, "Evaluation — ground-truth anchored", ha="center",
            fontsize=8.4, fontweight="bold", color=PS.GREEN)

    # ---- the post-processing boundary ------------------------------------
    ax.plot([0.418, 0.418], [0.06, 0.915], ls=(0, (4, 3)), lw=1.2,
            color="#B23A48", zorder=5)
    ax.text(0.418, 0.012, "post-processing boundary — nothing to the right reads "
            "weights, logits or internal state",
            ha="center", va="bottom", fontsize=6.8, color="#B23A48")

    # ---- generation side --------------------------------------------------
    box(ax, 0.02, 0.70, 0.36, 0.155,
        "Prompt from the twelve-template matrix\n(Type A / B / C / D)", "blue")
    box(ax, 0.02, 0.435, 0.36, 0.175,
        "TARGET LLM\nlocal checkpoint or commercial API\nweights never read or modified",
        "red", bold=True)
    box(ax, 0.02, 0.185, 0.36, 0.15,
        "Completion text\n(the only thing that crosses)", "grey")
    arrow(ax, (0.20, 0.70), (0.20, 0.612))
    arrow(ax, (0.20, 0.435), (0.20, 0.337))
    arrow(ax, (0.382, 0.255), (0.446, 0.305))

    # ---- ground truth entering -------------------------------------------
    box(ax, 0.448, 0.725, 0.545, 0.145,
        "Known ground-truth entry from M1\n(person, category, exact value)",
        "amber", fs=7.0)
    arrow(ax, (0.58, 0.725), (0.58, 0.652), ls=(0, (2, 2)))
    arrow(ax, (0.86, 0.725), (0.86, 0.652), ls=(0, (2, 2)))

    # ---- dual track -------------------------------------------------------
    box(ax, 0.448, 0.475, 0.265, 0.175,
        "Track 1 — structured\nidentifier\nexact token match, then\n"
        "checksum re-validation", "green", fs=6.6)
    box(ax, 0.728, 0.475, 0.265, 0.175,
        "Track 2 — person name\nexact match only\n(fuzzy match withdrawn,\n§4.5)",
        "green", fs=6.6)
    arrow(ax, (0.58, 0.475), (0.58, 0.387))
    arrow(ax, (0.86, 0.475), (0.86, 0.387))

    box(ax, 0.448, 0.235, 0.545, 0.150,
        "Verdict per query: hit or miss\nMER · matched-pair CLMD · RW-MER = MER × PRI(c)",
        "grey", fs=6.9)
    ax.text(0.720, 0.175, "no probabilistic classifier, no entity model, "
            "no threshold to tune", ha="center", fontsize=6.6, color="0.35")
    ax.text(0.720, 0.115, "a hit requires the exact ground-truth token, so precision "
            "is bounded by construction", ha="center", fontsize=6.6, color="0.35")

    PS.finish(fig)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_detection_pipeline.{ext}")
    plt.close(fig)
    print(f"wrote fig_detection_pipeline.png / .pdf in {OUT}")


if __name__ == "__main__":
    main()
