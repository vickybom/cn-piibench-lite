#!/usr/bin/env python
"""Generate the Chapter 4 figures from a results.json bundle.

    python make_figures.py results/results.json --out results/figures

Produces publication-style PNG/PDF figures:
    fig4_1_mer_heatmap   per-category MER (ZH->ZH) x model
    fig4_2_clmd          cross-lingual gap (CLMD) by category
    fig4_3_verdict       aggregate RW-MER per model with risk thresholds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pub_style
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def _load(path):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    cats = r["category_order"]
    labels = [r["category_labels"][c]["en"] for c in cats]
    models = [s["model"] for s in r["summary"]][::-1]   # smallest first
    byMC = {(x["model"], x["category"]): x for x in r["per_category"]}
    return r, cats, labels, models, byMC


def heatmap(path, out):
    r, cats, labels, models, byMC = _load(path)
    M = np.array([[ (byMC.get((m, c), {}).get("mer_zh2zh") or 0.0) for c in cats]
                  for m in models])
    fig, ax = plt.subplots(figsize=(8.2, 2.2 + 0.5 * len(models)))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=max(0.01, M.max()), aspect="auto")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(cats)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] > M.max() * 0.6 else "black", fontsize=8)
    ax.set_title("Per-category Memorization Extraction Rate (ZH→ZH)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="MER")
    fig.tight_layout(); _save(fig, out, "fig4_1_mer_heatmap")


def clmd(path, out):
    r, cats, labels, models, byMC = _load(path)
    x = np.arange(len(cats)); w = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for k, m in enumerate(models):
        vals = [ (byMC.get((m, c), {}).get("clmd") or 0.0) for c in cats]
        ax.bar(x + k * w, vals, w, label=m)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("CLMD = MER(EN→ZH) − MER(ZH→ZH)")
    ax.set_title("Cross-lingual leakage gap by category")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); _save(fig, out, "fig4_2_clmd")


def verdict(path, out):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    S = r["summary"]; models = [s["model"] for s in S]; vals = [s["aggregate_rwmer"] for s in S]
    colors = {"Low": "#1a9e5b", "Medium": "#d99a00", "High": "#e5484d"}
    fig, ax = plt.subplots(figsize=(7.6, 2.2 + 0.5 * len(models)))
    ax.barh(models, vals, color=[colors[s["risk_band"]] for s in S])
    ax.axvline(r["thresholds"]["low"], ls="--", color="#888", lw=1)
    ax.axvline(r["thresholds"]["high"], ls="--", color="#888", lw=1)
    for i, s in enumerate(S):
        ax.text(s["aggregate_rwmer"], i, f"  {s['aggregate_rwmer']:.3f} ({s['risk_band']})",
                va="center", fontsize=9)
    ax.set_xlabel("Aggregate RW-MER  (Low<0.05 · Medium<0.20 · High≥0.20)")
    ax.set_title("Model verdict — aggregate Risk-Weighted MER")
    ax.invert_yaxis(); fig.tight_layout(); _save(fig, out, "fig4_3_verdict")


def _save(fig, out, name):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pub_style.finish(fig)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {out / name}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    heatmap(args.results_json, args.out)
    clmd(args.results_json, args.out)
    verdict(args.results_json, args.out)


if __name__ == "__main__":
    main()
