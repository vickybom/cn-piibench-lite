#!/usr/bin/env python
"""Redraw the schematic diagrams whose content no longer matches the thesis.

These five were carried over from the proposal and never updated. Every one of
them states facts the finished study contradicts:

  * "10 templates (A/B/C)"  -- the matrix is twelve, and the Type D templates
    are what make the matched-pair CLMD of Section 5.8 possible. Omitting them
    removes a headline contribution from the figure that introduces the design.
  * "1,000-3,000 entries"   -- the corpus is 140 records, 980 entries.
  * "7B optional"           -- never evaluated; Section 7.5 lists it as a limit.
  * "Privacy defenses (DP-SGD, machine unlearning) ... reserved for future work
    and are not implemented in this study" -- Chapter 6 evaluates three of them
    across roughly 60,000 probes.

A reader who compares Figure 2.1 with Chapter 6 finds a flat contradiction, so
these are corrected here rather than merely re-rendered at higher resolution.
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
        "amber": "#FCF3DF", "purple": "#F1E9F4", "grey": "#F0F1F2"}
EDGE = {"blue": PS.BLUE, "red": PS.RED, "green": PS.GREEN,
        "amber": "#9C7A15", "purple": PS.PURPLE, "grey": "#7A7F86"}


def box(ax, x, y, w, h, text, tone="blue", fs=8, weight="normal",
        title=None, title_fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                fc=FILL[tone], ec=EDGE[tone], lw=0.9, zorder=2))
    if title:
        band = min(0.30 * h, 0.075)
        ax.text(x + w / 2, y + h - band * 0.55, title, ha="center", va="center",
                fontsize=title_fs, color=EDGE[tone], fontweight="bold",
                zorder=3)
        ax.text(x + w / 2, y + (h - band) / 2, text, ha="center", va="center",
                fontsize=fs, color="#1b1f24", linespacing=1.45, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#1b1f24", fontweight=weight,
                linespacing=1.45, zorder=3)


def arrow(ax, p1, p2, ls="-", color="#3a3f45", lw=1.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9,
                                 lw=lw, color=color, ls=ls,
                                 shrinkA=1, shrinkB=1, zorder=4))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.grid(False)
    return fig, ax


def save(fig, name):
    PS.finish(fig)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    # SVG too: three of these figures are embedded in the document as SVG with a
    # PNG fallback, and Word renders the SVG. Replacing only the fallback leaves
    # the page unchanged, which is how the first attempt failed silently.
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print(f"  wrote {name}")



def no_overlaps(fig, ax, label, allow=()):
    """Fail if any two pieces of text in the figure collide, or one is clipped.

    The first draft of Figure 3.1 was tuned by eye: box titles sat on the first
    body line, a sub-band's right label ran into its left one, and the last
    line of a six-line box fell out through the bottom of the box. Eyeballing a
    figure this dense does not scale, so the geometry is measured instead.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes = [(t, t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
    bad = []
    for t, bb in boxes:
        if not (fb.x0 - 1 <= bb.x0 and bb.x1 <= fb.x1 + 1
                and fb.y0 - 1 <= bb.y0 and bb.y1 <= fb.y1 + 1):
            bad.append(f"clipped: {t.get_text()[:48]!r}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.overlaps(b):
                ta = boxes[i][0].get_text()[:34]
                tb = boxes[j][0].get_text()[:34]
                if (ta, tb) in allow or (tb, ta) in allow:
                    continue
                bad.append(f"overlap: {ta!r} x {tb!r}")
    if bad:
        joined = "\n    ".join(bad[:12])
        raise SystemExit(f"  {label}: {len(bad)} layout problem(s)\n    "
                         + joined)


# =================================================================== Fig 1.1
def threat_model():
    fig, ax = canvas(6.0, 2.35)
    box(ax, 0.005, 0.60, 0.235, 0.24,
        "Monolingual adversary\n(ZH $\\rightarrow$ ZH)", "red", fs=8.2)
    box(ax, 0.005, 0.16, 0.235, 0.24,
        "Cross-lingual adversary\n(EN $\\rightarrow$ ZH)", "amber", fs=8.2)
    box(ax, 0.335, 0.24, 0.30, 0.52,
        "Qwen2.5-1.5B / 3B\nQwen2.5-1.5B-Instruct\ntwo commercial endpoints\n\n"
        "black box: output channel only,\nweights never accessed",
        "blue", fs=8, title="Domestic Chinese LLM")
    box(ax, 0.73, 0.36, 0.265, 0.32,
        "MER  ·  matched-pair CLMD\nRW-MER  ·  PII Risk Index",
        "green", fs=8, title="CN-PIIBench-Lite")
    for i, (lab, tone) in enumerate((("Low", "green"), ("Medium", "amber"),
                                     ("High", "red"))):
        box(ax, 0.735 + i * 0.088, 0.10, 0.080, 0.115, lab, tone, fs=7.6)
    # not "PIPL Article 51 risk tiers": the bands are defined by this study and
    # carry no legal force, which the label has to say as plainly as the text
    ax.text(0.867, 0.045, "study-defined, PIPL-oriented risk bands",
            ha="center", fontsize=6.6, color="0.4")
    arrow(ax, (0.24, 0.72), (0.335, 0.60))
    arrow(ax, (0.24, 0.28), (0.335, 0.40))
    arrow(ax, (0.635, 0.50), (0.73, 0.50))
    arrow(ax, (0.867, 0.36), (0.867, 0.222))
    ax.text(0.288, 0.90, "ZH prefix and\nassociation prompts", fontsize=6.8,
            color="0.4", ha="center", va="center", linespacing=1.2)
    ax.text(0.288, 0.10, "EN chat-template\ninstruction", fontsize=6.8,
            color="0.4", ha="center", va="center", linespacing=1.2)
    ax.text(0.683, 0.565, "PII in\noutput", fontsize=6.8, color="0.4",
            ha="center", va="bottom", linespacing=1.2)
    ax.annotate("", xy=(0.12, 0.585), xytext=(0.12, 0.415),
                arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.8,
                                linestyle=(0, (3, 2))))
    ax.text(0.135, 0.50, "CLMD", fontsize=7.5, color="0.35", va="center")
    save(fig, "fig_threat_model")


# =================================================================== Fig 2.1
def conceptual_framework():
    fig, ax = canvas(6.0, 3.5)
    box(ax, 0.02, 0.775, 0.96, 0.20,
        "black-box output-channel adversary\n"
        "ZH$\\rightarrow$ZH monolingual and EN$\\rightarrow$ZH cross-lingual axes\n"
        "seven Chinese PII categories, GB 11643-1999 and GB 32100-2015",
        "red", fs=7.6, title="Layer 1 — Threat model")
    box(ax, 0.02, 0.485, 0.96, 0.265, "", "blue", fs=8,
        title="Layer 2 — Evaluation pipeline (CN-PIIBench-Lite)")
    cells = ["Synthetic corpus\n140 records\n980 entries",
             "Prompt matrix\n12 templates\nTypes A/B/C/D",
             "MER and\nmatched-pair\nCLMD",
             "RW-MER\nMER × PRI(c)",
             "study-defined band\nLow/Med/High"]
    for i, t in enumerate(cells):
        box(ax, 0.038 + i * 0.1885, 0.505, 0.171, 0.155, t, "grey", fs=7.0)
    box(ax, 0.02, 0.245, 0.96, 0.225,
        "per-category MER, matched-pair CLMD, RW-MER = MER × PRI(c)\n"
        "study-defined, PIPL-oriented bands:   Low < 0.05    Medium 0.05–0.20    "
        "High $\\geq$ 0.20",
        "amber", fs=7.6, title="Layer 3 — Risk scoring")
    box(ax, 0.02, 0.015, 0.96, 0.20,
        "PII-Auditor-CN-Lite — offline, GPU-optional audit toolkit\n"
        "three defenses on one fixed instrument: output filtering,\n"
        "machine unlearning, differentially private fine-tuning (Chapter 6)",
        "green", fs=7.6, title="Layer 4 — Defense evaluation and deliverable")
    for y0, y1 in ((0.775, 0.752), (0.485, 0.472), (0.245, 0.217)):
        arrow(ax, (0.5, y0), (0.5, y1))
    save(fig, "fig_conceptual_framework")


# =================================================================== Fig 3.1
# Figure 3.1 had no generator at all. It was authored by hand as an SVG in May
# (topic/cn_piibench_threat_taxonomy_v2.svg) and embedded as a picture, so it
# was the one figure in the dissertation that no checker could read and no
# script could redraw. It still described the proposal: defenses "reserved for
# future work" when Chapter 6 evaluates three of them, a corpus of "1,000-3,000
# entries" against the final 140 records and 980 entries, and "7B optional" for
# a scale that was in fact reached, through Yi-1.5-6B and DeepSeek-LLM-7B. Its
# PRI severity scale also read as a second, competing classification beside the
# study's own RW-MER bands, and its closing line asserted a "mandatory PIPIA
# under PIPL" that the manuscript is careful never to claim.
#
# The text is English throughout. This file sets Times New Roman, matplotlib
# does not fall back per glyph, and the CJK terms of the original SVG render as
# empty boxes; Section 1.7 carries the Chinese category names instead.
#
# Two things about the size are load-bearing. Box heights are computed from the
# line count rather than guessed, because the first draft guessed and the
# titles sat on the first body line. And the canvas is 6.0 x 7.1 inches, which
# is what pub_style.finish() leaves alone: it rescales to the 6.0 in text block
# and caps height at 7.6 in, so a 6.3 x 10.5 draft came out squashed to
# 4.56 x 7.6 with the point sizes unchanged, and everything collided. The
# geometry check therefore runs after finish(), on the real output.
FIGH_31 = 7.1                  # inches: the height the old figure occupied, so
                               # image and caption still share one page
FS_BODY, FS_TITLE = 4.8, 5.6   # a full-page taxonomy at the text-block width


def threat_taxonomy():
    fig, ax = canvas(6.0, FIGH_31)

    def pitch(fs, k=1.35):
        """Vertical step for stacked single-line text.

        A measured line occupies about 1.35 x its point size once ascenders and
        descenders are counted; stepping by less than that is what put seven
        overlapping pairs into the first geometry-checked draft. Lines carrying
        an arrow need more: U+2192 is absent from Times New Roman, matplotlib
        substitutes a font with taller metrics for it, and the whole line box
        grows with it.
        """
        return fs * k / (72.0 * FIGH_31)

    def box_h(bodies, fs=FS_BODY, pad=0.008):
        # box() spends min(0.30h, 0.075) of its height on the title band, so
        # h = 0.30h + body + pad gives the height the content actually needs.
        n = max(t.count("\n") + 1 for t in bodies)
        return (n * fs * 1.45 / (72.0 * FIGH_31) + pad) / 0.70

    def hdr(y, text, h=0.020):
        ax.add_patch(FancyBboxPatch((0.008, y - h), 0.984, h,
                                    boxstyle="round,pad=0.002",
                                    fc="#3a3f45", ec="#3a3f45", lw=0, zorder=2))
        ax.text(0.5, y - h / 2, text, ha="center", va="center", fontsize=6.4,
                color="white", fontweight="bold", zorder=3)
        return y - h - 0.004

    def sub(y, left, right, h=0.016):
        ax.add_patch(FancyBboxPatch((0.008, y - h), 0.984, h,
                                    boxstyle="round,pad=0.002",
                                    fc="#EDEEF0", ec="#C6CACE", lw=0.7, zorder=2))
        ax.text(0.020, y - h / 2, left, ha="left", va="center", fontsize=5.2,
                fontweight="bold", color="#2b3036", zorder=3)
        ax.text(0.980, y - h / 2, right, ha="right", va="center", fontsize=4.6,
                color="0.45", zorder=3)
        return y - h - 0.003

    def row(y, items, tone="grey", fs=FS_BODY, gap=0.009):
        h = box_h([b for _, b in items], fs)
        n = len(items)
        w = (0.984 - gap * (n - 1)) / n
        for i, (t, body) in enumerate(items):
            box(ax, 0.008 + i * (w + gap), y - h, w, h, body, tone,
                fs=fs, title=t, title_fs=FS_TITLE)
        return y - h - 0.007

    def note(y, text, h=0.012):
        ax.text(0.5, y - h / 2, text, ha="center", va="center", fontsize=4.8,
                color="0.35", zorder=3)
        return y - h - 0.003

    y = 0.995
    ax.text(0.5, y - 0.011, "CN-PIIBench-Lite threat landscape and PII "
            "classification taxonomy", ha="center", va="center", fontsize=7.6,
            fontweight="bold", color="#1b1f24")
    ax.text(0.5, y - 0.030, "Three-dimensional threat model for Section 3.1",
            ha="center", va="center", fontsize=5.8, color="0.42")
    y -= 0.040

    # ---------------------------------------------------- Dimension 1
    y = hdr(y, "Dimension 1 — PII attribute categories")
    y = sub(y, "Sensitive personal information", "PIPL Art. 28")
    y = row(y, [
        ("Biometric & medical",
         "Fingerprint · iris · face ID\nDNA · medical record\n"
         "Diagnosis · mental health\nPRI > 0.80"),
        ("Financial & location",
         "Bank acct. · card no.\nCredit score · tax record\n"
         "GPS · whereabouts track\nPRI 0.60 – 0.80"),
        ("Religious & minors",
         "Religious belief\nSpecific identity\nMinors under age 14\n"
         "Separate consent required"),
    ], "red")
    y = sub(y, "General personal information", "GB/T 35273")
    y = row(y, [
        ("Basic identity",
         "Name · ID card no.\nPassport · driver lic.\nPRI 0.30 – 0.60"),
        ("Contact details",
         "Mobile no. · email\nAddress · postal code\nPRI 0.25 – 0.50"),
        ("Demographic",
         "DOB · gender · age\nNationality\nPRI 0.15 – 0.35"),
        ("Combined PII",
         "Name + address\nName + bank acct.\n"
         "Aggregation penalty $\\lambda$ = 0.025"),
    ], "grey")
    y = note(y, "PRI ranges here are attribute-level values from the "
                "seven-factor framework of Jeon et al. (2026). They are not "
                "the risk bands this study assigns to a deployment.")

    # ---------------------------------------------------- Dimension 2
    y = hdr(y, "Dimension 2 — attacker capabilities")
    y = sub(y, "Training data extraction attacks", "Carlini · Nasr · Cheng")
    y = row(y, [
        ("Prefix probing",
         "Unconditional sampling\nPPL + zlib filter\nVerbatim suffix match"),
        ("Scalable divergence",
         "EOS-token divergence\nRLHF bypass strategy\n"
         "150$\\times$ emission rate"),
        ("Augmented few-shot",
         "Online example selection\nPrompt-chain augmentation\nASR 48.9%"),
    ], "amber")
    y = sub(y, "Membership inference attacks",
            "surveyed only — this study implements extraction probing")
    y = row(y, [
        ("LiRA (likelihood-ratio)",
         "Shadow-model approach\nTPR @ 0.1% FPR metric\n"
         "Online + offline variants"),
        ("Min-K% probability",
         "Non-member outlier test\nAUC-based evaluation\n"
         "Unlearning verification"),
        ("Foundation role",
         "Filters memorised outputs\nHigh precision at low FPR\n"
         "Extraction attack step 2"),
    ], "purple")
    # ---------------------------------------------------- Dimension 3
    y = hdr(y, "Dimension 3 — adversarial access levels")
    y = row(y, [
        ("Black-box API access",
         "Query-only; weights never read\n"
         "Output text is the only channel\n"
         "Evaluated here: Qwen2.5-1.5B / 3B / 1.5B-Instruct,\n"
         "Yi-1.5-6B, DeepSeek-LLM-7B and two hosted\n"
         "DashScope endpoints (Sections 5.4, 5.12)"),
        ("Gray-box / architecture knowledge",
         "Open-weight; training data private\n"
         "MLA / GQA architecture knowledge\n"
         "Shadow-model MIA, activation analysis,\n"
         "safety-neuron editing — taxonomy only,\n"
         "not exercised in this black-box study"),
    ], "blue")

    # ---------------------------------------------------- intersection
    y = hdr(y, "Central intersection — the CN-PIIBench-Lite attack surface")
    scope = ("Structured Chinese PII of low intrinsic dimension and high "
             "memorisation, probed black-box over a\n"
             "twelve-template matrix (Types A/B/C/D) and scored by the "
             "seven-factor PRI, against small and\n"
             "medium domestic Chinese LLMs. Three defenses are evaluated in "
             "Chapter 6: output filtering,\n"
             "machine unlearning, and differentially private fine-tuning.\n"
             "140 person records · 980 entries · seven categories · "
             "70,560 primary probes")
    h = box_h([scope])
    box(ax, 0.09, y - h, 0.82, h, scope, "green", fs=FS_BODY,
        title="CN-PIIBench-Lite benchmark scope", title_fs=FS_TITLE)
    y = y - h - 0.008

    # ---------------------------------------------------- PRI scale
    y = hdr(y, "PRI attribute-level background scale — not this study's "
               "deployment bands")
    y = row(y, [
        ("Critical  > 0.80", "Biometric · ID no. · passport"),
        ("High  0.55 – 0.80", "Financial · medical"),
        ("Medium  0.30 – 0.55", "Location · contact info"),
        ("Low  < 0.30", "Given name · gender · postal"),
    ], "grey")
    hb = 0.034
    ax.add_patch(FancyBboxPatch((0.06, y - hb), 0.88, hb,
                                boxstyle="round,pad=0.004",
                                fc="#FFFFFF", ec="#9C7A15", lw=1.0, zorder=2))
    ax.text(0.5, y - hb / 2,
            "The deployment risk bands of CN-PIIBench-Lite are defined on "
            "aggregate RW-MER, not on PRI:\n"
            "Low < 0.05   ·   Medium 0.05 – 0.20   ·   High $\\geq$ 0.20"
            "      study-defined operational thresholds (Section 3.1.2)",
            ha="center", va="center", fontsize=4.9, color="#1b1f24",
            linespacing=1.6, zorder=3)
    y = y - hb - 0.009

    # ---------------------------------------------------- connectors
    ax.text(0.008, y, "Cross-dimension paths", ha="left", va="center",
            fontsize=5.0, fontweight="bold", color="#2b3036")
    y -= pitch(5.0)
    for line in (
        "Low-ID PII (Dim. 1)  →  high memorisation  →  extraction attack "
        "(Dim. 2)  →  black-box probing (Dim. 3)  →  PRI weighting  →  RW-MER",
        "Combination PII (Dim. 1)  +  attribute-conditioned association (Dim. 2)  +  "
        "black-box access (Dim. 3)  →  high attribute-level PRI  →  may "
        "relate to PIPL impact-assessment",
        "obligations, subject to the applicable processing context. "
        "No rating produced by this benchmark carries legal force.",
    ):
        ax.text(0.008, y, line, ha="left", va="center", fontsize=4.6,
                color="0.30")
        y -= pitch(4.6, k=2.1)
    y -= 0.004

    ax.plot([0.008, 0.992], [y, y], color="#C6CACE", lw=0.7)
    y -= 0.010
    ax.text(0.008, y, "Sources: Carlini et al. · Nasr et al. · Cheng et al. "
            "· Shi et al. · Lukas et al. · Arnold (2025) · Liang et al. (2026) "
            "· Jeon et al. (2026) · GB/T 35273 · TC260 · PIPL",
            ha="left", va="center", fontsize=4.3, color="0.42")
    y -= pitch(4.3)

    assert y > 0.0, (f"Figure 3.1 layout overflowed the canvas by {-y:.3f} of "
                     f"its height; raise FIGH_31 or cut content")
    PS.finish(fig)                  # measure the geometry that is actually saved
    no_overlaps(fig, ax, "Figure 3.1")
    save(fig, "fig_threat_taxonomy")

# =================================================================== Fig 3.2
def phases():
    fig, ax = canvas(6.0, 1.75)
    ph = [("Phase 1", "Corpus construction\n140 records, 980 entries\n"
                      "checksum-validated", "green"),
          ("Phase 2", "Memorization probing\n1.5B / 3B / Instruct\n"
                      "12 templates", "red"),
          ("Phase 3", "Risk scoring\nMER · CLMD · RW-MER\nPIPL tiers",
           "purple"),
          ("Phase 4", "Defense evaluation\nfilter · unlearn · DP-SGD\n"
                      "toolkit release", "blue")]
    w = 0.232
    for i, (t, body, tone) in enumerate(ph):
        x = 0.008 + i * 0.2475
        box(ax, x, 0.30, w, 0.60, body, tone, fs=7.4, title=t, title_fs=8.5)
        if i < 3:
            arrow(ax, (x + w, 0.60), (x + w + 0.0155, 0.60))
    ax.text(0.5, 0.13, "70,560 probing queries across the six primary conditions, plus "
            "1,680 commercial; every stage writes a serialized artefact and can be re-run alone",
            ha="center", fontsize=7.2, color="0.35")
    save(fig, "fig_phases")


# =================================================================== Fig 3.4
def workflow():
    fig, ax = canvas(6.0, 5.2)
    NL = "\n"
    steps = [
        ("1. Data collection",
         "GB 11643-1999, GB 32100-2015, carrier prefixes," + NL +
         "UnionPay BIN, hospital and SOE schemas", "green"),
        ("2. Synthetic PII generation",
         "140 fictitious records across seven" + NL +
         "categories, 980 entries in total", "green"),
        ("3. Corpus validation",
         "checksum, regex and lookup validators," + NL +
         "plus web-absence verification", "green"),
        ("4. Prompt-matrix construction",
         "twelve templates: A prefix, B association," + NL +
         "C English chat-template, D Chinese (matched to C)", "red"),
        ("5. Model setup",
         "4-bit quantised Qwen2.5-1.5B / 3B / Instruct;" + NL +
         "greedy decoding throughout", "red"),
        ("6. Memorization probing",
         "black-box completion under" + NL +
         "ZH$\\rightarrow$ZH and EN$\\rightarrow$ZH", "red"),
        ("7. Leakage detection",
         "exact match with checksum re-validation;" + NL +
         "exact matching for names", "purple"),
        ("8. Metric computation",
         "MER, matched-pair CLMD, and RW-MER" + NL +
         "weighted by the PII Risk Index", "purple"),
        ("9. Risk classification",
         "aggregate RW-MER mapped to study-defined," + NL +
         "PIPL-oriented Low / Medium / High bands", "purple"),
        ("10. Defense evaluation",
         "output filtering, machine unlearning and" + NL +
         "DP-SGD on one fixed instrument", "blue"),
        ("11. Validation and release",
         "reproducibility re-run, replication at a second" + NL +
         "training seed, public toolkit release", "blue"),
    ]
    n = len(steps)
    top, gap = 0.985, 0.0055
    h = (top - 0.01 - gap * (n - 1)) / n
    lanes = [("Phase 1", "green", 0, 2), ("Phase 2", "red", 3, 5),
             ("Phase 3", "purple", 6, 8), ("Phase 4", "blue", 9, 10)]
    for lab, tone, a, b in lanes:
        y1 = top - a * (h + gap)
        y0 = top - b * (h + gap) - h
        ax.add_patch(FancyBboxPatch((0.005, y0), 0.115, y1 - y0,
                                    boxstyle="round,pad=0.004",
                                    fc=FILL[tone], ec=EDGE[tone], lw=0.7,
                                    alpha=0.55, zorder=1))
        ax.text(0.0625, (y0 + y1) / 2, lab, ha="center", va="center",
                fontsize=8, color=EDGE[tone], fontweight="bold", rotation=90,
                zorder=3)
    for i, (title, body, tone) in enumerate(steps):
        y = top - i * (h + gap) - h
        ax.add_patch(FancyBboxPatch((0.135, y), 0.858, h,
                                    boxstyle="round,pad=0.004",
                                    fc=FILL[tone], ec=EDGE[tone], lw=0.8,
                                    zorder=2))
        ax.text(0.150, y + h / 2, title, ha="left", va="center", fontsize=7.3,
                fontweight="bold", color=EDGE[tone], zorder=3)
        ax.text(0.475, y + h / 2, body, ha="left", va="center", fontsize=6.9,
                color="#1b1f24", linespacing=1.3, zorder=3)
        if i < n - 1:
            arrow(ax, (0.60, y), (0.60, y - gap))
    save(fig, "fig_workflow")


# =================================================================== Fig 3.5
def architecture():
    fig, ax = canvas(6.0, 2.9)
    zones = [("BEFORE the model\n(input preparation)", 0.005, 0.325, "blue"),
             ("AROUND the model\n(query execution)", 0.338, 0.325, "amber"),
             ("AFTER the model\n(leakage analysis)", 0.671, 0.324, "green")]
    for lab, x, w, tone in zones:
        ax.add_patch(FancyBboxPatch((x, 0.02), w, 0.96,
                                    boxstyle="round,pad=0.004",
                                    fc=FILL[tone], ec="none", alpha=0.45,
                                    zorder=0))
        ax.text(x + w / 2, 0.945, lab, ha="center", va="center", fontsize=8,
                color=EDGE[tone], fontweight="bold", linespacing=1.3, zorder=3)
    box(ax, 0.022, 0.60, 0.29, 0.235,
        "M1  Synthetic PII generator\n7 categories, checksum-validated,\n"
        "140 fictitious records", "blue", fs=7.2)
    box(ax, 0.022, 0.30, 0.29, 0.235,
        "M2  Prompt-matrix builder\nTypes A/B/C/D, ZH and EN,\n12 templates",
        "blue", fs=7.2)
    box(ax, 0.352, 0.60, 0.29, 0.20,
        "M3  Inference controller\ngreedy decoding; local or API", "amber", fs=7.2)
    box(ax, 0.352, 0.235, 0.29, 0.235,
        "TARGET LLM\nQwen2.5-1.5B / 3B / Instruct\nweights NOT modified",
        "red", fs=7.2, weight="bold")
    box(ax, 0.682, 0.665, 0.30, 0.185,
        "M4  Leakage detector\nexact match + checksum", "green", fs=7.2)
    box(ax, 0.682, 0.435, 0.30, 0.185,
        "M5  Metric engine\nMER · CLMD · RW-MER", "green", fs=7.2)
    box(ax, 0.682, 0.205, 0.30, 0.185,
        "M6  Risk classifier\nLow / Medium / High (PIPL)", "green", fs=7.2)
    arrow(ax, (0.167, 0.60), (0.167, 0.535))
    arrow(ax, (0.312, 0.40), (0.352, 0.34))
    arrow(ax, (0.497, 0.60), (0.497, 0.47))
    arrow(ax, (0.642, 0.34), (0.682, 0.70))
    arrow(ax, (0.832, 0.665), (0.832, 0.62))
    arrow(ax, (0.832, 0.435), (0.832, 0.39))
    ax.text(0.335, 0.375, "prompts", fontsize=6.8, color="0.4", ha="center")
    ax.text(0.503, 0.535, "queries", fontsize=6.8, color="0.4", ha="left")
    ax.text(0.664, 0.53, "outputs", fontsize=6.8, color="0.4", ha="left")
    ax.text(0.5, 0.055, "the toolkit is an external wrapper: it surrounds the "
            "model and never reads or alters its weights",
            ha="center", fontsize=7.2, color="0.35")
    save(fig, "fig_architecture_m1m6")


if __name__ == "__main__":
    print("redrawing the schematics whose content contradicted the thesis")
    threat_model()
    conceptual_framework()
    threat_taxonomy()
    phases()
    workflow()
    architecture()
    print(f"\noutput in {OUT}")
