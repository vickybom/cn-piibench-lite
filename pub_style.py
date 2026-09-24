"""Publication styling shared by every figure script in this project.

The figures in the first version of the thesis were authored at 8-9 inches
wide, saved at the matplotlib default resolution, and then scaled down to the
6-inch text block when they were placed in the document. That has two
consequences an IEEE or ACM reviewer would flag immediately: the effective
resolution lands between 170 and 220 dpi against a 600 dpi requirement for line
art, and the type is reduced to roughly two thirds of its authored size, which
puts most axis labels under 7 pt.

Importing this module and calling `finish(fig)` before saving fixes both. The
figure is rescaled to exactly the text-block width, so nothing is scaled at
placement time and 9 pt authored is 9 pt printed.

    import pub_style
    ...
    pub_style.finish(fig)
    fig.savefig(path)          # dpi comes from rcParams
"""
from __future__ import annotations

import matplotlib as mpl

# 6.0 in is the thesis text block. Rendering at 700 dpi leaves headroom so that
# bbox_inches="tight", which trims the canvas and therefore the pixel count,
# still delivers more than 600 dpi across the placed width.
TEXT_WIDTH_IN = 6.0
RENDER_DPI = 700

RC = {
    "figure.dpi": 150,
    "savefig.dpi": RENDER_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,          # TrueType: text stays selectable and editable
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "legend.frameon": False,
}

# Paul Tol bright: colour-blind safe. Kept here so every script draws from one
# palette rather than three different ad-hoc ones.
BLUE, RED, GREEN, YELLOW, CYAN, PURPLE, GREY = (
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB")
BAND = {"Low": GREEN, "Medium": YELLOW, "High": RED}


def apply():
    mpl.rcParams.update(RC)


def finish(fig, width_in: float = TEXT_WIDTH_IN, max_height_in: float = 7.6):
    """Rescale to the text-block width, preserving aspect, before saving.

    Height is capped so that a tall figure cannot run past a single page once
    the caption is placed underneath it.
    """
    w, h = fig.get_size_inches()
    scale = width_in / w
    new_h = h * scale
    if new_h > max_height_in:
        scale = max_height_in / h
        width_in = w * scale
        new_h = max_height_in
    fig.set_size_inches(width_in, new_h)
    return fig


apply()
