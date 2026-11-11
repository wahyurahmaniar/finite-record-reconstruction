"""Shared figure conventions for the QIP manuscript.

Every figure is drawn at the same width (FIG_W inches) and included at the full
text width, so one point size gives one on-page size in all figures. Serif text
(Times New Roman when installed, its metric twin Liberation Serif otherwise),
STIX mathematics, bright pure colors, panel labels centered below each panel.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_W = 6.0
DPI = 600
COL = {
    "blue": "#0000FF",
    "red": "#FF0000",
    "green": "#00A000",
    "magenta": "#E000E0",
    "orange": "#FF8C00",
    "purple": "#8000FF",
    "cyan": "#00B0F0",
    "black": "#000000",
    "gray": "#808080",
}


def apply_style(base=9.5):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base,
        "legend.fontsize": base - 1,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.6",
        "legend.fancybox": False,
        "lines.linewidth": 1.5,
        "savefig.dpi": DPI,
    })


def panel_label(fig, ax, text, y, size=10.5):
    bb = ax.get_position()
    fig.text(0.5 * (bb.x0 + bb.x1), y, text, ha="center", va="bottom", fontsize=size)


LABEL_GAP = 16.0      # points between the lowest axis text of a panel and its panel label


def labels_below(fig, pairs, gap_pt=LABEL_GAP, size=10.5):
    """Center each panel label under its axes, all labels of a row on one line, gap_pt
    below the lowest tick label or axis label of the row. Returns the bottom of the
    labels in figure coordinates."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    low = min(ax.get_tightbbox(r).y0 for ax, _ in pairs)
    y = (low - gap_pt * fig.dpi / 72.0) / fig.bbox.height
    texts = []
    for ax, lab in pairs:
        bb = ax.get_position()
        texts.append(fig.text(0.5 * (bb.x0 + bb.x1), y, lab, ha="center", va="top",
                              fontsize=size))
    fig.canvas.draw()
    bottom = min(t.get_window_extent(r).y0 for t in texts) / fig.bbox.height
    assert bottom > 0, "panel label falls below the canvas"
    return bottom


def save(fig, path):
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
