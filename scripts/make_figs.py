"""Figures of the manuscript.

fig1_protocol.png    protocol: one copy, destructive measurement, classical record, preparation
fig2_bounds.png      visible bounds; two-state optimum V(c); ellipse of achievable states
fig3_circuit.png     circuit realization of the two-state optimum
fig4_baselines.png   comparisons: two states against overlap; trine and tetrahedron against d
fig5_particle.png    one-particle realization: temporal-mode sorting and shaped emission

Numerical inputs come from data/results_*.json (written by scripts/verify.py);
the figures are written to figures/.
"""
import json
import os
import sys
from math import comb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import (Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Patch,
                                Rectangle)
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # figstyle.py sits beside this script (also under python -I)
from figstyle import COL, FIG_W, apply_style, labels_below, save  # noqa: E402

NUM = os.path.join(HERE, "..", "data")
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figures"))
os.makedirs(OUT, exist_ok=True)
apply_style()
LAB = 10.5          # panel-label size, identical in every figure (all drawn at FIG_W)


def load(name):
    with open(os.path.join(NUM, f"results_{name}.json")) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ analytics
K, RATIO = 16, 0.8
prior = RATIO ** np.arange(K)
prior /= prior.sum()
cum = np.cumsum(prior)


def g_fun(p, c):
    return (1 - c * c) * np.cos(p) - c * np.sin(p) + c * c * np.sin(p) * np.cos(p)


def phistar(c):
    return brentq(lambda p: g_fun(p, c), 0.0, np.pi / 2, xtol=1e-15)


def V(c):
    if c <= 0 or c >= 1:
        return 0.0
    S = np.sqrt(1 - c * c)
    p = phistar(c)
    return 0.5 * np.sqrt(S * S * (1 - np.sin(p)) ** 2 + (c - np.cos(p)) ** 2)


def gp(c):
    S = np.sqrt(1 - c * c)
    return 0.5 * (1 - S) * S


def fidopt(c):
    S = np.sqrt(1 - c * c)
    p = np.arctan(S * S / c)
    return 0.5 * np.sqrt(S * S * (1 - np.sin(p)) ** 2 + (c - np.cos(p)) ** 2)


cgrid = np.linspace(0, 1, 801)
Vg = np.array([V(c) for c in cgrid])


# ------------------------------------------------------------------ drawing helpers
def canvas(h):
    fig = plt.figure(figsize=(FIG_W, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, ec="black", fc="white", lw=1.0, r=0.04, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, zorder=z))


def qwire(ax, xs, ys, color="black", lw=1.1, z=2):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=z)


def cwire(ax, xs, ys, color=COL["red"], z=2):
    """Classical wire: two parallel strokes (the circuit convention)."""
    ax.plot(xs, ys, color=color, lw=3.0, solid_capstyle="butt", solid_joinstyle="miter", zorder=z)
    ax.plot(xs, ys, color="white", lw=1.2, solid_capstyle="butt", solid_joinstyle="miter",
            zorder=z + 0.1)


def arrow(ax, p, q, color="black", lw=1.1, ms=9, z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=z))


def meter(ax, x, y, w, h, color="black"):
    box(ax, x, y, w, h, ec=color)
    cx, cy = x + w / 2, y + 0.26 * h
    ax.add_patch(Arc((cx, cy), 0.62 * w, 0.62 * w, theta1=0, theta2=180, lw=1.0, color=color,
                     zorder=4))
    arrow(ax, (cx, cy), (cx + 0.30 * w, cy + 0.42 * h), color=color, lw=1.0, ms=6)


def bloch(ax, x, y, r, ang, color, label=None, lab_xy=None, fs=10):
    ax.add_patch(Circle((x, y), r, fc=color + "14", ec=color, lw=1.0, zorder=3))
    th = np.linspace(0, np.pi, 60)
    ax.plot(x + r * np.cos(th), y - 0.30 * r * np.sin(th), color=color, lw=0.7, zorder=3)
    ax.plot(x + r * np.cos(th), y + 0.30 * r * np.sin(th), color=color, lw=0.7, ls=(0, (2, 2)),
            zorder=3)
    ax.plot([x], [y], "o", color=color, ms=2.2, zorder=5)
    arrow(ax, (x, y), (x + 0.92 * r * np.cos(ang), y + 0.92 * r * np.sin(ang)), color=color,
          lw=1.2, ms=7, z=5)
    if label:
        ax.text(*lab_xy, label, ha="center", va="center", fontsize=fs, color=color, zorder=6)


def packet(ax, x0, x1, yb, amp, color, wl=0.13):
    x = np.linspace(x0, x1, 600)
    xc, sig = 0.5 * (x0 + x1), 0.17 * (x1 - x0)
    env = amp * np.exp(-((x - xc) ** 2) / (2 * sig ** 2))
    ax.fill_between(x, yb - env, yb + env, color=color, alpha=0.10, lw=0, zorder=2)
    ax.plot(x, yb + env * np.cos(2 * np.pi * (x - xc) / wl), color=color, lw=1.1, zorder=3)
    ax.plot(x, yb + env, color=color, lw=0.6, alpha=0.6, zorder=3)
    ax.plot(x, yb - env, color=color, lw=0.6, alpha=0.6, zorder=3)


def detector(ax, x, y, s, color="black", fired=False):
    ax.add_patch(Rectangle((x, y - s / 2), s * 0.35, s, fc="white", ec=color, lw=0.9, zorder=4))
    ax.add_patch(Arc((x + s * 0.35, y), s, s, theta1=-90, theta2=90, color=color, lw=0.9,
                     zorder=4))
    if fired:
        ax.add_patch(Circle((x + s * 0.47, y), s * 0.17, fc=COL["red"], ec="none", zorder=5))


def heading(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color="black")


# ------------------------------------------------------------------ Fig. 1
def fig1():
    H = 2.25
    fig, ax = canvas(H)
    yq = 1.10
    ax.text(1.62, 2.12, "ORIGIN", ha="center", va="center", fontsize=9, color="black")
    ax.text(4.95, 2.12, "DESTINATION", ha="center", va="center", fontsize=9, color="black")
    ax.plot([3.42, 3.42], [0.12, 1.98], color="0.7", lw=1.0, ls=(0, (4, 3)), zorder=1)
    # source
    heading(ax, 0.45, 1.72, "source")
    bloch(ax, 0.45, yq, 0.30, np.deg2rad(62), COL["blue"], r"$\rho_i$", (0.45, 0.62))
    qwire(ax, [0.75, 1.02], [yq, yq])
    # destructive measurement
    heading(ax, 1.30, 1.72, "acquisition")
    meter(ax, 1.02, yq - 0.24, 0.56, 0.48)
    ax.text(1.30, 0.62, r"POVM $\{Q_m\}$", ha="center", va="center", fontsize=9.5)
    # classical record
    cwire(ax, [1.58, 1.92], [yq, yq])
    heading(ax, 2.37, 1.72, "classical record")
    labels = ["1", "2", r"$\cdots$", r"$d$"]
    for k, lab in enumerate(labels):
        x0 = 1.92 + 0.225 * k
        fc = COL["red"] + "33" if k == 1 else "white"
        ax.add_patch(Rectangle((x0, yq - 0.16), 0.225, 0.32, fc=fc, ec=COL["red"], lw=0.9, zorder=4))
        ax.text(x0 + 0.1125, yq, lab, ha="center", va="center", fontsize=9, zorder=5)
    ax.text(2.37, 0.62, r"$m\in\{1,\ldots,d\}$", ha="center", va="center", fontsize=9.5,
            color=COL["red"])
    # transmission of the record value
    cwire(ax, [2.82, 4.05], [yq, yq])
    ax.text(3.42, yq + 0.20, "record value only", ha="center", va="bottom", fontsize=8.5,
            color=COL["red"], bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)
    # preparation
    heading(ax, 4.45, 1.72, "preparation")
    box(ax, 4.05, yq - 0.24, 0.80, 0.48)
    ax.text(4.45, yq, r"prepare $\tau_m$", ha="center", va="center", fontsize=9.5, zorder=5)
    # local resource: an excited emitter supplies the excitation
    ax.plot([4.27, 4.63], [0.40, 0.40], color=COL["green"], lw=1.4, zorder=3)
    ax.plot([4.27, 4.63], [0.16, 0.16], color=COL["green"], lw=1.4, zorder=3)
    ax.add_patch(Circle((4.45, 0.40), 0.045, fc=COL["green"], ec="none", zorder=4))
    arrow(ax, (4.45, 0.45), (4.45, yq - 0.24), color=COL["green"], lw=1.1, ms=8)
    ax.text(4.72, 0.33, "local resources:\nexcitation, energy,\ncharges", ha="left", va="center",
            fontsize=8.0, color=COL["green"], linespacing=1.15)
    # output
    qwire(ax, [4.85, 5.24], [yq, yq])
    heading(ax, 5.55, 1.72, "output")
    bloch(ax, 5.55, yq, 0.30, np.deg2rad(70), COL["purple"], r"$\Phi(\rho_i)$", (5.55, 0.62))
    save(fig, os.path.join(OUT, "fig1_protocol.png"))


# ------------------------------------------------------------------ Fig. 2
def fig2():
    fig = plt.figure(figsize=(FIG_W, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.80], left=0.085, right=0.99,
                          bottom=0.30, top=0.97, wspace=0.42)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    # (a) visible bounds
    d = np.arange(1, K + 1)
    up = 1 - cum
    lo90 = np.maximum(0, 0.90 - cum)
    lo75 = np.maximum(0, 0.75 - cum)
    ax1.fill_between(d, lo90, up, color=COL["blue"], alpha=0.08, lw=0)
    ax1.plot(d, up, "-o", color="black", ms=3.6, lw=1.3, label="upper bound")
    ax1.plot(d, up, "s", mfc="none", mec=COL["red"], mew=1.0, ms=5.4, label=r"$P_{\rm guess}=1$")
    ax1.plot(d, lo90, "--^", color=COL["blue"], ms=3.6, lw=1.2, label=r"$P_{\rm guess}=0.90$")
    ax1.plot(d, lo75, ":D", color=COL["green"], ms=3.2, lw=1.4, label=r"$P_{\rm guess}=0.75$")
    ax1.set_xlim(0.4, 16.6)
    ax1.set_ylim(-0.03, 1.12)
    ax1.set_xticks([1, 4, 8, 12, 16])
    ax1.set_xlabel(r"record size $d$")
    ax1.set_ylabel(r"visible distortion $\Delta^{\rm vis}_d$")
    ax1.legend(loc="upper right", fontsize=7.6, handlelength=1.8, borderpad=0.3,
               labelspacing=0.25, handletextpad=0.4)
    # (b) two-state optimum
    ax2.plot(cgrid, [gp(c) for c in cgrid], "--", color=COL["blue"], lw=1.4,
             label="guess-and-prepare")
    ax2.plot(cgrid, Vg, "-", color=COL["red"], lw=2.0, label=r"optimum $V(c)$")
    ax2.plot([0, 1], [0, 0], ":", color="black", lw=1.6, label="visible benchmark")
    cs = 0.861526
    ax2.plot([cs], [V(cs)], "o", color=COL["red"], ms=5, mec="white", mew=0.6, zorder=5)
    ax2.plot([np.sqrt(3) / 2], [0.125], "o", color=COL["blue"], ms=5, mec="white", mew=0.6,
             zorder=5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.008, 0.205)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    ax2.set_xlabel(r"overlap $c$")
    ax2.set_ylabel("average distortion")
    ax2.legend(loc="upper left", fontsize=7.6, handlelength=1.9, borderpad=0.3,
               labelspacing=0.25, handletextpad=0.4)
    # (c) ellipse of achievable symmetrized states at c = 0.861
    c = 0.861
    S = np.sqrt(1 - c * c)
    p = phistar(c)
    ax3.set_aspect("equal")
    ax3.axis("off")
    ax3.set_xlim(-1.12, 1.12)
    ax3.set_ylim(-2.45, 1.15)
    th = np.linspace(0, 2 * np.pi, 400)
    ax3.plot(np.cos(th), np.sin(th), color="0.65", lw=0.9)
    ax3.fill(S * np.sin(th), np.cos(th), color=COL["orange"], alpha=0.18, lw=0)
    ax3.plot(S * np.sin(th), np.cos(th), color=COL["orange"], lw=1.1)
    ax3.plot([-1.05, 1.05], [0, 0], color="0.85", lw=0.6, zorder=0)
    ax3.plot([0, 0], [-1.05, 1.05], color="0.85", lw=0.6, zorder=0)
    ax3.plot([-S, S], [c, c], "o", color=COL["blue"], ms=6, zorder=5)
    ax3.plot([-np.sin(p), np.sin(p)], [np.cos(p)] * 2, "s", color=COL["red"], ms=5.5, zorder=5)
    ax3.plot([-S * np.sin(p), S * np.sin(p)], [np.cos(p)] * 2, "D", mfc="white", mec=COL["red"],
             mew=1.2, ms=4.8, zorder=6)
    hs = [Line2D([], [], ls="", marker="o", color=COL["blue"], ms=6, label="targets"),
          Line2D([], [], ls="", marker="s", color=COL["red"], ms=5.5, label="prepared"),
          Line2D([], [], ls="", marker="D", mfc="white", mec=COL["red"], mew=1.2, ms=4.8,
                 label="rebuilt"),
          Patch(fc=(1.0, 0.549, 0.0, 0.18), ec=COL["orange"], lw=1.1, label="achievable")]
    ax3.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.03), frameon=False,
               fontsize=7.6, handletextpad=0.7, labelspacing=0.3, ncol=1, handlelength=1.4)
    labels_below(fig, [(ax1, "(a)"), (ax2, "(b)"), (ax3, "(c)")], size=LAB)
    save(fig, os.path.join(OUT, "fig2_bounds.png"))


# ------------------------------------------------------------------ Fig. 3
def fig3():
    H = 2.0
    fig, ax = canvas(H)
    yt, yb = 1.30, 0.52
    ax.text(1.55, 1.88, "source processor", ha="center", va="center", fontsize=9, color="black")
    ax.text(4.55, 1.88, "destination processor", ha="center", va="center", fontsize=9,
            color="black")
    ax.plot([3.10, 3.10], [0.10, 1.74], color="0.7", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.50, yt, r"$|\psi_i\rangle$", ha="right", va="center", fontsize=11)
    ax.text(0.50, yb, r"$|0\rangle$", ha="right", va="center", fontsize=11)
    qwire(ax, [0.58, 1.10], [yt, yt])
    box(ax, 1.10, yt - 0.19, 0.38, 0.38)
    ax.text(1.29, yt, r"$H$", ha="center", va="center", fontsize=11, zorder=5)
    qwire(ax, [1.48, 1.78], [yt, yt])
    meter(ax, 1.78, yt - 0.19, 0.46, 0.38)
    ax.text(1.66, yt - 0.36, "measurement in the $X$ basis", ha="center", va="top", fontsize=8.5,
            color="black")
    cwire(ax, [2.24, 4.40, 4.40], [yt, yt, yb + 0.21])
    ax.add_patch(Circle((4.40, yb + 0.21), 0.035, fc=COL["red"], ec="none", zorder=5))
    ax.text(3.24, yt + 0.09, r"$m\in\{+,{-}\}$, one bit", ha="left", va="bottom", fontsize=9.5,
            color=COL["red"])
    qwire(ax, [0.58, 3.98], [yb, yb])
    box(ax, 3.98, yb - 0.21, 0.84, 0.42)
    ax.text(4.40, yb, r"$R_y(\pm\varphi^\star)$", ha="center", va="center", fontsize=10.5,
            zorder=5)
    qwire(ax, [4.82, 5.40], [yb, yb])
    ax.text(5.47, yb, r"$|\tau_m\rangle$", ha="left", va="center", fontsize=11)
    ax.text(4.40, yb - 0.30, "feed-forward rotation", ha="center", va="top", fontsize=8.5,
            color="black")
    save(fig, os.path.join(OUT, "fig3_circuit.png"))


# ------------------------------------------------------------------ Fig. 4
def fig4():
    sdp = load("sdp")
    tri = load("trine")
    sic = load("sic")
    H = 5.60                       # inches; panel heights and margins below are in inches
    fig = plt.figure(figsize=(FIG_W, H))
    ax1 = fig.add_axes([0.085, (H - 0.07 - 1.63) / H, 0.50, 1.63 / H])
    ax2 = fig.add_axes([0.085, 0.735 / H, 0.385, 1.53 / H])
    ax3 = fig.add_axes([0.600, 0.735 / H, 0.385, 1.53 / H])
    # (a) two pure states against overlap
    cc = cgrid[1:-1]
    S = np.sqrt(1 - cc ** 2)
    ax1.plot(cc, 0.5 * cc * S, ":", color=COL["green"], lw=1.6,
             label="unambiguous, then prepare ($d\\geq3$)")
    ax1.plot(cc, S / 2 - S ** 2 / 4, ls=(0, (5, 1.5, 1, 1.5, 1, 1.5)), color=COL["magenta"], lw=1.3,
             label="Haar-random basis ($d=2$)")
    ax1.plot(cc, [fidopt(c) for c in cc], "-.", color=COL["orange"], lw=1.5,
             label="fidelity-optimal ($d=2$)")
    ax1.plot(cc, [gp(c) for c in cc], "--", color=COL["blue"], lw=1.4,
             label="guess-and-prepare ($d=2$)")
    ax1.plot(cgrid, Vg, "-", color=COL["red"], lw=2.0, label="optimum $V(c)$ ($d\\geq2$)")
    rows = [r for r in sdp["rows"] if round(r["c"], 4) in (0.1, 0.3, 0.5, 0.7, 0.8615, 0.95)]
    ax1.plot([r["c"] for r in rows], [r["sdp"] for r in rows], "o", mfc="none", mec="black",
             mew=1.0, ms=5.5, label="SDP, all channels")
    ax1.plot([0, 1], [0, 0], ":", color="black", lw=1.6, label="visible benchmark")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.012, 0.27)
    ax1.set_xlabel(r"overlap $c$")
    ax1.set_ylabel("average distortion")
    h1, l1 = ax1.get_legend_handles_labels()
    b1 = ax1.get_position()
    fig.legend(h1, l1, loc="center left", bbox_to_anchor=(0.605, 0.5 * (b1.y0 + b1.y1)),
               fontsize=8.0, handlelength=2.6, borderpad=0.4, labelspacing=0.35)
    # (b) trine, (c) tetrahedron, against the record size
    for ax, res, K_, title_bound in ((ax2, tri, 3, 0.25), (ax3, sic, 4, 1 / 3)):
        dd = np.arange(1, 6)
        scan = [res["scan"][str(d)] if str(d) in res["scan"] else res["scan"][str(K_)]
                for d in dd]
        vis = [res["vis"][str(d)] if str(d) in res["vis"] else 0.0 for d in dd]
        bound = [0.5] + [title_bound] * 4
        ax.plot(dd, vis, ":s", color="black", lw=1.5, ms=4.5)
        ax.plot(dd, bound, "--", color=COL["blue"], lw=1.4)
        ax.plot(dd, scan, "-o", color=COL["red"], lw=1.8, ms=5)
        ax.set_xticks(dd)
        ax.set_xlim(0.6, 5.4)
        ax.set_xlabel(r"record size $d$")
        ax.set_ylabel("average distortion")
    ax2.set_ylim(-0.03, 0.58)
    ax3.set_ylim(-0.03, 0.58)
    # baselines in both panels: fidelity-optimal two-outcome protocols (bar at d = 2),
    # fidelity-optimal measurement with K outcomes (triangle at d = K), Haar-random basis
    # averaged over fixed bases (cross, two outcomes) and as a randomized channel (plus,
    # the guessed label sent with K record values); DX separates the Haar markers from
    # the bar and the optima drawn at the same record size
    DX = 0.16
    for ax, res, K_, fopt in ((ax2, tri, 3, 0.25), (ax3, sic, 4, 1 / 3)):
        ax.plot([2, 2], [res["fid2_min"], res["fid2_max"]], "-", color=COL["orange"], lw=4,
                alpha=0.55, solid_capstyle="butt")
        ax.plot([K_], [fopt], "^", color=COL["orange"], ms=7, mec="black", mew=0.5, zorder=6)
        ax.plot([2 + DX], [res["haar_fixed"]], "x", color=COL["magenta"], ms=7, mew=1.6,
                zorder=6)
        ax.plot([K_ + DX], [res["haar_rand"]], "+", color=COL["magenta"], ms=8, mew=1.6,
                zorder=6)
    hs = [Line2D([], [], ls=":", marker="s", color="black", lw=1.5, ms=4.5,
                 label=r"visible $\Delta^{\rm vis}_d$"),
          Line2D([], [], ls="-", marker="o", color=COL["red"], lw=1.8, ms=5,
                 label=r"acquisition-limited $\Delta^{\rm scan}_d$"),
          Line2D([], [], ls="--", color=COL["blue"], lw=1.4, label=r"bound $1-F_{{\rm acc},d}$"),
          Line2D([], [], ls="-", color=COL["orange"], lw=4, alpha=0.55,
                 label="fidelity-optimal, two outcomes"),
          Line2D([], [], ls="", marker="^", color=COL["orange"], ms=7, mec="black", mew=0.5,
                 label=r"fidelity-optimal, $K$ outcomes"),
          Line2D([], [], ls="", marker="x", color=COL["magenta"], ms=7, mew=1.6,
                 label="Haar basis, average"),
          Line2D([], [], ls="", marker="+", color=COL["magenta"], ms=8, mew=1.6,
                 label="Haar basis, channel")]
    ya = labels_below(fig, [(ax1, "(a)")], size=LAB)
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.535, ya - 10.0 / 72 / H),
               ncol=3, fontsize=7.8, handlelength=2.2, borderpad=0.4, labelspacing=0.35,
               columnspacing=1.0)
    labels_below(fig, [(ax2, "(b)"), (ax3, "(c)")], size=LAB)
    save(fig, os.path.join(OUT, "fig4_baselines.png"))


# ------------------------------------------------------------------ Fig. 5
def fig5():
    H = 2.45
    fig, ax = canvas(H)
    yq = 1.12
    ax.text(1.55, 2.33, "ORIGIN", ha="center", va="center", fontsize=9, color="black")
    ax.text(4.75, 2.33, "DESTINATION", ha="center", va="center", fontsize=9, color="black")
    ax.plot([3.20, 3.20], [0.10, 2.20], color="0.7", lw=1.0, ls=(0, (4, 3)), zorder=1)
    # incoming one-particle wave packet
    packet(ax, 0.08, 1.12, yq, 0.26, COL["blue"])
    ax.text(0.60, 1.58, r"$|1_{f_i}\rangle$", ha="center", va="center", fontsize=11,
            color=COL["blue"])
    arrow(ax, (1.14, yq), (1.36, yq), color=COL["blue"])
    # temporal-mode sorter with one detector per mode
    heading(ax, 1.76, 1.98, "mode-resolving measurement")
    box(ax, 1.36, yq - 0.42, 0.78, 0.84)
    ax.text(1.75, yq + 0.13, "temporal-", ha="center", va="center", fontsize=8.5, zorder=5)
    ax.text(1.75, yq - 0.11, "mode sorter", ha="center", va="center", fontsize=8.5, zorder=5)
    for k, yy in enumerate((yq + 0.28, yq, yq - 0.28)):
        qwire(ax, [2.14, 2.34], [yy, yy])
        detector(ax, 2.34, yy, 0.20, fired=(k == 1))
    ax.text(2.46, yq - 0.56, r"outcome $m$", ha="center", va="center", fontsize=9.5,
            color=COL["red"])
    # classical record to the control of the emitter
    cwire(ax, [2.56, 3.75, 3.75, 4.12], [yq, yq, 1.88, 1.88])
    ax.text(3.20, yq + 0.10, "record $m$", ha="center", va="bottom", fontsize=8.5,
            color=COL["red"], bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)
    # control-pulse generator
    box(ax, 4.12, 1.70, 0.86, 0.36)
    ax.text(4.55, 1.88, r"pulse $u_m(t)$", ha="center", va="center", fontsize=9, zorder=5)
    arrow(ax, (4.55, 1.70), (4.55, 1.40), color="black", lw=1.0, ms=7)
    # cavity with a two-level emitter that holds the local excitation
    from matplotlib.patches import Polygon
    ys = np.linspace(yq - 0.38, yq + 0.38, 40)
    for x0, sgn in ((4.12, 1), (4.98, -1)):     # concave faces towards the emitter
        face = x0 + sgn * 0.45 * (ys - yq) ** 2
        pts = [(x0 - sgn * 0.08, ys[0])] + list(zip(face, ys)) + [(x0 - sgn * 0.08, ys[-1])]
        ax.add_patch(Polygon(pts, closed=True, fc="0.80", ec="0.30", lw=0.9, zorder=4))
    ax.plot([4.38, 4.72], [yq + 0.15, yq + 0.15], color=COL["green"], lw=1.5, zorder=5)
    ax.plot([4.38, 4.72], [yq - 0.15, yq - 0.15], color=COL["green"], lw=1.5, zorder=5)
    ax.add_patch(Circle((4.55, yq + 0.15), 0.045, fc=COL["green"], ec="none", zorder=6))
    ax.text(4.77, yq + 0.15, r"$|e\rangle$", ha="left", va="center", fontsize=9,
            color=COL["green"])
    ax.text(4.77, yq - 0.15, r"$|g\rangle$", ha="left", va="center", fontsize=9,
            color=COL["green"])
    ax.text(4.55, yq - 0.60, "local emitter", ha="center", va="center", fontsize=8.5,
            color=COL["green"])
    # emitted wave packet in the selected mode
    arrow(ax, (5.07, yq), (5.22, yq), color=COL["purple"])
    packet(ax, 5.20, 5.98, yq, 0.24, COL["purple"], wl=0.11)
    ax.text(5.60, 1.58, r"$|1_{f_m}\rangle$", ha="center", va="center", fontsize=11,
            color=COL["purple"])
    save(fig, os.path.join(OUT, "fig5_particle.png"))


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        globals()["fig" + w]()
    print("figures written:", ", ".join(which))
