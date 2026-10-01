# -*- coding: utf-8 -*-
"""
P8 (Major Revision 1, R1-7): revidirane slike.

- fig1_ea_vs_nodes_v2  (Figura 4 rada): 2x2 — gornji red apsolutni Ea (kao dosad),
  donji red RAZLIKA (Ea,local − Ea,global)/Ea,global u % (zahtjev recenzenta 1).
- fig5_combined_h4_v2  (Figura 8 rada): k=6 crtkano (patoloski slucaj vizualno
  izdvojen); ostalo identicno.
Stil identican Analiza/slike.py. Izlaz u ../Slike/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "figures")

C_BASE = "#2a78d6"
COLOR_K = {6: "#eb6834", 8: "#1baf7a", 12: "#eda100", 20: "#e87ba4"}
MARKER_K = {6: "s", 8: "^", 12: "D", 20: "v"}
KSUB = [6, 8, 12, 20]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.bbox": "tight",
})

FULL_W = 6.7


def load(alg, mode, space):
    return pd.read_csv(os.path.join(DATA, f"{alg}{mode}{space}.csv"))


def mean_by(df, col, k=None):
    if k is not None:
        df = df[df.k_neighbors == k]
    return df.groupby("NumberOfNodes")[col].mean()


def style_ax(ax, panel, xlabel, ylabel):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"({panel})", loc="left", fontweight="bold")
    ax.margins(x=0.02)


def plot_k_series(ax, base_df, local_df, col, base_label, dashed_k=()):
    handles = {}
    for i, k in enumerate(reversed(KSUB)):
        m = mean_by(local_df, col, k)
        # k=6 (patoloski slucaj): tockasta (dotted) linija, deblja, iznad ostalih;
        # baseline je crtkan pa dotted ostaje jednoznacan
        ls = (0, (1.2, 1.6)) if k in dashed_k else "-"
        lw = 2.4 if k in dashed_k else plt.rcParams["lines.linewidth"]
        (handles[k],) = ax.plot(
            m.index, m.values, color=COLOR_K[k], marker=MARKER_K[k], linestyle=ls,
            linewidth=lw, markersize=3.5, markevery=(i, 4),
            label=f"Local, $k$ = {k}", zorder=3.5 if k in dashed_k else 3)
    m = mean_by(base_df, col)
    (hb,) = ax.plot(m.index, m.values, color=C_BASE, linestyle=(0, (4, 2)),
                    marker="o", markersize=3.5, markevery=(2, 4),
                    label=base_label, zorder=4)
    ax._legend_order = [hb] + [handles[k] for k in KSUB]


def figlegend(fig, handles, labels, ncol, y=1.02):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=ncol, frameon=False, columnspacing=1.4, handletextpad=0.6)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=600)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)
    print(f"  {name}.png + .pdf")


print("Generiram revidirane slike u ../Slike/ :")

# --- Figura 4 rada (v2): 2x2 — apsolutni Ea + relativna razlika --------------
fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.4))
for j, space in enumerate((200, 400)):
    base = load("GridScan", "Global", space)
    local = load("GridScan", "Local", space)
    # (a, b): apsolutni Ea
    ax = axes[0, j]
    plot_k_series(ax, base, local, "AverageEstimationError", "Global Voronoi")
    style_ax(ax, "ab"[j], "",
             "Normalized estimation error $E_a$" if j == 0 else "")
    # (c, d): relativna razlika u %
    ax = axes[1, j]
    mb = mean_by(base, "AverageEstimationError")
    for i, k in enumerate(reversed(KSUB)):
        ml = mean_by(local, "AverageEstimationError", k)
        rel = 100.0 * (ml - mb) / mb
        ax.plot(rel.index, rel.values, color=COLOR_K[k], marker=MARKER_K[k],
                markersize=3.5, markevery=(i, 4), zorder=3)
    ax.axhline(0.0, color=C_BASE, linestyle=(0, (4, 2)), linewidth=1.2, zorder=2)
    style_ax(ax, "cd"[j], "Number of nodes",
             "$E_a$ increase vs. Global (%)" if j == 0 else "")
h = axes[0, 0]._legend_order
figlegend(fig, h, [x.get_label() for x in h], ncol=5, y=1.0)
fig.subplots_adjust(hspace=0.3)
save(fig, "fig1_ea_vs_nodes_v2")

# --- Figura 8 rada (v2): kombinirana optimizacija, k=6 crtkano ---------------
fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.4))
for j, space in enumerate((200, 400)):
    base = load("GridScan", "Global", space)
    comb = load("SweepLine", "Local", space)
    for i, (col, ylab, log) in enumerate(
            [("TotalTime", "Total time (s)", True),
             ("AverageEstimationError", "Normalized estimation error $E_a$", False)]):
        ax = axes[i, j]
        plot_k_series(ax, base, comb, col, "Grid scan + Global", dashed_k=(6,))
        if log:
            ax.set_yscale("log")
        panel = "abcd"[i * 2 + j]
        style_ax(ax, panel, "Number of nodes" if i == 1 else "",
                 ylab if j == 0 else "")
h = axes[0, 0]._legend_order
l = [x.get_label().replace("Local", "Sweep line + Local") for x in h]
figlegend(fig, h, l, ncol=3, y=1.0)
fig.subplots_adjust(hspace=0.35)
save(fig, "fig5_combined_h4_v2")

print("Gotovo.")
