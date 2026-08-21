# -*- coding: utf-8 -*-
"""
Slike publikacijske kvalitete za članak (MDPI Sensors) iz VelikiRun podataka.

Ulaz : ../data/*.csv  (+ tab_exp1_voronoi.csv iz analiza.py za sliku 3)
Izlaz: ../figures/fig*.png (600 dpi) + fig*.pdf (vektor)

Pokretanje:  python slike.py   (iz mape Analiza/)

Stil (vidi PLAN.md, FAZA 2 + Odluke 7-8):
  - engleske oznake, dva panela (a) 200 m / (b) 400 m,
  - krivulje po k: podskup k = 6, 8, 12, 20 + Global baseline,
  - paleta validirana dataviz validatorom (CVD-safe, uz markere kao
    sekundarno kodiranje jer su 3 boje ispod 3:1 kontrasta na bijelom).
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "figures")

# boja prati entitet (ista u svim slikama); markeri = sekundarno kodiranje
C_BASE = "#2a78d6"   # baseline (Global Voronoi / Grid Scanning)
C_ALT = "#eb6834"    # alternativa (SweepLine u fig4; 400 m u fig3)
COLOR_K = {6: "#eb6834", 8: "#1baf7a", 12: "#eda100", 20: "#e87ba4"}
MARKER_K = {6: "s", 8: "^", 12: "D", 20: "v"}
KSUB = [6, 8, 12, 20]

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
})

FULL_W = 6.7  # inča ~ 17 cm (puna širina MDPI stranice)


def load(alg, mode, space):
    df = pd.read_csv(os.path.join(DATA, f"{alg}{mode}{space}.csv"))
    return df


def mean_by(df, col, k=None):
    """Srednja vrijednost metrike po broju čvorova (preko 3 runa)."""
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


def plot_k_series(ax, base_df, local_df, col, base_label):
    """Baseline (Global) + Local za k iz podskupa.

    Local se crta silazno po k (krivulje za velike k poklapaju se s baselineom,
    a k=6 najviše odskače pa završava na vrhu snopa); Global ide zadnji,
    isprekidano, da ostane vidljiv i kad se krivulje preklapaju. Markeri su
    razmaknuti po serijama (markevery offset) da se ne gomilaju na istim x.
    """
    handles = {}
    for i, k in enumerate(reversed(KSUB)):
        m = mean_by(local_df, col, k)
        (handles[k],) = ax.plot(
            m.index, m.values, color=COLOR_K[k], marker=MARKER_K[k],
            markersize=3.5, markevery=(i, 4), label=f"Local, $k$ = {k}", zorder=3)
    m = mean_by(base_df, col)
    (hb,) = ax.plot(m.index, m.values, color=C_BASE, linestyle=(0, (4, 2)),
                    marker="o", markersize=3.5, markevery=(2, 4),
                    label=base_label, zorder=4)
    # legenda: baseline pa k uzlazno
    ax._legend_order = [hb] + [handles[k] for k in KSUB]


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=600)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)
    print(f"  {name}.png + .pdf")


def two_panel(ylabel, log=False):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
    for ax, panel, space in zip(axes, "ab", (200, 400)):
        style_ax(ax, panel, "Number of nodes", ylabel if panel == "a" else "")
        if log:
            ax.set_yscale("log")
    return fig, axes


def figlegend(fig, handles, labels, ncol, y=1.02):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=ncol, frameon=False, columnspacing=1.4, handletextpad=0.6)


# ----------------------------------------------------------------------------
print("Generiram slike u ../figures/ :")

# --- Fig 1: Ea vs N — Global vs Local(k), GridScan (Eksperiment 1) ----------
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
for ax, panel, space in zip(axes, "ab", (200, 400)):
    plot_k_series(ax, load("GridScan", "Global", space),
                  load("GridScan", "Local", space),
                  "AverageEstimationError", "Global Voronoi")
    style_ax(ax, panel, "Number of nodes",
             "Normalized estimation error $E_a$" if panel == "a" else "")
h = axes[0]._legend_order
figlegend(fig, h, [x.get_label() for x in h], ncol=5)
save(fig, "fig1_ea_vs_nodes")

# --- Fig 2: vrijeme Voronoi faze vs N — Global vs Local(k) ------------------
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
for ax, panel, space in zip(axes, "ab", (200, 400)):
    plot_k_series(ax, load("GridScan", "Global", space),
                  load("GridScan", "Local", space),
                  "VoronoiProc", "Global Voronoi")
    ax.set_yscale("log")
    style_ax(ax, panel, "Number of nodes",
             "Voronoi phase time (s)" if panel == "a" else "")
h = axes[0]._legend_order
figlegend(fig, h, [x.get_label() for x in h], ncol=5)
save(fig, "fig2_voronoi_time")

# --- Fig 3: kompromis točnost/brzina po k (svi k; iz tab_exp1) --------------
exp1 = pd.read_csv(os.path.join(HERE, "tab_exp1_voronoi.csv"))
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
for ax, panel, (col, ylab) in zip(
        axes, "ab",
        [("VoronoiSpeedup_pct", "Voronoi phase speed-up (%)"),
         ("dEa_pct", "Increase in $E_a$ vs. Global (%)")]):
    for space, color, mk in ((200, C_BASE, "o"), (400, C_ALT, "s")):
        e = exp1[exp1.Space == space].sort_values("k")
        ax.plot(e.k, e[col], color=color, marker=mk, markersize=4.5,
                label=f"{space} m × {space} m", zorder=3)
    ax.axvspan(10, 12, color=GRID, alpha=0.55, zorder=1)
    ax.set_xticks(range(6, 21, 2))
    style_ax(ax, panel, "Number of nearest neighbours $k$", ylab)
axes[1].annotate("recommended\n$k$ = 10–12", xy=(11, axes[1].get_ylim()[1] * 0.75),
                 ha="center", fontsize=7.5, color=MUTED)
h, l = axes[0].get_legend_handles_labels()
figlegend(fig, h, l, ncol=2)
save(fig, "fig3_tradeoff_k")

# --- Fig 4: vrijeme presjeka regija — GridScan vs SweepLine (Global) --------
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
for ax, panel, space in zip(axes, "ab", (200, 400)):
    for name, label, color, mk in (("GridScan", "Grid scanning", C_BASE, "o"),
                                   ("SweepLine", "Sweep line", C_ALT, "s")):
        m = mean_by(load(name, "Global", space), "IntersectionCalc")
        ax.plot(m.index, m.values, color=color, marker=mk, markersize=3.5,
                markevery=4, label=label, zorder=3)
    style_ax(ax, panel, "Number of nodes",
             "Region intersection time (s)" if panel == "a" else "")
h, l = axes[0].get_legend_handles_labels()
figlegend(fig, h, l, ncol=2)
save(fig, "fig4_intersection_time")

# --- Fig 5: kombinirana optimizacija (H4): TotalTime i Ea vs N --------------
fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.4))
for j, space in enumerate((200, 400)):
    base = load("GridScan", "Global", space)
    comb = load("SweepLine", "Local", space)
    for i, (col, ylab, log) in enumerate(
            [("TotalTime", "Total time (s)", True),
             ("AverageEstimationError", "Normalized estimation error $E_a$", False)]):
        ax = axes[i, j]
        plot_k_series(ax, base, comb, col, "Grid scan + Global")
        if log:
            ax.set_yscale("log")
        panel = "abcd"[i * 2 + j]
        style_ax(ax, panel, "Number of nodes" if i == 1 else "",
                 ylab if j == 0 else "")
h = axes[0, 0]._legend_order
l = [x.get_label().replace("Local", "Sweep line + Local") for x in h]
figlegend(fig, h, l, ncol=3, y=1.0)
fig.subplots_adjust(hspace=0.35)
save(fig, "fig5_combined_h4")

print("Gotovo.")
