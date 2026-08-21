# -*- coding: utf-8 -*-
"""
Ilustrativne (pedagoške) slike za Methods — zamjena za slike iz drafta
(image1–image4 iz `Slike/iz_drafta/`), regenerirane u konzistentnom stilu.

Izlaz (u ../figures/):
  ill_voronoi_diagram  — primjer Voronoi dijagrama (zamjena za image1)
  ill_grouping_g       — dijagram s generatorima p1..p6 i tockom g (image2)
  ill_local_voronoi    — (a) odabir k najblizih susjeda u punom dijagramu,
                         (b) lokalni Voronoi samo iz k susjeda (image3+image4)

Fortuneov algoritam (image5) NE regeneriramo — zadrzati original iz drafta.

Pokretanje:  python slike_ilustracije.py   (iz mape Analiza/)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")

INK = "#0b0b0b"
MUTED = "#52514e"
STAR = "#eda100"
# mekane ispune iz kategoričke palete (identitet regija je nebitan — samo razlikovanje)
FILLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#4a3aa7", "#e34948"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
})

FULL_W = 6.7


def bounded_voronoi(pts, bbox):
    """Voronoi s konačnim regijama: točke se zrcale preko rubova okvira."""
    x0, x1, y0, y1 = bbox
    mirrors = [np.c_[2 * x0 - pts[:, 0], pts[:, 1]],
               np.c_[2 * x1 - pts[:, 0], pts[:, 1]],
               np.c_[pts[:, 0], 2 * y0 - pts[:, 1]],
               np.c_[pts[:, 0], 2 * y1 - pts[:, 1]]]
    return Voronoi(np.vstack([pts] + mirrors))


def region_polygon(vor, i):
    """Poligon regije i-te ULAZNE točke (konačan zbog zrcaljenja)."""
    reg = vor.regions[vor.point_region[i]]
    return vor.vertices[reg]


def new_ax(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.8)
    return fig, ax


def draw_cells(ax, vor, n, bbox, facecolors=None, edge=INK, lw=1.0):
    for i in range(n):
        poly = region_polygon(vor, i)
        fc = facecolors(i) if facecolors else "none"
        ax.fill(poly[:, 0], poly[:, 1], facecolor=fc, edgecolor=edge,
                linewidth=lw, zorder=2)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=600)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)
    print(f"  {name}.png + .pdf")


rng = np.random.default_rng(4275)  # isti seed kao simulacije, radi prepoznatljivosti
BBOX = (0.0, 1.0, 0.0, 1.0)


def sample_min_dist(n, lo, hi, dmin):
    """n slučajnih točaka s minimalnim međusobnim razmakom (rejection sampling)."""
    pts = []
    while len(pts) < n:
        cand = rng.uniform(lo, hi, size=2)
        if all(np.hypot(*(cand - p)) >= dmin for p in pts):
            pts.append(cand)
    return np.array(pts)

print("Generiram ilustracije u ../figures/ :")

# --- 1) Primjer Voronoi dijagrama (zamjena za image1) ------------------------
pts = sample_min_dist(8, 0.05, 0.95, 0.28)
vor = bounded_voronoi(pts, BBOX)
fig, ax = new_ax((3.3, 3.3))
draw_cells(ax, vor, len(pts), BBOX,
           facecolors=lambda i: FILLS[i % len(FILLS)] + "55")
ax.plot(pts[:, 0], pts[:, 1], "o", color=INK, markersize=4, zorder=3)
save(fig, "ill_voronoi_diagram")

# --- 2) Generatori p1..p6 i točka g (zamjena za image2) ----------------------
pts2 = np.array([[0.18, 0.68], [0.52, 0.85], [0.88, 0.55],
                 [0.50, 0.50], [0.17, 0.25], [0.47, 0.14]])
g = np.array([0.40, 0.58])
vor2 = bounded_voronoi(pts2, BBOX)
fig, ax = new_ax((3.3, 3.3))
draw_cells(ax, vor2, len(pts2), BBOX, edge=INK, lw=1.0)
ax.plot(pts2[:, 0], pts2[:, 1], "o", color=INK, markersize=4, zorder=3)
for i, (x, y) in enumerate(pts2, start=1):
    ax.annotate(f"$p_{i}$", (x, y), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=11)
ax.plot(*g, marker="D", color=INK, markersize=5, zorder=3)
ax.annotate("$g$", g, xytext=(0, 7), textcoords="offset points",
            ha="center", fontsize=11)
save(fig, "ill_grouping_g")

# --- 3) Lokalni Voronoi: (a) odabir k susjeda, (b) lokalni dijagram ----------
K = 6
pts3 = sample_min_dist(26, 0.03, 0.97, 0.14)
star = np.array([0.50, 0.52])
tree = cKDTree(pts3)
_, idx = tree.query(star, k=K + 1)          # najbliži čvor + njegovih K susjeda
nearest = idx[0]
neigh = idx[1:]

fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.4))
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.8)

# (a) puni dijagram: tamna = regija najbližeg čvora, svijetle = k susjeda
vor3 = bounded_voronoi(pts3, BBOX)
def fc_full(i):
    if i == nearest:
        return "#7a7a76"
    if i in neigh:
        return "#cfcec9"
    return "none"
ax = axes[0]
draw_cells(ax, vor3, len(pts3), BBOX, facecolors=fc_full, edge=INK, lw=0.8)
ax.plot(pts3[:, 0], pts3[:, 1], "o", color=INK, markersize=3, zorder=3)
ax.plot(*star, marker="*", color=STAR, markersize=13, markeredgecolor=INK,
        markeredgewidth=0.5, zorder=4)
ax.set_title("(a)", loc="left", fontweight="bold", fontsize=10)

# (b) lokalni dijagram iz samo k+1 čvorova
sub = pts3[idx]
vor4 = bounded_voronoi(sub, BBOX)
ax = axes[1]
draw_cells(ax, vor4, len(sub), BBOX,
           facecolors=lambda i: "#7a7a76" if i == 0 else "#e6e5e0",
           edge=INK, lw=0.8)
ax.plot(sub[:, 0], sub[:, 1], "o", color=INK, markersize=3, zorder=3)
ax.plot(*star, marker="*", color=STAR, markersize=13, markeredgecolor=INK,
        markeredgewidth=0.5, zorder=4)
ax.set_title("(b)", loc="left", fontweight="bold", fontsize=10)

save(fig, "ill_local_voronoi")

print("Gotovo.")
