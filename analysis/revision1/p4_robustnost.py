# -*- coding: utf-8 -*-
"""
P4 (Major Revision 1, R2-5): robusnost statistickih zakljucaka na pooling 144
parova preko razlicitih velicina mreze.

Ponavlja glavne uparene usporedbe iz rada na TERCILIMA velicine mreze
(60-360, 380-680, 700-1000 cvorova; 16 velicina x 3 ponavljanja = 48 parova
po tercilu), za obje povrsine. Wilcoxon signed-rank + Holm unutar obitelji
(usporedba x povrsina, preko tercila) + Cohen dz + srednje relativno smanjenje.

Izlaz: tab_p4_robustnost.csv + ispis.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")

FILES = {
    ("Grid Scanning", "Global", 200): "GridScanGlobal200.csv",
    ("Grid Scanning", "Global", 400): "GridScanGlobal400.csv",
    ("Grid Scanning", "Local", 200): "GridScanLocal200.csv",
    ("Grid Scanning", "Local", 400): "GridScanLocal400.csv",
    ("Sweep Line", "Global", 200): "SweepLineGlobal200.csv",
    ("Sweep Line", "Global", 400): "SweepLineGlobal400.csv",
    ("Sweep Line", "Local", 200): "SweepLineLocal200.csv",
    ("Sweep Line", "Local", 400): "SweepLineLocal400.csv",
}

TERCILES = [("small (60-360)", 60, 360), ("medium (380-680)", 380, 680),
            ("large (700-1000)", 700, 1000)]


def load(alg, mode, space):
    df = pd.read_csv(os.path.join(DATA, FILES[(alg, mode, space)]))
    df["Pe_pct"] = df["Pe"].str.rstrip("%").astype(float)
    df["run"] = df.groupby(["k_neighbors", "NumberOfNodes"]).cumcount()
    return df


def pairs(base, opt, metric, k_opt=None):
    b = base[base["k_neighbors"] == base["k_neighbors"].iloc[0]] if False else base
    o = opt if k_opt is None else opt[opt["k_neighbors"] == k_opt]
    m = pd.merge(b[["NumberOfNodes", "run", metric]],
                 o[["NumberOfNodes", "run", metric]],
                 on=["NumberOfNodes", "run"], suffixes=("_base", "_opt"))
    return m


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


COMPARISONS = [
    ("Voronoi time, local k=12 vs global (grid)", "Grid Scanning", "Global", None,
     "Grid Scanning", "Local", 12, "VoronoiProc"),
    ("Ea, local k=12 vs global (grid)", "Grid Scanning", "Global", None,
     "Grid Scanning", "Local", 12, "AverageEstimationError"),
    ("Intersection time, sweep vs grid (global)", "Grid Scanning", "Global", None,
     "Sweep Line", "Global", None, "IntersectionCalc"),
    ("Total time, sweep+local k=12 vs grid+global", "Grid Scanning", "Global", None,
     "Sweep Line", "Local", 12, "TotalTime"),
]

rows = []
for space in (200, 400):
    for (name, alg_b, mode_b, k_b, alg_o, mode_o, k_o, metric) in COMPARISONS:
        base = load(alg_b, mode_b, space)
        opt = load(alg_o, mode_o, space)
        m = pairs(base, opt, metric, k_opt=k_o)
        fam_p = []
        fam_rows = []
        for (tname, lo, hi) in TERCILES:
            g = m[(m["NumberOfNodes"] >= lo) & (m["NumberOfNodes"] <= hi)]
            d = g[f"{metric}_opt"].values - g[f"{metric}_base"].values
            stat, p = stats.wilcoxon(d)
            dz = d.mean() / d.std(ddof=1)
            rel = 100.0 * (d / g[f"{metric}_base"].values).mean()
            sign_neg = int((d < 0).sum())
            fam_p.append(p)
            fam_rows.append(dict(space=space, comparison=name, tercile=tname,
                                 n_pairs=len(g), mean_rel_change_pct=round(rel, 2),
                                 dz=round(dz, 3), p_raw=p, sign_neg=sign_neg))
        for r, padj in zip(fam_rows, holm(np.array(fam_p))):
            r["p_holm"] = padj
            rows.append(r)

out = pd.DataFrame(rows)
out["p_raw"] = out["p_raw"].map(lambda x: f"{x:.2e}")
out["p_holm"] = out["p_holm"].map(lambda x: f"{x:.2e}")
out.to_csv(os.path.join(HERE, "tab_p4_robustnost.csv"), index=False)
pd.set_option("display.width", 200)
for space in (200, 400):
    print(f"\n=== {space} m ===")
    print(out[out["space"] == space].to_string(index=False))
