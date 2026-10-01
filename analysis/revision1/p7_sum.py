# -*- coding: utf-8 -*-
"""
P7 (Major Revision 1, R1-4B / R3-2 / R3-3): analiza SIM-D (RSSI shadowing sum).

1) Interakcija sum x lokalna aproksimacija: uparena razlika Ea (sweep-local:k vs
   grid-global) po sigma ∈ {0,4,8} dB; sigma=0 parovi iz Simulacije/sim_c.csv
   (isti seedovi 4275, 2001, 2002). Ocekivanje: penalty aproksimacije ostaje mali.
2) Utjecaj suma na baseline: Ea grid-global po sigma.
3) Validacijska veza: Li-postavka (100x100, N=140, ra=28) — Ea po sigma vs. 0.361
   iz Li et al. (sim_a1 = sigma 0, 5 seedova; sim_d_li = sigma 4/8, isti seedovi).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SIMD = os.path.join(HERE, "..", "..", "data", "supplementary")

SEEDS = [4275, 2001, 2002]
NS = [200, 600, 1000]


def prep(df):
    df = df.copy()
    df["Pe_pct"] = df["Pe"].str.rstrip("%").astype(float)
    return df

d = prep(pd.read_csv(os.path.join(SIMD, "sim_d.csv")))
c = prep(pd.read_csv(os.path.join(SIMD, "sim_c.csv")))
c = c[c["Seed"].isin(SEEDS)]
c["RssiNoiseSigma"] = 0.0

rows = []
print("=== 1) Uparena razlika Ea: sweep-local:k vs grid-global, po sigma ===")
for sigma, src in [(0.0, c), (4.0, d), (8.0, d)]:
    src_s = src[src["RssiNoiseSigma"] == sigma] if "RssiNoiseSigma" in src else src
    base = src_s[(src_s["Algorithm"] == "Grid Scanning") & (src_s["VoronoiMode"] == "Global")]
    for k in (10, 12):
        opt = src_s[(src_s["Algorithm"] == "Sweep Line") & (src_s["VoronoiMode"] == "Local")
                    & (src_s["k_neighbors"] == k)]
        m = pd.merge(base[["Seed", "NumberOfNodes", "AverageEstimationError", "Pe_pct"]],
                     opt[["Seed", "NumberOfNodes", "AverageEstimationError", "Pe_pct"]],
                     on=["Seed", "NumberOfNodes"], suffixes=("_g", "_l"))
        dEa = 100.0 * (m["AverageEstimationError_l"] - m["AverageEstimationError_g"]) / m["AverageEstimationError_g"]
        dPe = (m["Pe_pct_l"] - m["Pe_pct_g"]).abs().max()
        try:
            p = stats.wilcoxon(m["AverageEstimationError_l"] - m["AverageEstimationError_g"])[1]
        except ValueError:
            p = float("nan")
        print(f"sigma={sigma:3.0f} dB k={k}: n={len(m)}  dEa mean={dEa.mean():+.2f} %  "
              f"(min {dEa.min():+.2f}, max {dEa.max():+.2f})  maxdPe={dPe:.2f}  p={p:.3g}")
        rows.append(dict(analysis="dEa_local_vs_global", sigma=sigma, k=k, n=len(m),
                         mean_dEa_pct=round(dEa.mean(), 3), min_dEa_pct=round(dEa.min(), 3),
                         max_dEa_pct=round(dEa.max(), 3), max_abs_dPe=round(dPe, 3),
                         p_wilcoxon=p))

print("\n=== 2) Baseline Ea (grid-global) po sigma ===")
base0 = c[(c["Algorithm"] == "Grid Scanning") & (c["VoronoiMode"] == "Global")]
for sigma, src in [(0.0, base0), (4.0, None), (8.0, None)]:
    if src is None:
        src = d[(d["RssiNoiseSigma"] == sigma) & (d["Algorithm"] == "Grid Scanning")
                & (d["VoronoiMode"] == "Global")]
    for N in NS:
        g = src[src["NumberOfNodes"] == N]
        print(f"sigma={sigma:3.0f} dB N={N:4d}: Ea={g['AverageEstimationError'].mean():.4f} "
              f"± {g['AverageEstimationError'].std(ddof=1):.4f}  Pe={g['Pe_pct'].mean():.2f} %")
        rows.append(dict(analysis="baseline_Ea", sigma=sigma, N=N,
                         Ea_mean=round(g["AverageEstimationError"].mean(), 5),
                         Ea_sd=round(g["AverageEstimationError"].std(ddof=1), 5),
                         Pe_mean=round(g["Pe_pct"].mean(), 2)))

print("\n=== 3) Li-postavka: Ea po sigma vs. Li et al. 0.361 ===")
a1 = prep(pd.read_csv(os.path.join(SIMD, "sim_a1.csv")))
li = prep(pd.read_csv(os.path.join(SIMD, "sim_d_li.csv")))
for sigma, src in [(0.0, a1), (4.0, li[li["RssiNoiseSigma"] == 4.0]),
                   (8.0, li[li["RssiNoiseSigma"] == 8.0])]:
    print(f"sigma={sigma:3.0f} dB: n={len(src)}  Ea={src['AverageEstimationError'].mean():.4f} "
          f"± {src['AverageEstimationError'].std(ddof=1):.4f}  "
          f"Pe={src['Pe_pct'].mean():.2f} ± {src['Pe_pct'].std(ddof=1):.2f} %")
    rows.append(dict(analysis="li_setup", sigma=sigma, n=len(src),
                     Ea_mean=round(src["AverageEstimationError"].mean(), 4),
                     Ea_sd=round(src["AverageEstimationError"].std(ddof=1), 4),
                     Pe_mean=round(src["Pe_pct"].mean(), 2)))

pd.DataFrame(rows).to_csv(os.path.join(HERE, "tab_p7_sum.csv"), index=False)
print("\nZapisano: tab_p7_sum.csv")
