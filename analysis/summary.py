"""
Sazetak simulacijskih CSV-ova iz mape Simulacije/.

Pokretati IZ mape Simulacije nakon sto su simulacije gotove:
    python summary.py

Za svaki sim_*.csv generira summary_<ime>.csv s prosjecima po
(PercentageAnchor, NumberOfNodes, Algorithm, VoronoiMode, k_neighbors)
preko svih seedova, te ispisuje broj redaka po datoteci (kontrola potpunosti).
Ocekivani brojevi redaka: sim_a1=5, sim_a2=35, sim_b=270, sim_c=90.
"""
import glob
import os

import pandas as pd

GROUP = ["PercentageAnchor", "NumberOfNodes", "Algorithm", "VoronoiMode", "k_neighbors"]
METRICS = ["AverageEstimationError", "Pe_num", "VoronoiProc", "IntersectionCalc", "TotalTime"]

for path in sorted(glob.glob("sim_*.csv")):
    df = pd.read_csv(path)
    df["Pe_num"] = df["Pe"].astype(str).str.rstrip("%").astype(float)
    for col in ["AverageEstimationError", "VoronoiProc", "IntersectionCalc", "TotalTime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_seeds = df["Seed"].nunique()
    print(f"{path}: {len(df)} redaka, {n_seeds} seedova")

    agg = df.groupby(GROUP)[METRICS].agg(["mean", "std", "count"]).round(6)
    agg.columns = ["_".join(c) for c in agg.columns]
    out = "summary_" + os.path.basename(path)
    agg.reset_index().to_csv(out, index=False)
    print(f"  -> {out}")

print("Gotovo.")
