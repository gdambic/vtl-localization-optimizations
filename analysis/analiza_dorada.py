"""
Uparena statisticka analiza doradnih simulacija (SIM-A1/A2/B/C) za fazu W.

Pokretati iz mape Simulacije:
    python analiza_dorada.py

Izlazi:
    rezultati_dorada.md      — sve brojke za tekstove W3-W5
    tab_dorada_validacija.csv, tab_dorada_sensitivity.csv,
    tab_dorada_sensitivity_poN.csv, tab_dorada_seeds.csv

Metodologija = ista kao Analiza/analiza.py: upareni dizajn preko (Seed, N),
dvostrani Wilcoxon signed-rank, Holmova korekcija po obitelji testova,
Cohen dz na uparenim razlikama (negativan = optimizirana konfiguracija brza).
Relativne promjene = prosjek po-parnih omjera (kao u radu).
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

OUT_MD = []


def log(s=""):
    OUT_MD.append(s)
    print(s)


def load(path):
    df = pd.read_csv(path)
    df["Pe_num"] = df["Pe"].astype(str).str.rstrip("%").astype(float)
    for c in ["AverageEstimationError", "VoronoiProc", "IntersectionCalc", "TotalTime"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def holm(pvals):
    """Holmova korekcija; vraca adjusted p-values u izvornom poretku."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def dz(diffs):
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else 0.0


def paired(df, base_mask, opt_mask, keys=("Seed", "NumberOfNodes")):
    b = df[base_mask].set_index(list(keys)).sort_index()
    o = df[opt_mask].set_index(list(keys)).sort_index()
    common = b.index.intersection(o.index)
    return b.loc[common], o.loc[common]


# ----------------------------------------------------------------------------
log("# Rezultati doradnih simulacija (upareni testovi) — generirano analiza_dorada.py")
log()

# ==== SIM-A1: Li et al. postavka =============================================
a1 = load("sim_a1.csv")
log("## SIM-A1 — postavka Li et al. (100x100 m, N=140, 40 sidrenih, 5 seedova)")
log()
log(f"- Nas baseline: Ea = {a1.AverageEstimationError.mean():.3f} ± {a1.AverageEstimationError.std(ddof=1):.3f}, "
    f"Pe = {a1.Pe_num.mean():.1f} ± {a1.Pe_num.std(ddof=1):.1f} %  (n=5 seedova)")
log(f"- Raspon Ea po seedovima: {a1.AverageEstimationError.min():.3f}–{a1.AverageEstimationError.max():.3f}")
log("- Li et al. (Electronics 2022, Table 4): Ea = 0.361, Pe = 100 %")
log()
a1_out = a1[["Seed", "AverageEstimationError", "Pe_num"]].copy()
a1_out.to_csv("tab_dorada_validacija.csv", index=False)

# ==== SIM-A2: trend po anchor ratiju =========================================
a2 = load("sim_a2.csv")
g = a2.groupby("PercentageAnchor").agg(
    Ea_mean=("AverageEstimationError", "mean"), Ea_sd=("AverageEstimationError", "std"),
    Pe_mean=("Pe_num", "mean"), Pe_sd=("Pe_num", "std")).round(3)
log("## SIM-A2 — trend po udjelu sidrenih cvorova (100x100 m, N=100, 5 seedova)")
log()
log(g.to_string())
rho_ea, p_ea = spearmanr(a2.PercentageAnchor, a2.AverageEstimationError)
rho_pe, p_pe = spearmanr(a2.PercentageAnchor, a2.Pe_num)
log()
log(f"- Spearman (ra, Ea): rho = {rho_ea:.3f}, p = {p_ea:.2e} (pada s gustocom sidara)")
log(f"- Spearman (ra, Pe): rho = {rho_pe:.3f}, p = {p_pe:.2e} (raste s gustocom sidara)")
log("- Kvalitativno isti trend kao Li et al., Fig. 11.")
log()

# ==== SIM-B: sensitivity (ra 20/60, k=8..16) =================================
b = load("sim_b.csv")
log("## SIM-B — osjetljivost na gustocu sidrenih cvorova (200x200 m, N=100-900, 3 seeda)")
log("Baseline = grid-global; optimizirano = sweep-local:k (kombinirana optimizacija).")
log("Pairing po (Seed, N): n = 27 parova po (ra, k).")
log()
rows = []
per_n_rows = []
for ra in sorted(b.PercentageAnchor.unique()):
    dfr = b[b.PercentageAnchor == ra]
    base_mask = (dfr.Algorithm == "Grid Scanning") & (dfr.VoronoiMode == "Global")
    for k in [8, 10, 12, 16]:
        opt_mask = (dfr.Algorithm == "Sweep Line") & (dfr.VoronoiMode == "Local") & (dfr.k_neighbors == k)
        bb, oo = paired(dfr, base_mask, opt_mask)
        n = len(bb)
        dea_pct = ((oo.AverageEstimationError / bb.AverageEstimationError) - 1) * 100
        vor_speed = (1 - oo.VoronoiProc / bb.VoronoiProc) * 100
        tot_speed = (1 - oo.TotalTime / bb.TotalTime) * 100
        pe_maxdiff = float((oo.Pe_num - bb.Pe_num).abs().max())
        ea_d = (oo.AverageEstimationError - bb.AverageEstimationError).values
        t_d = (oo.TotalTime - bb.TotalTime).values
        p_ea_w = wilcoxon(ea_d, alternative="two-sided").pvalue if np.any(ea_d != 0) else 1.0
        p_t_w = wilcoxon(t_d, alternative="two-sided").pvalue
        rows.append(dict(ra=ra, k=k, n=n,
                         dEa_pct=round(dea_pct.mean(), 2),
                         vor_speedup_pct=round(vor_speed.mean(), 1),
                         tot_speedup_pct=round(tot_speed.mean(), 1),
                         p_Ea=p_ea_w, dz_Ea=round(dz(ea_d), 2),
                         p_time=p_t_w, dz_time=round(dz(t_d), 2),
                         Pe_maxdiff=pe_maxdiff))
        # po-N razrada (prosjek po seedovima unutar N) za nijansiranje teksta
        for nn in sorted(dfr.NumberOfNodes.unique()):
            bsel = bb[bb.index.get_level_values("NumberOfNodes") == nn]
            osel = oo[oo.index.get_level_values("NumberOfNodes") == nn]
            per_n_rows.append(dict(ra=ra, k=k, N=nn,
                                   dEa_pct=round(float(((osel.AverageEstimationError / bsel.AverageEstimationError) - 1).mean() * 100), 2),
                                   tot_speedup_pct=round(float((1 - osel.TotalTime / bsel.TotalTime).mean() * 100), 1)))

tab = pd.DataFrame(rows)
tab["p_Ea_holm"] = holm(tab.p_Ea)
tab["p_time_holm"] = holm(tab.p_time)
tab = tab[["ra", "k", "n", "dEa_pct", "p_Ea_holm", "dz_Ea",
           "vor_speedup_pct", "tot_speedup_pct", "p_time_holm", "dz_time", "Pe_maxdiff"]]
log(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
tab.to_csv("tab_dorada_sensitivity.csv", index=False)
pd.DataFrame(per_n_rows).to_csv("tab_dorada_sensitivity_poN.csv", index=False)
log()
log("Po-N razrada u tab_dorada_sensitivity_poN.csv (za nijanse u tekstu).")
log()

# ==== SIM-C: stabilnost preko 10 seedova =====================================
c = load("sim_c.csv")
log("## SIM-C — stabilnost preko 10 nezavisnih seedova (200x200 m, ra=40 %)")
log()
rows = []
base_mask = (c.Algorithm == "Grid Scanning") & (c.VoronoiMode == "Global")
for k in [10, 12]:
    opt_mask = (c.Algorithm == "Sweep Line") & (c.VoronoiMode == "Local") & (c.k_neighbors == k)
    for nn in sorted(c.NumberOfNodes.unique()):
        bb, oo = paired(c[c.NumberOfNodes == nn], base_mask[c.NumberOfNodes == nn],
                        opt_mask[c.NumberOfNodes == nn], keys=("Seed",))
        dea_pct = ((oo.AverageEstimationError / bb.AverageEstimationError) - 1) * 100
        tot_speed = (1 - oo.TotalTime / bb.TotalTime) * 100
        rows.append(dict(k=k, N=nn, n=len(bb),
                         Ea_base=f"{bb.AverageEstimationError.mean():.4f}±{bb.AverageEstimationError.std(ddof=1):.4f}",
                         Ea_opt=f"{oo.AverageEstimationError.mean():.4f}±{oo.AverageEstimationError.std(ddof=1):.4f}",
                         dEa_pct=round(dea_pct.mean(), 2),
                         tot_speedup_pct=f"{tot_speed.mean():.1f}±{tot_speed.std(ddof=1):.1f}",
                         Pe_maxdiff=float((oo.Pe_num - bb.Pe_num).abs().max())))
tabc = pd.DataFrame(rows)
log(tabc.to_string(index=False))
tabc.to_csv("tab_dorada_seeds.csv", index=False)
log()
log("Napomena: apsolutna vremena ove kampanje nisu usporediva s VelikiRun kampanjom")
log("(razlicito opterecenje stroja); mjerodavne su RELATIVNE redukcije po paru.")

with open("rezultati_dorada.md", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT_MD) + "\n")
print("\nZapisano: rezultati_dorada.md + tab_dorada_*.csv")
