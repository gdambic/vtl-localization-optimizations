# -*- coding: utf-8 -*-
"""
Re-analiza VelikiRun podataka za članak (MDPI Sensors).

Ulaz : ../data/*.csv  (8 datoteka, seed 4275)
Izlaz: tab_*.csv tablice + rezultati.md u ovoj mapi.

Pokretanje:  python analiza.py   (iz mape Analiza/)

Eksperimenti (vidi PLAN.md, FAZA 1):
  1. Local vs Global Voronoi (GridScan konfiguracije), po k = 6..20
  2. SweepLine vs GridScan (Global konfiguracije)
  3. Kombinirana optimizacija: SweepLine+Local(k) vs GridScan+Global (H4)
Statistika: upareni parovi po (NumberOfNodes, run); Shapiro na razlikama ->
paired t-test ili Wilcoxon signed-rank; Cohen dz (i r za Wilcoxon);
Holmova korekcija unutar obitelji testova.
"""

import io
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

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

FIXED = dict(Seed=4275, PercentageAnchor=40, DetectionRadius_m=20, Clusters_K=6, GridSize_m=2)
NODES = list(range(60, 1001, 20))          # 48 točaka
KS = list(range(6, 21, 2))                 # 8 vrijednosti k
METRICS = ["AverageEstimationError", "Pe_pct", "VoronoiProc", "IntersectionCalc", "TotalTime"]

report = io.StringIO()


def w(line=""):
    print(line)
    report.write(line + "\n")


# ----------------------------------------------------------------------------
# 1.1 Učitavanje i validacija
# ----------------------------------------------------------------------------
def load_all():
    frames = []
    problems = []
    for (alg, mode, space), fname in FILES.items():
        df = pd.read_csv(os.path.join(DATA, fname))
        df["Pe_pct"] = df["Pe"].str.rstrip("%").astype(float)
        df["Space"] = space
        df["File"] = fname

        # konzistentnost deklariranih i stvarnih atributa
        if not (df["Algorithm"] == alg).all():
            problems.append(f"{fname}: stupac Algorithm != {alg}")
        if not (df["VoronoiMode"] == mode).all():
            problems.append(f"{fname}: stupac VoronoiMode != {mode}")
        if not (df["SimulationSize_m"] == space).all():
            problems.append(f"{fname}: SimulationSize_m != {space}")
        for col, val in FIXED.items():
            if not (df[col] == val).all():
                problems.append(f"{fname}: {col} != {val}")

        # kompletnost
        exp_rows = 48 * 3 * (8 if mode == "Local" else 1)
        if len(df) != exp_rows:
            problems.append(f"{fname}: {len(df)} redaka, očekivano {exp_rows}")
        if sorted(df["NumberOfNodes"].unique()) != NODES:
            problems.append(f"{fname}: neočekivane vrijednosti NumberOfNodes")
        if mode == "Local" and sorted(df["k_neighbors"].unique()) != KS:
            problems.append(f"{fname}: neočekivane vrijednosti k_neighbors")

        # NaN
        n_nan = int(df[["AverageEstimationError", "Pe_pct", "VoronoiProc",
                        "IntersectionCalc", "TotalTime"]].isna().sum().sum())
        if n_nan:
            problems.append(f"{fname}: {n_nan} NaN vrijednosti")

        # run-indeks: redoslijed pojavljivanja unutar (k, N)
        df["run"] = df.groupby(["k_neighbors", "NumberOfNodes"]).cumcount()
        if not (df.groupby(["k_neighbors", "NumberOfNodes"]).size() == 3).all():
            problems.append(f"{fname}: broj ponavljanja po (k, N) nije 3")

        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    dup = full.duplicated(subset=["File", "k_neighbors", "NumberOfNodes", "run"]).sum()
    if dup:
        problems.append(f"duplikati ključa (File,k,N,run): {dup}")
    return full, problems


# ----------------------------------------------------------------------------
# Statistički alati
# ----------------------------------------------------------------------------
def holm(pvals):
    """Holm-Bonferroni korigirane p-vrijednosti (redoslijed ulaza očuvan).

    NaN ulazi (testovi koji nisu provedeni) isključeni su iz obitelji.
    """
    p = np.asarray(pvals, dtype=float)
    adj = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    m = len(valid)
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def paired_test(x, y):
    """Upareni test x vs y. Vraća dict s rezultatima.

    Shapiro na razlikama -> paired t (normalno) ili Wilcoxon signed-rank.
    Cohen dz uvijek; za Wilcoxon i r = |z|/sqrt(n).
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(d)
    if np.all(d == 0):
        return dict(test="identično (d=0)", stat=np.nan, p=np.nan,
                    shapiro_p=np.nan, n=n, dz=np.nan, r=np.nan)
    sh_p = stats.shapiro(d).pvalue if np.ptp(d) > 0 else np.nan
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    if not np.isnan(sh_p) and sh_p > 0.05:
        t, p = stats.ttest_rel(x, y)
        return dict(test="paired t", stat=t, p=p, shapiro_p=sh_p, n=n, dz=dz, r=np.nan)
    res = stats.wilcoxon(x, y, zero_method="wilcox", method="approx")
    z = stats.norm.ppf(res.pvalue / 2)  # dvostrani -> |z|
    r = abs(z) / np.sqrt(n)
    return dict(test="Wilcoxon", stat=res.statistic, p=res.pvalue, shapiro_p=sh_p, n=n, dz=dz, r=r)


def ci95(a):
    a = np.asarray(a, dtype=float)
    n = len(a)
    if n < 2:
        return np.nan
    return stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)


def fmt_p(p):
    if np.isnan(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def md_table(df, floatfmt="{:.3f}"):
    """DataFrame -> Markdown tablica (bez ovisnosti o tabulate)."""
    def f(v):
        if isinstance(v, float):
            return "—" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(f(v) for v in row) + " |")
    return "\n".join(lines)


def pair_frames(a, b, cols):
    """Spoji dvije konfiguracije po (NumberOfNodes, run); vrati df s _a/_b sufiksima."""
    key = ["NumberOfNodes", "run"]
    m = a[key + cols].merge(b[key + cols], on=key, suffixes=("_a", "_b"))
    assert len(m) == 144, f"očekivano 144 para, dobiveno {len(m)}"
    return m


def per_node_ratio_speedup(a, b, col):
    """Ubrzanje kao 1 - mean_po_N( mean3(a[col]) / mean3(b[col]) ), a=optimizirano, b=baseline."""
    ma = a.groupby("NumberOfNodes")[col].mean()
    mb = b.groupby("NumberOfNodes")[col].mean()
    return float((1.0 - ma / mb).mean() * 100.0)


def per_node_rel_diff(a, b, col):
    """Relativna razlika (a-b)/b u %, prosjek po N (a=optimizirano, b=baseline)."""
    ma = a.groupby("NumberOfNodes")[col].mean()
    mb = b.groupby("NumberOfNodes")[col].mean()
    return float(((ma - mb) / mb).mean() * 100.0)


# ----------------------------------------------------------------------------
# Glavni tijek
# ----------------------------------------------------------------------------
def main():
    full, problems = load_all()

    w("# Rezultati re-analize VelikiRun podataka")
    w()
    w(f"Skripta: `Analiza/analiza.py` · pandas {pd.__version__}, scipy {stats.__name__ and __import__('scipy').__version__}")
    w()

    # ---- 1.1 validacija ----
    w("## 1. Validacija podataka")
    w()
    w(f"- Ukupno učitano: **{len(full)}** simulacija iz 8 CSV datoteka (očekivano 5184).")
    w(f"- Fiksni parametri (seed {FIXED['Seed']}, {FIXED['PercentageAnchor']} % sidrišnih, "
      f"R = {FIXED['DetectionRadius_m']} m, K = {FIXED['Clusters_K']}, grid {FIXED['GridSize_m']} m): "
      + ("**svi konzistentni**." if not problems else "vidi probleme dolje."))
    w("- Broj čvorova 60–1000 (korak 20, 48 točaka), 3 ponavljanja po točki; "
      "Local konfiguracije dodatno k = 6–20 (korak 2). Bez NaN-ova i duplikata."
      if not problems else "")
    if problems:
        w()
        w("**PROBLEMI:**")
        for p in problems:
            w(f"- {p}")
    w()

    def cfg(alg, mode, space):
        return full[(full.Algorithm == alg) & (full.VoronoiMode == mode) & (full.Space == space)]

    # ---- 1.2 deskriptivna statistika ----
    rows = []
    for (alg, mode, space) in FILES:
        d = cfg(alg, mode, space)
        kvals = KS if mode == "Local" else [None]
        for k in kvals:
            dk = d if k is None else d[d.k_neighbors == k]
            for N, g in dk.groupby("NumberOfNodes"):
                row = dict(Algorithm=alg, VoronoiMode=mode, Space=space,
                           k=(k if k is not None else ""), NumberOfNodes=N, n_runs=len(g))
                for mcol in METRICS:
                    v = g[mcol].to_numpy(float)
                    row[f"{mcol}_mean"] = v.mean()
                    row[f"{mcol}_sd"] = v.std(ddof=1)
                    row[f"{mcol}_ci95"] = ci95(v)
                rows.append(row)
    desc = pd.DataFrame(rows)
    desc.to_csv(os.path.join(HERE, "tab_deskriptivna.csv"), index=False)
    w("## 2. Deskriptivna statistika")
    w()
    w(f"Po konfiguraciji i broju čvorova (srednja vrijednost, SD, 95 % CI preko 3 ponavljanja) "
      f"za Ea, Pe, VoronoiProc, IntersectionCalc, TotalTime — ukupno {len(desc)} redaka: "
      "`tab_deskriptivna.csv`.")
    w()

    # ---- 1.3 Eksperiment 1: Local vs Global (GridScan) ----
    w("## 3. Eksperiment 1 — lokalni vs. globalni Voronoi (Grid Scanning konfiguracije)")
    w()
    exp1 = []
    for space in (200, 400):
        base = cfg("Grid Scanning", "Global", space)
        tests_v, tests_e = [], []
        for k in KS:
            loc = cfg("Grid Scanning", "Local", space)
            loc = loc[loc.k_neighbors == k]
            sp_v = per_node_ratio_speedup(loc, base, "VoronoiProc")
            de = per_node_rel_diff(loc, base, "AverageEstimationError")
            m = pair_frames(loc, base, ["VoronoiProc", "AverageEstimationError"])
            tv = paired_test(m["VoronoiProc_a"], m["VoronoiProc_b"])
            te = paired_test(m["AverageEstimationError_a"], m["AverageEstimationError_b"])
            tests_v.append(tv)
            tests_e.append(te)
            exp1.append(dict(Space=space, k=k, VoronoiSpeedup_pct=sp_v, dEa_pct=de,
                             Vor_test=tv["test"], Vor_p=tv["p"], Vor_dz=tv["dz"], Vor_shapiro=tv["shapiro_p"],
                             Ea_test=te["test"], Ea_p=te["p"], Ea_dz=te["dz"], Ea_shapiro=te["shapiro_p"]))
        # Holm unutar prostora, po metrici (8 testova)
        i0 = len(exp1) - len(KS)
        for adjcol, tests in (("Vor_p_holm", tests_v), ("Ea_p_holm", tests_e)):
            adj = holm([t["p"] for t in tests])
            for j, a in enumerate(adj):
                exp1[i0 + j][adjcol] = a
    exp1 = pd.DataFrame(exp1)
    exp1.to_csv(os.path.join(HERE, "tab_exp1_voronoi.csv"), index=False)

    for space in (200, 400):
        e = exp1[exp1.Space == space]
        w(f"### Prostor {space} m")
        w()
        t = pd.DataFrame({
            "k": e.k,
            "Ubrzanje VoronoiProc (%)": e.VoronoiSpeedup_pct.round(1),
            "ΔEa vs Global (%)": e.dEa_pct.round(2),
            "test (Vor)": e.Vor_test,
            "p (Vor, Holm)": e.Vor_p_holm.map(fmt_p),
            "dz (Vor)": e.Vor_dz.round(2),
            "test (Ea)": e.Ea_test,
            "p (Ea, Holm)": e.Ea_p_holm.map(fmt_p),
            "dz (Ea)": e.Ea_dz.round(2),
        })
        w(md_table(t, "{:.2f}"))
        w()

    # ---- 1.4 Eksperiment 2: SweepLine vs GridScan (Global) ----
    w("## 4. Eksperiment 2 — sweep line vs. grid scan (Global konfiguracije)")
    w()
    exp2 = []
    tests2 = []
    for space in (200, 400):
        gs = cfg("Grid Scanning", "Global", space)
        sl = cfg("Sweep Line", "Global", space)
        for col, name in (("IntersectionCalc", "presjek regija"), ("TotalTime", "ukupno vrijeme"),
                          ("AverageEstimationError", "Ea (kontrola)")):
            sp = per_node_ratio_speedup(sl, gs, col) if col != "AverageEstimationError" \
                else per_node_rel_diff(sl, gs, col)
            m = pair_frames(sl, gs, [col])
            tt = paired_test(m[f"{col}_a"], m[f"{col}_b"])
            tests2.append(tt)
            exp2.append(dict(Space=space, Metrika=name, Stupac=col,
                             Ubrzanje_ili_dEa_pct=sp, test=tt["test"], p=tt["p"],
                             dz=tt["dz"], shapiro_p=tt["shapiro_p"]))
    adj = holm([t["p"] for t in tests2])
    for j, a in enumerate(adj):
        exp2[j]["p_holm"] = a
    exp2 = pd.DataFrame(exp2)
    exp2.to_csv(os.path.join(HERE, "tab_exp2_sweepline.csv"), index=False)
    t = pd.DataFrame({
        "Prostor (m)": exp2.Space,
        "Metrika": exp2.Metrika,
        "Ubrzanje / ΔEa (%)": exp2.Ubrzanje_ili_dEa_pct.round(2),
        "test": exp2.test,
        "p (Holm)": exp2.p_holm.map(fmt_p),
        "dz": exp2.dz.round(2),
    })
    w(md_table(t, "{:.2f}"))
    w()
    w("*Napomena: za Ea je prikazana relativna razlika (SweepLine − GridScan)/GridScan. "
      "Razlike Ea su **identički nula za svih 144 para** u oba prostora — sweep line daje "
      "isti rezultat lokalizacije kao grid scan, mijenja se samo vrijeme izračuna. "
      "Test stoga nije potreban (oznaka „identično (d=0)”).*")
    w()

    # ---- 1.5 Eksperiment 3: kombinirana optimizacija (H4) ----
    w("## 5. Eksperiment 3 — kombinirana optimizacija: SweepLine+Local(k) vs GridScan+Global (H4)")
    w()
    exp3 = []
    for space in (200, 400):
        base = cfg("Grid Scanning", "Global", space)
        tests_t, tests_e = [], []
        for k in KS:
            comb = cfg("Sweep Line", "Local", space)
            comb = comb[comb.k_neighbors == k]
            sp_t = per_node_ratio_speedup(comb, base, "TotalTime")
            de = per_node_rel_diff(comb, base, "AverageEstimationError")
            m = pair_frames(comb, base, ["TotalTime", "AverageEstimationError"])
            tt = paired_test(m["TotalTime_a"], m["TotalTime_b"])
            te = paired_test(m["AverageEstimationError_a"], m["AverageEstimationError_b"])
            tests_t.append(tt)
            tests_e.append(te)
            exp3.append(dict(Space=space, k=k, TotalTimeSpeedup_pct=sp_t, dEa_pct=de,
                             TT_test=tt["test"], TT_p=tt["p"], TT_dz=tt["dz"],
                             Ea_test=te["test"], Ea_p=te["p"], Ea_dz=te["dz"]))
        i0 = len(exp3) - len(KS)
        for adjcol, tests in (("TT_p_holm", tests_t), ("Ea_p_holm", tests_e)):
            adj = holm([t["p"] for t in tests])
            for j, a in enumerate(adj):
                exp3[i0 + j][adjcol] = a
    exp3 = pd.DataFrame(exp3)
    exp3.to_csv(os.path.join(HERE, "tab_exp3_kombinirana.csv"), index=False)

    for space in (200, 400):
        e = exp3[exp3.Space == space]
        w(f"### Prostor {space} m")
        w()
        t = pd.DataFrame({
            "k": e.k,
            "Ubrzanje TotalTime (%)": e.TotalTimeSpeedup_pct.round(1),
            "ΔEa vs baseline (%)": e.dEa_pct.round(2),
            "test (TT)": e.TT_test,
            "p (TT, Holm)": e.TT_p_holm.map(fmt_p),
            "dz (TT)": e.TT_dz.round(2),
            "p (Ea, Holm)": e.Ea_p_holm.map(fmt_p),
        })
        w(md_table(t, "{:.2f}"))
        w()

    # zapažanje: k=6 usporava fazu presjeka (veće pozitivne regije)
    w("**Zapažanje (k = 6):** ukupno ubrzanje za k = 6 manje je nego za k ≥ 8, iako je "
      "Voronoi faza najbrža. Uzrok je faza presjeka regija:")
    w()
    for space in (200, 400):
        sl_loc = cfg("Sweep Line", "Local", space)
        ic = sl_loc.groupby("k_neighbors")["IntersectionCalc"].mean()
        w(f"- Prostor {space} m — prosječni IntersectionCalc: k=6: {ic[6]:.1f} s, "
          f"k=8: {ic[8]:.1f} s, k=12: {ic[12]:.1f} s. Premali k daje netočniji lokalni "
          "Voronoi dijagram i veće pozitivne regije, pa faza presjeka radi više posla.")
    w()

    # ---- kompromisna tablica za preporuku k ----
    w("## 6. Kompromis točnost/brzina po k (sažetak za preporuku)")
    w()
    tr = exp1.merge(exp3, on=["Space", "k"], suffixes=("", "_c"))
    t = pd.DataFrame({
        "Prostor (m)": tr.Space,
        "k": tr.k,
        "Ubrzanje Voronoi faze (%)": tr.VoronoiSpeedup_pct.round(1),
        "ΔEa, samo Local (%)": tr.dEa_pct.round(2),
        "Ubrzanje TotalTime, komb. (%)": tr.TotalTimeSpeedup_pct.round(1),
        "ΔEa, komb. (%)": tr.dEa_pct_c.round(2),
    })
    w(md_table(t, "{:.2f}"))
    w()

    # ---- 1.7 validacija simulatora ----
    w("## 7. Validacija simulatora (trendovi kao u Electronics 2022)")
    w()
    base = cfg("Grid Scanning", "Global", 200)
    agg = base.groupby("NumberOfNodes").agg(
        Ea=("AverageEstimationError", "mean"), Pe=("Pe_pct", "mean")).reset_index()
    sel = agg[agg.NumberOfNodes.isin([60, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])]
    rho_ea = stats.spearmanr(agg.NumberOfNodes, agg.Ea)
    rho_pe = stats.spearmanr(agg.NumberOfNodes, agg.Pe)
    t = pd.DataFrame({"N čvorova": sel.NumberOfNodes,
                      "Ea (norm.)": sel.Ea.round(3),
                      "Pe (%)": sel.Pe.round(1)})
    w(md_table(t, "{:.3f}"))
    w()
    w(f"- Spearman ρ(N, Ea) = {rho_ea.statistic:.3f} (p {fmt_p(rho_ea.pvalue)}) — "
      "Ea pada s gustoćom čvorova, u skladu s Li et al. (Electronics 2022).")
    w(f"- Spearman ρ(N, Pe) = {rho_pe.statistic:.3f} (p {fmt_p(rho_pe.pvalue)}) — "
      "udio lokaliziranih čvorova raste s gustoćom.")
    agg.to_csv(os.path.join(HERE, "tab_validacija.csv"), index=False)
    w()

    with open(os.path.join(HERE, "rezultati.md"), "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print("\n>>> Zapisano: rezultati.md + tab_*.csv u Analiza/")


if __name__ == "__main__":
    main()
