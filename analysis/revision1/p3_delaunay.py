"""
P3 (Major Revision 1, R2-3): veza kNN odabira i Delaunayjevih susjeda.

Teorija: Voronoi celija sidra a* odredjena je ISKLJUCIVO njegovim Delaunayjevim
susjedima. local_voronoi() gradi dijagram nad k tocaka najblizih a* (ukljucujuci
a*). Ako tih k-1 susjeda sadrzi sve Delaunayjeve susjede a*, lokalna celija je
IDENTICNA globalnoj (klipanoj na prostor). Skript Monte Carlom na geometrijama
eksperimenta (uniformno u kvadratu, m = broj sidara u grupi ~ 0.2N) mjeri:
  - broj Delaunayjevih susjeda a* (distribucija),
  - pokrivenost: |Del(a*) ∩ (k-1)NN(a*)| / |Del(a*)| za k = 6..20,
  - P(celija egzaktna) = P(pokrivenost == 1).
a* = sidro najblize slucajnom nepoznatom cvoru (uvjet: unutar R = 20 m).
"""
import numpy as np
from scipy.spatial import Delaunay
import csv, collections

AREA = 200.0
R = 20.0
TRIALS = 2000
KS = list(range(6, 21, 2))
# m = velicina grupe = (0.4 * N) / 2 za N iz eksperimenta
GROUPS = {200: 40, 600: 120, 1000: 200}

rng = np.random.default_rng(4275)
rows = []
for N, m in GROUPS.items():
    cov_sum = {k: 0.0 for k in KS}
    exact_cnt = {k: 0 for k in KS}
    ndel = []
    t = 0
    while t < TRIALS:
        pts = rng.uniform(0, AREA, size=(m, 2))
        u = rng.uniform(0, AREA, size=2)
        d2u = ((pts - u) ** 2).sum(axis=1)
        i_star = int(np.argmin(d2u))
        if d2u[i_star] > R * R:
            continue  # nepoznati cvor bez sidra u radijusu -> odbaci
        t += 1
        tri = Delaunay(pts)
        indptr, indices = tri.vertex_neighbor_vertices
        neigh = set(indices[indptr[i_star]:indptr[i_star + 1]])
        ndel.append(len(neigh))
        d2a = ((pts - pts[i_star]) ** 2).sum(axis=1)
        order = np.argsort(d2a)  # order[0] == i_star (dist 0)
        for k in KS:
            knn = set(order[:k].tolist()) - {i_star}   # k tocaka UKLJUCUJUCI a*
            cov = len(neigh & knn) / len(neigh)
            cov_sum[k] += cov
            exact_cnt[k] += (cov == 1.0)
    ndel = np.array(ndel)
    print(f"N={N} (m={m}): Delaunay susjeda a*: mean={ndel.mean():.2f}, "
          f"median={np.median(ndel):.0f}, p95={np.percentile(ndel, 95):.0f}, max={ndel.max()}")
    for k in KS:
        rows.append({"N": N, "m": m, "k": k,
                     "mean_coverage_pct": round(100 * cov_sum[k] / TRIALS, 2),
                     "cell_exact_pct": round(100 * exact_cnt[k] / TRIALS, 2),
                     "mean_delaunay_neighbors": round(float(ndel.mean()), 2)})
        print(f"  k={k:2d}: pokrivenost={rows[-1]['mean_coverage_pct']:6.2f} %  "
              f"celija egzaktna={rows[-1]['cell_exact_pct']:6.2f} %")

with open("tab_p3_delaunay.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("Zapisano: tab_p3_delaunay.csv")
