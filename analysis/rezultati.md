# Rezultati re-analize VelikiRun podataka

Skripta: `Analiza/analiza.py` · pandas 3.0.0, scipy 1.17.0

## 1. Validacija podataka

- Ukupno učitano: **5184** simulacija iz 8 CSV datoteka (očekivano 5184).
- Fiksni parametri (seed 4275, 40 % sidrišnih, R = 20 m, K = 6, grid 2 m): **svi konzistentni**.
- Broj čvorova 60–1000 (korak 20, 48 točaka), 3 ponavljanja po točki; Local konfiguracije dodatno k = 6–20 (korak 2). Bez NaN-ova i duplikata.

## 2. Deskriptivna statistika

Po konfiguraciji i broju čvorova (srednja vrijednost, SD, 95 % CI preko 3 ponavljanja) za Ea, Pe, VoronoiProc, IntersectionCalc, TotalTime — ukupno 1728 redaka: `tab_deskriptivna.csv`.

## 3. Eksperiment 1 — lokalni vs. globalni Voronoi (Grid Scanning konfiguracije)

### Prostor 200 m

| k | Ubrzanje VoronoiProc (%) | ΔEa vs Global (%) | test (Vor) | p (Vor, Holm) | dz (Vor) | test (Ea) | p (Ea, Holm) | dz (Ea) |
|---|---|---|---|---|---|---|---|---|
| 6 | 81.50 | 12.30 | Wilcoxon | <0.001 | -1.07 | Wilcoxon | <0.001 | 2.51 |
| 8 | 82.70 | 4.01 | Wilcoxon | <0.001 | -1.08 | Wilcoxon | <0.001 | 2.15 |
| 10 | 74.30 | 1.61 | Wilcoxon | <0.001 | -1.07 | Wilcoxon | <0.001 | 1.49 |
| 12 | 73.40 | 0.71 | Wilcoxon | <0.001 | -1.05 | Wilcoxon | <0.001 | 1.21 |
| 14 | 72.60 | 0.30 | Wilcoxon | <0.001 | -1.05 | Wilcoxon | <0.001 | 0.90 |
| 16 | 71.10 | 0.13 | Wilcoxon | <0.001 | -1.04 | Wilcoxon | <0.001 | 0.79 |
| 18 | 69.10 | 0.04 | Wilcoxon | <0.001 | -1.04 | Wilcoxon | <0.001 | 0.55 |
| 20 | 68.90 | 0.01 | Wilcoxon | <0.001 | -1.03 | Wilcoxon | <0.001 | 0.33 |

### Prostor 400 m

| k | Ubrzanje VoronoiProc (%) | ΔEa vs Global (%) | test (Vor) | p (Vor, Holm) | dz (Vor) | test (Ea) | p (Ea, Holm) | dz (Ea) |
|---|---|---|---|---|---|---|---|---|
| 6 | 81.40 | 4.21 | Wilcoxon | <0.001 | -0.99 | Wilcoxon | <0.001 | 1.52 |
| 8 | 79.00 | 1.21 | Wilcoxon | <0.001 | -1.00 | Wilcoxon | <0.001 | 1.27 |
| 10 | 76.60 | 0.40 | Wilcoxon | <0.001 | -0.99 | Wilcoxon | <0.001 | 1.15 |
| 12 | 74.00 | 0.13 | Wilcoxon | <0.001 | -0.98 | Wilcoxon | <0.001 | 0.94 |
| 14 | 73.50 | 0.04 | Wilcoxon | <0.001 | -0.97 | Wilcoxon | <0.001 | 0.65 |
| 16 | 75.80 | 0.01 | Wilcoxon | <0.001 | -0.98 | Wilcoxon | <0.001 | 0.48 |
| 18 | 74.20 | 0.00 | Wilcoxon | <0.001 | -0.98 | Wilcoxon | <0.001 | 0.32 |
| 20 | 74.30 | 0.00 | Wilcoxon | <0.001 | -0.98 | Wilcoxon | 0.002 | 0.22 |

## 4. Eksperiment 2 — sweep line vs. grid scan (Global konfiguracije)

| Prostor (m) | Metrika | Ubrzanje / ΔEa (%) | test | p (Holm) | dz |
|---|---|---|---|---|---|
| 200 | presjek regija | 7.70 | Wilcoxon | <0.001 | -0.98 |
| 200 | ukupno vrijeme | 3.81 | Wilcoxon | <0.001 | -0.39 |
| 200 | Ea (kontrola) | 0.00 | identično (d=0) | — | — |
| 400 | presjek regija | 18.11 | Wilcoxon | <0.001 | -1.20 |
| 400 | ukupno vrijeme | 10.17 | Wilcoxon | <0.001 | -0.67 |
| 400 | Ea (kontrola) | 0.00 | identično (d=0) | — | — |

*Napomena: za Ea je prikazana relativna razlika (SweepLine − GridScan)/GridScan. Razlike Ea su **identički nula za svih 144 para** u oba prostora — sweep line daje isti rezultat lokalizacije kao grid scan, mijenja se samo vrijeme izračuna. Test stoga nije potreban (oznaka „identično (d=0)”).*

## 5. Eksperiment 3 — kombinirana optimizacija: SweepLine+Local(k) vs GridScan+Global (H4)

### Prostor 200 m

| k | Ubrzanje TotalTime (%) | ΔEa vs baseline (%) | test (TT) | p (TT, Holm) | dz (TT) | p (Ea, Holm) |
|---|---|---|---|---|---|---|
| 6 | 37.90 | 12.30 | Wilcoxon | <0.001 | -0.90 | <0.001 |
| 8 | 60.80 | 4.01 | Wilcoxon | <0.001 | -1.15 | <0.001 |
| 10 | 55.80 | 1.61 | Wilcoxon | <0.001 | -1.13 | <0.001 |
| 12 | 47.00 | 0.71 | Wilcoxon | <0.001 | -0.98 | <0.001 |
| 14 | 55.50 | 0.30 | Wilcoxon | <0.001 | -1.11 | <0.001 |
| 16 | 56.90 | 0.13 | Wilcoxon | <0.001 | -1.12 | <0.001 |
| 18 | 56.20 | 0.04 | Wilcoxon | <0.001 | -1.12 | <0.001 |
| 20 | 55.40 | 0.01 | Wilcoxon | <0.001 | -1.11 | <0.001 |

### Prostor 400 m

| k | Ubrzanje TotalTime (%) | ΔEa vs baseline (%) | test (TT) | p (TT, Holm) | dz (TT) | p (Ea, Holm) |
|---|---|---|---|---|---|---|
| 6 | 37.10 | 4.21 | Wilcoxon | <0.001 | -0.85 | <0.001 |
| 8 | 48.50 | 1.21 | Wilcoxon | <0.001 | -1.17 | <0.001 |
| 10 | 36.60 | 0.40 | Wilcoxon | <0.001 | -0.89 | <0.001 |
| 12 | 44.90 | 0.13 | Wilcoxon | <0.001 | -1.06 | <0.001 |
| 14 | 45.20 | 0.04 | Wilcoxon | <0.001 | -1.06 | <0.001 |
| 16 | 51.20 | 0.01 | Wilcoxon | <0.001 | -1.07 | <0.001 |
| 18 | 55.10 | 0.00 | Wilcoxon | <0.001 | -1.14 | <0.001 |
| 20 | 54.70 | 0.00 | Wilcoxon | <0.001 | -1.13 | 0.002 |

**Zapažanje (k = 6):** ukupno ubrzanje za k = 6 manje je nego za k ≥ 8, iako je Voronoi faza najbrža. Uzrok je faza presjeka regija:

- Prostor 200 m — prosječni IntersectionCalc: k=6: 29.4 s, k=8: 16.1 s, k=12: 18.9 s. Premali k daje netočniji lokalni Voronoi dijagram i veće pozitivne regije, pa faza presjeka radi više posla.
- Prostor 400 m — prosječni IntersectionCalc: k=6: 36.9 s, k=8: 29.3 s, k=12: 27.4 s. Premali k daje netočniji lokalni Voronoi dijagram i veće pozitivne regije, pa faza presjeka radi više posla.

## 6. Kompromis točnost/brzina po k (sažetak za preporuku)

| Prostor (m) | k | Ubrzanje Voronoi faze (%) | ΔEa, samo Local (%) | Ubrzanje TotalTime, komb. (%) | ΔEa, komb. (%) |
|---|---|---|---|---|---|
| 200.00 | 6.00 | 81.50 | 12.30 | 37.90 | 12.30 |
| 200.00 | 8.00 | 82.70 | 4.01 | 60.80 | 4.01 |
| 200.00 | 10.00 | 74.30 | 1.61 | 55.80 | 1.61 |
| 200.00 | 12.00 | 73.40 | 0.71 | 47.00 | 0.71 |
| 200.00 | 14.00 | 72.60 | 0.30 | 55.50 | 0.30 |
| 200.00 | 16.00 | 71.10 | 0.13 | 56.90 | 0.13 |
| 200.00 | 18.00 | 69.10 | 0.04 | 56.20 | 0.04 |
| 200.00 | 20.00 | 68.90 | 0.01 | 55.40 | 0.01 |
| 400.00 | 6.00 | 81.40 | 4.21 | 37.10 | 4.21 |
| 400.00 | 8.00 | 79.00 | 1.21 | 48.50 | 1.21 |
| 400.00 | 10.00 | 76.60 | 0.40 | 36.60 | 0.40 |
| 400.00 | 12.00 | 74.00 | 0.13 | 44.90 | 0.13 |
| 400.00 | 14.00 | 73.50 | 0.04 | 45.20 | 0.04 |
| 400.00 | 16.00 | 75.80 | 0.01 | 51.20 | 0.01 |
| 400.00 | 18.00 | 74.20 | 0.00 | 55.10 | 0.00 |
| 400.00 | 20.00 | 74.30 | 0.00 | 54.70 | 0.00 |

## 7. Validacija simulatora (trendovi kao u Electronics 2022)

| N čvorova | Ea (norm.) | Pe (%) |
|---|---|---|
| 60.000 | 0.511 | 41.700 |
| 100.000 | 0.519 | 80.000 |
| 200.000 | 0.401 | 81.700 |
| 300.000 | 0.298 | 98.900 |
| 400.000 | 0.237 | 96.700 |
| 500.000 | 0.197 | 100.000 |
| 600.000 | 0.167 | 100.000 |
| 700.000 | 0.145 | 100.000 |
| 800.000 | 0.146 | 100.000 |
| 900.000 | 0.132 | 100.000 |
| 1000.000 | 0.118 | 100.000 |

- Spearman ρ(N, Ea) = -0.993 (p <0.001) — Ea pada s gustoćom čvorova, u skladu s Li et al. (Electronics 2022).
- Spearman ρ(N, Pe) = 0.883 (p <0.001) — udio lokaliziranih čvorova raste s gustoćom.

