# Dataset: VTL simulation results (5184 runs)

Raw results of all simulation runs reported in the paper. Each CSV file
contains one algorithm configuration; the file name encodes it as
`{IntersectionAlgorithm}{VoronoiMode}{AreaSide}.csv` — e.g.
`SweepLineLocal400.csv` = sweep line intersection + local Voronoi computation
in the 400 m × 400 m area.

| File | Rows | Contents |
|---|---|---|
| `GridScanGlobal200.csv`, `GridScanGlobal400.csv`, `SweepLineGlobal200.csv`, `SweepLineGlobal400.csv` | 144 each | 48 network sizes × 3 repetitions |
| `GridScanLocal200.csv`, `GridScanLocal400.csv`, `SweepLineLocal200.csv`, `SweepLineLocal400.csv` | 1152 each | 48 network sizes × 8 values of k × 3 repetitions |

Fixed parameters of all runs: seed 4275; 40% anchor nodes; communication
radius R = 20 m; K = 6 groups; grid cell size 2 m; network sizes 60–1000 in
steps of 20; uniform random deployment in a square area. For a given network
size and repetition index, all configurations share an identical node
deployment (paired design).

## Data dictionary

| Column | Unit | Description |
|---|---|---|
| `Seed` | – | Pseudo-random seed (4275 in all runs) |
| `SimulationSize_m` | m | Side length of the square deployment area (200 or 400) |
| `NumberOfNodes` | – | Total number of nodes in the network |
| `PercentageAnchor` | % | Share of anchor nodes (40 in all runs) |
| `DetectionRadius_m` | m | Communication radius R (20 in all runs) |
| `Clusters_K` | – | Number of anchor groups K (6 in all runs) |
| `GridSize_m` | m | Grid cell size of the grid scan (2 in all runs) |
| `Algorithm` | – | Intersection algorithm: `Grid Scanning` or `Sweep Line` |
| `VoronoiMode` | – | Voronoi computation mode: `Global` or `Local` |
| `k_neighbors` | – | Neighbourhood size k (local mode; 6–20 in steps of 2) |
| `AverageEstimationError` | – | Average normalized localization error E_a (mean absolute error divided by R) |
| `LocalizationError` | m | Average absolute localization error (= E_a × R) |
| `Pe` | % | Percentage of localized nodes (P_e) |
| `MainLoop` | s | Duration of the main localization loop |
| `DetectableCalc` | s | Time to determine detectable anchors |
| `GroupSplitting` | s | Time to form the K anchor groups |
| `VoronoiProc` | s | Total time of the Voronoi phase (preparation + computation + bounding of regions) |
| `VoronoiCalc` | s | Time of the Voronoi diagram computation itself (subset of `VoronoiProc`) |
| `IntersectionCalc` | s | Time of the positive-region intersection phase (grid scan or sweep line) |
| `ShapeCalc` | s | Time of geometric shape processing of regions |
| `DrawingOps` | s | Time spent on drawing operations (excluded from algorithmic comparisons) |
| `UIUpdates` | s | Time spent on GUI updates (excluded from algorithmic comparisons) |
| `TotalTime` | s | Total execution time of the simulation run |
| `UnknownNodes` | – | Number of unknown (non-anchor) nodes |
| `LocalizedNodes` | – | Number of successfully localized unknown nodes |
| `UnlocalizableNodes` | – | Number of unknown nodes that could not be localized |

## License

This dataset is licensed under the Creative Commons Attribution 4.0
International licence (CC BY 4.0) — see `LICENSE` in this directory. When
using the data, please cite the paper and/or the repository (see
`CITATION.cff` in the repository root).
