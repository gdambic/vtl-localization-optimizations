"""
Headless batch runner za VTL simulator.

Pokrece STVARNI kod simulatora (simulator.py / helper.py) bez otvaranja
prozora, preko Qt "offscreen" platforme. Rezultate zapisuje u CSV nakon
SVAKOG runa (sigurno prekinuti u bilo kojem trenutku; kod ponovnog
pokretanja vec izracunati redovi se preskacu).

Primjeri:
  python headless_batch.py --out ..\\..\\Simulacije\\sim_a1.csv --area 100 --sizes 140 ^
      --seeds 4275,1001,1002,1003,1004 --anchor-ratio 28 --configs grid-global

  python headless_batch.py --out ..\\..\\Simulacije\\sim_b.csv --area 200 --sizes 100:900:100 ^
      --seeds 4275,1001,1002 --anchor-ratio 20 ^
      --configs grid-global,sweep-local:8,sweep-local:10,sweep-local:12,sweep-local:16

Sintaksa:
  --sizes    jedan broj ("140"), lista ("60,100,200") ili raspon "start:stop:step" ("60:1000:20", ukljucivo)
  --seeds    lista seedova, npr. "4275,1001,1002" (dozvoljeno 0-10000)
  --configs  lista konfiguracija <alg>-<mode>[:k]
             alg = grid | sweep ; mode = global | local (local OBAVEZNO s :k)
             npr. "grid-global,sweep-global,grid-local:12,sweep-local:12"

Redoslijed petlji: seed -> velicina -> konfiguracija. Prije svake
konfiguracije cvorovi se ponovno generiraju s istim seedom, pa sve
konfiguracije za dani (seed, velicina) rade na IDENTICNOM razmjestaju
(upareni dizajn, isto kao u radu).
"""
import argparse
import contextlib
import csv
import io
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIELDS_KEY = ("Seed", "SimulationSize_m", "NumberOfNodes", "PercentageAnchor",
              "Algorithm", "VoronoiMode", "k_neighbors")


def parse_sizes(spec):
    if ":" in spec:
        start, stop, step = (int(x) for x in spec.split(":"))
        return list(range(start, stop + 1, step))
    return [int(x) for x in spec.split(",")]


def parse_configs(spec):
    configs = []
    for token in spec.split(","):
        token = token.strip()
        base, _, k = token.partition(":")
        alg, _, mode = base.partition("-")
        alg_name = {"grid": "Grid Scanning", "sweep": "Sweep Line"}[alg]
        mode_name = {"global": "Global", "local": "Local"}[mode]
        if mode_name == "Local" and not k:
            raise SystemExit(f"Konfiguracija '{token}': local zahtijeva :k (npr. sweep-local:12)")
        configs.append((alg_name, mode_name, int(k) if k else None))
    return configs


def existing_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                keys.add(tuple(str(row.get(k, "")) for k in FIELDS_KEY))
    return keys


def main():
    ap = argparse.ArgumentParser(description="Headless batch runner za VTL simulator")
    ap.add_argument("--out", required=True, help="Izlazni CSV (append + resume)")
    ap.add_argument("--area", type=int, required=True, help="Velicina prostora u m (50-500)")
    ap.add_argument("--sizes", required=True, help="Broj cvorova: N | N1,N2,... | start:stop:step")
    ap.add_argument("--seeds", required=True, help="Seedovi, npr. 4275,1001,1002")
    ap.add_argument("--anchor-ratio", type=int, default=40, help="Postotak sidrenih cvorova (default 40)")
    ap.add_argument("--configs", required=True, help="grid-global,sweep-local:12,...")
    ap.add_argument("--radius", type=int, default=20, help="Komunikacijski radijus R u m (default 20)")
    ap.add_argument("--clusters", type=int, default=6, help="Broj grupa K (default 6)")
    ap.add_argument("--grid", type=int, default=2, help="Velicina grid celije u m (default 2)")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)
    seeds = [int(s) for s in args.seeds.split(",")]
    configs = parse_configs(args.configs)

    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    import simulator
    simulator.app = app  # calculate() referencira globalni 'app'

    win = simulator.MainWindow()
    # Isprazni event queue: konstruktor zakaze QTimer.singleShot(generate_nodes)
    # koji bi se inace okinuo usred calculate() (processEvents) i resetirao RNG.
    app.processEvents()
    win.param_inputs["Simulation size in meters"].setValue(args.area)
    win.param_inputs["Percentage of anchor nodes"].setValue(args.anchor_ratio)
    win.param_inputs["Detection radius (m)"].setValue(args.radius)
    win.param_inputs["Number of clusters"].setValue(args.clusters)
    win.param_inputs["Grid size (m)"].setValue(args.grid)

    done = existing_keys(args.out)
    total = len(seeds) * len(sizes) * len(configs)
    print(f"Ukupno runova: {total}; vec gotovo u CSV-u: {len(done)}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    header_written = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    counter = 0
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = None
        for seed in seeds:
            for nt in sizes:
                for alg_name, mode_name, k in configs:
                    counter += 1
                    key = (str(seed), str(args.area), str(nt), str(args.anchor_ratio),
                           alg_name, mode_name, str(k if k is not None else 5))
                    label = f"[{counter}/{total}] seed={seed} N={nt} {alg_name}+{mode_name}" + \
                            (f" k={k}" if k else "")
                    if key in done:
                        print(f"{label} -> preskacem (vec u CSV-u)", flush=True)
                        continue

                    win.seed_input.setValue(seed)
                    win.param_inputs["Number of nodes"].setValue(nt)
                    win.param_inputs["grid_algorithm"].setCurrentText(alg_name)
                    if mode_name == "Global":
                        win.global_radio.setChecked(True)
                        win.k_neighbors_spin.setValue(5)  # neiskoristeno; fiksno radi resume kljuca
                    else:
                        win.local_radio.setChecked(True)
                        win.k_neighbors_spin.setValue(k)

                    t0 = time.perf_counter()
                    # generate_nodes resetira np.random.seed -> identican razmjestaj
                    # i identicno grupiranje za sve konfiguracije istog (seed, N)
                    with contextlib.redirect_stdout(io.StringIO()):
                        win.generate_nodes(noPlot=True)
                        win.calculate(noPlot=True)
                    wall = time.perf_counter() - t0

                    rec = win.collect_current_run_record()
                    rec["WallClock_s"] = f"{wall:.1f}"
                    rec["Timestamp"] = datetime.now().isoformat(timespec="seconds")

                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(rec.keys()))
                        if not header_written:
                            writer.writeheader()
                    writer.writerow(rec)
                    f.flush()
                    print(f"{label} -> Ea={rec['AverageEstimationError']} Pe={rec['Pe']} "
                          f"({wall:.1f} s)", flush=True)

    print("GOTOVO.", flush=True)


if __name__ == "__main__":
    main()
