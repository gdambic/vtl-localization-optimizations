"""
Headless batch runner za VTL simulator — varijanta s RSSI shadowing sumom (SIM-D,
Major Revision 1). Isto ponasanje kao headless_batch.py, uz dodatno:

  --rssi-noise-sigma S   sigma log-normalnog shadowinga u dB (default 0 = bez suma)
  --path-loss-exp ETA    path-loss eksponent (default 3.0)

Sum se primjenjuje ISKLJUCIVO na odabir najjaceg sidra unutar grupe (path-loss
ekvivalent 10*eta*log10(d) + N(0, sigma^2) dB); detektabilnost ostaje geometrijska
(binarni disk model). Sum dolazi iz zasebnog RNG-a (np.random.default_rng(seed)),
reinicijaliziranog prije SVAKOG runa, pa sve konfiguracije istog (seed, N, sigma)
vide identicnu realizaciju suma -> upareni dizajn ocuvan. Za sigma=0 kod je
bit-for-bit identican originalu.

Primjer:
  python headless_batch_noise.py --out "..\\..\\Major Revision 1\\simd\\sim_d.csv" ^
      --area 200 --sizes 200,600,1000 --seeds 4275,2001,2002 --anchor-ratio 40 ^
      --configs grid-global,sweep-local:10,sweep-local:12 --rssi-noise-sigma 4
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
              "Algorithm", "VoronoiMode", "k_neighbors", "RssiNoiseSigma")


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
    ap = argparse.ArgumentParser(description="Headless batch runner za VTL simulator (+RSSI sum)")
    ap.add_argument("--out", required=True, help="Izlazni CSV (append + resume)")
    ap.add_argument("--area", type=int, required=True, help="Velicina prostora u m (50-500)")
    ap.add_argument("--sizes", required=True, help="Broj cvorova: N | N1,N2,... | start:stop:step")
    ap.add_argument("--seeds", required=True, help="Seedovi, npr. 4275,2001,2002")
    ap.add_argument("--anchor-ratio", type=int, default=40, help="Postotak sidrenih cvorova (default 40)")
    ap.add_argument("--configs", required=True, help="grid-global,sweep-local:12,...")
    ap.add_argument("--radius", type=int, default=20, help="Komunikacijski radijus R u m (default 20)")
    ap.add_argument("--clusters", type=int, default=6, help="Broj grupa K (default 6)")
    ap.add_argument("--grid", type=int, default=2, help="Velicina grid celije u m (default 2)")
    ap.add_argument("--rssi-noise-sigma", type=float, default=0.0,
                    help="Sigma shadowinga u dB (default 0 = bez suma)")
    ap.add_argument("--path-loss-exp", type=float, default=3.0,
                    help="Path-loss eksponent eta (default 3.0)")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)
    seeds = [int(s) for s in args.seeds.split(",")]
    configs = parse_configs(args.configs)

    import numpy as np
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
    win.path_loss_exponent = args.path_loss_exp

    done = existing_keys(args.out)
    total = len(seeds) * len(sizes) * len(configs)
    print(f"Ukupno runova: {total}; vec gotovo u CSV-u: {len(done)}; "
          f"sigma={args.rssi_noise_sigma} dB, eta={args.path_loss_exp}", flush=True)

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
                           alg_name, mode_name, str(k if k is not None else 5),
                           f"{args.rssi_noise_sigma:g}")
                    label = f"[{counter}/{total}] seed={seed} N={nt} {alg_name}+{mode_name}" + \
                            (f" k={k}" if k else "") + f" sigma={args.rssi_noise_sigma:g}"
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

                    # Zaseban RNG za sum, reinicijaliziran PRIJE svakog runa:
                    # identicna realizacija suma za sve konfiguracije istog (seed, N, sigma).
                    win.rssi_noise_sigma = args.rssi_noise_sigma
                    win.rssi_noise_rng = np.random.default_rng(seed)

                    t0 = time.perf_counter()
                    # generate_nodes resetira np.random.seed -> identican razmjestaj
                    # i identicno grupiranje za sve konfiguracije istog (seed, N)
                    with contextlib.redirect_stdout(io.StringIO()):
                        win.generate_nodes(noPlot=True)
                        win.calculate(noPlot=True)
                    wall = time.perf_counter() - t0

                    rec = win.collect_current_run_record()
                    rec["RssiNoiseSigma"] = f"{args.rssi_noise_sigma:g}"
                    rec["PathLossExp"] = f"{args.path_loss_exp:g}"
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
