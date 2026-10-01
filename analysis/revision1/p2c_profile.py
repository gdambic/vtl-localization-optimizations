"""
P2c (Major Revision 1, R1-4A): profiler sanity check.

cProfile jednog runa (N=600, 200 m, seed 4275) za grid-global i sweep-local:12.
Cilj: pokazati da vremena faza dominiraju geometrijske operacije (qhull/Shapely,
C ekstenzije), a ne Python-objektni overhead, te da obje varijante dijele iste
kodne putanje zajednickih faza.
"""
import cProfile
import contextlib
import io
import os
import pstats
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "simulator")
sys.path.insert(0, SIM)

from PyQt6.QtWidgets import QApplication
app = QApplication([])
import simulator
simulator.app = app

win = simulator.MainWindow()
app.processEvents()
win.param_inputs["Simulation size in meters"].setValue(200)
win.param_inputs["Percentage of anchor nodes"].setValue(40)
win.param_inputs["Detection radius (m)"].setValue(20)
win.param_inputs["Number of clusters"].setValue(6)
win.param_inputs["Grid size (m)"].setValue(2)
win.seed_input.setValue(4275)
win.param_inputs["Number of nodes"].setValue(600)

for label, alg, mode, k in [("grid-global", "Grid Scanning", "Global", None),
                            ("sweep-local:12", "Sweep Line", "Local", 12)]:
    win.param_inputs["grid_algorithm"].setCurrentText(alg)
    if mode == "Global":
        win.global_radio.setChecked(True)
        win.k_neighbors_spin.setValue(5)
    else:
        win.local_radio.setChecked(True)
        win.k_neighbors_spin.setValue(k)

    pr = cProfile.Profile()
    with contextlib.redirect_stdout(io.StringIO()):
        win.generate_nodes(noPlot=True)
        pr.enable()
        win.calculate(noPlot=True)
        pr.disable()

    st = pstats.Stats(pr)
    total = st.total_tt

    # agregacija tottime po "komponenti"
    comp = {}
    for (fname, lineno, func), (cc, nc, tt, ct, callers) in st.stats.items():
        f = fname.replace("\\", "/").lower()
        if "shapely" in f:
            key = "shapely (C geometrija)"
        elif "qhull" in f or (f == "~" and "qhull" in func.lower()):
            key = "scipy qhull (C)"
        elif "scipy" in f:
            key = "scipy (ostalo)"
        elif "foronoi" in f:
            key = "foronoi (Python)"
        elif "numpy" in f:
            key = "numpy"
        elif "helper.py" in f:
            key = "helper.py (nas kod)"
        elif "simulator.py" in f:
            key = "simulator.py (nas kod)"
        elif f == "~":
            key = f"builtin: {func}" if tt / total > 0.03 else "builtins (ostalo)"
        else:
            key = "ostalo (stdlib/Qt)"
        comp[key] = comp.get(key, 0.0) + tt

    print(f"\n=== {label}: ukupno {total:.1f} s (profilirano) ===")
    for key, tt in sorted(comp.items(), key=lambda kv: -kv[1]):
        if tt / total >= 0.005:
            print(f"  {key:35s} {tt:8.2f} s  {100*tt/total:5.1f} %")

    print("  -- top 10 funkcija po tottime --")
    for (fname, lineno, func), (cc, nc, tt, ct, callers) in sorted(
            st.stats.items(), key=lambda kv: -kv[1][2])[:10]:
        short = fname.replace("\\", "/").split("/")[-1]
        print(f"    {tt:8.2f} s  {nc:>9d}x  {short}:{lineno} {func}")
