from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QProgressBar, QLabel, QSpinBox,
                             QGroupBox, QFormLayout, QTabWidget, QTreeWidget, QTreeWidgetItem, QComboBox, QRadioButton, QButtonGroup,
                             QFileDialog, QCheckBox)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor
import sys
from matplotlib.pyplot import grid
import matplotlib.pyplot as plt
import csv
import numpy as np
import random
import pyqtgraph as pg
import math
import time
import copy
import os
from scipy.spatial import Voronoi
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from helper import (NodeList, grid_scanning, voronoi_finite_polygons_2d, get_region_coordinates, local_voronoi, sweep_line_algorithm)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voronoi Diagram Simulator")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
                
        self.left_layout = QVBoxLayout()
        self.main_layout.addLayout(self.left_layout)
        
        self.plot_tabs = QTabWidget()
        self.left_layout.addWidget(self.plot_tabs)
        
        self.plot_widgets = []
        self.param_inputs = {}

        self.create_nodes_plot_widget()
        self.create_result_plot_widget()
        self.create_voronoi_plot_widget()
        
        QTimer.singleShot(0, self.generate_nodes)

        
    def create_result_plot_widget(self):
        tab = QWidget()
        tab_layout = QHBoxLayout()
        tab.setLayout(tab_layout)
        
        left_layout = QVBoxLayout()
        tab_layout.addLayout(left_layout)
            
        self.plot_widget_result = pg.PlotWidget()
        self.plot_widget_result.setBackground('w')
        self.plot_widget_result.showGrid(x=True, y=True)
        self.plot_widget_result.setXRange(0, 200)
        self.plot_widget_result.setYRange(0, 200)
        self.plot_widget_result.enableAutoRange(x=False, y=False)
        
        left_layout.addWidget(self.plot_widget_result)
        self.plot_tabs.addTab(tab, f"Results")
        self.plot_widgets.append(self.plot_widget_result)

        self.button = QPushButton("Generate Voronoi Diagram")
        left_layout.addWidget(self.button)
        self.button.clicked.connect(self.calculate)
        
        params = {
            "Detection radius (m)": (1, 100, 20),
            "Number of clusters": (1, 20, 6),
            "Grid size (m)": (1, 50, 2)
        }

        self.vertical_layout = QVBoxLayout()
        tab_layout.addLayout(self.vertical_layout)

        self.param_group = QGroupBox("Properties")
        self.param_layout = QFormLayout()
        self.param_group.setLayout(self.param_layout)
        self.vertical_layout.addWidget(self.param_group)

        for name, (min_val, max_val, default) in params.items():
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            self.param_layout.addRow(QLabel(name), spin)
            self.param_inputs[name] = spin
            
        self.algorithm_select = QComboBox()
        self.algorithm_select.addItems(["Grid Scanning", "Sweep Line"])
        self.param_layout.addRow(QLabel("Intersection algorithm"), self.algorithm_select)
        self.param_inputs["grid_algorithm"] = self.algorithm_select
        
        self.global_radio = QRadioButton("Global Voronoi")
        self.local_radio = QRadioButton("Local Voronoi")
        
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.global_radio)
        self.button_group.addButton(self.local_radio)
        
        self.global_radio.setChecked(True)
        
        self.param_layout.addRow(self.global_radio)
        self.param_layout.addRow(self.local_radio)
        
        self.k_neighbors_spin = QSpinBox()
        self.k_neighbors_spin.setRange(1, 50)
        self.k_neighbors_spin.setValue(5)
        self.k_neighbors_spin.setEnabled(False)
        self.param_layout.addRow(QLabel("Local voronoi k neighbors"), self.k_neighbors_spin)
        self.param_inputs["k_neighbors"] = self.k_neighbors_spin
        
        # Enableanje polja za k susjede ovisno o odabranom algoritmu
        self.global_radio.toggled.connect(self.toggle_k_neighbors)
        self.local_radio.toggled.connect(self.toggle_k_neighbors)
        
        self.results_group = QGroupBox("Results")
        self.results_layout = QFormLayout()
        self.results_group.setLayout(self.results_layout)
        self.vertical_layout.addWidget(self.results_group)
        self.param_inputs["AverageEstimationError"] = QLabel("N/A")
        self.param_inputs["LocalizationError"] = QLabel("N/A")
        self.param_inputs["Pe"] = QLabel("N/A")
        self.param_inputs["UnknownNodes"] = QLabel("N/A")
        self.param_inputs["UnknownNodes"].setVisible(False)
        self.param_inputs["LocalizedNodes"] = QLabel("N/A")
        self.param_inputs["LocalizedNodes"].setVisible(False)
        self.param_inputs["UnlocalizableNodes"] = QLabel("N/A")
        self.param_inputs["UnlocalizableNodes"].setVisible(False) 
        self.param_inputs["MainLoop"] = QLabel("N/A")
        self.param_inputs["DetectableCalc"] = QLabel("N/A")
        self.param_inputs["GroupSplitting"] = QLabel("N/A")
        self.param_inputs["VoronoiProc"] = QLabel("N/A")
        self.param_inputs["VoronoiCalc"] = QLabel("N/A")
        self.param_inputs["IntersectionCalc"] = QLabel("N/A")
        self.param_inputs["ShapeCalc"] = QLabel("N/A")
        self.param_inputs["DrawingOps"] = QLabel("N/A")
        self.param_inputs["UIUpdates"] = QLabel("N/A")
        self.param_inputs["TotalTime"] = QLabel("N/A")

        self.results_layout.addRow(QLabel("Average estimation error (Ea):"), self.param_inputs["AverageEstimationError"])
        self.results_layout.addRow(QLabel("Localization error:"), self.param_inputs["LocalizationError"])
        self.results_layout.addRow(QLabel("Percentage localized:"), self.param_inputs["Pe"])
        self.results_layout.addRow(QLabel("Main processing loop:"), self.param_inputs["MainLoop"])
        self.results_layout.addRow(QLabel("  - Detectable nodes calc:"), self.param_inputs["DetectableCalc"])
        self.results_layout.addRow(QLabel("  - Group splitting:"), self.param_inputs["GroupSplitting"])
        self.results_layout.addRow(QLabel("  - Voronoi processing:"), self.param_inputs["VoronoiProc"])
        self.results_layout.addRow(QLabel("  - Voronoi cell calculations:"), self.param_inputs["VoronoiCalc"])
        self.results_layout.addRow(QLabel("  - Intersections calculations:"), self.param_inputs["IntersectionCalc"])
        self.results_layout.addRow(QLabel("  - Shape calculations:"), self.param_inputs["ShapeCalc"])
        self.results_layout.addRow(QLabel("  - Drawing operations:"), self.param_inputs["DrawingOps"])
        self.results_layout.addRow(QLabel("  - UI updates:"), self.param_inputs["UIUpdates"])
        self.results_layout.addRow(QLabel("Total:"), self.param_inputs["TotalTime"])

        self.progressBar = QProgressBar(minimum=0, maximum=120)
        left_layout.addWidget(self.progressBar)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        
        self.batch_group = QGroupBox("Batch options")
        self.batch_layout = QFormLayout(self.batch_group)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 500)
        self.batch_spin.setValue(5)
        self.batch_layout.addRow("Batch iterations:", self.batch_spin)

        self.increment_checkbox = QCheckBox("Increment nodes each iteration")
        self.increment_checkbox.toggled.connect(lambda checked: self.toggle_batch_options(checked))
        self.batch_layout.addRow(self.increment_checkbox)
        
        self.plot_voronoi_checkbox = QCheckBox("Save Voronoi plots (debug)")
        self.param_layout.addRow(self.plot_voronoi_checkbox)

        self.start_nodes_spin = QSpinBox()
        self.start_nodes_spin.setRange(1, 10000)
        self.start_nodes_spin.setValue(self.param_inputs["Number of nodes"].value())
        self.start_nodes_spin.setEnabled(False)
        self.batch_layout.addRow("Start number of nodes:", self.start_nodes_spin)

        self.increment_step_spin = QSpinBox()
        self.increment_step_spin.setRange(1, 1000)
        self.increment_step_spin.setValue(10)
        self.increment_step_spin.setEnabled(False)
        self.batch_layout.addRow("Increment step:", self.increment_step_spin)

        self.batchProgressBar = QProgressBar(minimum=0, maximum=100)
        self.batchProgressBar.setValue(0)
        self.batchProgressBar.setVisible(False)
        self.batch_layout.addRow(self.batchProgressBar)

        self.batch_button = QPushButton("Run Batch")
        self.batch_button.clicked.connect(self.run_batch)
        self.batch_layout.addRow(self.batch_button)

        self.export_button = QPushButton("Export Batch CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_batch_csv)
        self.batch_layout.addRow(self.export_button)

        self.vertical_layout.addWidget(self.batch_group)
        
        self.export_all_button = QPushButton("Export All Simulations CSV")
        self.vertical_layout.addWidget(self.export_all_button)
        self.export_all_button.clicked.connect(self.export_all_csv)

        self.batch_results = []
        self.all_runs_history = []
        
    def toggle_k_neighbors(self):
        self.k_neighbors_spin.setEnabled(self.local_radio.isChecked())    
    
    def toggle_batch_options(self, enabled):
        self.start_nodes_spin.setEnabled(enabled)
        self.increment_step_spin.setEnabled(enabled)
    
    def run_batch(self):
        self.export_button.setEnabled(False)
        self.batch_results.clear()
        iterations = self.batch_spin.value()
        self.batchProgressBar.setVisible(True)
        self.batchProgressBar.setMaximum(iterations)
        self.batchProgressBar.setValue(0)
        
        if self.increment_checkbox.isChecked():
            self.param_inputs["Number of nodes"].setValue(self.start_nodes_spin.value())
            self.generate_nodes(noPlot=True)

        for i in range(iterations):
            self.batchProgressBar.setValue(i + 1)
            self.calculate(noPlot=True)
            result = {
                "Iteration": i + 1,
                "Ea": self.param_inputs["AverageEstimationError"].text(),
                "LocalizationError": self.param_inputs["LocalizationError"].text(),
                "Pe": self.param_inputs["Pe"].text(),
                "TotalTime": self.param_inputs["TotalTime"].text(),
                "NumberOfNodes": self.param_inputs["Number of nodes"].value()
            }
            self.batch_results.append(result)
            
            if self.increment_checkbox.isChecked():
                current_nodes = self.param_inputs["Number of nodes"].value()
                self.param_inputs["Number of nodes"].setValue(current_nodes + self.increment_step_spin.value())  
                self.generate_nodes(noPlot=True)
                              
            
        self.export_button.setEnabled(True)
        self.batchProgressBar.setVisible(False)

    def export_batch_csv(self):
        if not self.batch_results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Batch Results", "", "CSV Files (*.csv)")
        if path:
            with open(path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.batch_results[0].keys())
                writer.writeheader()
                writer.writerows(self.batch_results)
    
    def create_nodes_plot_widget(self):
        tab = QWidget()
        tab_layout = QVBoxLayout()
        tab.setLayout(tab_layout)
            
        self.plot_widget_nodes = pg.PlotWidget()
        self.plot_widget_nodes.setBackground('w')
        self.plot_widget_nodes.showGrid(x=True, y=True)
        self.plot_widget_nodes.setXRange(0, 200)
        self.plot_widget_nodes.setYRange(0, 200)
        self.plot_widget_nodes.enableAutoRange(x=False, y=False)
        
        tab_layout.addWidget(self.plot_widget_nodes)
        self.plot_tabs.addTab(tab, f"Nodes")
        self.plot_widgets.append(self.plot_widget_nodes)

        hlayout = QHBoxLayout()
        tab_layout.addLayout(hlayout)

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 10000)
        self.seed_input.setValue(42)

        form_layout = QFormLayout()
        form_layout.addRow("Seed:", self.seed_input)
        
        
        params = {
            "Simulation size in meters": (50, 500, 200),
            "Number of nodes": (10, 1000, 120),
            "Percentage of anchor nodes": (0, 100, 40),
        }

        for name, (min_val, max_val, default) in params.items():
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            form_layout.addRow(QLabel(name), spin)
            self.param_inputs[name] = spin


        hlayout.addLayout(form_layout)

        self.button = QPushButton("Random Seed")
        hlayout.addWidget(self.button)
        self.button.clicked.connect(self.randomize_seed)

        self.button = QPushButton("Generate Nodes")
        hlayout.addWidget(self.button)
        self.button.clicked.connect(self.generate_nodes)

    def randomize_seed(self):
        new_seed = random.randint(0, 10000)
        self.seed_input.setValue(new_seed)

    def create_node_plot(self, nodes):
        self.plot_widget_nodes.clear()
        for node in nodes.all_nodes:
            if node.type == "A":
                self.plot_widget_nodes.plot([node.x], [node.y], pen=None, symbol='o', symbolBrush='b', symbolSize=10)
            else:
                self.plot_widget_nodes.plot([node.x], [node.y], pen=None, symbol='x', symbolBrush='r', symbolSize=10)
            text = pg.TextItem(text=node.name, anchor=(0, 1), color='g')
            text.setPos(node.x, node.y)
            self.plot_widget_nodes.addItem(text)
        
    def generate_nodes(self, noPlot=False):
        S_m = self.param_inputs["Simulation size in meters"].value()
        Nt = self.param_inputs["Number of nodes"].value()
        ra_percent = self.param_inputs["Percentage of anchor nodes"].value()
        nodes = NodeList(S_m, Nt, ra_percent, seed=self.seed_input.value())
        self.nodes = nodes
        if not noPlot:
            self.create_node_plot(nodes)
        return nodes
    
    def create_voronoi_plot_widget(self):
        self.voronoi_plots = {}
        
        tab = QWidget()
        tab_layout = QHBoxLayout()
        tab.setLayout(tab_layout)
        
        tab_left_layout = QVBoxLayout()
        tab_layout.addLayout(tab_left_layout)

        self.plot_widget_voronoi = pg.PlotWidget()
        self.plot_widget_voronoi.setBackground('w')
        self.plot_widget_voronoi.showGrid(x=True, y=True)
        self.plot_widget_voronoi.setXRange(0, 200)
        self.plot_widget_voronoi.setYRange(0, 200)
        self.plot_widget_voronoi.enableAutoRange(x=False, y=False)

        tab_left_layout.addWidget(self.plot_widget_voronoi)
        self.plot_tabs.addTab(tab, f"Voronoi")
        self.plot_widgets.append(self.plot_widget_voronoi)

        self.button = QPushButton("Generate Voronoi Diagram")
        tab_left_layout.addWidget(self.button)
        # self.button.clicked.connect(self.generate_voronoi)
        
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("Voronoi Plots")
        self.tree_widget.itemClicked.connect(self.on_item_clicked)

        tab_layout.addWidget(self.tree_widget)

    def on_item_clicked(self, item, column):
        data = item.data(0, 1)
        if data:  # klusterima je data postavljena
            node_name, k = data
            regions, vertices = self.voronoi_plots[node_name][k]
            self.generate_voronoi(regions, vertices, color='b')

    def setup_tree_widget(self):
        self.tree_widget.clear()
        self.plot_widget_voronoi.clear()
        for node in self.nodes.all_nodes:
            if node.type == "A":
                self.plot_widget_voronoi.plot([node.x], [node.y], pen=None, symbol='o', symbolBrush='b', symbolSize=10)
            else:
                self.plot_widget_voronoi.plot([node.x], [node.y], pen=None, symbol='x', symbolBrush='r', symbolSize=10)
            text = pg.TextItem(text=node.name, anchor=(0, 1), color='g')
            text.setPos(node.x, node.y)
            self.plot_widget_voronoi.addItem(text)
            
        for node_name, clusters in self.voronoi_plots.items():
            node_item = QTreeWidgetItem([node_name])
            node_item.setExpanded(False)
            for k in clusters:
                cluster_item = QTreeWidgetItem([f"Cluster {k}"])
                cluster_item.setData(0, 1, (node_name, k))
                node_item.addChild(cluster_item)
            self.tree_widget.addTopLevelItem(node_item)
    
    def generate_voronoi(self, regions, vertices, color):
        self.plot_widget_voronoi.clear()
        for node in self.nodes.all_nodes:
            if node.type == "A":
                self.plot_widget_voronoi.plot([node.x], [node.y], pen=None, symbol='o', symbolBrush='b', symbolSize=10)
            else:
                self.plot_widget_voronoi.plot([node.x], [node.y], pen=None, symbol='x', symbolBrush='r', symbolSize=10)
            text = pg.TextItem(text=node.name, anchor=(0, 1), color='g')
            text.setPos(node.x, node.y)
            self.plot_widget_voronoi.addItem(text)
        for region in regions.values():
            polygon = [vertices[i] for i in region]
            if len(polygon) >= 3:
                self.plot_widget_voronoi.plot([p[0] for p in polygon], [p[1] for p in polygon], pen=pg.mkPen(), symbol='o')

    def calculate(self, noPlot=False, __DEBUG__ = False):
        # image_px = 700 # rezolucija slike
        # S_m = 200 # veličina simulacije u metrima
        # Nt = 120 # broj čvorova
        # ra_percent = 40 # postotak Anchor čvorova
        # #R_m = 20
        # Ra_m = 20 # detekcijski radijus u metrima
        # EGPS_m = 0
        # K = 6 # broj grupa
        # l_m2 = 2 # veličina mreže u metrima
        # algorithm izbor algoritma za presjek celija 
        # k_neibgours parametar za lokalni voronoi
        simulation_size_meters = self.param_inputs["Simulation size in meters"].value()
        number_of_nodes = self.param_inputs["Number of nodes"].value()
        ra_percent = self.param_inputs["Percentage of anchor nodes"].value()
        detection_radius_meters = self.param_inputs["Detection radius (m)"].value()
        number_of_clusters = self.param_inputs["Number of clusters"].value()
        l_m2 = self.param_inputs["Grid size (m)"].value()
        algorithm = self.param_inputs["grid_algorithm"].currentText()
        k_neighbours = self.k_neighbors_spin.value()
        global_voronoi = self.global_radio.isChecked()
        
        number_of_unknown_nodes = number_of_nodes * (1 - ra_percent / 100)
        
        time_voronoi_diagram = 0
        time_detectable_calc = 0
        time_group_splitting = 0
        time_voronoi_processing = 0
        time_intersection_algorithm = 0
        time_shape_calc = 0
        time_drawing = 0
        time_ui_updates = 0
        
        all_start_time = time.perf_counter()

        ui_start = time.perf_counter()
        self.plot_widget_result.clear()
        self.progressBar.setValue(0)
        self.progressBar.setVisible(True)
        self.progressBar.setMaximum(int(number_of_unknown_nodes))
        time_ui_updates += time.perf_counter() - ui_start

        nodes = copy.deepcopy(self.nodes)

        sum_of_all_errors = 0

        # Sad prođemo kroz sve unknown čvorove i izvrtimo algoritam
        main_loop_start = time.perf_counter()
        for current_node in nodes.unknown_nodes():
            
            # print(f"\nCurrent node: {current_node}")
            # print(f"All anchors: {NodeList.stringify(nodes.anchors())}")

            detectable_start = time.perf_counter()
            detectable_anchors, undetectable_anchors = nodes.calculate_detectable_nodes(detection_radius_meters, current_node)
            time_detectable_calc += time.perf_counter() - detectable_start
            
            # print(f"Detectable nodes: {NodeList.stringify(detectable_anchors)}")
            # print(f"Undetectable nodes: {NodeList.stringify(undetectable_anchors)}")

            ui_start = time.perf_counter()
            self.progressBar.setValue(self.progressBar.value() + 1)
            time_ui_updates += time.perf_counter() - ui_start

            if len(detectable_anchors) == 0:
                print(f"Node {current_node.name} is unlocalizable")
                nodes.unlocalizable_nodes.append(current_node)
                
                draw_start = time.perf_counter()
                if not noPlot:
                    self.plot_widget_result.plot([current_node.x], [current_node.y], pen=None, symbol='s', symbolBrush='r', symbolSize=10)
                    text_calc = pg.TextItem(text=f"{current_node.name}", anchor=(0, 1), color='r')
                    text_calc.setPos(current_node.x, current_node.y)
                    self.plot_widget_result.addItem(text_calc)
                time_drawing += time.perf_counter() - draw_start
                continue

            positive_regions = {}

            for k in range(int(number_of_clusters / 2)):
                # print(f"k={k}")

                split_start = time.perf_counter()
                group_one, group_two, detectable_in_one, detectable_in_two = nodes.split_anchors(detectable_anchors, undetectable_anchors)
                time_group_splitting += time.perf_counter() - split_start

                for group_number, group, detectables in [(1, group_one, detectable_in_one), (2, group_two, detectable_in_two)]:
                    # print(f"\tGroup {k * 2 + group_number} (n={len(group)}): {NodeList.stringify(group)}")

                    # Ako nema detectable čvorova, ova grupa nema pozitivnu regiju
                    if detectables == 0:
                        print("\tNo detectable anchors, so no positive region")
                        continue

                    voronoi_proc_start = time.perf_counter()
                    
                    # Voronoi biblioteka očekuje listu listi s x,y koordinatama
                    # [[np.float64(59.86584841970366), np.float64(35.67533266935893)], ...]
                    group_v = [[a[0].x, a[0].y] for a in group]
                    
                    # Najbliži čvor
                    closest_node = min([n for n in group if n[1] <= detection_radius_meters], key=lambda item: item[1])
                    # print(f"\t\tClosest node: {NodeList.stringify([closest_node])}")

                    simulation_size = self.param_inputs["Simulation size in meters"].value()
                    bounds = (0, 0, simulation_size, simulation_size)

                    # regions kao ključ ima indeks u only_anchors koji kaže koji čvor je vlasnik regije
                    start_time = time.perf_counter()
                    if global_voronoi:
                        voronoi = Voronoi(group_v, furthest_site=False)
                    else:
                        voronoi = local_voronoi(group_v, (closest_node[0].x, closest_node[0].y), k=k_neighbours)

                    diff1 = time.perf_counter() - start_time
                    time_voronoi_diagram += diff1
                    regions, vertices = voronoi_finite_polygons_2d(voronoi, simulation_bounds=bounds)

                    # print_regions(voronoi.points, nodes.anchors(), regions, vertices)
                        
                    if current_node.name not in self.voronoi_plots:
                        self.voronoi_plots[current_node.name] = {}
                    
                    self.voronoi_plots[current_node.name][k] = (regions.copy(), vertices.copy())
                    
                    if self.plot_voronoi_checkbox.isChecked():
                        voronoi_type = "Global" if global_voronoi else "Local"
                        self.plot_voronoi_diagram(regions, vertices, voronoi.points,
                                                current_node.name, k * 2 + group_number, voronoi_type)

                    # Želimo preslikati točku iz voronoi objekta u naš nodes.all_anchor objekt
                    positive_region = nodes.map(regions, voronoi.points, closest_node[0])

                    positive_regions[k * 2 + group_number] = get_region_coordinates(positive_region, vertices)
                    # print_region(positive_region, vertices)
                    
                    time_voronoi_processing += time.perf_counter() - voronoi_proc_start

            sweep_start = time.perf_counter()
            if algorithm == "Grid Scanning":
                grid = grid_scanning(simulation_size_meters, l_m2,
                                     positive_regions, 
                                     unknown_node_xy=(current_node.x, current_node.y), 
                                     detect_radius=detection_radius_meters)
            else:
                grid = sweep_line_algorithm(simulation_size_meters, l_m2, positive_regions,
                                            unknown_node_xy=(current_node.x, current_node.y), 
                                            detect_radius=detection_radius_meters)

            time_intersection_algorithm += time.perf_counter() - sweep_start

            shape_start = time.perf_counter()
            max_value = np.max(grid)
            cell_boxes = []
            cell_centers = []

            for r in range(len(grid)):
                for c in range(len(grid[r])):
                    if grid[r][c] == max_value:
                        x0, y0 = c * l_m2, r * l_m2
                        x1, y1 = x0 + l_m2, y0 + l_m2
                        cell_boxes.append(box(x0, y0, x1, y1))

                        cell_centers.append((c * l_m2 + l_m2 / 2,
                                            r * l_m2 + l_m2 / 2))

            if len(cell_boxes) == 0:
                centroid = None
            elif len(cell_boxes) == 2:
                shape = LineString(cell_centers)
                centroid = shape.centroid
            else:
                shape = unary_union(cell_boxes)
                centroid = shape.centroid

            location_calculated = (centroid.x, centroid.y) if centroid else (None, None)
            location_actual = (current_node.x, current_node.y)
            time_shape_calc += time.perf_counter() - shape_start

            # print(f"Calculated location: {location_calculated[0]}, {location_calculated[1]}")
            # print(f"Actual location: {location_actual[0]}, {location_actual[1]}")

            draw_start = time.perf_counter()
            if not noPlot:
                self.draw_points(current_node.name, location_calculated[0], location_calculated[1], location_actual[0], location_actual[1])
            time_drawing += time.perf_counter() - draw_start
            
            ui_start = time.perf_counter()
            app.processEvents()
            time_ui_updates += time.perf_counter() - ui_start

            sum_of_all_errors += math.sqrt(math.pow(location_calculated[0] - location_actual[0], 2) +
                                        math.pow(location_calculated[1] - location_actual[1], 2))

        main_loop_time = time.perf_counter() - main_loop_start
        
        number_of_localized_nodes = number_of_unknown_nodes - len(nodes.unlocalizable_nodes)
        localization_error = sum_of_all_errors / number_of_localized_nodes
        percent_localized = (number_of_localized_nodes / number_of_unknown_nodes) * 100
        avg_norm_loc_error = sum_of_all_errors / (number_of_localized_nodes * detection_radius_meters)

        total_time = time.perf_counter() - all_start_time

        self.progressBar.setValue(self.progressBar.maximum())

        self.progressBar.setVisible(False)
        self.param_inputs["AverageEstimationError"].setText(f"{avg_norm_loc_error:.8f}")
        self.param_inputs["LocalizationError"].setText(f"{localization_error:.8f}")
        self.param_inputs["Pe"].setText(f"{percent_localized:.2f}%")
        self.param_inputs["UnknownNodes"].setText(f"{int(number_of_unknown_nodes)}")
        self.param_inputs["LocalizedNodes"].setText(f"{int(number_of_localized_nodes)}")
        self.param_inputs["UnlocalizableNodes"].setText(f"{int(len(nodes.unlocalizable_nodes))}")
        self.param_inputs["MainLoop"].setText(f"{main_loop_time:.10f}")
        self.param_inputs["DetectableCalc"].setText(f"{time_detectable_calc:.10f}")
        self.param_inputs["GroupSplitting"].setText(f"{time_group_splitting:.10f}")
        self.param_inputs["VoronoiProc"].setText(f"{time_voronoi_processing:.10f}")
        self.param_inputs["VoronoiCalc"].setText(f"{time_voronoi_diagram:.10f}")
        self.param_inputs["IntersectionCalc"].setText(f"{time_intersection_algorithm:.10f}")
        self.param_inputs["ShapeCalc"].setText(f"{time_shape_calc:.10f}")
        self.param_inputs["DrawingOps"].setText(f"{time_drawing:.10f}")
        self.param_inputs["UIUpdates"].setText(f"{time_ui_updates:.10f}")
        self.param_inputs["TotalTime"].setText(f"{total_time:.10f}")

        print(f"\n=== Rezultati ===")
        print(f"Localization error (m) = {localization_error}")
        print(f"Percentage localized = {percent_localized}%")
        print(f"\n=== TIMING BREAKDOWN ===")
        print(f"Main processing loop:           {main_loop_time:.4f}s ({main_loop_time/total_time*100:.1f}%)")
        print(f"  - Detectable nodes calc:      {time_detectable_calc:.4f}s ({time_detectable_calc/total_time*100:.1f}%)")
        print(f"  - Group splitting:            {time_group_splitting:.4f}s ({time_group_splitting/total_time*100:.1f}%)")
        print(f"  - Voronoi processing:         {time_voronoi_processing:.4f}s ({time_voronoi_processing/total_time*100:.1f}%)")
        print(f"  - Voronoi cell calculations:  {time_voronoi_diagram:.4f}s ({time_voronoi_diagram/total_time*100:.1f}%)")
        print(f"  - Grid scanning:              {time_intersection_algorithm:.4f}s ({time_intersection_algorithm/total_time*100:.1f}%)")
        print(f"  - Shape calculations:         {time_shape_calc:.4f}s ({time_shape_calc/total_time*100:.1f}%)")
        print(f"  - Drawing operations:         {time_drawing:.4f}s ({time_drawing/total_time*100:.1f}%)")
        print(f"  - UI updates:                 {time_ui_updates:.4f}s ({time_ui_updates/total_time*100:.1f}%)")
        print(f"TOTAL TIME:                     {total_time:.4f}s")
        print(f"=== END TIMING BREAKDOWN ===\n")
        

        if not noPlot:
            self.setup_tree_widget()
        record = self.collect_current_run_record()
        self.all_runs_history.append(record)

        
    def get_random_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return QColor(r, g, b)

    def draw_points(self, name, calc_x, calc_y, actual_x, actual_y):
        self.plot_widget_result.plot([calc_x], [calc_y], pen=None, symbol='o', symbolBrush='b', symbolSize=10)
        text_calc = pg.TextItem(text=f"{name}", anchor=(0, 1), color='b')
        text_calc.setPos(calc_x, calc_y)
        self.plot_widget_result.addItem(text_calc)

        self.plot_widget_result.plot([actual_x], [actual_y], pen=None, symbol='x', symbolBrush='g', symbolSize=10)
        text_actual = pg.TextItem(text=f"{name}", anchor=(0, 0), color='g')
        text_actual.setPos(actual_x, actual_y)
        self.plot_widget_result.addItem(text_actual)

        dx = actual_x - calc_x
        dy = actual_y - calc_y
                    
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        angle_deg = (angle_deg * -1) +180
        
        line = pg.PlotDataItem([calc_x, actual_x], [calc_y, actual_y], pen=pg.mkPen('k', width=2))
        self.plot_widget_result.addItem(line)
        
        arrow = pg.ArrowItem(
            pos=(actual_x, actual_y),
            angle=angle_deg,
            headLen=10,
            tipAngle=30,
            baseAngle=20,
            brush='k'
        )
        self.plot_widget_result.addItem(arrow)  
        
        # print(f"Distance: {distance:.2f} at angle: {angle_deg:.2f} degrees")
        # if angle_deg > 180:
            # print(f"Angle is greater than 180, adjusting to {angle_deg - 180:.2f} degrees")
            # angle_deg = angle_deg - 180

        # label = pg.TextItem(text=f"{distance:.2f}", color='b', anchor=(0.5, 0.5))
        # label.setPos(mid_x, mid_y)
        # label.setRotation(angle_deg)
        # self.plot_widget_result.addItem(label)
        
    def collect_current_run_record(self, iteration=None):
        record = {}

        try:
            record["Seed"] = int(self.seed_input.value())
        except Exception:
            record["Seed"] = ""

        record["SimulationSize_m"] = int(self.param_inputs["Simulation size in meters"].value())
        record["NumberOfNodes"] = int(self.param_inputs["Number of nodes"].value())
        record["UnknownNodes"] = int(self.param_inputs["UnknownNodes"].text())
        record["LocalizedNodes"] = int(self.param_inputs["LocalizedNodes"].text())
        record["UnlocalizableNodes"] = int(self.param_inputs["UnlocalizableNodes"].text())
        record["PercentageAnchor"] = int(self.param_inputs["Percentage of anchor nodes"].value())

        record["DetectionRadius_m"] = int(self.param_inputs["Detection radius (m)"].value())
        record["Clusters_K"] = int(self.param_inputs["Number of clusters"].value())
        record["GridSize_m"] = int(self.param_inputs["Grid size (m)"].value())
        record["Algorithm"] = str(self.param_inputs["grid_algorithm"].currentText())
        record["VoronoiMode"] = "Global" if self.global_radio.isChecked() else "Local"
        record["k_neighbors"] = int(self.k_neighbors_spin.value())

        for key in ["AverageEstimationError", "LocalizationError", "Pe", "MainLoop",
                    "DetectableCalc", "GroupSplitting", "VoronoiProc", "VoronoiCalc",
                    "IntersectionCalc", "ShapeCalc", "DrawingOps", "UIUpdates", "TotalTime"]:
            label = self.param_inputs.get(key)
            if label is None:
                record[key] = ""
            else:
                try:
                    record[key] = label.text()
                except Exception:
                    record[key] = str(label)

        if iteration is not None:
            record["Iteration"] = int(iteration)

        ordered = {}
        if "Iteration" in record:
            ordered["Iteration"] = record.pop("Iteration")
        for k in ("Seed", "SimulationSize_m", "NumberOfNodes", "PercentageAnchor"):
            ordered[k] = record.pop(k, "")
        for k in ("DetectionRadius_m", "Clusters_K", "GridSize_m", "Algorithm", "VoronoiMode", "k_neighbors"):
            ordered[k] = record.pop(k, "")
        for k in ("AverageEstimationError", "LocalizationError", "Pe", "MainLoop",
                  "DetectableCalc", "GroupSplitting", "VoronoiProc", "VoronoiCalc",
                  "IntersectionCalc", "ShapeCalc", "DrawingOps", "UIUpdates", "TotalTime"):
            ordered[k] = record.pop(k, "")
        for k, v in record.items():
            ordered[k] = v

        return ordered

    def export_all_csv(self):
        
        if not self.all_runs_history:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save All Simulations", "", "CSV Files (*.csv)")
        if not path:
            return

        first = self.all_runs_history[0]
        fieldnames = list(first.keys())

        try:
            with open(path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.all_runs_history:
                    writer.writerow({fn: row.get(fn, "") for fn in fieldnames})
        except Exception as e:
            print(f"Failed to write CSV: {e}")

    def plot_voronoi_diagram(self, regions, vertices, points, node_name, cluster_k, voronoi_type):
        """
        Simple plotting function for Voronoi diagrams.
        Saves plots to the plots/ directory for debugging purposes.
        """
        if not os.path.exists("plots"):
            os.makedirs("plots")
        
        plt.figure(figsize=(8, 6))
        
        # Plot Voronoi diagram
        for region_idx, region_vertices in regions.items():
            if len(region_vertices) >= 3:
                polygon = [vertices[i] for i in region_vertices]
                polygon.append(polygon[0])  # Close the polygon
                xs, ys = zip(*polygon)
                plt.plot(xs, ys, 'b-', alpha=0.7)
                plt.fill(xs, ys, alpha=0.2, color='lightblue')
        
        # Plot the points
        if len(points) > 0:
            xs, ys = zip(*points)
            plt.scatter(xs, ys, c='red', s=50, zorder=5)
            
            # Add labels for points
            for i, (x, y) in enumerate(points):
                plt.annotate(f'P{i}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(f'{voronoi_type} Voronoi - Node: {node_name}, Cluster: {cluster_k}')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Save the plot
        filename = f"plots/voronoi_{voronoi_type.lower()}_{node_name}_cluster_{cluster_k}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Voronoi plot: {filename}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())