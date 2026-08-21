import numpy as np
import math
import foronoi
from scipy.spatial import Voronoi, KDTree
from shapely.geometry import Polygon, box

np.float = float 

class Node:
    def __init__(self, name, x, y, type):
        self.name = name
        self.x = x
        self.y = y
        self.type = type

    def __str__(self):
        return f"{self.name}({int(round(self.x, 0))},{int(round(self.y, 0))})"

    def get_distance_to(self, other_node):
        return math.sqrt(math.pow(self.x - other_node.x, 2) + math.pow(self.y - other_node.y, 2))


class NodeList:
    def __init__(self, s_m, nt, ra_percent, seed=None):
        self.all_nodes = []
        self.unlocalizable_nodes = []

        if not seed is None:
            np.random.seed(seed)

        generated_x = np.random.uniform(low=0, high=s_m, size=nt)
        generated_y = np.random.uniform(low=0, high=s_m, size=nt)
        order_number_u = 1
        order_number_a = 1
        t = None
        name = None
        for i in range(nt):
            if i < nt * (ra_percent / 100):
                t = "A"
                name = f"A{order_number_a}"
                order_number_a += 1
            else:
                t = "U"
                name = f"U{order_number_u}"
                order_number_u += 1

            self.all_nodes.append(Node(name, generated_x[i], generated_y[i], t))

    def anchors(self):
        return [n for n in self.all_nodes if n.type == "A"]

    def unknown_nodes(self):
        return [n for n in self.all_nodes if n.type == "U"]

    def calculate_detectable_nodes(self, R_m, unknown_node):
        detectable_anchors = []
        undetectable_anchors = []

        for anchor_node in self.anchors():
            distance = unknown_node.get_distance_to(anchor_node)
            if distance <= R_m:
                detectable_anchors.append((anchor_node, distance))
            else:
                undetectable_anchors.append((anchor_node, distance))

        return detectable_anchors, undetectable_anchors

    def split_anchors(self, detectable_anchors, undetectable_anchors):
        group_one = []
        group_two = []

        to_assign = detectable_anchors.copy()

        # Prvo stavimo random pola u prvu grupu
        how_many_in_first = int(math.ceil(len(detectable_anchors) / 2))
        for i in range(how_many_in_first):
            index = np.random.randint(low=0, high=len(to_assign))
            group_one.append([to_assign[index][0], to_assign[index][1]])
            del to_assign[index]

        # Sve ostale detectable čvorove stavimo u drugu grupu
        for n in to_assign:
            group_two.append([n[0], n[1]])

        detectable_in_one = len(group_one)
        detectable_in_two = len(group_two)

        # Sad preostalih pola stavimo u prvu, a ostatak u drugu grupu
        to_assign = undetectable_anchors.copy()

        how_many_in_first = int(len(to_assign) / 2)
        for i in range(how_many_in_first):
            index = np.random.randint(low=0, high=len(to_assign))
            group_one.append([to_assign[index][0], to_assign[index][1]])
            del to_assign[index]

        for n in to_assign:
            group_two.append([n[0], n[1]])

        return group_one, group_two, detectable_in_one, detectable_in_two

    def map(self, regions, voronoi_points, closest_node):
        positive_region = None
        for i in range(len(voronoi_points)):
            if math.isclose(closest_node.x, voronoi_points[i][0]) and math.isclose(closest_node.y, voronoi_points[i][1]):
                positive_region = regions[i]
        return positive_region

    @staticmethod
    def stringify(node_list):
        if len(node_list) == 0:
            return "<prazno>"

        r = ""
        for a in node_list:
            if type(a) in [list, tuple]:
                r += str(a[0])
                r += " DIST=" + str(int(a[1]))
            else:
                r += str(a)
            r += ", "

        if len(r) >= 2:
            return r.rstrip(", ")
        else:
            return r


def voronoi_finite_polygons_2d(vor, radius=None, simulation_bounds=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = {}
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        if simulation_bounds is not None:
            min_x, min_y, max_x, max_y = simulation_bounds
            radius = max(max_x - min_x, max_y - min_y) * 2
        else:
            radius = min(max(np.ptp(vor.points).max(), 200), 1000)

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions[p1] = vertices
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions[p1] = new_region.tolist()

    if simulation_bounds is not None:
        min_x, min_y, max_x, max_y = simulation_bounds
        bounds_poly = box(min_x, min_y, max_x, max_y)

        clipped_regions = {}
        clipped_vertices = []

        vmap = {}

        for pid, region in new_regions.items():
            poly_coords = [tuple(new_vertices[i]) for i in region]
            poly = Polygon(poly_coords)

            clipped = poly.intersection(bounds_poly)

            if clipped.is_empty:
                clipped_regions[pid] = []
                continue

            if clipped.geom_type == 'MultiPolygon':
                clipped = max(clipped.geoms, key=lambda g: g.area)

            region_indices = []
            for x, y in clipped.exterior.coords[:-1]: #Preskoci zadnju tocku jer je ista kao prva
                key = (x, y)
                if key not in vmap:
                    vmap[key] = len(clipped_vertices)
                    clipped_vertices.append([x, y])
                region_indices.append(vmap[key])

            clipped_regions[pid] = region_indices

        return clipped_regions, np.asarray(clipped_vertices)

    return new_regions, np.asarray(new_vertices)


def map(voronoi_node, anchors):
    for a in anchors:
        if math.isclose(voronoi_node[0], a.x) and math.isclose(voronoi_node[1], a.y):
            return a
    return None


def print_regions(vor_anchors, our_anchors, regions, vertices):
    # vor_anchors: [[59.86584842, 35.67533267], [15.60186404, 28.09345097], [95.07143064, 27.13490318]]
    # our_anchors: [<helper.Node object at 0x000001CFEBBB3150>, ... ]
    # regions: {0: [0, 1, 2], 1: [4, 0, 3], 2: [6, 5, 0]}
    # vertices: [[54.22674999, -64.40320809], [72.96168046, 12.82641314], ...]
    print("\t\tRegions:")
    for i, region_vertices in regions.items():
        node = map(vor_anchors[i], our_anchors)
        r = ""
        for v in region_vertices:
            r += f"({int(vertices[v][0])},{int(vertices[v][1])}), "
        print(f"\t\t\t{node}: {r.rstrip(', ')}")


def print_region(region, vertices):
    # regions: {0: [0, 1, 2], 1: [4, 0, 3], 2: [6, 5, 0]}
    # vertices: [[54.22674999, -64.40320809], [72.96168046, 12.82641314], ...]
    print("\t\tClosest (positive) region:", end=" ")
    r = ""
    for v in region:
        r += f"({int(vertices[v][0])},{int(vertices[v][1])}), "
    print(f"{r.rstrip(', ')}")


def get_region_coordinates(region, vertices):
    # regions: {0: [0, 1, 2], 1: [4, 0, 3], 2: [6, 5, 0]}
    # vertices: [[54.22674999, -64.40320809], [72.96168046, 12.82641314], ...]
    coords = []
    for v in region:
        coords.append([ vertices[v][0], vertices[v][1] ])
    return coords


def print_grid(grid):
    for line in grid:
        print(line)
    
def local_voronoi(points, target, k=10):
    if len(points) <= k:
        print(f"Not enough points for local Voronoi, returning global Voronoi: {points}")
        return Voronoi(points, furthest_site=False)
    
    # dist calc
    target_x, target_y = target
    distances = []
    for i, point in enumerate(points):
        dist_sq = (point[0] - target_x)**2 + (point[1] - target_y)**2
        distances.append((i, dist_sq))
    
    distances.sort(key=lambda x: x[1])
    local_points = [points[distances[i][0]] for i in range(k)]
    
    return Voronoi(local_points, furthest_site=False)

def grid_scanning(s_m, l_m2, positive_regions, unknown_node_xy, detect_radius):
    """
    Perform grid scanning to find the intersection shape,
    using a local bounding box to dramatically reduce the scan area.

    :param s_m: Size of the area
    :param l_m2: Length of the grid cell
    :param positive_regions: Dictionary of positive-region polygon point lists
    :param unknown_node_xy: (x, y) coordinates of the unknown node
    :param shrink_radius: Distance defining the local bounding region (e.g. detection_radius*1.5)
    :return: Grid with intersection shape counts
    """
    W = int(s_m / l_m2)
    grid = np.zeros((W, W), dtype=int)

    shrink_radius = detect_radius * 2

    # Local bounding region around the unknown node
    ux, uy = unknown_node_xy
    local_box = box(
        ux - shrink_radius,
        uy - shrink_radius,
        ux + shrink_radius,
        uy + shrink_radius
    )

    for group_number, positive_region in positive_regions.items():

        # Convert region into polygon
        reg_poly = Polygon([(float(c[0]), float(c[1])) for c in positive_region])

        reg_poly = reg_poly.intersection(local_box)

        # print("Bounds:", reg_poly.bounds)
        # print("Area:", reg_poly.area)
        
        if reg_poly.is_empty:
            continue

        reg_poly = reg_poly.buffer(0)

        minx, miny, maxx, maxy = reg_poly.bounds

        row_start = max(0, int(miny // l_m2))
        row_end   = min(W, int(maxy // l_m2) + 1)
        col_start = max(0, int(minx // l_m2))
        col_end   = min(W, int(maxx // l_m2) + 1)

        # Grid intersection scanning
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                cell = box(c * l_m2, r * l_m2,
                           (c + 1) * l_m2, (r + 1) * l_m2)
                if reg_poly.intersects(cell):
                    grid[r][c] += 1

    return grid

def sweep_line_algorithm(s_m, l_m2, positive_regions, unknown_node_xy, detect_radius):
    """
    Perform a sweep line algorithm to find the intersection shape.

    :param s_m: Size of the area
    :param l_m2: Length of the grid cell
    :param positive_regions: Dictionary of positive regions
    :return: Grid with intersection shape
    """
    W = int(s_m / l_m2)
    grid = np.zeros((W, W), dtype=int)
    
    shrink_radius = detect_radius * 2

    # Local bounding region around the unknown node
    ux, uy = unknown_node_xy
    local_box = box(
        ux - shrink_radius,
        uy - shrink_radius,
        ux + shrink_radius,
        uy + shrink_radius
    )

    for group_number, positive_region in positive_regions.items():
        reg_poly = Polygon([(float(c[0]), float(c[1])) for c in positive_region])
        reg_poly = reg_poly.intersection(local_box)

        if reg_poly.is_empty:
            continue

        reg_poly = reg_poly.buffer(0)
        
        grid_temp = np.zeros((W, W), dtype=int)

        minx, miny, maxx, maxy = reg_poly.bounds
        col_start = max(0, int(minx // l_m2))
        col_end = min(W, int(maxx // l_m2) + 1)

        for c in range(col_start, col_end):
            x0 = c * l_m2
            x1 = (c + 1) * l_m2
            strip = box(x0, miny, x1, maxy)

            sliced = reg_poly.intersection(strip)

            if not sliced.is_empty:
                if sliced.geom_type == 'Polygon':
                    polys = [sliced]
                elif sliced.geom_type == 'MultiPolygon':
                    polys = list(sliced.geoms)
                else:
                    continue

                for poly in polys:
                    min_sy, max_sy = poly.bounds[1], poly.bounds[3]
                    row_start = max(0, int(min_sy // l_m2))
                    row_end = min(W, int(max_sy // l_m2) + 1)

                    for r in range(row_start, row_end):
                        cell = box(x0, r * l_m2, x1, (r + 1) * l_m2)
                        if poly.intersects(cell):
                            grid[r][c] += 1
                            grid_temp[r][c] = 1
    return grid
