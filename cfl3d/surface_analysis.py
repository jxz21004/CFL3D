"""Conservative strip loads and connected cuts of cached structured wall faces.

No SciPy, MPI, file parsing or native solver state is needed here. Quads use
the fixed (00,10,11)/(00,11,01) triangulation for geometric clipping only;
strip loads are fractions of the original native integrated face loads.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from .solution_io import wall_vertices

WALL_VARIABLES = ("Cp", "T", "Cfx", "Cfy", "Cfz", "y+")
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def direction_index(direction):
    if not isinstance(direction, str) or direction.lower() not in ("x", "y", "z"):
        raise ValueError("direction must be one of 'x', 'y', or 'z'")
    return "xyz".index(direction.lower())


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    a = np.asarray(points)
    return 0.5 * np.linalg.norm(np.sum(np.cross(a[1:-1] - a[0], a[2:] - a[0]), axis=0))


def clip_polygon(points, axis, position, greater):
    result = []
    if not len(points):
        return result
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        da, db = a[axis] - position, b[axis] - position
        inside_a = da >= 0 if greater else da <= 0
        inside_b = db >= 0 if greater else db <= 0
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            result.append(a + (b - a) * (da / (da - db)))
    return result


@dataclass
class Patch:
    key: tuple[str, str]
    quads: np.ndarray
    values: np.ndarray
    loads: np.ndarray
    topology: tuple


class SurfaceMesh:
    def __init__(self, mesh, walls, forces=None, sref=1.0, vertices=None):
        self.patches = {}
        for zone in mesh.zones:
            for boundary in zone.boundaries:
                key = zone.name, boundary.name
                if key not in walls:
                    continue
                xyz = wall_vertices(zone, boundary) if vertices is None else vertices[key]
                quads = np.stack((xyz[:-1, :-1], xyz[1:, :-1], xyz[1:, 1:], xyz[:-1, 1:]), axis=-2).reshape(-1, 4, 3)
                values = np.asarray(walls[key]).reshape(-1, 6)
                if forces is not None:
                    loads = np.asarray(forces[key]).reshape(-1, 6)
                else:
                    # Geometric fallback for reference geometry and analytical tests.
                    axis = int(np.flatnonzero(boundary.point_range[:, 0] == boundary.point_range[:, 1])[0])
                    side = 1 if boundary.point_range[axis, 0] == 1 else -1
                    vector = 0.5 * np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1])
                    normal = vector * ((-1) ** axis) * side
                    pressure = -values[:, 0, None] * normal / sref
                    viscous = values[:, 2:5] * np.linalg.norm(vector, axis=-1)[:, None] / sref
                    loads = np.column_stack((pressure, viscous))
                topology = (zone.dimensions, tuple(np.sort(boundary.point_range, axis=1).flat))
                self.patches[key] = Patch(key, quads, values, loads, topology)

    def selected(self, members):
        patches = [self.patches[key] for key in sorted(set(members) & self.patches.keys())]
        if not patches:
            raise ValueError("Selected group has no wall surfaces")
        return patches

    def bounds(self, members):
        patches = self.selected(members)
        lo = np.min([p.quads.min(axis=(0, 1)) for p in patches], axis=0)
        hi = np.max([p.quads.max(axis=(0, 1)) for p in patches], axis=0)
        return lo, hi

    def tolerance(self, members):
        lo, hi = self.bounds(members)
        return (
            64
            * np.finfo(float).eps
            * max(np.max(hi - lo), np.max(np.abs(lo)), np.max(np.abs(hi)), np.finfo(float).tiny)
        )


def lift_distribution(surface, members, nsegments, axis, angles, y_up=False, scale=None):
    lo, hi = surface.bounds(members)
    if hi[axis] - lo[axis] <= surface.tolerance(members):
        raise ValueError("Selected walls have zero extent in the distribution direction")
    edges = np.linspace(lo[axis], hi[axis], nsegments + 1)
    width = np.diff(edges)
    bins = np.zeros((nsegments, 6))
    for patch in surface.selected(members):
        for quad, load in zip(patch.quads, patch.loads, strict=True):
            minimum, maximum = quad[:, axis].min(), quad[:, axis].max()
            start = int(np.clip(np.searchsorted(edges, minimum, side="right") - 1, 0, nsegments - 1))
            stop = int(np.clip(np.searchsorted(edges, maximum, side="left"), start + 1, nsegments))
            if stop - start == 1:
                bins[start] += load
                continue
            triangles = quad[TRIANGLES]
            total_area = sum(polygon_area(t) for t in triangles)
            if total_area == 0:
                continue
            weights = []
            for index in range(start, stop):
                area = 0.0
                for triangle in triangles:
                    cut = clip_polygon(triangle, axis, edges[index], True)
                    cut = clip_polygon(cut, axis, edges[index + 1], False)
                    area += polygon_area(cut)
                weights.append(area / total_area)
            weights = np.array(weights)
            # Normalize only geometric roundoff. Each native cell load is allocated exactly once.
            if weights.sum() > 0:
                weights /= weights.sum()
                bins[start:stop] += weights[:, None] * load
    alpha, beta = angles
    lift = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])
    drag = np.array([np.cos(alpha) * np.cos(beta), -np.sin(beta), np.sin(alpha) * np.cos(beta)])
    if y_up:
        lift = lift[[0, 2, 1]] * np.array([1, 1, -1])
        drag = drag[[0, 2, 1]] * np.array([1, 1, -1])
    pressure, viscous = bins[:, :3], bins[:, 3:]
    result = {
        "edges": edges,
        "stations": 0.5 * (edges[:-1] + edges[1:]),
        "width": width,
        "pressure_coefficients": pressure,
        "viscous_coefficients": viscous,
    }
    values = {
        "CL": (pressure + viscous) @ lift,
        "CD": (pressure + viscous) @ drag,
        "CDp": pressure @ drag,
        "CDv": viscous @ drag,
    }
    for name, array in values.items():
        result["bin_" + name] = array
        result["d" + name + "_ds"] = array / width
    result["units"] = {"stations": "m", "width": "m", **{"d" + name + "_ds": "1/m" for name in values}}
    if scale is not None:
        for name in ("L", "D", "Dp", "Dv"):
            result["d" + name + "_ds"] = result["dC" + name + "_ds"] * scale
            result["units"]["d" + name + "_ds"] = "N/m"
    return result


@dataclass
class CutReference:
    key: tuple[str, str]
    cell: int
    triangle: int
    weights: np.ndarray
    topology: tuple


def triangle_cut(triangle, axis, position, tolerance):
    signed = triangle[:, axis] - position
    if np.all(np.abs(signed) <= tolerance):
        return None  # Coplanar faces have a 2D intersection, not a section line.
    candidates = []
    eye = np.eye(3)
    for i in range(3):
        j = (i + 1) % 3
        if abs(signed[i]) <= tolerance:
            candidates.append(eye[i])
        if (signed[i] < -tolerance and signed[j] > tolerance) or (signed[j] < -tolerance and signed[i] > tolerance):
            t = signed[i] / (signed[i] - signed[j])
            candidates.append((1 - t) * eye[i] + t * eye[j])
    unique = []
    for weight in candidates:
        point = weight @ triangle
        if not any(np.linalg.norm(point - w @ triangle) <= tolerance for w in unique):
            unique.append(weight)
    if len(unique) != 2 or np.linalg.norm((unique[0] - unique[1]) @ triangle) <= tolerance:
        return None
    return np.asarray(unique)


def slice_references(surface, members, axis, position):
    tolerance = surface.tolerance(members)
    result = []
    for patch in surface.selected(members):
        bounds = patch.quads[:, :, axis]
        selected = np.flatnonzero(
            (bounds.min(axis=1) <= position + tolerance) & (bounds.max(axis=1) >= position - tolerance)
        )
        for cell in selected:
            for index, triangle in enumerate(patch.quads[cell][TRIANGLES]):
                weights = triangle_cut(triangle, axis, position, tolerance)
                if weights is not None:
                    result.append(CutReference(patch.key, int(cell), index, weights, patch.topology))
    return result


def evaluate_slice(surface, members, references, axis, position, absolute):
    """Weld coincident cut endpoints and duplicate edges across conformal patches.

    Values remain cell-centered; no smoothing across sharp edges is introduced.
    A cut exactly on a shared edge uses the area-weighted two-sided face value.
    """
    tolerance = surface.tolerance(members)
    origin = surface.bounds(members)[0]
    nodes, buckets, edges = [], {}, {}

    def vertex(point):
        point = np.array(point, copy=True)
        if absolute:
            point[axis] = position
        location = tuple(np.floor((point - origin) / tolerance).astype(np.int64))
        for offset in product((-1, 0, 1), repeat=3):
            neighbor = tuple(a + b for a, b in zip(location, offset, strict=True))
            for index in buckets.get(neighbor, ()):
                if np.linalg.norm(nodes[index] - point) <= tolerance:
                    return index
        index = len(nodes)
        nodes.append(point)
        buckets.setdefault(location, []).append(index)
        return index

    for reference in references:
        patch = surface.patches.get(reference.key)
        if patch is None or patch.topology != reference.topology:
            raise ValueError("Relative slice topology changed; register the slice again")
        quad = patch.quads[reference.cell]
        points = reference.weights @ quad[TRIANGLES[reference.triangle]]
        if np.linalg.norm(points[1] - points[0]) <= tolerance:
            continue
        a, b = (vertex(point) for point in points)
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        area = np.linalg.norm(0.5 * np.cross(quad[2] - quad[0], quad[3] - quad[1]))
        if area == 0:
            continue
        if key not in edges:
            edges[key] = [np.zeros(6), 0.0]
        edges[key][0] += area * patch.values[reference.cell]
        edges[key][1] += area
    connectivity = np.asarray(list(edges), dtype=np.int32).reshape(-1, 2)
    points = np.asarray(nodes).reshape(-1, 3)
    values = np.asarray([v[0] / v[1] for v in edges.values()]).reshape(-1, 6)
    coordinates = points[connectivity].mean(axis=1) if len(connectivity) else np.empty((0, 3))
    return {
        "points": points,
        "segments": connectivity,
        "coordinates": coordinates,
        **{name: values[:, i] for i, name in enumerate(WALL_VARIABLES)},
        "units": {"points": "m", "coordinates": "m", "T": "K"},
    }
