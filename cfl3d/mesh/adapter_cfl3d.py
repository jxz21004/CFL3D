"""Native CFL3D exponential/FMEM control displacement and arc-length TFI.

Opt-in steady mesh deformation, separate from the resident flow workspace.
Only the coarse control system and prescribed surface coordinates are collected
on root. Full volume interpolation stays on each zone owner. No x0 is retained:
absolute targets are applied relative to the mesh's current accepted coordinates.
"""

from itertools import product
import numpy as np
from .volume_layout import VolumeMeshLayout
from ._interface_validation import validate_interfaces
from ..surface_mesh import SurfaceTopology


class CFL3DMeshAdapter:
    """Warp point-matched structured meshes using original CFL3D kernels.

    ``method`` is ``'exponential'`` or ``'fmem'``. ``control_stride`` sets the
    initial control lattice spacing; interface and patch endpoints are inserted
    automatically. Smaller strides cost more, especially for the root FMEM
    solve. Physical boundary points absent from ``surface`` are held fixed.
    The serial native extension is required, including in MPI jobs. CFD
    installation/metric checks remain a separate setVolumeCoordinates/solve step.
    """

    def __init__(
        self,
        mesh,
        surface,
        comm=None,
        *,
        method="exponential",
        control_stride=4,
        beta1=5.0,
        alpha1=0.5,
        beta2=5.0,
        alpha2=0.1,
        smoothing_iterations=0,
    ):
        self.mesh, self.comm = mesh, comm
        self.rank = 0 if comm is None else comm.rank
        self.size = 1 if comm is None else comm.size

        def prepare():
            if method not in ("exponential", "fmem"):
                raise ValueError("method must be 'exponential' or 'fmem'")
            if isinstance(control_stride, bool) or not isinstance(control_stride, int) or control_stride < 1:
                raise ValueError("control_stride must be a positive integer")
            if (
                isinstance(smoothing_iterations, bool)
                or not isinstance(smoothing_iterations, int)
                or smoothing_iterations < 0
            ):
                raise ValueError("smoothing_iterations must be a nonnegative integer")
            parameters = np.array([beta1, alpha1, beta2, alpha2], dtype=float)
            if (
                not np.isfinite(parameters).all()
                or beta1 <= 0
                or beta2 <= 0
                or not 0 < alpha1 <= 1
                or not 0 <= alpha2 < 1
            ):
                raise ValueError("Require positive beta1/beta2, 0<alpha1<=1 and 0<=alpha2<1")
            self.layout = VolumeMeshLayout(mesh)
            self.topology = SurfaceTopology(mesh)
            if any(min(z.dimensions) < 2 for z in mesh.zones):
                raise ValueError("Every structured dimension must contain at least two vertices")
            self._ids = np.asarray(surface["node_ids"], dtype=np.int64).copy()
            self._faces = np.asarray(surface["connectivity"], dtype=np.int64).copy()
            self._surface(surface)
            return parameters

        self.parameters = self._checked(prepare)
        self.method, self.sweeps = method, smoothing_iterations
        settings = (method, control_stride, tuple(self.parameters), smoothing_iterations, self.topology.signature)
        if comm is not None and len(set(comm.allgather(settings))) != 1:
            raise ValueError("Mesh adapter settings/topology differ between ranks")
        parts = [self.layout.owned] if comm is None else comm.allgather(self.layout.owned)
        self.owners = {g: r for r, part in enumerate(parts) for g in part}
        if len(self.owners) != len(mesh.zones) or sum(map(len, parts)) != len(self.owners):
            raise ValueError("Each volume zone must have exactly one coordinate owner")
        self.axes = self._axes(control_stride)
        self.nodes = []
        self.zone_nodes = []
        for g, axes in enumerate(self.axes):
            nodes = list(product(*axes))
            self.zone_nodes.append(np.arange(len(self.nodes), len(self.nodes) + len(nodes)))
            self.nodes.extend((g, p) for p in nodes)
        if len(self.nodes) > np.iinfo(np.int32).max // 245:
            raise ValueError("Native control system exceeds int32 workspace indexing")
        self.canonical = np.array([self.topology.canonical(g, p) for g, p in self.nodes], dtype=np.int64)
        self._graph = self._build_graph()
        # Identity is topology-based; unrelated coincident surfaces stay separate.
        self._skin = {}
        for g in self.layout.owned:
            dims = mesh.zones[g].dimensions
            points = set()
            for axis in range(3):
                for end in (0, dims[axis] - 1):
                    ranges = [range(n) for n in dims]
                    ranges[axis] = (end,)
                    points.update(product(*ranges))
            self._skin[g] = [(p, self.topology.canonical(g, p)) for p in sorted(points)]
        self._targets(surface)  # Validate unique surface ownership collectively.
        self._fixed_skin = {g: {p for p, _ in skin if self._physical(g, p)} for g, skin in self._skin.items()}

    def _checked(self, function):
        error = None
        value = None
        try:
            value = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [error] if self.comm is None else self.comm.allgather(error)
        if any(errors):
            raise ValueError("CFL3D mesh adapter: " + next(e for e in errors if e))
        return value

    def _surface(self, surface):
        ids = np.asarray(surface["node_ids"])
        xyz = np.asarray(surface["coordinates"])
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids)
            or not np.array_equal(ids, self._ids)
            or not np.array_equal(surface["connectivity"], self._faces)
            or xyz.shape != (len(ids), 3)
            or not np.isrealobj(xyz)
            or not np.isfinite(xyz).all()
        ):
            raise ValueError("Surface partition/topology must stay fixed with finite (N,3) coordinates")
        return dict(zip(map(int, ids), xyz))

    def _targets(self, surface):
        local = self._checked(lambda: self._surface(surface))
        parts = [local] if self.comm is None else self.comm.allgather(local)
        targets = {}
        for part in parts:
            if targets.keys() & part.keys():
                raise ValueError("Surface node IDs must have unique owners")
            targets.update(part)
        for node in targets:
            g, p = self.topology.decode(node)
            if (
                not any(p[a] in (0, self.topology.dimensions[g][a] - 1) for a in range(3))
                or self.topology.canonical(g, p) != node
            ):
                raise ValueError("Surface IDs must identify canonical block-boundary vertices")
        return targets

    def _axes(self, stride):
        # FMEM material decay needs more than one distance layer. Include at
        # least five control indices when available, even for small fixtures.
        def spacing(n):
            return min(stride, max(1, (n - 1) // 4)) if self.method == "fmem" else stride

        sets = [[set(range(0, n, spacing(n))) | {n - 1} for n in z.dimensions] for z in self.mesh.zones]
        for g, z in enumerate(self.mesh.zones):
            for bc in z.boundaries:
                p = self.topology.point_range(g, bc.point_range)
                for a in range(3):
                    sets[g][a].update(map(int, p[a]))
        changed = True
        while changed:
            changed = False
            for g, links in enumerate(self.topology.links):
                for target, lo, hi, start, donor, transform in links:
                    for a, t in enumerate(transform):
                        values = {
                            int(donor[abs(t) - 1] + (1 if t > 0 else -1) * (v - start[a]))
                            for v in sets[g][a]
                            if lo[a] <= v <= hi[a]
                        }
                        dest = sets[target][abs(t) - 1]
                        if not values <= dest:
                            dest.update(values)
                            changed = True
                    for a in range(3):
                        extra = {int(lo[a]), int(hi[a])} - sets[g][a]
                        if extra:
                            sets[g][a].update(extra)
                            changed = True
        if any(len(axis) > 500 for zone in sets for axis in zone):
            raise ValueError("Native TFI supports at most 500 control indices per axis")
        return [[tuple(sorted(a)) for a in axes] for axes in sets]

    def _physical(self, g, p):
        for bc in self.mesh.zones[g].boundaries:
            q = np.asarray(bc.point_range) - 1
            if np.all(np.array(p) >= q.min(axis=1)) and np.all(np.array(p) <= q.max(axis=1)):
                return True
        return False

    def _build_graph(self):
        n = len(self.nodes)
        graph = np.zeros((n, 21), dtype=np.int32)
        groups = {}
        for idx, node in enumerate(self.canonical):
            groups.setdefault(int(node), []).append(idx)
        for g, indices in enumerate(self.zone_nodes):
            lookup = {self.nodes[idx][1]: int(idx) for idx in indices}
            axes = self.axes[g]
            for pos in product(*(range(len(a)) for a in axes)):
                p = tuple(axes[a][pos[a]] for a in range(3))
                idx = lookup[p]
                graph[idx, 0] = p[1] + self.mesh.zones[g].dimensions[1] * (
                    p[2] + self.mesh.zones[g].dimensions[2] * p[0]
                )
                for slot, a, sign in ((1, 1, -1), (2, 1, 1), (3, 2, -1), (4, 2, 1), (5, 0, -1), (6, 0, 1)):
                    q = list(p)
                    q[a] = axes[a][min(max(pos[a] + sign, 0), len(axes[a]) - 1)]
                    graph[idx, slot] = lookup[tuple(q)] + 1
                graph[idx, 7] = 0 if self._physical(g, p) else 1
                graph[idx, 8] = g + 1
                # Native 27-point elastic stencil reserves 3 coefficients per
                # neighbouring node and two off-diagonal components at self.
                neighbours = np.prod([1 + (pos[a] > 0) + (pos[a] < len(axes[a]) - 1) for a in range(3)]) - 1
                graph[idx, 9] = 3 * neighbours + 2
        for group in groups.values():
            if len(group) > 10:
                raise ValueError("Native FMEM supports at most 10 coincident control vertices")
            if any(graph[k, 7] == 0 for k in group):
                graph[group, 7] = 0
            for idx in group:
                others = [k + 1 for k in group if k != idx]
                graph[idx, 10] = len(group)
                graph[idx, 11 : 11 + len(others)] = others
        return graph

    def warp(self, surface):
        """Return absolute owned coordinates without changing mesh or CFD state.

        All ranks participate. Current accepted volume coordinates are the base;
        users needing total displacement from x0 retain x0 themselves. A failed
        native control solve raises collectively and leaves the solver untouched.
        """
        targets = self._targets(surface)

        def local_controls():
            if (
                tuple((z.name, tuple(z.dimensions)) for z in self.mesh.zones) != self.layout.topology
                or tuple(g for g, z in enumerate(self.mesh.zones) if z.coordinates is not None) != self.layout.owned
            ):
                raise ValueError("Mesh ownership or dimensions changed; rebuild the adapter")
            if SurfaceTopology(self.mesh).signature != self.topology.signature:
                raise ValueError("Mesh topology changed; rebuild the adapter")
            return {
                int(idx): self.mesh.zones[g].coordinates[p].copy()
                for g in self.layout.owned
                for idx in self.zone_nodes[g]
                for _, p in (self.nodes[idx],)
            }

        local = self._checked(local_controls)
        parts = [local] if self.comm is None else self.comm.gather(local, root=0)

        def solve_controls():
            if self.rank != 0:
                return None
            from .. import _native
            from ..solver import _NATIVE_LOCK

            if not hasattr(_native, "py_warp_controls"):
                raise RuntimeError("Rebuild the serial native extension for CFL3D mesh adapters")
            points = np.empty((len(self.nodes), 3))
            for part in parts:
                for idx, value in part.items():
                    points[idx] = value
            graph = self._graph.copy()
            u = np.zeros_like(points)
            for idx, node in enumerate(self.canonical):
                if int(node) in targets:
                    graph[idx, 7] = 0
                    u[idx] = targets[int(node)] - points[idx]
            graph[graph[:, 7] == 0, 9] = 0
            if np.all(u == 0) or np.all(graph[:, 7] == 0):
                return u
            if not _NATIVE_LOCK.acquire(blocking=False):
                raise RuntimeError("Native solver is busy")
            try:
                result, status = _native.py_warp_controls(
                    points, graph, u, int(self.method == "fmem"), self.parameters, self.sweeps, len(self.mesh.zones) + 1
                )
            finally:
                _NATIVE_LOCK.release()
            if status:
                raise RuntimeError(f"Native {self.method} control solve failed ({status})")
            # Dirichlet data are authoritative even at block joins.
            result[graph[:, 7] == 0] = u[graph[:, 7] == 0]
            return result

        controls = self._checked(solve_controls)
        if self.comm is not None:
            outgoing = None
            if self.rank == 0:
                outgoing = [
                    {g: controls[self.zone_nodes[g]] for g in range(len(self.mesh.zones)) if self.owners[g] == r}
                    for r in range(self.size)
                ]
            controls = self.comm.scatter(outgoing, root=0)
        else:
            controls = {g: controls[self.zone_nodes[g]] for g in self.layout.owned}

        def interpolate():
            from .. import _native
            from ..solver import _NATIVE_LOCK

            updated = {}
            if not _NATIVE_LOCK.acquire(blocking=False):
                raise RuntimeError("Native solver is busy")
            try:
                for g in self.layout.owned:
                    z = self.mesh.zones[g]
                    xyz = z.coordinates
                    delta = np.zeros_like(xyz)
                    fixed = np.zeros(z.dimensions, dtype=np.int32)
                    for idx, u in zip(self.zone_nodes[g], controls[g]):
                        delta[self.nodes[idx][1]] = u
                    for p, node in self._skin[g]:
                        if node in targets:
                            delta[p] = targets[node] - xyz[p]
                            fixed[p] = 1
                        elif p in self._fixed_skin[g]:
                            delta[p] = 0
                            fixed[p] = 1
                    out, status = _native.py_warp_tfi(
                        np.asfortranarray(xyz.transpose(1, 2, 0, 3)),
                        np.asfortranarray(delta.transpose(1, 2, 0, 3)),
                        np.asfortranarray(fixed.transpose(1, 2, 0)),
                        *(np.asarray(a, dtype=np.int32) + 1 for a in self.axes[g]),
                    )
                    if status:
                        raise RuntimeError(f"Native TFI failed for {z.name} ({status})")
                    updated[z.name] = out.transpose(2, 0, 1, 3).copy()
            finally:
                _NATIVE_LOCK.release()
            return updated

        updated = self._checked(interpolate)
        validate_interfaces(
            self.mesh, updated, tuple(self.owners[g] for g in range(len(self.mesh.zones))), self.comm, self._checked
        )
        parts = [updated[self.mesh.zones[g].name].reshape(-1, 3, order="F") for g in self.layout.owned]
        return np.concatenate(parts) if parts else np.empty((0, 3))
