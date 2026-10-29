"""Native CFL3D exponential/FMEM control displacement and arc-length TFI.

Opt-in steady mesh deformation, separate from the resident flow workspace.
Default FMEM gathers only coarse controls for original native Gauss-Seidel;
explicit petsc-gmres keeps control assembly/solving distributed. Volume TFI stays
on zone owners for both backends. No x0 is retained:
absolute targets are applied relative to the mesh's current accepted coordinates.
"""

from itertools import product
from collections.abc import Mapping
from numbers import Integral
from types import MappingProxyType
import numpy as np
from .volume_layout import VolumeMeshLayout
from ._interface_validation import validate_interfaces
from ..surface_mesh import SurfaceTopology
from ._distributed_warp import SharedNodes, exchange, scalar, nearest


class CFL3DMeshAdapter:
    """Warp point-matched structured meshes using original CFL3D kernels.

    ``method`` is ``'exponential'`` or ``'fmem'``. ``control_mode`` selects
    ``'minimal'`` (segment endpoints), ``'auto_stride'`` (automatic uniform
    spacing), ``'stride'`` (default spacing 4), or ``'indices'`` (explicit
    zero-based per-zone index lists). Strides may be a
    scalar, an (i,j,k) tuple, or a zone-name mapping of these. Explicit indices
    require an entry for every zone. Only the active mode's option may be given.
    Physical/interface endpoints and matching indices are inserted automatically;
    inspect the effective lattice with ``getControlIndices()``. Sparse FMEM
    lattices can be degenerate and require explicit refinement. Smaller strides
    cost more. Boundary controls follow native ``setslave`` classification: symmetry
    planes are free, while prescribed wall/farfield controls remain constrained.
    No additional plane projection is applied by this adapter.
    FMEM defaults to ``linear_solver='native-gauss-seidel'`` (original control
    assembly/GS on root). Explicit ``'petsc-gmres'`` requires petsc4py and uses
    distributed GMRES/Jacobi with ``linear_solver_options={"restart": 30}``. ``alpha1``
    controls native GS relaxation only. No backend is selected by availability,
    and failures never trigger automatic fallback. Exponential decay needs no
    linear solver; selecting petsc-gmres with it is rejected.
    The serial native extension is required, including in MPI jobs. CFD
    installation/metric checks remain a separate setVolumeCoordinates/solve step.
    """

    def __init__(
        self,
        mesh,
        surface,
        comm=None,
        *,
        boundary_overrides=None,
        method="exponential",
        linear_solver="native-gauss-seidel",
        linear_solver_options=None,
        control_stride=None,
        control_mode="stride",
        control_indices=None,
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
            if linear_solver not in ("native-gauss-seidel", "petsc-gmres"):
                raise ValueError("linear_solver must be 'native-gauss-seidel' or 'petsc-gmres'")
            if method != "fmem" and linear_solver != "native-gauss-seidel":
                raise ValueError("linear_solver='petsc-gmres' applies only to method='fmem'")
            if linear_solver_options is not None and not isinstance(linear_solver_options, Mapping):
                raise ValueError("linear_solver_options must be a mapping or None")
            options = {} if linear_solver_options is None else dict(linear_solver_options)
            allowed = {"restart"} if linear_solver == "petsc-gmres" else set()
            unknown = set(options) - allowed
            if unknown:
                raise ValueError(f"Unsupported linear_solver_options for {linear_solver}: {sorted(unknown, key=str)}")
            if linear_solver == "petsc-gmres":
                restart = options.get("restart", 30)
                if isinstance(restart, (bool, np.bool_)) or not isinstance(restart, Integral) or restart < 1:
                    raise ValueError("linear_solver_options['restart'] must be a positive integer")
                options["restart"] = int(restart)
            # Copy/freeze configuration: caller mutations must not silently change
            # one MPI rank's solver settings after collective validation.
            self.linear_solver_options = MappingProxyType(options)
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
            from ..solver import _BC_TYPES
            overrides = {} if boundary_overrides is None else dict(boundary_overrides)
            # Native setslave.F uses this list, not membership in any BC patch.
            fixed_codes = {1000, 1002, 1003, 1005, 1006, 1008, 2003, 2004,
                           2006, 2007, 2008, 2014, 2016, 2024, 2034, 2102, 2103}
            is_2d = all(z.dimensions[0] == 2 for z in mesh.zones)
            self._fixed_ranges = {}
            for g, zone in enumerate(mesh.zones):
                ranges = []
                for bc in zone.boundaries:
                    key = (zone.name, bc.name)
                    entry = overrides.get(key, _BC_TYPES.get(bc.bc_type))
                    if entry is None:
                        raise ValueError(f"{key}: unknown BC needs boundary_overrides")
                    code = entry[0]
                    if isinstance(code, bool) or not isinstance(code, Integral):
                        raise ValueError(f"{key}: boundary code must be an integer")
                    q = np.asarray(bc.point_range) - 1
                    axis = int(np.flatnonzero(q[:, 0] == q[:, 1])[0])
                    # setslave skips I-face constraints for a 2-D mesh.
                    if is_2d and axis == 0:
                        continue
                    if code in fixed_codes or (code == 1013 and axis != 0 and zone.dimensions[0] == 2):
                        ranges.append((q.min(axis=1), q.max(axis=1)))
                self._fixed_ranges[g] = ranges
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
        self.linear_solver = linear_solver
        self._petsc = None
        self.axes = self._checked(lambda: self._axes(control_mode, control_stride, control_indices))
        self.control_mode = control_mode
        settings = (
            method,
            linear_solver,
            tuple(sorted(self.linear_solver_options.items())),
            control_mode,
            tuple(tuple(a) for a in self.axes),
            tuple(tuple((tuple(lo), tuple(hi)) for lo, hi in self._fixed_ranges[g])
                  for g in range(len(mesh.zones))),
            tuple(self.parameters),
            smoothing_iterations,
            self.topology.signature,
        )
        if comm is not None and len(set(comm.allgather(settings))) != 1:
            raise ValueError("Mesh adapter settings/topology differ between ranks")
        if linear_solver == "petsc-gmres":
            from ._petsc_warp import load_petsc

            self._petsc = self._checked(load_petsc)
        parts = [self.layout.owned] if comm is None else comm.allgather(self.layout.owned)
        self.owners = {g: r for r, part in enumerate(parts) for g in part}
        if len(self.owners) != len(mesh.zones) or sum(map(len, parts)) != len(self.owners):
            raise ValueError("Each volume zone must have exactly one coordinate owner")
        self.nodes = []
        self.zone_nodes = {}
        self.graphs = {}
        for g in self.layout.owned:
            nodes = list(product(*self.axes[g]))
            self.zone_nodes[g] = np.arange(len(self.nodes), len(self.nodes) + len(nodes))
            self.nodes.extend((g, p) for p in nodes)
            self.graphs[g] = self._build_graph(g, nodes)
        self.canonical = np.array([self.topology.canonical(g, p) for g, p in self.nodes], dtype=np.int64)
        self.unique, self.inverse = np.unique(self.canonical, return_inverse=True)
        self.shared = SharedNodes(self, self.unique)
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

        def route():
            outgoing = [{} for _ in range(self.size)]
            for node, xyz in local.items():
                g, p = self.topology.decode(node)
                if (
                    not any(p[a] in (0, self.topology.dimensions[g][a] - 1) for a in range(3))
                    or self.topology.canonical(g, p) != node
                ):
                    raise ValueError("Surface IDs must identify canonical block-boundary vertices")
                outgoing[self.owners[g]][node] = xyz
            return outgoing

        received = exchange(self.comm, self._checked(route))

        def directory():
            owned = {}
            for part in received:
                if owned.keys() & part.keys():
                    raise ValueError("Surface node IDs must have unique owners")
                owned.update(part)
            return owned

        owned = self._checked(directory)
        requests = [[] for _ in range(self.size)]
        for node in sorted({node for skin in self._skin.values() for _, node in skin}):
            requests[self.owners[self.topology.decode(node)[0]]].append(node)
        incoming = exchange(self.comm, requests)
        replies = [{node: owned[node] for node in part if node in owned} for part in incoming]
        return {node: xyz for part in exchange(self.comm, replies) for node, xyz in part.items()}

    def getControlIndices(self):
        """Return effective zero-based (i, j, k) index tuples by zone name.

        Includes automatically inserted physical/interface segment endpoints and
        indices propagated across matching interfaces. This returns only index
        metadata, never grid coordinates, and does not require MPI communication.
        """
        return {z.name: tuple(tuple(axis) for axis in axes) for z, axes in zip(self.mesh.zones, self.axes)}

    def _axes(self, mode, stride, indices):
        if mode not in ("minimal", "auto_stride", "stride", "indices"):
            raise ValueError("control_mode must be 'minimal', 'auto_stride', 'stride' or 'indices'")
        if mode != "stride" and stride is not None:
            raise ValueError("control_stride is only valid with control_mode='stride'")
        if mode != "indices" and indices is not None:
            raise ValueError("control_indices is only valid with control_mode='indices'")
        names = {z.name for z in self.mesh.zones}

        def mapping(value, label):
            if not isinstance(value, Mapping) or set(value) != names:
                raise ValueError(f"{label} must contain exactly all zone names: {sorted(names)}")

        def integer(v):
            return isinstance(v, Integral) and not isinstance(v, (bool, np.bool_))

        if mode == "indices":
            mapping(indices, "control_indices")
        if mode == "stride":
            stride = 4 if stride is None else stride
            if isinstance(stride, Mapping):
                mapping(stride, "control_stride")
        sets = []
        for z in self.mesh.zones:
            dims = z.dimensions
            if mode in ("minimal", "auto_stride"):
                axes = [{0, n - 1} for n in dims]
            elif mode == "stride":
                value = stride[z.name] if isinstance(stride, Mapping) else stride
                values = (value,) * 3 if integer(value) else tuple(value) if isinstance(value, (tuple, list)) else ()
                if len(values) != 3 or any(not integer(v) or v < 1 for v in values):
                    raise ValueError("Each control_stride must be a positive integer or three positive integers")
                axes = [set(range(0, n, int(step))) | {n - 1} for n, step in zip(dims, values)]
            else:
                values = indices[z.name]
                if not isinstance(values, (tuple, list)) or len(values) != 3:
                    raise ValueError("Each control_indices entry must contain i, j and k index sequences")
                axes = []
                for n, value in zip(dims, values):
                    a = np.asarray(value)
                    if (
                        a.ndim != 1
                        or a.size == 0
                        or a.dtype.kind not in "iu"
                        or np.any(a < 0)
                        or np.any(a >= n)
                        or np.any(a[1:] <= a[:-1])
                    ):
                        raise ValueError(
                            "Control indices must be nonempty, strictly increasing integers within each axis"
                        )
                    axes.append(set(map(int, a)) | {0, n - 1})
            sets.append(axes)
        # No hidden FMEM minimum count: the requested lattice is augmented only
        # to meet block/physical/interface topology constraints. Invalid coarse
        # macro-elements must be refined explicitly by the caller.
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
            if mode == "auto_stride":
                # Largest uniform stride containing every required endpoint.
                # Repeat interface propagation after filling these index lists.
                for zone in sets:
                    for axis in zone:
                        ordered = sorted(axis)
                        step = int(np.gcd.reduce(np.diff(ordered)))
                        uniform = set(range(ordered[0], ordered[-1] + 1, step))
                        if not uniform <= axis:
                            axis.update(uniform)
                            changed = True
        if any(len(axis) > 500 for zone in sets for axis in zone):
            raise ValueError("Native TFI supports at most 500 control indices per axis")
        return [[tuple(sorted(a)) for a in axes] for axes in sets]

    def _physical(self, g, p):
        return any(np.all(np.asarray(p) >= lo) and np.all(np.asarray(p) <= hi)
                   for lo, hi in self._fixed_ranges[g])

    def _build_graph(self, g, nodes):
        n = len(nodes)
        if n > np.iinfo(np.int32).max // 245:
            raise ValueError("Local native control system exceeds int32 indexing")
        graph = np.zeros((n, 21), dtype=np.int32)
        lookup = {p: idx for idx, p in enumerate(nodes)}
        axes = self.axes[g]
        for pos in product(*(range(len(a)) for a in axes)):
            p = tuple(axes[a][pos[a]] for a in range(3))
            idx = lookup[p]
            for slot, a, sign in ((1, 1, -1), (2, 1, 1), (3, 2, -1), (4, 2, 1), (5, 0, -1), (6, 0, 1)):
                q = list(p)
                q[a] = axes[a][min(max(pos[a] + sign, 0), len(axes[a]) - 1)]
                graph[idx, slot] = lookup[tuple(q)] + 1
            graph[idx, 7] = 0 if self._physical(g, p) else 1
            graph[idx, 8] = 1
            neighbours = np.prod([1 + (pos[a] > 0) + (pos[a] < len(axes[a]) - 1) for a in range(3)]) - 1
            graph[idx, 9] = 3 * neighbours + 2
            # Interface contributions are summed by canonical ID outside the
            # local native assembly, never through a replicated global graph.
            graph[idx, 10] = 1
        return graph

    def _sum(self, values):
        local = np.zeros((len(self.unique),) + values.shape[1:], dtype=values.dtype)
        np.add.at(local, self.inverse, values)
        return self.shared.sum(local)

    def _native_controls(self, points, fixed, prescribed):
        """Gather only the coarse system and call the original FMEM/GS driver.

        Ordering is block, i, k, j (j fastest), as in native setslave. Full volume
        grids and TFI buffers remain on zone owners. This explicitly chosen
        compatibility backend trades root control-system memory for native GS.
        """
        local = {
            g: (points[idx], fixed[idx], prescribed[idx], self.canonical[idx], self.graphs[g])
            for g, idx in self.zone_nodes.items()
        }
        parts = [local] if self.comm is None else self.comm.gather(local, root=0)

        def native():
            if self.rank != 0:
                return None
            from .. import _native
            from ..solver import _NATIVE_LOCK

            blocks = {g: data for part in parts for g, data in part.items()}
            xyz, u, graphs, ids, slices, permutations = [], [], [], [], {}, {}
            offset = 0
            lookups = {}
            for g in range(len(self.mesh.zones)):
                p, f, displacement, canonical, graph = blocks[g]
                axes = self.axes[g]
                positions = list(product(*axes))
                order = np.array(
                    sorted(range(len(positions)), key=lambda n: (positions[n][0], positions[n][2], positions[n][1]))
                )
                inverse = np.empty_like(order)
                inverse[order] = np.arange(len(order))
                q = graph[order].copy()
                q[:, 1:7] = inverse[q[:, 1:7] - 1] + offset + 1
                q[:, 7] = (~f[order]).astype(int)
                q[:, 8] = g + 1
                q[f[order], 9] = 0
                lookups[g] = {positions[old]: offset + i for i, old in enumerate(order)}
                xyz.append(p[order])
                u.append(displacement[order])
                ids.extend(canonical[order])
                graphs.append(q)
                slices[g] = slice(offset, offset + len(order))
                permutations[g] = inverse
                offset += len(order)
            if offset > np.iinfo(np.int32).max // 245:
                raise ValueError("Global native control system exceeds int32 workspace indexing; use petsc-gmres")
            graph = np.concatenate(graphs)
            # Native setslave connects the outward stencil slot to the next
            # control point inside the donor block, not to the coincident point.
            # This also determines sparse row capacity at block interfaces.
            slots = ((5, 6), (1, 2), (3, 4))  # i-/+, j-/+, k-/+ (zero-based columns)
            for g, links in enumerate(self.topology.links):
                for target, lo, hi, start, donor, transform in links:
                    normals = [
                        a for a in range(3) if lo[a] == hi[a] and lo[a] in (0, self.mesh.zones[g].dimensions[a] - 1)
                    ]
                    if len(normals) != 1:
                        raise ValueError("Native GS requires rectangular block-boundary interfaces")
                    axis = normals[0]
                    side = int(lo[axis] != 0)
                    daxis = abs(transform[axis]) - 1
                    for position, row in lookups[g].items():
                        if not all(lo[a] <= position[a] <= hi[a] for a in range(3)):
                            continue
                        q = [0, 0, 0]
                        for a, t in enumerate(transform):
                            q[abs(t) - 1] = int(donor[abs(t) - 1] + np.sign(t) * (position[a] - start[a]))
                        donor_axis = self.axes[target][daxis]
                        if q[daxis] == donor_axis[0]:
                            q[daxis] = donor_axis[1]
                        elif q[daxis] == donor_axis[-1]:
                            q[daxis] = donor_axis[-2]
                        else:
                            raise ValueError("Native GS interface must terminate on a donor boundary")
                        graph[row, slots[axis][side]] = lookups[target][tuple(q)] + 1
            for row in range(offset):
                neighbours = (
                    np.prod([1 + int(graph[row, a] != row + 1) + int(graph[row, b] != row + 1) for a, b in slots]) - 1
                )
                graph[row, 9] = 3 * neighbours + 2 if graph[row, 7] else 0
            groups = {}
            for i, node in enumerate(ids):
                groups.setdefault(int(node), []).append(i)
            for group in groups.values():
                if len(group) > 10:
                    raise ValueError("Native GS supports at most 10 coincident controls; use petsc-gmres")
                for i in group:
                    others = [j + 1 for j in group if j != i]
                    graph[i, 10] = len(group)
                    graph[i, 11 : 11 + len(others)] = others
            if not _NATIVE_LOCK.acquire(blocking=False):
                raise RuntimeError("Native solver is busy")
            try:
                result, status = _native.py_warp_controls(
                    np.concatenate(xyz),
                    graph,
                    np.concatenate(u),
                    1,
                    self.parameters,
                    self.sweeps,
                    len(self.mesh.zones) + 1,
                )
            finally:
                _NATIVE_LOCK.release()
            if status:
                raise ValueError(f"Native FMEM/Gauss-Seidel failed ({status}); refine controls or inspect geometry")
            return [
                {g: result[slices[g]][permutations[g]] for g in range(len(self.mesh.zones)) if self.owners[g] == rank}
                for rank in range(self.size)
            ]

        outgoing = self._checked(native)
        return outgoing[0] if self.comm is None else self.comm.scatter(outgoing, root=0)

    def _controls(self, targets):
        from .. import _native
        from ..solver import _NATIVE_LOCK

        points = np.array([self.mesh.zones[g].coordinates[p] for g, p in self.nodes]).reshape(-1, 3)
        fixed = np.array(
            [self._physical(g, p) or int(node) in targets for (g, p), node in zip(self.nodes, self.canonical)]
        )
        fixed = self._sum(fixed.astype(int))[self.inverse] > 0
        prescribed = np.array(
            [
                targets[int(node)] - xyz if int(node) in targets else np.zeros(3)
                for node, xyz in zip(self.canonical, points)
            ]
        ).reshape(-1, 3)
        count = self._sum(np.ones(len(points)))
        prescribed_unique = self._sum(prescribed) / count[:, None]
        prescribed = prescribed_unique[self.inverse]
        no_motion = scalar(self.comm, float(np.max(abs(prescribed), initial=0)), "max") == 0
        no_unknowns = scalar(self.comm, int(np.count_nonzero(~fixed))) == 0
        if no_motion or no_unknowns:
            return {g: prescribed[idx] for g, idx in self.zone_nodes.items()}
        if self.method == "fmem" and self.linear_solver == "native-gauss-seidel":
            return self._native_controls(points, fixed, prescribed)
        if self.method == "exponential":
            distance, displacement = nearest(self, points, points[fixed], prescribed[fixed], self.canonical[fixed])
            distance[fixed] = 0
            rmax = scalar(self.comm, float(np.max(distance, initial=0)), "max")
            rmin = scalar(self.comm, float(np.min(distance[distance > 1e-13], initial=1e30)), "min")
            u = (
                displacement
                * np.exp(np.minimum(0.0, -self.parameters[2] * (distance / max(rmax, 1e-30) - self.parameters[3])))[
                    :, None
                ]
            )
            u[fixed] = prescribed[fixed]
            # Original spring smoothing is simultaneous within each block.
            for _ in range(self.sweeps):
                new = u.copy()
                for g, idx in self.zone_nodes.items():
                    neighbours = self.graphs[g][:, 1:7] - 1
                    d = np.sqrt(np.sum((points[idx, None, :] - points[idx[neighbours]]) ** 2, axis=2) + rmin / 10000.0)
                    d[neighbours == np.arange(len(idx))[:, None]] = 1e30
                    weights = 1 / d
                    new[idx] = np.sum(u[idx[neighbours]] * weights[:, :, None], axis=1) / weights.sum(axis=1)[:, None]
                u = new
                u[fixed] = prescribed[fixed]
            u = (self._sum(u) / count[:, None])[self.inverse]
        else:
            # Native hookefe measures element-centre distance to fixed control
            # points and eligible fixed face centres, including remote blocks.
            centres, samples, cells = [], [points[fixed]], []
            for g, idx in self.zone_nodes.items():
                graph = self.graphs[g]
                plus = graph[:, [2, 4, 6]] - 1
                for n, (j, k, i) in enumerate(plus):
                    corners = np.array([n, j, k, plus[j, 1], i, plus[k, 2], plus[j, 2], plus[plus[j, 2], 1]])
                    if n != j and n != k and n != i:
                        centres.append(points[idx[corners]].mean(axis=0))
                        cells.append(int(idx[n]))
                    if fixed[idx[n]]:
                        for a, b, c in ((1, 2, 3), (2, 4, 5), (1, 4, 6)):
                            if fixed[idx[corners[a]]] and fixed[idx[corners[b]]]:
                                samples.append(points[idx[corners[[0, a, b, c]]]].mean(axis=0)[None, :])
            distances, _ = nearest(self, np.array(centres).reshape(-1, 3), np.concatenate(samples))
            rmax = scalar(self.comm, float(np.max(distances, initial=0)), "max")
            rmin = scalar(self.comm, float(np.min(distances[distances > 1e-13], initial=1e30)), "min")
            beta = self.parameters[0]
            logarithm = np.log(1.0 - 10.0 / 500000.0)
            r0 = (rmax * logarithm + beta * rmin) / (logarithm + beta)
            if not np.isfinite(r0) or abs(rmax - r0) < 1e-30:
                raise ValueError("Degenerate FMEM material-distance range")
            modulus = np.zeros(len(points))
            modulus[cells] = 10.0 / np.maximum(1e-13, 1.0 - np.exp(-beta * (distances - r0) / (rmax - r0)))

            def assemble():
                matrices = []
                if not _NATIVE_LOCK.acquire(blocking=False):
                    raise RuntimeError("Native solver is busy")
                try:
                    for g, idx in self.zone_nodes.items():
                        graph = self.graphs[g].copy()
                        graph[:, 7] = (~fixed[idx]).astype(int)
                        graph[fixed[idx], 9] = 0
                        sa, ija, status = _native.py_warp_matrix(points[idx], graph, modulus[idx])
                        if status:
                            raise ValueError(f"Native local FMEM assembly failed ({status})")
                        ndof = 3 * len(idx)
                        rows = np.repeat(np.arange(ndof), np.diff(ija[: ndof + 1]))
                        start, end = int(ija[0] - 1), int(ija[ndof] - 1)
                        cols = ija[start:end] - 1
                        values = sa[start:end]
                        valid = cols >= 0
                        matrices.append((idx, sa[:ndof].copy(), rows[valid], cols[valid], values[valid]))
                finally:
                    _NATIVE_LOCK.release()
                return matrices

            matrices = self._checked(assemble)
            fixed_unique = self._sum(fixed.astype(int)) > 0

            def matvec(vector):
                local = vector[self.inverse]
                output = np.zeros_like(local)
                for idx, diagonal, rows, cols, values in matrices:
                    v = local[idx].ravel()
                    out = diagonal * v + np.bincount(rows, weights=values * v[cols], minlength=len(v))
                    output[idx] = out.reshape(-1, 3)
                result = self._sum(output)
                result[fixed_unique] = vector[fixed_unique]
                return result

            diagonal = np.zeros_like(points)
            for idx, d, _, _, _ in matrices:
                diagonal[idx] = d.reshape(-1, 3)
            diagonal = self._sum(diagonal)
            diagonal[fixed_unique] = 1
            boundary = prescribed_unique.copy()
            boundary[~fixed_unique] = 0
            rhs = -matvec(boundary)
            rhs[fixed_unique] = 0
            from ._petsc_warp import solve_petsc

            result = (
                solve_petsc(self._petsc, self.shared, matvec, rhs, diagonal, self.linear_solver_options["restart"])
                + boundary
            )
            result[fixed_unique] = boundary[fixed_unique]
            u = result[self.inverse]
        return {g: u[idx] for g, idx in self.zone_nodes.items()}

    def warp(self, surface):
        """Return absolute owned coordinates without changing mesh or CFD state.

        All ranks participate. Current accepted volume coordinates are the base;
        users needing total displacement from x0 retain x0 themselves. A failed
        native control solve raises collectively and leaves the solver untouched.
        """

        def backend():
            from .. import _native

            if _native.py_api_version() != 22 or not hasattr(_native, "py_warp_matrix"):
                raise RuntimeError("Rebuild both native extensions for distributed mesh deformation API 22")

        self._checked(backend)
        targets = self._targets(surface)

        def local_controls():
            if (
                tuple((z.name, tuple(z.dimensions)) for z in self.mesh.zones) != self.layout.topology
                or tuple(g for g, z in enumerate(self.mesh.zones) if z.coordinates is not None) != self.layout.owned
            ):
                raise ValueError("Mesh ownership or dimensions changed; rebuild the adapter")
            if SurfaceTopology(self.mesh).signature != self.topology.signature:
                raise ValueError("Mesh topology changed; rebuild the adapter")
            for g in self.layout.owned:
                z = self.mesh.zones[g]
                if z.coordinates.shape != (*z.dimensions, 3) or not np.isfinite(z.coordinates).all():
                    raise ValueError("Current volume coordinates must be finite with unchanged shape")

        self._checked(local_controls)
        controls = self._controls(targets)

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
