"""Initialization-only block splitting using NASA's unchanged SPLITTER kernels.

Only metadata enters the native bridge. ``sources`` records original file indices
so owners slice coordinates locally and IDWarp retains its original numbering.
"""
from copy import deepcopy
from dataclasses import replace
from threading import Lock
import numpy as np
from ..cgns import CGNSMesh, Connection

_LOCK = Lock()  # SPLITTER's COMMON state is separate from solver state.
_TANGENTS = ((1, 2), (2, 0), (0, 1))


def _face(points, dims):
    p = np.asarray(points)
    if p.shape != (3, 2) or not np.issubdtype(p.dtype, np.integer):
        raise ValueError('Split requires integer (3,2) PointRanges')
    fixed = np.flatnonzero(p[:, 0] == p[:, 1])
    if len(fixed) != 1 or np.any(p < 1) or np.any(p > np.asarray(dims)[:, None]):
        raise ValueError('Split requires nondegenerate rectangular boundary faces')
    axis = int(fixed[0])
    if p[axis, 0] not in (1, dims[axis]):
        raise ValueError('Split requires exterior boundary faces')
    return 2*axis + (2 if p[axis, 0] == dims[axis] else 1)


def _range(face, bounds, dims):
    axis = (abs(face)-1)//2
    p = np.zeros((3, 2), dtype=np.int64)
    p[axis] = 1 if abs(face) % 2 else dims[axis]
    p[list(_TANGENTS[axis])] = np.asarray(bounds).reshape(2, 2)
    return p


def _pack(mesh):
    names = {z.name: i for i, z in enumerate(mesh.zones)}
    if len(names) != len(mesh.zones):
        raise ValueError('Duplicate zone names')
    rows, owners, originals, links = [], [], [], {}
    def add_link(g, p, d, q, transform):
        f, df = _face(p, mesh.zones[g].dimensions), _face(q, mesh.zones[d].dimensions)
        t = np.asarray(transform, dtype=int)
        if sorted(abs(t)) != [1, 2, 3]:
            raise ValueError('Invalid connection transform')
        expected = q[:, 0].copy()
        for a in range(3):
            expected[abs(t[a])-1] += np.sign(t[a])*(p[a, 1]-p[a, 0])
        if not np.array_equal(expected, q[:, 1]):
            raise ValueError('Connection ranges and transform disagree')
        tangents, dt = _TANGENTS[(f-1)//2], _TANGENTS[(df-1)//2]
        if set(abs(t[list(tangents)])-1) != set(dt):
            raise ValueError('Connection transform does not map face tangents')
        p0 = np.min(p, axis=1)
        q0 = q[:, 0].copy()
        for a in range(3):
            q0[abs(t[a])-1] += np.sign(t[a])*(p0[a]-p[a, 0])
        qe = q0.copy()
        for a in tangents:
            qe[abs(t[a])-1] += np.sign(t[a])*abs(p[a, 1]-p[a, 0])
        r = np.zeros(24, np.int32)
        r[:2] = (1, f)
        r[2:6] = np.sort(p[list(tangents)], axis=1).ravel()
        r[6:8] = d+1, df if abs(t[tangents[0]])-1 == dt[0] else -df
        r[8:12] = np.column_stack((q0[list(dt)], qe[list(dt)])).ravel()
        key = (g, f, *r[2:6])
        if key in links:
            if not np.array_equal(rows[links[key]], r):
                raise ValueError('Conflicting reciprocal connections')
            return
        links[key] = len(rows); rows.append(r); owners.append(g+1)
    for g, z in enumerate(mesh.zones):
        for bc in z.boundaries:
            if any(np.asarray(v).size > 1 for v in bc.data.values()):
                raise NotImplementedError('Automatic splitting does not support spatial BC datasets')
            r = np.zeros(24, np.int32)
            face = _face(bc.point_range, z.dimensions)
            r[:2] = 1002, face  # Opaque physical segment; actual BC preserved by tag.
            r[2:6] = np.sort(bc.point_range[list(_TANGENTS[(face-1)//2])], axis=1).ravel()
            originals.append(deepcopy(bc)); r[23] = len(originals)
            rows.append(r); owners.append(g+1)
        for c in z.connections:
            d = names[c.donor]
            p, q, t = c.point_range, c.donor_range, np.asarray(c.transform)
            add_link(g, p, d, q, t)
            inverse = np.zeros(3, dtype=int)
            for a, v in enumerate(t):
                inverse[abs(v)-1] = int(np.sign(v))*(a+1)
            add_link(d, q, g, p, inverse)
    peers = np.zeros(len(rows), np.int32)
    for key, s in links.items():
        r = rows[s]
        donor_bounds = np.sort(r[8:12].reshape(2, 2), axis=1).ravel()
        peers[s] = links[(int(r[6])-1, abs(int(r[7])), *donor_bounds)] + 1
    return np.asarray(rows, np.int32).T.copy(order='F'), np.asarray(owners, np.int32), peers, originals


def _split_once(mesh, block, axis, index):
    from .. import _native
    if not hasattr(_native, 'py_split_topology'):
        raise RuntimeError('Automatic splitting requires rebuilding the serial extension with tools/build_python.py')
    rows, owners, peers, originals = _pack(mesh)
    dims = np.asarray([z.dimensions for z in mesh.zones], np.int32).T.copy(order='F')
    with _LOCK:
        result, owner, peer, bounds, count, status = _native.py_split_topology(
            dims, rows, owners, peers, block+1, axis+1, index)
    if status:
        raise ValueError(f'Native SPLITTER rejected cut: status {status}')
    parent = mesh.zones[block]
    # The bridge keeps existing IDs and appends exactly one child.
    number = len(mesh.zones)
    used = {z.name for z in mesh.zones}
    name = parent.name[:20] + f'_split{number}'
    while name in used:
        number += 1
        name = parent.name[:20] + f'_split{number}'
    zones = [replace(z, coordinates=None, boundaries=[], connections=[],
                     coordinate_dimensions=tuple(map(int, bounds[1::2, i]-bounds[::2, i]+1)))
             for i, z in enumerate([*mesh.zones, replace(parent, name=name)])]
    for s in range(count):
        r = result[:, s]; z = zones[owner[s]-1]
        p = _range(int(r[1]), r[2:6], z.dimensions)
        if peer[s]:
            d = int(owner[peer[s]-1])-1
            q = _range(int(r[7]), r[8:12], zones[d].dimensions)
            ta, tb = _TANGENTS[(r[1]-1)//2], _TANGENTS[(abs(r[7])-1)//2]
            if r[7] < 0:
                tb = tb[::-1]
            t = np.zeros(3, dtype=int)
            for a, b in zip(ta, tb):
                t[a] = (b+1)*(1 if q[b, 1] > q[b, 0] else -1)
            a, b = (int(r[1])-1)//2, (abs(int(r[7]))-1)//2
            t[a] = (b+1)*(-1 if (r[1] % 2) == (abs(r[7]) % 2) else 1)
            z.connections.append(Connection(zones[d].name, p, q, t))
        else:
            bc = deepcopy(originals[int(r[23])-1]); bc.point_range = p
            z.boundaries.append(bc)
    sources = deepcopy(mesh.split_sources)
    for i in (block, len(zones)-1):
        origin, original_dims, offset, lo, hi = mesh.split_sources[parent.name]
        start = np.asarray(lo) + bounds[::2, i]-1
        stop = np.asarray(lo) + bounds[1::2, i]
        sources[zones[i].name] = (origin, original_dims, offset, tuple(start), tuple(stop))
    return CGNSMesh(zones, mesh.source, sources)


def split_metadata(mesh, workers, levels):
    """Return a detached metadata plan; preserve requested levels, never downgrade."""
    if workers <= len(mesh.zones):
        return mesh
    if any(z.coordinates is not None for z in mesh.zones):
        raise ValueError('Split planning accepts metadata only')
    factor = 2**(levels-1)
    offset, sources = 0, {}
    for z in mesh.zones:
        dims = tuple(z.dimensions)
        sources[z.name] = (z.name, dims, offset, (0, 0, 0), dims)
        offset += int(np.prod(dims))
        active = [a for a in range(3) if not (a == 0 and dims[a] == 2)]
        if any((dims[a]-1) % factor or (dims[a]-1)//factor < 2 for a in active):
            raise ValueError(f'{z.name}: dimensions do not support {levels} multigrid levels')
        for p in [b.point_range for b in z.boundaries] + [c.point_range for c in z.connections]:
            face = _face(p, dims)
            if any((int(v)-1) % factor for a in _TANGENTS[(face-1)//2]
                   if a in active for v in p[a]):
                raise ValueError(f'{z.name}: boundary/interface does not align with {levels} levels')
    result = CGNSMesh(deepcopy(mesh.zones), mesh.source, sources)
    while len(result.zones) < workers:
        candidates = []
        for b, z in enumerate(result.zones):
            cells = np.asarray(z.dimensions)-1
            for a in range(3):
                # Both children need at least two cells on the coarsest level.
                if cells[a] < 4*factor:
                    continue
                cut = max(2*factor, min(cells[a]-2*factor, (cells[a]//(2*factor))*factor))
                candidates.append((int(np.prod(cells)), int(cells[a]), -b, -a, int(cut)+1))
        if not candidates:
            raise ValueError(f'Cannot create {workers} blocks while retaining {levels} multigrid levels; '
                             f'only {len(result.zones)} legal blocks available. Reduce MPI ranks or change the mesh.')
        _, _, b, a, index = max(candidates)
        result = _split_once(result, -b, -a, index)
    return result


def install_split(solver):
    """Collective installation after original owner-local coordinate loading."""
    comm = solver.comm
    # Restart payloads are keyed by the original topology; never reinterpret them.
    if solver.options.restartFile:
        raise ValueError('Automatic block splitting cannot repartition restartFile; use fewer MPI ranks')
    plan = solver._checked(lambda: split_metadata(
        solver._mesh_metadata(solver.mesh), comm.size-1, solver.options.multigrid_levels)
        if comm.rank == 0 else None)
    plan = comm.bcast(plan, root=0)
    owners = tuple(1+i % (comm.size-1) for i in range(len(plan.zones)))
    def slices():
        local = {z.name:z.coordinates for z in solver.mesh.zones if z.coordinates is not None}
        outgoing = [{} for _ in range(comm.size)]
        for i,z in enumerate(plan.zones):
            origin, _, _, lo, hi = plan.split_sources[z.name]
            if origin in local:
                selection = tuple(slice(a,b) for a,b in zip(lo,hi))
                outgoing[owners[i]][z.name] = np.array(local[origin][selection], copy=True)
        return outgoing
    outgoing = solver._checked(slices)
    received = comm.alltoall(outgoing)
    local = {k:v for part in received for k,v in part.items()}
    for z in plan.zones:
        z.coordinates = local.get(z.name)
    def overrides():
        # A fresh file reload may arrive with overrides expanded by a previous split.
        originals = {value[0] for value in plan.split_sources.values()}
        previous = getattr(solver, "_split_override_keys", set())
        base = {k:v for k,v in solver.boundary_overrides.items() if k[0] in originals or k not in previous}
        known = {(z.name,b.name) for z in solver.mesh.zones for b in z.boundaries}
        unknown = set(base)-known
        if unknown:
            raise ValueError(f'Unknown boundary override keys: {unknown}')
        return {(z.name,b.name):deepcopy(base[(plan.split_sources[z.name][0],b.name)])
                for z in plan.zones for b in z.boundaries
                if (plan.split_sources[z.name][0],b.name) in base}
    expanded = solver._checked(overrides)
    solver.mesh = plan
    solver.boundary_overrides = expanded
    solver._split_override_keys = set(expanded)
    solver._mesh_owners = owners
    solver._progress(f'Automatically split into {len(plan.zones)} blocks for {comm.size-1} workers; '
                     f'retained {solver.options.multigrid_levels} multigrid levels')
