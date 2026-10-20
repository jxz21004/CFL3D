"""Partitioned coupling geometry, independent of native zone load balancing.

Node identities follow CGNS one-to-one topology, never coordinate proximity.
Only local wall nodes/faces and small replicated zone/connectivity metadata are
constructed. There is no global surface-coordinate gather in this module.
"""
from collections import Counter, deque
import hashlib

import numpy as np


class SurfaceTopology:
    """Map connected CGNS vertices to the smallest whole-mesh vertex ID."""

    def __init__(self, mesh):
        signature = hashlib.sha256()
        self.dimensions = [tuple(z.dimensions) for z in mesh.zones]
        self.offsets = [0]
        for dimensions in self.dimensions:
            self.offsets.append(self.offsets[-1] + int(np.prod(dimensions, dtype=object)))
        if self.offsets[-1] > np.iinfo(np.int64).max:
            raise ValueError("Surface node IDs exceed int64 capacity")
        names = {zone.name: index for index, zone in enumerate(mesh.zones)}
        if len(names) != len(mesh.zones):
            raise ValueError("CGNS zone names must be unique")
        self.links = [[] for _ in mesh.zones]
        seen = set()
        for source, zone in enumerate(mesh.zones):
            signature.update(repr((zone.name, self.dimensions[source])).encode())
            for boundary in zone.boundaries:
                signature.update(repr((boundary.name, boundary.bc_type,
                                       np.asarray(boundary.point_range).tolist())).encode())
            for connection in zone.connections:
                if connection.donor not in names:
                    raise ValueError(f"Unknown donor zone {connection.donor}")
                target = names[connection.donor]
                points = self.point_range(source, connection.point_range)
                donor = self.point_range(target, connection.donor_range)
                transform = np.asarray(connection.transform)
                if (transform.shape != (3,) or transform.dtype.kind not in 'iu'
                        or sorted(abs(int(t)) for t in transform) != [1, 2, 3]):
                    raise ValueError("Connection Transform must be a signed permutation of 1,2,3")
                transform = tuple(map(int, transform))
                mapped = self._map(points[:, 1], points[:, 0], donor[:, 0], transform)
                if not np.array_equal(mapped, donor[:, 1]):
                    raise ValueError("Connection Transform and donor range disagree")
                inverse = [0, 0, 0]
                for axis, value in enumerate(transform):
                    inverse[abs(value)-1] = (axis+1) * (1 if value > 0 else -1)
                for a, b, p, q, t in ((source, target, points, donor, transform),
                                      (target, source, donor, points, tuple(inverse))):
                    token = (a, b, tuple(p.flat), tuple(q.flat), t)
                    if token not in seen:
                        seen.add(token)
                        signature.update(repr(token).encode())
                        self.links[a].append((b, p.min(axis=1), p.max(axis=1), p[:, 0], q[:, 0], t))
        self.signature = signature.hexdigest()

    def point_range(self, zone, value):
        points = np.asarray(value)
        if (points.shape != (3, 2) or points.dtype.kind not in 'iu'
                or np.any(points < 1) or np.any(points > np.array(self.dimensions[zone])[:, None])
                or np.count_nonzero(points[:, 0] == points[:, 1]) != 1):
            raise ValueError("PointRange must describe a rectangular boundary face within its zone")
        points = points.astype(np.int64) - 1
        axis = int(np.flatnonzero(points[:, 0] == points[:, 1])[0])
        if points[axis, 0] not in (0, self.dimensions[zone][axis]-1):
            raise ValueError("PointRange must lie on an exterior zone face")
        return points

    @staticmethod
    def _map(point, start, donor, transform):
        result = np.array(donor, copy=True)
        for axis, value in enumerate(transform):
            result[abs(value)-1] += (1 if value > 0 else -1) * (point[axis]-start[axis])
        return result

    def vertex_id(self, zone, point):
        _, nj, nk = self.dimensions[zone]
        i, j, k = map(int, point)
        return self.offsets[zone] + (i*nj+j)*nk+k

    def decode(self, node):
        from bisect import bisect_right
        zone = bisect_right(self.offsets, int(node))-1
        if zone < 0 or zone >= len(self.dimensions):
            raise ValueError("Invalid surface node ID")
        _, nj, nk = self.dimensions[zone]
        i, rest = divmod(int(node)-self.offsets[zone], nj*nk)
        j, k = divmod(rest, nk)
        return zone, (i, j, k)

    def canonical(self, zone, point):
        pending = deque([(zone, tuple(map(int, point)))])
        visited = set()
        smallest = self.vertex_id(zone, point)
        while pending:
            current, vertex = pending.popleft()
            token = current, vertex
            if token in visited:
                continue
            visited.add(token)
            smallest = min(smallest, self.vertex_id(current, vertex))
            point = np.asarray(vertex)
            for target, lo, hi, start, donor, transform in self.links[current]:
                if np.all(point >= lo) and np.all(point <= hi):
                    mapped = self._map(point, start, donor, transform)
                    pending.append((target, tuple(map(int, mapped))))
        return smallest

    def node_owner(self, node, size):
        zone, _ = self.decode(node)
        return 0 if size == 1 else 1 + zone % (size-1)


def build_surface_mesh(mesh, members, wall_keys, coordinate_owners, comm, checked):
    """Return owned unique nodes and locally owned faces with global connectivity.

    Faces follow current native zone placement. Coupling nodes instead use a
    fixed owner derived from their canonical zone, so setup-time vector shapes
    remain valid after the first native load-balance. Canonical coordinates are
    requested directly from the current zone owner, not selected by message order.
    """
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
    topology = checked(lambda: SurfaceTopology(mesh))
    if comm is not None:
        signatures = comm.allgather((topology.signature, tuple(coordinate_owners)))
        if len(set(signatures)) != 1:
            raise RuntimeError("Surface mesh topology/coordinate ownership differs within comm")

    def local_faces():
        coordinates, faces, ids, keys = {}, [], [], []
        offset = 0
        for g, zone in enumerate(mesh.zones):
            if (zone.coordinates is not None) != (coordinate_owners[g] == rank):
                raise RuntimeError("Surface coordinate payload differs from its declared zone owner")
            for boundary in zone.boundaries:
                key = zone.name, boundary.name
                points = topology.point_range(g, boundary.point_range)
                axis = int(np.flatnonzero(points[:, 0] == points[:, 1])[0])
                tangents = [a for a in range(3) if a != axis]
                lo, hi = points.min(axis=1), points.max(axis=1)
                nu, nv = (int(hi[a]-lo[a]) for a in tangents)
                if key in members and key in wall_keys:
                    if zone.coordinates is None:
                        raise RuntimeError("Owned wall has no coordinate payload")
                    nodes = np.empty((nu+1, nv+1), dtype=np.int64)
                    for u, v in np.ndindex(nodes.shape):
                        point = lo.copy()
                        point[tangents] += (u, v)
                        node = topology.canonical(g, point)
                        xyz = zone.coordinates[tuple(point)]
                        if not np.isfinite(xyz).all():
                            raise ValueError("Surface coordinates must be finite")
                        if node in coordinates and not np.allclose(coordinates[node], xyz, rtol=1e-10, atol=1e-10):
                            raise ValueError("One-to-one interface coordinates do not coincide")
                        coordinates[node] = xyz.copy()
                        nodes[u, v] = node
                    # C ordering matches the two cell axes in result.wall_forces.
                    patch = np.stack((nodes[:-1, :-1], nodes[1:, :-1],
                                      nodes[1:, 1:], nodes[:-1, 1:]), axis=-1).reshape(-1, 4)
                    # Into the fluid domain for a positive-handed structured grid.
                    if (-1)**axis * (1 if lo[axis] == 0 else -1) < 0:
                        patch = patch[:, [0, 3, 2, 1]]
                    faces.extend(patch)
                    ids.extend(range(offset, offset+nu*nv))
                    keys.extend([key]*(nu*nv))
                offset += nu*nv
        return coordinates, np.asarray(faces, dtype=np.int64).reshape(-1, 4), np.asarray(ids, dtype=np.int64), tuple(keys)

    local, faces, face_ids, face_keys = checked(local_faces)
    total_faces = len(faces) if comm is None else comm.allreduce(len(faces))
    if not total_faces:
        # No rank raises while another is about to enter a collective.
        error = ValueError if comm is None else RuntimeError
        raise error("Selected group has no wall surfaces")

    def registrations():
        outgoing = [{} for _ in range(size)]
        for node, xyz in local.items():
            outgoing[topology.node_owner(node, size)][node] = xyz
        return outgoing

    outgoing = checked(registrations)
    received = outgoing if comm is None else comm.alltoall(outgoing)

    def merge():
        owned = {}
        for part in received:
            for node, xyz in part.items():
                if node in owned and not np.allclose(owned[node], xyz, rtol=1e-10, atol=1e-10):
                    raise ValueError("One-to-one interface coordinates do not coincide")
                owned[node] = xyz
        return np.array(sorted(owned), dtype=np.int64), owned

    node_ids, samples = checked(merge)

    def coordinate_requests():
        requests = [[] for _ in range(size)]
        for node in node_ids:
            zone, _ = topology.decode(node)
            requests[coordinate_owners[zone]].append(int(node))
        return requests

    requests = checked(coordinate_requests)
    incoming = requests if comm is None else comm.alltoall(requests)

    def coordinate_replies():
        replies = [{} for _ in range(size)]
        for requester, nodes in enumerate(incoming):
            for node in nodes:
                zone, point = topology.decode(node)
                xyz = mesh.zones[zone].coordinates
                if xyz is None or coordinate_owners[zone] != rank:
                    raise RuntimeError("Surface coordinate request reached a non-owner")
                replies[requester][node] = xyz[point].copy()
        return replies

    replies = checked(coordinate_replies)
    received = replies if comm is None else comm.alltoall(replies)

    def result():
        values = {node:xyz for part in received for node,xyz in part.items()}
        xyz = np.array([values[int(node)] for node in node_ids], dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(xyz).all() or any(
                not np.allclose(samples[int(node)], value, rtol=1e-10, atol=1e-10)
                for node, value in zip(node_ids, xyz)):
            raise ValueError("One-to-one interface coordinates do not coincide")
        return dict(coordinates=xyz, node_ids=node_ids,
                    node_owners=np.full(len(node_ids), rank, dtype=np.int32),
                    connectivity=faces, face_ids=face_ids, face_keys=face_keys,
                    units={"coordinates": "m"})

    return checked(result)


def distribute_surface_forces(mesh, surface, wall_forces, scale, comm, checked):
    """Assemble pressure/viscous nodal forces on the fixed coupling owners.

    Return (N,6), pressure XYZ followed by viscous XYZ, in N. The quad-center
    displacement interpolation uses four weights of 1/4; its transpose assigns
    the integrated force to the same four corners. No area weighting or neighbour
    averaging belongs in this nodal FORCE assembly. It is also consistent with
    native CFL3D's arithmetic four-corner moment arm, including warped quads.
    """
    size = 1 if comm is None else comm.size
    topology = checked(lambda: SurfaceTopology(mesh))

    def contributions():
        outgoing = [{} for _ in range(size)]
        offset = 0
        # getSurfaceMesh orders whole patches consecutively, with C-ordered cell
        # rows matching wall_forces. Select the group's faces BEFORE scattering:
        # shared nodes must not pick up loads from unselected adjacent groups.
        for key, count in Counter(surface["face_keys"]).items():
            values = np.asarray(wall_forces[key])
            if (values.ndim != 3 or values.shape[-1] != 6 or values.size != count*6
                    or not np.isfinite(values).all()):
                raise ValueError(f"Native wall-force shape/values differ from surface cells for {key}")
            forces = values.reshape(-1, 6) * (0.25 * scale)
            faces = surface["connectivity"][offset:offset+count]
            if tuple(surface["face_keys"][offset:offset+count]) != (key,)*count:
                raise ValueError("Surface face rows must retain contiguous native patch order")
            for face, force in zip(faces, forces, strict=True):
                for node in face:
                    target = outgoing[topology.node_owner(node, size)]
                    if node not in target:
                        target[node] = np.zeros(6)
                    target[node] += force
            offset += count
        if offset != len(surface["connectivity"]):
            raise ValueError("Surface face keys and connectivity sizes differ")
        return outgoing

    outgoing = checked(contributions)
    # Unlike ADflow's patch-local force copies, our interface exposes one unique
    # owner per shared node. ADD remote contributions on that owner, not root.
    received = outgoing if comm is None else comm.alltoall(outgoing)

    def assemble():
        ids = surface["node_ids"]
        result = np.zeros((len(ids), 6))
        for part in received:
            if not part:
                continue
            nodes = np.fromiter(part, dtype=np.int64)
            rows = np.searchsorted(ids, nodes)
            if np.any(rows >= len(ids)) or not np.array_equal(ids[rows], nodes):
                raise RuntimeError("Nodal force reached a rank that does not own the surface node")
            result[rows] += np.asarray(list(part.values()))
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite dimensional nodal force")
        return result

    return checked(assemble)
