"""Optional IDWarp adapter; importing CFL3D does not import IDWarp.

The caller supplies an existing USMesh on the same communicator and from the
same CGNS zone order as the CFL3D mesh. This layer returns staged coordinates;
install them with solver.setVolumeCoordinates() followed by solve().
"""
import numpy as np
from .volume_layout import VolumeMeshLayout


def local_surface(surface, comm=None):
    """Convert global coupling IDs to face-local points using sparse requests.

    Faces may reference nodes owned by other ranks. Exchange only those nodes,
    not a replicated whole surface. IDWarp itself may subsequently gather its
    surface definition internally. All ranks must participate, including host.
    """
    size = 1 if comm is None else comm.size
    ids = np.asarray(surface['node_ids'], dtype=np.int64)
    xyz = np.asarray(surface['coordinates'])
    faces = np.asarray(surface['connectivity'], dtype=np.int64)
    invalid = (ids.ndim != 1 or xyz.shape != (len(ids), 3)
               or faces.ndim != 2 or faces.shape[1] != 4
               or not np.isfinite(xyz).all())
    if comm is not None:
        invalid = any(comm.allgather(invalid))
    if invalid:
        raise ValueError("Expected finite surface coordinates (N,3) and quad connectivity (M,4)")
    wanted = np.unique(faces)
    # IDs alone form the ownership directory; no global coordinate gather.
    partitions = [ids] if comm is None else comm.allgather(ids)
    owners = {}
    for rank, part in enumerate(partitions):
        for node in part:
            if int(node) in owners:
                raise ValueError("Surface node IDs must have unique owners")
            owners[int(node)] = rank
    missing = any(int(node) not in owners for node in wanted)
    if comm is not None:
        missing = any(comm.allgather(missing))
    if missing:
        raise ValueError("Surface connectivity references an unknown node")
    requests = [[] for _ in range(size)]
    for node in wanted:
        requests[owners[int(node)]].append(int(node))
    incoming = requests if comm is None else comm.alltoall(requests)
    local = dict(zip(map(int, ids), xyz))
    replies = [{node: local[node] for node in part} for part in incoming]
    received = replies if comm is None else comm.alltoall(replies)
    points = {node: value for part in received for node, value in part.items()}
    coordinates = np.array([points[int(node)] for node in wanted], dtype=float).reshape(-1, 3)
    connectivity = np.searchsorted(wanted, faces).astype(np.int32).ravel()
    return coordinates, connectivity, np.full(len(faces), 4, dtype=np.int32)


class IDWarpAdapter:
    """Map an IDWarp mesh to a fixed local CFL3D volume layout.

    No x0 is retained. Save reference coordinates explicitly in the calling
    Python application if needed. Construct again after ownership changes.
    """

    def __init__(self, warper, mesh, surface, comm=None):
        self.comm = comm
        self._surface_ids = np.asarray(surface["node_ids"]).copy()
        self._faces = np.asarray(surface["connectivity"]).copy()
        self.warper = warper
        self.layout = VolumeMeshLayout(mesh)
        indices = self.layout.component_indices()
        # IDWarp's usual build uses 32-bit Fortran integers. Refuse overflow.
        too_large = bool(len(indices) and indices.max() > np.iinfo(np.int32).max)
        if comm is not None:
            too_large = any(comm.allgather(too_large))
        if too_large:
            raise ValueError("IDWarp coordinate indices exceed int32 capacity")
        points, conn, sizes = local_surface(surface, comm)
        warper.setExternalMeshIndices(indices.astype(np.int32))
        warper.setSurfaceDefinition(points, conn, sizes)

    def warp(self, surface):
        """Collectively deform and return (N,3) absolute volume coordinates.

        Surface topology/partition must match construction. This method does
        not update CFL3D's native geometry or invalidate its cached results.
        """
        comm = self.comm
        changed = (not np.array_equal(surface["node_ids"], self._surface_ids)
                   or not np.array_equal(surface["connectivity"], self._faces))
        if comm is not None:
            changed = any(comm.allgather(changed))
        if changed:
            raise ValueError("Surface topology or partition changed; rebuild the adapter")
        points, _, _ = local_surface(surface, comm)
        self.warper.setSurfaceCoordinates(points)
        self.warper.warpMesh()
        xyz = np.asarray(self.warper.getSolverGrid()).reshape(-1, 3)
        invalid = xyz.shape != (self.layout.size, 3) or not np.isfinite(xyz).all()
        if comm is not None:
            invalid = any(comm.allgather(invalid))
        if invalid:
            raise ValueError("IDWarp returned invalid volume coordinates")
        return xyz.copy()
