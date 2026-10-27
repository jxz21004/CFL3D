"""Actual IDWarp zero-motion mapping; run serially or with mpirun -np 3."""
from pathlib import Path
import tempfile
import numpy as np
from mpi4py import MPI
from idwarp import USMesh
from cfl3d import CFL3D
from cfl3d.mesh import IDWarpAdapter
from cgns_fixtures import write_adf
from test_surface_mesh import wall_boxes

comm = MPI.COMM_WORLD
root = tempfile.TemporaryDirectory() if comm.rank == 0 else None
path = Path(comm.bcast(root.name if root else None, root=0)) / 'mesh.cgns'
if comm.rank == 0:
    write_adf(wall_boxes(), path)
comm.Barrier()
solver = CFL3D(str(path), comm=comm)
warper = USMesh(options={'gridFile': str(path)}, comm=comm)
adapter = IDWarpAdapter(warper, solver.mesh, solver.getSurfaceMesh(), comm)
original = adapter.layout.coordinates(solver.mesh)
result = adapter.warp(solver.getSurfaceMesh())
np.testing.assert_allclose(result, original, rtol=0, atol=1e-12)
# Nonzero motion verifies that the adapter is not simply echoing initial data.
surface = solver.getSurfaceMesh()
surface['coordinates'][:, 0] += 0.01
moved = adapter.warp(surface)
assert comm.allreduce(float(np.max(np.abs(moved-original))) if moved.size else 0., op=MPI.MAX) > 1e-5
np.testing.assert_array_equal(adapter.layout.coordinates(solver.mesh), original)
solver.close()
comm.Barrier()
if comm.rank == 0:
    print('IDWarp zero-motion mapping and nonzero staging passed')
    root.cleanup()
