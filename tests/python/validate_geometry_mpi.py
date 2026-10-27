"""Public geometry updates, all-level refresh and unchanged continuation."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
from mpi4py import MPI
from cfl3d import CFL3D, MultigridStage
from test_surface_forces import grouped_plate

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='sst')
parser.add_argument('--y-up', action='store_true')
args = parser.parse_args()
comm = MPI.COMM_WORLD
root = tempfile.TemporaryDirectory() if comm.rank == 0 else None
path = Path(comm.bcast(root.name if root else None, root=0))
config = dict(resident=True, iterations=2, turbulence=args.model, mach=.2,
              coarse_grid_stages=(MultigridStage(1, 1), MultigridStage(1, 2)),
              y_up=args.y_up,
              multigrid_levels=3, multigrid_cycle=2, cfl=1.,
              writevolumesolution=False, writeTecplotSurfaceSolution=False, progress=False)
mesh = grouped_plate()
if args.y_up:
    for zone in mesh.zones:
        old = zone.coordinates.copy()
        zone.coordinates[..., 1] = old[..., 2]
        zone.coordinates[..., 2] = -old[..., 1]
with CFL3D(mesh, config, comm=comm) as solver:
    solver.solve(output_directory=path/'updated')
    xyz = solver.getVolumeCoordinates()
    solver.setVolumeCoordinates(xyz)
    solver.solve(output_directory=path/'updated')
    assert solver.getRuntimeInfo()['geometry_updates'] == 0
    moved = xyz.copy()
    moved[:, 1 if args.y_up else 2] *= 1.01
    solver.setVolumeCoordinates(moved)
    result = solver.solve(output_directory=path/'updated')
    np.testing.assert_array_equal(solver.getVolumeCoordinates(), moved)
    assert solver.getRuntimeInfo()['geometry_updates'] == 1
    assert solver.getRuntimeInfo()['initializations'] == 1
    assert result.start_mode == 'restart'
    assert all(np.isfinite(q).all() for q in result.fields.values())
    solver.solve(output_directory=path/'updated')
    assert solver.getRuntimeInfo()['geometry_updates'] == 1
comm.Barrier()
if comm.rank == 0:
    print('Resident geometry update and no-change continuation passed')
    root.cleanup()
