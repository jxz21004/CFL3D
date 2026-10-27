"""Expected native geometry failures return on all ranks without MPI abort."""
import argparse
import argparse
from pathlib import Path
import tempfile
import numpy as np
from mpi4py import MPI
from cfl3d import CFL3D, SolveFailure
from cfl3d.mesh import VolumeLayout
from test_surface_forces import grouped_plate

parser = argparse.ArgumentParser()
parser.add_argument('--split', action='store_true')
args = parser.parse_args()
comm = (MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank // 2, MPI.COMM_WORLD.rank)
        if args.split else MPI.COMM_WORLD)
root = tempfile.TemporaryDirectory() if comm.rank == 0 else None
path = Path(comm.bcast(root.name if root else None, root=0))
options = dict(resident=True, iterations=2, multigrid_levels=3, turbulence='sst',
               writevolumesolution=False, writeTecplotSurfaceSolution=False, progress=False)
# First exercise volume rejection everywhere, then a metric failure on one owner.
for mode in ('volume', 'one-owner'):
    with CFL3D(grouped_plate(), options, comm=comm) as solver:
        initial = solver.solve(output_directory=path/mode)
        assert not initial.fail
        original = solver.getVolumeCoordinates()
        bad = original.copy()
        if mode == 'volume':
            bad[:, 2] *= -1
        else:
            zones = VolumeLayout(solver.mesh).unpack(bad)
            name = solver.mesh.zones[0].name
            if name in zones:
                zones[name][:, 2:-2, 1:-1, 2] *= -1
            pieces = [zones[z.name].reshape(-1, 3, order='F') for z in solver.mesh.zones
                      if z.coordinates is not None]
            bad = np.concatenate(pieces) if pieces else np.empty((0, 3))
        solver.setVolumeCoordinates(bad)
        info = solver.getRuntimeInfo()
        result = solver.solve(output_directory=path/mode)
        assert isinstance(result, SolveFailure) and result.fail and result.reset_required
        assert len(set(comm.allgather(result.native_status))) == 1
        assert solver.getRuntimeInfo()['cycles'] == info['cycles']
        assert solver.getRuntimeInfo()['geometry_updates'] == 0
        assert solver.nSolve == 1
        assert solver.solve(output_directory=path/mode).fail
        np.testing.assert_array_equal(solver.getVolumeCoordinates(), original)
        try:
            solver.getSolution()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Failed geometry exposed old loads')
        solver.resetFlow()
        # Python never installed the rejected coordinates; restart on its good mesh.
        recovered = solver.solve(output_directory=path/(mode+'-recovered'))
        assert not recovered.fail
comm.Barrier()
if comm.rank == 0:
    print('Collective geometry failure, no advance, cleanup and recovery passed')
    root.cleanup()
