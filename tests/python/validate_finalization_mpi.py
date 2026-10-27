"""MPI final/nonfinal chunks, collective validation, and warm-state parity."""
import argparse
from pathlib import Path
import numpy as np
from mpi4py import MPI
from cfl3d import CFL3D
from test_surface_forces import grouped_plate


def run(comm, root):
    for resident in (False, True):
        config = dict(turbulence='sst', mach=.2, cfl=1., iterations=2,
                      resident=resident, multigrid_levels=3, progress=False,
                      writevolumesolution=False, writeTecplotSurfaceSolution=True)
        directory = root / ('resident' if resident else 'restart')
        with CFL3D(grouped_plate(), config, comm=comm, name='coupled') as solver:
            try:
                solver.solve(final=comm.rank == 0, output_directory=directory)
            except RuntimeError as exc:
                assert 'finalization/chunk' in str(exc)
            else:
                raise AssertionError('Different final flags were accepted')
            assert solver.nSolve == 0
            first = solver.solve(final=False, output_directory=directory)
            if comm.rank == 0:
                assert not first.output_paths['surface'].exists()
            solver.options.writeIntermediateSolutions = True
            second = solver(final=False, output_directory=directory)
            assert (second.solve_index, second.chunk_index, second.iteration_offset) == (0, 1, 2)
            if comm.rank == 0:
                assert second.chunk_output_paths['surface'].exists()
                assert not second.output_paths['surface'].exists()
            solver.addSlices('y', -.5, 'absolute', 'Wing')
            if comm.rank == comm.size - 1:
                solver.clearSurfaceAnalysis()
            try:
                solver.finalizeSolution()
            except RuntimeError as exc:
                assert 'Surface registrations differ' in str(exc)
            else:
                raise AssertionError('Inconsistent surface registrations were accepted')
            solver.clearSurfaceAnalysis()
            before = solver.getRuntimeInfo().copy()
            assert solver.finalizeSolution() is second
            assert solver.getRuntimeInfo() == before
            solver.finalizeSolution()
            assert (solver.nSolve, solver.nSolutions) == (2, 1)
            if comm.rank == 0:
                assert second.output_paths['surface'].exists()
                rows = np.loadtxt(second.output_paths['residual'])
                np.testing.assert_array_equal(rows[:, 0], np.arange(1, 5))
            third = solver.solve(output_directory=directory)
            assert (third.solve_index, third.chunk_index, solver.nSolutions) == (1, 0, 2)
            if resident:
                assert solver.getRuntimeInfo()['initializations'] == 1
            combined = np.concatenate([first.history, second.history, third.history])
        with CFL3D(grouped_plate(), {**config, 'iterations': 6}, comm=comm) as solver:
            reference = solver.solve(output_directory=directory/'reference')
            np.testing.assert_array_equal(combined, reference.history)
            for zone in third.fields:
                np.testing.assert_array_equal(third.fields[zone], reference.fields[zone])
        comm.Barrier()
        if comm.rank == 0:
            print(f'PASS finalization: resident={resident}, ranks={comm.size}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    if args.split:
        comm = world.Split(world.rank // 2, world.rank)
        try:
            run(comm, args.directory.resolve()/f'group-{world.rank // 2}')
        finally:
            comm.Free()
    else:
        run(world, args.directory.resolve())
