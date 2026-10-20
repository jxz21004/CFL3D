"""Distributed nodal loads, native resultants and serial/MPI consistency."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from mpi4py import MPI

from cfl3d import CFL3D
from test_nodal_forces import cell_resultants, coupling_mesh


def check_forces(solver, group, reference=None):
    comm = solver.comm
    with patch.object(solver, '_gather_mesh', side_effect=AssertionError('global mesh gather')), \
         patch.object(solver, '_surface_collective', side_effect=AssertionError('global surface gather')):
        surface = solver.getSurfaceMesh(group)
        values = {kind: solver.getSurfaceForces(group, component=kind)
                  for kind in ('total', 'pressure', 'viscous')}
    def local():
        assert values['total'].shape == surface['coordinates'].shape
        np.testing.assert_array_equal(values['total'], values['pressure']+values['viscous'])
        if comm.rank == 0:
            assert values['total'].shape == (0, 3)
    solver._checked(local)
    force_error, moment_error = 0., 0.
    for kind, forces in values.items():
        nodal = np.r_[forces.sum(axis=0), np.cross(
            surface['coordinates']-solver._last_options.moment_center, forces).sum(axis=0)]
        total = comm.allreduce(nodal)
        expected = comm.allreduce(np.concatenate(cell_resultants(solver, group, kind)))
        solver._checked(lambda: np.testing.assert_allclose(total, expected, rtol=2e-12, atol=1e-10))
        if kind == 'total':
            loads = solver.getSolution(group)
            prefix = '' if group is None else group+'_'
            integrated = [loads[prefix+k+a] for k in ('F', 'M') for a in 'xyz']
            solver._checked(lambda: np.testing.assert_allclose(total, integrated, rtol=2e-12, atol=1e-10))
            force_error = float(np.max(np.abs(total[:3]-integrated[:3])))
            moment_error = float(np.max(np.abs(total[3:]-integrated[3:])))
    # Root gathers here ONLY to compare the complete vectors with serial output.
    pieces = comm.gather((surface['node_ids'], values), root=0)
    def serial_check():
        if comm.rank != 0 or reference is None:
            return
        nodes = np.concatenate([part[0] for part in pieces])
        order = np.argsort(nodes)
        assert len(nodes) == len(np.unique(nodes))
        np.testing.assert_array_equal(nodes[order], reference['ids'])
        for kind in values:
            nodal = np.concatenate([part[1][kind] for part in pieces])[order]
            np.testing.assert_allclose(nodal, reference[kind], rtol=2e-12, atol=1e-10)
    solver._checked(serial_check)
    return force_error, moment_error


def run(comm, directory):
    groups = {'Overlap': ['Wing', 'Forward', 'Aft', 'Wing']}
    selection = (None, 'Wing', 'Overlap', 'Forward', 'Aft')
    config = dict(turbulence='sst', mach=.2, alpha=2., beta=3., y_up=True,
                  moment_center=(.2, -.1, .3), sref=2., cref=.7, bref=1.3,
                  cfl=1., iterations=2, multigrid_levels=3, resident=True,
                  writevolumesolution=False, writeTecplotSurfaceSolution=False)
    reference = None
    if comm.rank == 0:
        with CFL3D(coupling_mesh(), {**config, 'resident': False, 'iterations': 5,
                                   'freestream_pressure': 90000.}, comm=None, groups=groups) as serial:
            serial.solve(output_directory=directory/'serial')
            reference = {group: dict(ids=serial.getSurfaceMesh(group)['node_ids'],
                                     **{kind: serial.getSurfaceForces(group, component=kind)
                                        for kind in ('total', 'pressure', 'viscous')}) for group in selection}
    with CFL3D(coupling_mesh() if comm.rank == 0 else None, config, comm=comm, groups=groups) as solver:
        initial_ids = solver.getSurfaceMesh('Wing')['node_ids'].copy()
        result = solver.solve(output_directory=directory/'mpi')
        try:
            solver.getSurfaceForces('Wing')
        except RuntimeError as exc:
            assert 'freestream pressure' in str(exc)
        else:
            raise AssertionError('Expected missing dimensional scale')
        solver.options.freestream_pressure = 180000. if comm.rank == comm.size-1 else 90000.
        try:
            solver.getSurfaceForces('Wing')
        except RuntimeError as exc:
            assert 'dimensional scale differs' in str(exc)
        else:
            raise AssertionError('Expected mismatched dimensional scale')
        solver.options.freestream_pressure = 90000.
        try:
            solver.getSurfaceForces('Wing', component='pressure' if comm.rank == comm.size-1 else 'total')
        except RuntimeError as exc:
            assert 'component/dimensional scale differs' in str(exc)
        else:
            raise AssertionError('Expected mismatched component')
        for group in selection:
            check_forces(solver, group)
        solver.options.iterations = 3
        result = solver.solve(output_directory=directory/'mpi')
        force_error, moment_error = 0., 0.
        for group in selection:
            a, b = check_forces(solver, group, reference[group] if comm.rank == 0 else None)
            force_error, moment_error = max(a, force_error), max(b, moment_error)
        saved = solver.getSurfaceForces('Wing')
        # A configured pressure from the successful solve outranks live edits.
        solver.options.freestream_pressure = 180000.
        current = solver.getSurfaceForces('Wing')
        solver._checked(lambda: np.testing.assert_array_equal(saved, current))
        # Collective queries precede local assertions and their error checks.
        current_ids = solver.getSurfaceMesh('Wing')['node_ids']
        solver._checked(lambda: np.testing.assert_array_equal(initial_ids, current_ids))
        assert solver.getRuntimeInfo()['initializations'] == 1
        assert solver.getRuntimeInfo()['cycles'] == 5
        ranks = comm.gather(dict(rank=comm.rank, nodes=len(saved), force_bytes=saved.nbytes), root=0)
        solver._checked(lambda: (directory/'nodal-force-report.json').write_text(json.dumps(dict(
            ranks=ranks, max_force_error_N=force_error, max_moment_error_Nm=moment_error,
            zone_owners=result.zone_owners, serial_vectors_match=True,
            groups=list(selection), full_surface_gather_in_query=False), indent=2)+'\n') if comm.rank == 0 else None)
    after_close = solver.getSurfaceForces('Wing')
    solver._checked(lambda: np.testing.assert_array_equal(saved, after_close))
    if comm.rank == 0:
        print(f'Nodal forces: serial/MPI match; maximum native resultant errors {force_error:.3e} N, {moment_error:.3e} N*m', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--exclude', action='store_true')
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    if args.split or args.exclude:
        assert world.size == 4
        color = (MPI.UNDEFINED if world.rank == 0 else 0) if args.exclude else world.rank//2
        comm = world.Split(color, world.rank)
    else:
        assert world.size == 3
        color, comm = 0, world
    try:
        if comm != MPI.COMM_NULL:
            run(comm, args.directory.resolve()/f'group-{color}')
            if comm != world:
                comm.Free()
        world.Barrier()
    except BaseException:
        import traceback
        traceback.print_exc()
        world.Abort(1)


if __name__ == '__main__':
    main()
