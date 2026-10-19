"""Experimental resident cycles: serial parity, native ownership and one setup."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
from mpi4py import MPI
from cfl3d import CFL3D, _native_mpi as native
from test_surface_forces import grouped_plate


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def run(comm, root, model, open_ended=False):
    mesh = grouped_plate(split=25)
    # Unequal 3D blocks, with y-up input and sideslip, test the coordinate path.
    for zone in mesh.zones:
        zone.coordinates = np.linspace(zone.coordinates[0], zone.coordinates[-1], 5)
        for boundary in zone.boundaries:
            boundary.point_range[0][boundary.point_range[0] == 2] = 5
            if boundary.point_range[0, 0] == boundary.point_range[0, 1]:
                boundary.bc_type = 'BCSymmetryPlane'
        for connection in zone.connections:
            for points in (connection.point_range, connection.donor_range):
                points[0][points[0] == 2] = 5
        xyz = zone.coordinates.copy()
        zone.coordinates[..., 1] = xyz[..., 2]
        zone.coordinates[..., 2] = -xyz[..., 1]
    options = dict(turbulence=model, mach=.2, alpha=2., beta=3., y_up=True,
        cfl=1., iterations=29 if open_ended else 9, multigrid_levels=3, multigrid_cycle=2,
        first_order_iterations=2, cfl_ramp_steps=3, cfl_ramp_factor=2.,
        writevolumesolution=False, writeTecplotSurfaceSolution=False)
    reference = error = None
    if comm.rank == 0:
        try:
            serial = CFL3D(mesh, options, comm=None)
            result = serial.solve(output_directory=root/'serial')
            reference = ({name:digest(values) for name,values in result.fields.items()},
                         result.history, result.surface_coefficients, result.fields)
        except Exception as exc:
            error = repr(exc)
    error, reference = comm.bcast((error, reference), root=0)
    if error:
        raise RuntimeError(error)
    if open_ended:
        options['iterations'] = 2  # Reserve only the first chunk, not the final budget.
    solver = CFL3D(mesh, options, comm=comm)
    arrays, xyz = solver._native_arrays()
    original_directory = Path.cwd()
    directory = root/f'rank-{comm.rank}'
    solver._checked(lambda: directory.mkdir(parents=True, exist_ok=True))
    native_comm = comm.Dup()
    try:
        assert native.py_set_communicator(native_comm.py2f()) == 0
        assert native.py_configure(**arrays) == 0
        native.py_set_coordinates(xyz)
        del xyz
        native.py_set_output_names('resident', 0, 0)
        assert native.py_start_resident(2, int(open_ended)) == 0
        os.chdir(directory)
        status, rows = native.py_solve()
        assert (status, rows) == (0, 2)
        owners, cells, wall_cells = native.py_get_partition(len(mesh.zones))
        if comm.size == 3:
            assert tuple(owners) == (2, 1)  # Original native load balancer, unequal fixture.
        solver._redistribute_coordinates(tuple(owners))
        initial_info = native.py_resident_info()
        capacities = [initial_info[4]]
        def check(total, final=False):
            info = native.py_resident_info()
            closed = final and not open_ended
            assert info[:4] == (0 if closed else 1, 0 if closed else 1, total, 1), info
            assert info[5:] == initial_info[5:]
            assert info[4] >= total
            capacities.append(info[4])
            assert info[6] == 0  # No retained whole-case input coordinates.
            assert native.py_get_partition(len(mesh.zones))[0].tolist() == owners.tolist()
            history = native.py_get_history(total)
            np.testing.assert_array_equal(history, reference[1][:total])
            assert not Path('.cfl3d-state.bin').exists()
            if final:
                q = native.py_get_solution(cells)
                # The native buffer uses CFL3D axes; CFL3D.solve applies this
                # conversion before exposing primitive fields for y-up input.
                q[:, [2,3]] = q[:, [3,2]]
                q[:, 3] *= -1
                offset = 0
                for index, zone in enumerate(solver.mesh.zones):
                    if owners[index] != comm.rank:
                        continue
                    count = int(np.prod(np.array(zone.dimensions)-1))
                    values = q[offset:offset+count].reshape((*tuple(n-1 for n in zone.dimensions),5),order='F').copy()
                    expected = reference[3][zone.name]
                    np.testing.assert_array_equal(values, expected)
                    assert digest(values) == reference[0][zone.name], (
                        zone.name, float(np.max(np.abs(values-expected))),
                        int(np.count_nonzero(np.signbit(values) != np.signbit(expected))))
                    offset += count
                assert offset == cells
                coefficients = native.py_get_surface_coefficients(len(solver._surface_keys))
                for column, key in enumerate(solver._surface_keys):
                    if key is not None:
                        np.testing.assert_array_equal(coefficients[:,column], reference[2][key])
        solver._checked(lambda: check(2))
        assert native.py_set_communicator(comm.py2f()) != 0  # Borrowed context must stay intact.
        schedule = ((3,5),(1,6),(7,13),(16,29)) if open_ended else ((3,5),(1,6),(3,9))
        for chunk, total in schedule:
            assert native.py_advance_resident(chunk, 0.) == (0, total)
            solver._checked(lambda: check(total, total == schedule[-1][1]))
            comm.Barrier()  # Python MPI operations remain usable between native chunks.
        if open_ended:
            assert len(set(capacities)) > 1
        assert native.py_close_resident() == 0
        assert native.py_resident_info()[:4] == (0,0,schedule[-1][1],1)
        native.py_release_buffers()
        # Early close must not advance a cycle or reduce published forces twice.
        arrays, xyz = solver._native_arrays()
        assert native.py_configure(**arrays) == 0
        native.py_set_coordinates(xyz)
        del xyz
        assert native.py_start_resident(2, int(open_ended)) == 0
        assert native.py_solve() == (0, 2)
        before = native.py_get_surface_coefficients(len(solver._surface_keys))
        assert native.py_close_resident() == 0
        assert native.py_resident_info()[:4] == (0, 0, 2, 1)
        np.testing.assert_array_equal(native.py_get_surface_coefficients(len(solver._surface_keys)), before)
    finally:
        try:
            native.py_close_resident()
            native.py_release_buffers()
        finally:
            os.chdir(original_directory)
            native_comm.Free()
    if comm.rank == 0:
        print('RESIDENT_PASS', model, root, 'open_ended=', open_ended,
              'setup=1, cycles=', schedule[-1][1], 'capacities=', capacities,
              'no retained input xyz', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--model', default='sst')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--open-ended', action='store_true')
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    if world.size not in (2,3) and not (world.size == 4 and args.split):
        parser.error('Use 2/3 processes, or 4 processes with --split')
    color = world.rank//2 if args.split else 0
    comm = world.Split(color, world.rank) if args.split else world
    try:
        run(comm, args.directory.resolve()/f'group-{color}', args.model, args.open_ended)
    except BaseException:
        import traceback
        traceback.print_exc()
        world.Abort(1)
    finally:
        if args.split:
            comm.Free()
