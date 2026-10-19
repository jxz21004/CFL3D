"""Real MPI storage, serial parity, portable and in-memory restart regression.

mpirun -np 3 python tests/python/validate_distributed_mpi.py --directory ...
Use -np 4 --split for independent two-rank solver communicators.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from mpi4py import MPI
from cfl3d import CFL3D
from test_surface_forces import grouped_plate


def hashes(fields):
    return {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in fields.items()}


def run(comm, directory, model, unequal=False, three_dimensional=False, y_up=False):
    def mesh():
        result = grouped_plate(split=25 if unequal else 33)
        for zone in result.zones:
            if three_dimensional:
                zone.coordinates = np.linspace(zone.coordinates[0], zone.coordinates[-1], 5)
                for bc in zone.boundaries:
                    bc.point_range[0][bc.point_range[0] == 2] = 5
                    if bc.point_range[0,0] == bc.point_range[0,1]:
                        bc.bc_type = 'BCSymmetryPlane'
                for cn in zone.connections:
                    for ranges in (cn.point_range, cn.donor_range):
                        ranges[0][ranges[0] == 2] = 5
            if y_up:
                xyz = zone.coordinates.copy()
                zone.coordinates[...,1] = xyz[...,2]
                zone.coordinates[...,2] = -xyz[...,1]
        return result
    options = dict(turbulence=model, mach=.2, alpha=2., beta=3. if three_dimensional else 0.,
                   y_up=y_up, cfl=1., iterations=3,
                   multigrid_levels=2, freestream_pressure=101325., writevolumesolution=False,
                   writeTecplotSurfaceSolution=False)
    serial = None
    if comm.rank == 0:
        serial = CFL3D(mesh(), options, comm=None)
        expected = serial.solve(output_directory=directory/'serial-first')
        reference = (hashes(expected.fields), expected.history,
                     serial.getSolution(), len(serial.getFlowState().native))
    else:
        reference = None
    reference = comm.bcast(reference, root=0)
    solver = CFL3D(mesh(), options, comm=comm)
    before_owners = solver._mesh_owners
    result = solver.solve(output_directory=directory/'mpi-first')
    if unequal and comm.size == 3:
        assert solver._mesh_owners != before_owners

    def check(result, reference):
        def local():
            return check_local(result, reference)
        native_bytes, field_bytes, coordinate_bytes, wall_bytes = solver._checked(local)
        sizes = comm.allgather((native_bytes, field_bytes, coordinate_bytes, wall_bytes))
        assert sum(size[0] for size in sizes) == reference[3]
        expected_field_bytes = sum(int(np.prod(np.asarray(z.dimensions)-1))*5*8 for z in solver.mesh.zones)
        assert sum(size[1] for size in sizes) == expected_field_bytes
        assert sum(size[2] for size in sizes) == sum(int(np.prod(z.dimensions))*3*8 for z in solver.mesh.zones)
        if comm.rank == 0:
            print(model, 'rank (restart, mean-flow, coordinates, wall fields+forces) bytes:', sizes, flush=True)

    def check_local(result, reference):
        expected_hashes, history, loads, global_native_bytes = reference
        owned = {name for name, owner in result.zone_owners.items() if owner == comm.rank}
        assert set(result.fields) == owned
        assert set(solver._last_fields) == owned
        assert {z.name for z in solver.mesh.zones if z.coordinates is not None} == owned
        assert {z.name for z in solver._last_mesh.zones if z.coordinates is not None} == owned
        assert all(key[0] in owned for key in result.wall_fields)
        assert all(key[0] in owned for key in solver._last_wall_forces)
        assert solver._surface_context is None
        state = solver.getFlowState()
        assert set(state.fields) == owned
        assert set(state.zone_indices) == {i for i, zone in enumerate(solver.mesh.zones)
                                          if zone.name in owned}
        assert hashes(result.fields) == {name: expected_hashes[name] for name in owned}
        np.testing.assert_array_equal(result.history, history)
        actual_loads = solver.getSolution()
        for name in loads:
            if name != 'units':
                np.testing.assert_allclose(actual_loads[name], loads[name], rtol=1e-13, atol=1e-14)
        if comm.rank == 0:
            assert result.fields == {} and state.native == b''
        return (len(state.native), sum(a.nbytes for a in result.fields.values()),
                sum(z.coordinates.nbytes for z in solver.mesh.zones if z.coordinates is not None),
                sum(a.nbytes for a in result.wall_fields.values())+sum(a.nbytes for a in result.wall_forces.values()))
    check(result, reference)
    portable = directory/'portable.npz'
    solver.writeRestartFile(portable)  # Collective: no complete checkpoint kept on host.
    restored = solver.getFlowState()
    solver.resetFlow().setFlowState(restored)
    solver.setOptions({'iterations': 2, 'cfl': 2.})
    if comm.rank == 0:
        serial.setOptions({'iterations': 2, 'cfl': 2.})
        expected = serial.solve(output_directory=directory/'serial-next')
        serial_file = CFL3D(mesh(), {**options, 'iterations': 2, 'cfl': 2.,
                            'restartFile': portable}, comm=None)
        from_mpi = serial_file.solve(output_directory=directory/'serial-from-mpi-file')
        assert hashes(from_mpi.fields) == hashes(expected.fields)
        np.testing.assert_array_equal(from_mpi.history, expected.history)
        reference = (hashes(expected.fields), expected.history,
                     serial.getSolution(), len(serial.getFlowState().native))
    reference = comm.bcast(reference if comm.rank == 0 else None, root=0)
    result = solver.solve(output_directory=directory/'mpi-next')
    check(result, reference)
    # Complete files are read once, scattered, then cached on actual native owners.
    other = CFL3D(mesh(), {**options, 'iterations': 2, 'cfl': 2.,
                                  'restartFile': portable}, comm=comm)
    actual = other.solve(output_directory=directory/'mpi-file')
    assert hashes(actual.fields) == hashes(result.fields)
    np.testing.assert_array_equal(actual.history, result.history)
    gathered = solver.gatherFields()
    if comm.rank == 0:
        assert hashes(gathered) == reference[0]
    else:
        assert gathered is None
    solver.writeVolumeSolution(directory/'deferred.cgns')
    solver.writeFlowSolution(directory/'deferred-flow')
    # A fresh two-rank object can consume a checkpoint from this communicator.
    # Also exercise automatic volume output with root's empty local fields.
    other.setOption('writevolumesolution', True)
    automatic = other.solve(output_directory=directory/'automatic')
    if comm.rank == 0:
        with np.load(automatic.output_paths['solution']) as saved:
            assert {name[5:] for name in saved.files if name.startswith('zone_')} == set(reference[0])
        assert automatic.output_paths['volume'].exists()
        assert automatic.output_paths['restart'].exists()
    if comm.rank == comm.size-1:
        other._flow_state = None
    try:
        other.solve(output_directory=directory/'mismatched')
    except RuntimeError as exc:
        assert 'states differ' in str(exc)
    else:
        raise AssertionError('Missing rank-local cache was not rejected')
    if comm.rank == 0:
        print('DISTRIBUTED_PASS', model, directory, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--model', default='sst')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--unequal', action='store_true')
    parser.add_argument('--3d', dest='three_dimensional', action='store_true')
    parser.add_argument('--y-up', action='store_true')
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    color = world.rank % 2 if args.split else 0
    comm = world.Split(color) if args.split else world
    try:
        run(comm, args.directory.resolve()/f'group-{color}', args.model, args.unequal, args.three_dimensional, args.y_up)
    except BaseException:
        import traceback
        traceback.print_exc()
        world.Abort(1)
    finally:
        if args.split:
            comm.Free()
