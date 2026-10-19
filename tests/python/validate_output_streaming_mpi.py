"""Check selective CGNS input and output without whole-case Python assembly.

Run with mpirun -np 3, or -np 4 --split for independent communicators.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from mpi4py import MPI
from cfl3d import CFL3D
from cfl3d._cgns_native import library
import cfl3d._streaming as streaming
import cfl3d.solution_io as output
from cgns_fixtures import write_adf
from test_solution_io import SolutionIOTests
from test_surface_forces import grouped_plate


def run(comm, root):
    options = dict(input_grid=root/'input.cgns', turbulence='sst', mach=.2, alpha=2.,
                   iterations=3, cfl=1., multigrid_levels=2, freestream_pressure=101325.,
                   writevolumesolution=True, writeTecplotSurfaceSolution=True, writeflowsolution=True)
    serial = None
    def reference():
        nonlocal serial
        if comm.rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            write_adf(grouped_plate(split=25), options['input_grid'])
            serial = CFL3D(options={**options, 'output_directory': root/'serial'}, comm=None)
            serial.solve()
    # No solver object is available for collective exception checking yet.
    error = None
    try:
        reference()
    except Exception as exc:
        error = repr(exc)
    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)

    api = library()
    original_call = api.call
    reads = []
    def selective_read(name, *args):
        if name == 'cg_coord_read':
            zone = args[2]-1
            assert comm.rank == 1+zone % (comm.size-1), (comm.rank, zone)
            reads.append(zone)
        return original_call(name, *args)
    with patch.object(api, 'call', selective_read):
        solver = CFL3D(options={**options, 'output_directory': root/'mpi'}, comm=comm)
    assert len(reads) == 3*sum(owner == comm.rank for owner in solver._mesh_owners)

    original_setup, original_gather = solver._native_arrays, solver._gather_mesh
    original_transfer = streaming.transfer_bytes
    packets = []
    def no_assembly(*args, **kwargs):
        raise AssertionError('Whole-case Python assembly was used by an output writer')
    def setup():
        # Native grid setup is a separate remaining migration item.
        with patch.object(solver, '_gather_mesh', original_gather):
            return original_setup()
    def transfer(c, source, destination, chunks, consume, length, checked):
        def bounded():
            for chunk in chunks:
                count = memoryview(chunk).nbytes
                assert 0 < count <= streaming.CHUNK_BYTES
                packets.append(count)
                yield chunk
        return original_transfer(c, source, destination, bounded() if c.rank == source else None,
                                 consume, length, checked)
    def expect_failure(operation, message):
        try:
            operation()
        except RuntimeError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError('Expected collective output failure')
        comm.Barrier()
    with patch.object(solver, 'gatherFields', no_assembly), \
         patch.object(solver, '_gather_mesh', no_assembly), \
         patch.object(solver, '_native_arrays', setup), \
         patch.object(solver, '_surface_collective', no_assembly), \
         patch.object(streaming, 'transfer_bytes', transfer):
        first = solver.solve()
        def compare():
            if comm.rank != 0:
                return
            for suffix in ('solution.npz', 'surf.plt', 'flow.x', 'flow.q'):
                name = 'out_000_'+suffix
                assert (root/'mpi'/name).read_bytes() == (root/'serial'/name).read_bytes(), name
            SolutionIOTests().check_volume(root/'mpi'/'out_000_volume.cgns', serial.mesh,
                                          serial._last_fields, serial.options)
        solver._checked(compare)
        first.zone_owners.clear()  # Public result edits must not change deferred output routing.
        volume = root/'mpi'/'out_000_volume.cgns'
        old = volume.read_bytes() if comm.rank == 0 else None
        def fail_volume(name, *args):
            if name == 'cg_field_write':
                raise OSError('injected volume write failure')
            return original_call(name, *args)
        with patch.object(api, 'call', fail_volume):
            expect_failure(solver.writeVolumeSolution, 'injected volume write failure')
        def preserved():
            if comm.rank == 0:
                assert volume.read_bytes() == old
                assert not list(volume.parent.glob('.*.tmp'))
        solver._checked(preserved)
        solver.writeVolumeSolution()

        def fail_patch(*args, **kwargs):
            raise OSError('injected surface spool failure')
        with patch.object(output, 'write_surface_patch', fail_patch):
            expect_failure(solver.writeTecplotSurfaceSolution, 'injected surface spool failure')
        solver.writeTecplotSurfaceSolution()

        original_result = output.result_output
        @contextmanager
        def fail_result(*args):
            with original_result(*args) as writer:
                def fail(zone, q):
                    raise OSError('injected solution NPZ write failure')
                yield fail
        path = root/'mpi'/'retry.npz'
        def write_result():
            solver._write_zones(lambda: output.result_output(path, first._saved_values()), coordinates=False)
        with patch.object(output, 'result_output', fail_result):
            expect_failure(write_result, 'injected solution NPZ write failure')
        def no_leaked_output():
            if comm.rank == 0:
                assert not path.exists()
                assert not list(path.parent.glob('.*.tmp'))
        solver._checked(no_leaked_output)
        write_result()
        solver.setOption('iterations', 1)
        continued = solver.solve()
        assert continued.start_mode == 'restart'
        assert bool(solver._last_fields) == (comm.rank != 0)
    sizes = comm.allgather(max(packets, default=0))
    if comm.rank == 0:
        print('OUTPUT_STREAM_PASS', root, 'max packet bytes by rank:', sizes, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    if world.size not in (2,3) and not (world.size == 4 and args.split):
        parser.error('Use 2/3 processes, or 4 processes with --split')
    color = world.rank//2 if args.split else 0
    comm = world.Split(color, world.rank) if args.split else world
    try:
        run(comm, args.directory.resolve()/f'group-{color}')
    except BaseException:
        import traceback
        traceback.print_exc()
        world.Abort(1)
    finally:
        if args.split:
            comm.Free()
