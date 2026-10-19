"""Reject whole-case checkpoint operations; test transfer bounds and I/O failures.

mpirun -np 3 python tests/python/validate_checkpoint_streaming_mpi.py --directory ...
Use -np 4 --split for two independent solver communicators.
"""
import argparse
import hashlib
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
from mpi4py import MPI
from cfl3d import CFL3D
import cfl3d._streaming as streaming
import cfl3d.restart as restart
from test_surface_forces import grouped_plate


def run(comm, root, model):
    options = dict(turbulence=model, mach=.2, cfl=1., iterations=3, multigrid_levels=2,
                   writevolumesolution=False, writeTecplotSurfaceSolution=False)
    solver = CFL3D(grouped_plate(), options, comm=comm)
    target, raw = root/'portable.npz', root/'native.bin'
    observed = []
    original_transfer, original_read, original_load = streaming.transfer_bytes, Path.read_bytes, np.load
    original_open, original_chunks = zipfile.ZipFile.open, streaming.file_chunks

    def bounded_transfer(c, source, destination, chunks, consume, length, checked):
        def monitor():
            for chunk in chunks:
                count = memoryview(chunk).nbytes
                assert 0 < count <= streaming.CHUNK_BYTES
                observed.append(count)
                yield chunk
        return original_transfer(c,source,destination,monitor() if c.rank == source else None,
                                 consume,length,checked)

    def guarded_read(path):
        if path.suffix in ('.bin','.npz'):
            raise AssertionError('Whole-case Path.read_bytes is forbidden for MPI checkpoints')
        return original_read(path)

    def guarded_load(source, *args, **kwargs):
        name = str(getattr(source,'name',source))
        if name.endswith('.npz') or name == 'native.npy' or name.startswith('field_'):
            raise AssertionError('Whole-case checkpoint array loading is forbidden')
        return original_load(source,*args,**kwargs)

    def no_assembly(*args, **kwargs):
        raise AssertionError('Whole-case merge/scatter path was used')

    class FailingMember:
        def __init__(self, member):
            self.member = member
        def write(self, value):
            if len(value) > 512:  # Let NPY header write succeed; fail in the data transfer.
                raise OSError('injected checkpoint write failure')
            return self.member.write(value)
        def close(self):
            self.member.close()

    def failing_open(archive, name, mode='r', *args, **kwargs):
        member = original_open(archive,name,mode,*args,**kwargs)
        return FailingMember(member) if mode == 'w' and name == 'native.npy' else member

    def failing_chunks(*args, **kwargs):
        iterator = iter(original_chunks(*args,**kwargs))
        first = next(iterator)
        yield first[:max(1,len(first)//2)]
        raise OSError('injected checkpoint read failure')

    def expect_failure(operation, message):
        try:
            operation()
        except RuntimeError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError('Expected collective I/O failure')
        comm.Barrier()  # Peers must have left their packet protocol too.

    with patch.object(streaming,'transfer_bytes',bounded_transfer), \
         patch.object(Path,'read_bytes',guarded_read), patch.object(np,'load',guarded_load), \
         patch.object(restart,'merge_states',no_assembly), patch.object(restart,'partition_state',no_assembly):
        first = solver.solve(output_directory=root/'first')
        state = solver.getFlowState()
        solver._checked(lambda: target.write_bytes(b'previous-checkpoint') if comm.rank == 0 else None)
        with patch.object(zipfile.ZipFile,'open',failing_open):
            expect_failure(lambda: solver.writeRestartFile(target),'injected checkpoint write failure')
        def preserved():
            if comm.rank == 0:
                with target.open('rb') as stream:
                    assert stream.read(100) == b'previous-checkpoint'
                assert not list(root.glob('.portable.npz-*.tmp'))
        solver._checked(preserved)
        solver.writeRestartFile(target)
        solver._stage_flow_state(state,raw)
        with patch.object(streaming,'file_chunks',failing_chunks):
            expect_failure(lambda: restart.read_restart_partitioned(raw,solver.mesh,solver.options,
                           solver._zone_owners,comm,solver._checked),'injected checkpoint read failure')
        loaded = restart.read_restart_partitioned(raw,solver.mesh,solver.options,
                                                 solver._zone_owners,comm,solver._checked)
        assert loaded.native == state.native
        solver.options.iterations = 2
        continued = solver.solve(output_directory=root/'continued')
        resumed = CFL3D(grouped_plate(),{**options,'iterations':2,'restartFile':target},comm=comm)
        actual = resumed.solve(output_directory=root/'from-file')
        for name in continued.fields:
            np.testing.assert_array_equal(actual.fields[name],continued.fields[name])
        np.testing.assert_array_equal(actual.history,continued.history)
        assert solver.getFlowState().native == resumed.getFlowState().native

    # Legacy reader compatibility is checked outside the bounded production path.
    hashes = {name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in first.fields.items()}
    hashes = {name:value for piece in comm.allgather(hashes) for name,value in piece.items()}
    def compatibility():
        if comm.rank == 0:
            saved = restart.read_restart(target,solver.mesh,solver.options)
            assert saved.digest == state.global_digest
            assert set(saved.fields) == set(hashes)
            for name, values in saved.fields.items():
                assert hashlib.sha256(values.tobytes()).hexdigest() == hashes[name]
    solver._checked(compatibility)
    packets = comm.allgather((max(observed,default=0),len(observed)))
    assert max(item[0] for item in packets) <= streaming.CHUNK_BYTES
    if comm.rank == 0:
        print('CHECKPOINT_STREAM_PASS', model, 'rank (max packet bytes, packets):', packets, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--split',action='store_true')
    parser.add_argument('--model',default='gamma-theta')
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    color = world.rank//2 if args.split else 0
    comm = world.Split(color,world.rank) if args.split else world
    try:
        run(comm,args.directory.resolve()/f'group-{color}',args.model)
    except BaseException:
        import traceback
        traceback.print_exc()
        world.Abort(1)
    finally:
        if args.split:
            comm.Free()
