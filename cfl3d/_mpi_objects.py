"""Large Python payload transport; never replaces the public/native communicator."""


def large_objects(comm):
    """Borrow an intracommunicator using pkl5 out-of-band array transport.

    No Dup/Free and no rank-local size-dependent protocol switching. Small
    control messages and explicitly streamed buffers use the original comm.
    Import lazily so serial users do not require mpi4py.
    """
    from mpi4py.util import pkl5
    required = ('gather', 'scatter', 'alltoall', 'allgather', 'bcast')
    if not all(hasattr(pkl5.Intracomm, name) for name in required):
        raise RuntimeError('CFL3D large-object communication requires mpi4py with '
                           'pkl5 collective support; install mpi4py >= 4.0')
    return pkl5.Intracomm(comm)
