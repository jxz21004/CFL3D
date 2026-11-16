"""Choose object transport once when installing mesh metadata."""
from math import prod


# Switch at the legacy signed 32-bit byte-count boundary (2 GiB).
_OBJECT_BYTE_BUDGET = 1 << 31
_BYTES_PER_POINT = 3 * 8  # Three double-precision coordinates, without overhead.


def mesh_communicator(comm, mesh):
    """Collectively select transport before coordinates are distributed.

    Count only global coordinate bytes, not fields or serialization overhead.
    This is not a guarantee for other payloads. No arrays are allocated.
    Borrow the communicator handle; never Dup/Free it here. Small meshes do not
    import pkl5. Call again only when a new mesh is installed, not per message.
    """
    if comm is None:
        return None
    estimated = sum(prod(int(n) for n in z.dimensions) for z in mesh.zones) * _BYTES_PER_POINT
    if not any(comm.allgather(estimated >= _OBJECT_BYTE_BUDGET)):
        return comm
    error = None
    wrapped = None
    try:
        from mpi4py.util import pkl5
        required = ('gather', 'scatter', 'alltoall', 'allgather', 'bcast')
        if not all(hasattr(pkl5.Intracomm, name) for name in required):
            raise ImportError('pkl5 collective methods are unavailable')
        wrapped = comm if isinstance(comm, pkl5.Intracomm) else pkl5.Intracomm(comm)
    except (ImportError, AttributeError) as exc:
        error = str(exc)
    # All ranks finish checking before any caller uses the selected protocol.
    errors = comm.allgather(error)
    if any(errors):
        raise RuntimeError('Large mesh object communication requires mpi4py >= 4.0 '
                           'with pkl5 collective support: ' + next(e for e in errors if e))
    return wrapped
