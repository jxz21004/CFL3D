"""Bounded byte transfers for collective file I/O on a borrowed communicator."""
import numpy as np

CHUNK_BYTES = 1024 * 1024


def byte_chunks(data):
    view = memoryview(data).cast('B')
    for offset in range(0, len(view), CHUNK_BYTES):
        yield view[offset:offset+CHUNK_BYTES]


def file_chunks(stream, start, length):
    stream.seek(start)
    remaining = length
    while remaining:
        data = stream.read(min(remaining, CHUNK_BYTES))
        if not data:
            raise ValueError('Truncated restart payload')
        remaining -= len(data)
        yield data


def array_chunks(array, *, fortran_order=False):
    """Serialize even strided arrays without creating a whole contiguous copy."""
    iterator = np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                         op_flags=['readonly'], order='F' if fortran_order else 'C',
                         buffersize=max(1, CHUNK_BYTES // array.dtype.itemsize))
    for part in iterator:
        # nditer may return a strided external loop. Bound its contiguous copy.
        yield from byte_chunks(np.ascontiguousarray(part))


def transfer_bytes(comm, source, destination, chunks, consume, length, checked):
    """Stream one item, agreeing on failures before starting the next item.

    Callers use a duplicated communicator to isolate packet tags. A receiver
    acknowledges each consumed chunk, so an I/O exception can stop its sender
    instead of stranding it in the next data send. Other ranks wait only at the
    per-item error agreement, not once for every chunk.
    """
    buffer = checked(lambda: np.empty(min(CHUNK_BYTES, length), dtype=np.uint8)
                     if comm.rank == destination and source != destination else None)

    def move():
        if comm.rank not in (source, destination):
            return
        position = 0
        if comm.rank == source:
            iterator = iter(chunks)
            while position < length:
                try:
                    part = memoryview(next(iterator)).cast('B')
                    count = len(part)
                    if not 0 < count <= min(CHUNK_BYTES, length-position):
                        raise ValueError('Invalid streamed chunk size')
                except Exception as exc:
                    if source != destination:
                        comm.send((0, f'{type(exc).__name__}: {exc}'), dest=destination, tag=0)
                    raise
                if source == destination:
                    consume(part)
                else:
                    comm.send((count, None), dest=destination, tag=0)
                    comm.Send(np.frombuffer(part, dtype=np.uint8), dest=destination, tag=1)
                    error = comm.recv(source=destination, tag=2)
                    if error:
                        raise RuntimeError(error)
                position += count
        else:
            while position < length:
                count, error = comm.recv(source=source, tag=0)
                if error:
                    raise RuntimeError(error)
                comm.Recv(buffer[:count], source=source, tag=1)
                error = None
                try:
                    consume(memoryview(buffer[:count]))
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                comm.send(error, dest=source, tag=2)
                if error:
                    raise RuntimeError(error)
                position += count
    checked(move)
