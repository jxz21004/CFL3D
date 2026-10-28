"""Owned control-system operations; never collect a whole mesh on one rank.

Only scalar reductions are global. Interface sums use canonical-node owners;
nearest-boundary queries consume bounded source batches without retaining them.
"""

import numpy as np


def exchange(comm, values):
    return values if comm is None else comm.alltoall(values)


def scalar(comm, value, operation="sum"):
    if comm is None:
        return value
    from mpi4py import MPI

    return comm.allreduce(value, op={"sum": MPI.SUM, "min": MPI.MIN, "max": MPI.MAX}[operation])


class SharedNodes:
    """Sum local element contributions at shared nodes, then return to sharers."""

    def __init__(self, adapter, ids):
        self.comm = adapter.comm
        self.rank = adapter.rank
        self.ids = np.asarray(ids, dtype=np.int64)
        owners = np.array([adapter.owners[adapter.topology.decode(int(i))[0]] for i in ids])
        self.owned = owners == self.rank
        self.out = [
            np.flatnonzero(owners == r) if r != self.rank else np.empty(0, dtype=int) for r in range(adapter.size)
        ]
        received = exchange(self.comm, [self.ids[i] for i in self.out])
        lookup = {int(node): i for i, node in enumerate(ids)}
        self.inc = adapter._checked(lambda: [np.array([lookup[int(i)] for i in part], dtype=int) for part in received])

    def sum(self, values):
        result = np.array(values, copy=True)
        received = exchange(self.comm, [values[i] for i in self.out])
        for indices, part in zip(self.inc, received):
            np.add.at(result, indices, part)
        returned = exchange(self.comm, [result[i] for i in self.inc])
        for indices, part in zip(self.out, returned):
            result[indices] = part
        return result


def nearest(adapter, queries, sources, source_values=None, source_keys=None):
    """Exact distributed nearest search in bounded batches (ties use global keys).

    Every rank evaluates only its own queries. A source batch is discarded before
    the next arrives. No rank builds a global boundary coordinate array/tree.
    """
    queries = np.asarray(queries).reshape(-1, 3)
    sources = np.asarray(sources).reshape(-1, 3)
    best = np.full(len(queries), np.inf)
    keys = np.full(len(queries), np.iinfo(np.int64).max, dtype=np.int64)
    values = np.zeros((len(queries), 3))
    if source_values is None:
        source_values = np.zeros_like(sources)
    if source_keys is None:
        source_keys = np.arange(len(sources), dtype=np.int64)
    batch = 256
    for rank in range(adapter.size):
        count = (
            len(sources)
            if adapter.comm is None
            else adapter.comm.bcast(len(sources) if adapter.rank == rank else None, root=rank)
        )
        for start in range(0, count, batch):
            payload = (
                (
                    sources[start : start + batch],
                    source_values[start : start + batch],
                    source_keys[start : start + batch],
                )
                if adapter.rank == rank
                else None
            )
            xyz, u, ids = payload if adapter.comm is None else adapter.comm.bcast(payload, root=rank)
            order = np.argsort(ids, kind="stable")
            xyz, u, ids = xyz[order], u[order], ids[order]
            for q in range(0, len(queries), batch):
                d = np.sum((queries[q : q + batch, None, :] - xyz[None, :, :]) ** 2, axis=2)
                col = np.argmin(d, axis=1)
                dist, key = d[np.arange(len(col)), col], ids[col]
                old = best[q : q + len(col)]
                replace = (dist < old) | ((dist == old) & (key < keys[q : q + len(col)]))
                old[replace] = dist[replace]
                keys[q : q + len(col)][replace] = key[replace]
                values[q : q + len(col)][replace] = u[col[replace]]
    if scalar(adapter.comm, int(np.any(~np.isfinite(best))), "max"):
        raise ValueError("No finite prescribed control points for mesh deformation")
    return np.sqrt(best), values
