# Distributed storage migration

Native API 12 completes persistent coordinate and detailed wall distribution,
following the API 11 mean-flow/restart migration. Temporary host memory during
setup and file I/O still needs further work.

## Persistent data ownership

- Coordinates, returned mean flow, saved geometry, detailed wall fields/forces,
  and authoritative restart records reside only on their zone owner. The native
  host has empty arrays for these data between calls.
- Global zone order/dimensions, connectivity, BC/group metadata, per-boundary force
  integrals, and per-boundary wetted areas remain available on all ranks.
- `result.zone_owners` gives the actual native load-balancer map in `solver.comm`.
  Coordinates are initially scattered to workers, then moved if the native owner
  differs. Non-owned `zone.coordinates` is None; `zone.dimensions` remains valid.
- f2py mean-flow and wall buffers are allocated after the native partition is
  known. Neither detailed array is Allreduced. Only the host accepts global input
  coordinates through `rp3d`; workers obtain their grid through native MPI setup.
- Complete native per-zone restart records retain turbulence, viscosity, and
  ghost/wall state. Native coarse/work arrays are rebuilt for continuation.
  Successful extraction releases f2py data buffers.
- Historical snapshots remain separate local copies so editing the current local
  mesh cannot change outputs associated with an earlier solve.

MPI snapshots belong to their communicator partition. Install each rank's own
snapshot on that rank. To change the worker count, write a portable checkpoint and
load it through `restartFile`.

## Explicit collection and API changes

`gatherFields()` collects whole-case mean-flow arrays only when requested and
returns them on the selected root. All writers are collective in MPI. Volume
writers temporarily gather coordinates and fields; surface writers gather only
wall vertices and detailed wall data.

Surface registration and detailed queries are now collective too:
`addLiftDistribution`, `addSlices`, `getLiftDistribution`, and `getSlices` must be
called by every solver rank. Geometry is gathered temporarily for registration or
post-processing. Only compact requests/cut references and computed section/strip
results are broadcast. The root's temporary full wall data are released after the
operation; they are never installed in the persistent wall caches.

`clearSurfaceAnalysis()` must update every rank; inconsistent registrations are
rejected before the next solve/output. `getSolution()` remains a local operation:
its force integrals and wetted area use small cached boundary summaries.

This replaces API 11's root-only surface calls. Place collective calls outside
`if rank == 0` blocks. See [python-wrapper.md](python-wrapper.md#mpi).

## Remaining work

1. Bound temporary host storage during grid setup. Input loading/validation gathers
   whole-case geometry on root, and the native host retains its input buffer during
   computation. Workers no longer have global coordinate copies, but the host's
   peak memory still depends on the global mesh.
2. Stream portable checkpoint loading/output and native output redistribution.
   Root currently reads a complete native output before scattering shards, and
   portable checkpoint output gathers state. The persistent host cache is empty.
3. Bound visualization-output memory or introduce parallel/streamed writers.
   Explicit gathering is correct but can still require global-size temporary
   arrays on the writer rank.
4. Measure process RSS/peak memory on representative production meshes. Small
   payload checks establish ownership, not an overall scaling/performance result.

The native host/worker solver and whole-zone partition strategy remain in use;
this migration does not add intra-zone decomposition or remove all legacy scratch
allocations. Global topology and small per-boundary summaries remain replicated.

## Evidence for the second stage

Two-zone SST fixture, three ranks, first solve (bytes):

| Rank | Cached native | Returned mean flow | Current coordinates | Wall fields + forces |
| --- | ---: | ---: | ---: | ---: |
| 0 (host) | 0 | 0 | 0 | 0 |
| 1 | 1,053,724 | 122,880 | 153,648 | 1,536 |
| 2 | 1,053,252 | 122,880 | 153,648 | 3,072 |

Unequal-zone gamma-Re_theta fixture, first solve:

| Rank | Cached native | Returned mean flow | Current coordinates | Wall fields + forces |
| --- | ---: | ---: | ---: | ---: |
| 0 (host) | 0 | 0 | 0 | 0 |
| 1 | 1,632,800 | 153,600 | 190,896 | 3,840 |
| 2 | 999,004 | 92,160 | 116,400 | 768 |

The unequal case deliberately causes native owners to differ from the initial
coordinate placement. Tests check that coordinates and saved geometry follow the
new owners, all native records survive continuation, and results match serial.
These numbers count the indicated data categories, not every local copy and not
process RSS.

Regression entry points, after rebuilding both native extensions:

```bash
python -m unittest discover -s tests/python -v
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_distributed_mpi.py --directory build/native-output-distributed2-mpi3
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_distributed_mpi.py --unequal --model gamma-theta --directory build/native-output-distributed2-unequal-fixed
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_solution_io_mpi.py --directory build/native-output-distributed2-io
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_runtime_mpi.py --exclude --model gamma-theta --directory build/native-output-distributed2-runtime
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_solution_io_mpi.py --split --directory build/native-output-distributed2-io-split
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_distributed_mpi.py --unequal --3d --y-up --model sst --directory build/native-output-distributed2-3d
```

Use fresh output directories when reproducing tests. Existing numbered snapshots
are deliberately preserved.
