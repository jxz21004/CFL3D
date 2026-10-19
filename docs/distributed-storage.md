# Distributed storage migration

Native API 12 completes persistent coordinate and detailed wall distribution,
following the API 11 mean-flow/restart migration. MPI checkpoint I/O now also uses
bounded streaming, without changing native API 12 or the NPZ checkpoint format.
File-based MPI input now reads coordinates on their initial owners. Volume output
is written one zone at a time, and wall ASCII output is streamed from its owners.
Native grid setup and surface analysis still need further work on peak host memory.

## Persistent data ownership

- Coordinates, returned mean flow, saved geometry, detailed wall fields/forces,
  and authoritative restart records reside only on their zone owner. The native
  host has empty arrays for these data between calls.
- Global zone order/dimensions, connectivity, BC/group metadata, per-boundary force
  integrals, and per-boundary wetted areas remain available on all ranks.
- `result.zone_owners` gives the actual native load-balancer map in `solver.comm`.
  File inputs load metadata on root and coordinates only on their initial workers;
  an explicitly supplied complete `CGNSMesh` retains the compatibility scatter.
  Coordinates then move if the native owner differs. Non-owned `zone.coordinates`
  is None; `zone.dimensions` remains valid.
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
returns them on the selected root. All writers are collective in MPI. CGNS,
PLOT3D and automatic solution NPZ output receive and release one zone at a time;
they do not call `gatherFields()` or collect a global coordinate dictionary. Root
still needs the largest zone's arrays plus format/conversion buffers. This is a
zone-sized bound, not a 1 MiB bound on total writer memory.

The wall Tecplot writer serializes each patch on its owner and transfers ASCII
chunks of at most 1 MiB. Each patch spool spills to a temporary file above 1 MiB.
The writer does not assemble complete wall geometry or fields on root. Registered
slices/lift distributions still invoke the surface analysis collection described
below before the wall file is written.

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

## Streaming checkpoint I/O

Normal MPI continuation, native result redistribution, `restartFile` loading, and
`writeRestartFile()` no longer create a whole-case native/field payload on root:

- Native files are indexed through record markers and small headers. Each owner
  receives its zone records in chunks of at most 1 MiB. The raw file's checksum is
  computed during transmission, without a separate full-file checksum read.
- Compressed NPZ input is decompressed incrementally to an owned temporary file,
  with its checksum verified before distribution. Redundant `field_*` readouts
  are skipped while loading a restart; authoritative native records restore the
  solver and fresh mean-flow readouts are produced after calculation.
- The NPZ writer streams `native.npy` and each `field_*.npy` into ZIP members.
  It retains the existing metadata/checksum/readout layout and supports ZIP64.
  Older complete/serial readers can still consume these files.
- Transfers use a duplicated communicator and per-chunk acknowledgements. Source
  read failures and destination write failures are agreed collectively. Checkpoint
  output uses a temporary file and replacement; a failed write preserves the old
  destination. This does not make an entire solve/output batch transactional.
- Only metadata fingerprints, zone lengths, and array descriptors are exchanged
  globally for checkpoint writing. Neither full `FlowState` objects nor replicated
  copies of the entire metadata dictionary are gathered from every rank.

The 1 MiB limit applies to individual data chunks, not total process RSS. ZIP
buffers and global topology metadata add storage, and each worker still needs its
owned state. Compressed input needs temporary disk space approximately equal to
its uncompressed native payload. Python's temporary directory selection respects
`TMPDIR`; choose a disk-backed directory with adequate capacity for large cases.
Temporary files are closed on both successful reads and handled errors.

The low-level serial `read_restart()` still returns a complete snapshot. Likewise,
installing a complete serial `FlowState` on every MPI rank explicitly supplies
full copies and retains its compatibility conversion path. For distributed usage,
load through `options['restartFile']` or install each rank's own cached shard.

## Remaining work

1. Bound temporary host storage during native grid setup. Initial file loading no
   longer assembles a complete mesh, but native array preparation/validation still
   gathers whole-case geometry, and the native host retains its coordinate input
   buffer during computation. The host's peak memory still depends on global size.
2. Distribute surface analysis itself. Slice registration/evaluation and lift
   distribution computation still temporarily collect complete wall data on root.
   Volume writers are bounded by the largest zone; a single large zone can still
   require substantial conversion memory and cannot be split across workers.
3. Measure process RSS/peak memory on representative production meshes. Small
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

## Checkpoint streaming regression

65 serial/unit tests: 62 pass and 3 optional HDF5 tests skip in the current
environment. Native indexing and NPZ extraction cover all nine exposed model
names. Three-rank gamma-Re_theta continuation and two independent two-rank SST
communicators pass the checkpoint fault-injection regression. A three-dimensional
unequal-zone SST case with y-up axes also passes cached/file continuation and
legacy-reader compatibility through the streaming path.

The MPI regression forbids `Path.read_bytes()` for checkpoint files, whole-array
NPZ loads, and the former merge/scatter route while exercising normal checkpoint
operations. Every observed transfer is at most 1,048,576 bytes. Injected source
read and destination write failures exit collectively; the prior checkpoint stays
intact, temporary output is removed, and a subsequent write/read/solve succeeds.
This is direct path/buffer verification, not a production RSS measurement.

```bash
PYTHONPATH=tests/python python -m unittest test_restart_streaming -v
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_checkpoint_streaming_mpi.py --directory build/native-output-streaming-errors
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_checkpoint_streaming_mpi.py --split --model sst --directory build/native-output-streaming-final-split
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_runtime_mpi.py --exclude --model gamma-theta --directory build/native-output-streaming-runtime
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_distributed_mpi.py --unequal --3d --y-up --model sst --directory build/native-output-streaming-3d
```

## Selective input and output regression

66 serial/unit tests: 63 pass and 3 optional HDF5 tests skip. The selective ADF
reader test intercepts `cg_coord_read` to verify that metadata-only loading does
not read coordinates and selected loading reads only the requested zone. Both
ADF and optional HDF5 readers accept `coordinate_zones`; unselected zones retain
their dimensions, connectivity, BCs and families with coordinates=None.

The output MPI regression rejects `gatherFields()`, `_gather_mesh()` and complete
surface collection in output paths. (Native grid preparation is explicitly
exempted.) Automatic solution NPZ, wall ASCII and PLOT3D bytes match serial output;
CGNS coordinates and fields match as arrays. It injects CGNS write, surface spool
and NPZ write failures, checks collective exit, temporary-file cleanup and retry,
and verifies continuation after output. Two independent communicators also pass.
Observed transfer packets are no larger than 1 MiB; this does not measure RSS.

Existing analysis regression confirms identical slice/lift data, numbered files
and appended histories in two subcommunicators. A three-dimensional unequal-zone
y-up regression covers volume output and memory/file continuation.

```bash
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_output_streaming_mpi.py --directory build/native-output-output-streaming-fixed
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_output_streaming_mpi.py --split --directory build/native-output-output-streaming-fault-split
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_solution_io_mpi.py --split --directory build/native-output-output-streaming-analysis-split
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_distributed_mpi.py --unequal --3d --y-up --model sst --directory build/native-output-output-streaming-3d
```

These changes keep native API 12; rebuilding the native extensions is unnecessary
when updating from an already-built API 12 checkout. Result NPZ member order is
not an interface guarantee; existing array names and values remain compatible.
