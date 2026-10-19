# Distributed storage migration

Native API 13 adds an experimental native resident driver on top of API 12's
persistent coordinate/wall distribution and API 11's mean-flow/restart migration.
The public `CFL3D.solve()` path still uses restart-based continuation. MPI
checkpoint I/O uses bounded streaming and retains the existing NPZ format.
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

1. Retain the native solver workspace between calculation chunks. The current
   restart-based continuation rebuilds setup on every `solve()` and incurs
   internal checkpoint I/O. This takes priority over changing the grid I/O path;
   see the resident-runtime design below.
2. Bound temporary host storage during native grid setup. Initial file loading no
   longer assembles a complete mesh, but native array preparation/validation still
   gathers whole-case geometry, and the native host retains its coordinate input
   buffer during computation. The host's peak memory still depends on global size.
3. Distribute surface analysis itself. Slice registration/evaluation and lift
   distribution computation still temporarily collect complete wall data on root.
   Volume writers are bounded by the largest zone; a single large zone can still
   require substantial conversion memory and cannot be split across workers.
4. Measure process RSS/peak memory on representative production meshes. Small
   payload checks establish ownership, not an overall scaling/performance result.

The native host/worker solver and whole-zone partition strategy remain in use;
this migration does not add intra-zone decomposition or remove all legacy scratch
allocations. Global topology and small per-boundary summaries remain replicated.

## Resident runtime: design for the next stage

Status: **native prototype implemented; public integration pending**. The public
`CFL3D.solve()` still performs restart-based continuation.
An uncommitted experiment that staged coordinates to a temporary file has been
withdrawn from the active implementation. It removed the global coordinate copy
but did not avoid setup on every call and added coordinate disk I/O. Merely moving
that file to a RAM-backed directory would not solve the lifecycle problem.

The requirement is to retain the original CFL3D numerical methods, host/worker
partition and block distribution, while separating native initialization,
iteration advancement and final cleanup. Preserving the numerical algorithm
does not require preserving the standalone program's allocate/run/free lifetime.

The source audit identifies these boundaries:

- `dist/main.F` enters the native top-level solver for every call.
- `dist/cfl3d.F` owns the main work arrays and many topology/model arrays. It calls
  `mgbl`, exports Python readouts after that call, then deallocates the arrays.
- `dist/mgbl.F` initializes global state, determines the partition, calls `setup`,
  runs initial mesh sequencing and calls `mgblk`. Its residual/history work arrays
  are also deallocated before returning.
- `dist/mgblk.F` contains the multigrid cycle loop and end-of-chunk restart output.
  Keeping only the finest-grid mean flow cannot preserve this entire workspace.

### Intended behavior

1. The first `solve()` lazily initializes the native context, reads/distributes
   geometry using the original host/worker scheme, builds the grid hierarchy and
   advances the requested schedule. Initial coarse-to-fine sequencing runs once.
2. Further compatible calls update supported runtime controls at a collective
   iteration boundary and advance the requested number of additional cycles.
   They retain coordinates, partition, communication state, metric/wall-distance
   data, turbulence/ghost state, and all required multigrid levels and work arrays.
   They must not rerun setup, reload geometry or serialize a restart to continue.
3. Existing Python result queries use rank-local readouts and small global load
   summaries. Returning to Python must leave the authoritative native context
   alive. A context handle is distinct from an exported `FlowState` snapshot;
   creating a binary snapshot on every call must not become an implicit new cost.
4. `resetFlow()` or incompatible configuration changes invalidate the context
   and trigger a clean initialization when required. Output names/directories
   and the number of requested iterations do not invalidate the physical state.
   Grid hierarchy or allocation changes need an explicit policy; the current
   broad "numerical" option category is not sufficient to prove safe updates.
5. Explicit checkpoint loading/saving remains file I/O. Automatic snapshot,
   history and visualization policies must be separable from advancement; an
   internal control loop with output disabled must not write a restart, grid spool
   or unconditional solution NPZ merely because `solve()` returned.

This needs changes to allocation ownership and driver control flow, guarded for
the Python build. Residual evaluation, fluxes, turbulence equations, restriction,
prolongation and the original load balancer should be reused. Leaving arrays
allocated without retaining the associated scalar/COMMON state is insufficient.
Threads, background solver processes and implicit file-based context switching
are not part of this proposed design.

Communicators must remain valid for the lifetime of the native context, rather
than being duplicated/freed per calculation chunk. Multi-object semantics must
be resolved before enabling this path: the legacy COMMON state is process-global,
and initializing another object must not silently overwrite a resident context.
History capacity, convergence normalization, CFL-ramp counters, first-order
startup and lift-control counters must also survive calculation boundaries or be
updated according to a documented explicit option change.

### Acceptance checks

- Instrument initialization, partitioning and hierarchy construction. Repeated
  compatible advancement increments none of those counts. A requested reset or
  incompatible change causes exactly one new initialization.
- Compare one continuous run against many chunks with identical controls. Check
  residual history, force integrals, mean flow, transported turbulence variables
  and the preserved multigrid state. Cover V/W cycles, nonzero startup settings,
  SA/SST/gamma-Re_theta, unequal zones, 3D and both axis conventions.
- Change CFL and chunk length between calls and compare with the same prescribed
  control sequence applied to a continuous native run. Do not compare against a
  constant-CFL reference or silently restart CFL ramps at every chunk.
- Trace solver file access with automatic output disabled. After initialization,
  compatible advancement performs no grid/restart/snapshot file I/O. Native MPI
  runtime/library file access is separate from solver payload I/O.
- Confirm original block ownership and rank-local storage. Measure setup count,
  advancement time, payload bytes and RSS separately; do not infer performance
  from array sizes alone.
- Check queries between chunks, reset/cleanup, communicator splits/excluded ranks,
  exceptions and multi-object behavior. Verify that the non-Python standalone
  build and its numerical results remain unchanged.

The design above is not yet an available `SolverOptions` runtime option. Native
API 13 exposes the first prototype described below for explicit development use.

### First native prototype (API 13)

`py_start_resident(cycles)` opts into a bounded native session after ordinary
configuration. The configured single-stage iteration count reserves the total
schedule/history capacity. The first `py_solve()` returns after the requested
number of complete V/W cycles. `py_advance_resident(cycles, cfl)` advances additional
cycles in the same workspace; cfl=0 preserves the existing control, while a
positive value applies a new CFL. `py_close_resident()` releases an unfinished
context without advancing another cycle. Finishing the reserved schedule also
performs cleanup. These functions are low-level development interfaces, not a
supported replacement for `CFL3D.solve()` yet.

The guarded Python driver retains its original arrays and scalar state across
normal returns. It resumes after a completed cycle rather than re-entering
initialization, partitioning, `setup`, or grid hierarchy construction. MPI ranks
retain the original block map and must all advance/close together. Input `py_xyz`
is released immediately after original setup has distributed the blocks. No
coordinate spool or internal restart file supplies an advance operation.

Native restart output is disabled by default for the prototype. Existing native text logs
remain open and receive iteration diagnostics; this is **not** a claim of zero
filesystem activity. The resident driver does not create automatic Python result
NPZ/CGNS/Tecplot files, since the public writer pipeline is not attached to it.
Early close preserves the last published readout and does not reduce MPI force
rows a second time. Configuration/communicator replacement is rejected while a
context is active. Another ordinary solver may initialize after explicit close.
Native timing summaries span time spent back in Python while the context is
paused. Measure native advance calls separately for performance comparisons.

`py_resident_info()` returns enabled/paused flags, completed cycles, setup-call
count, reserved native history capacity, main real/integer workspace capacity in
bytes, and retained input-coordinate value count. Workspace bytes describe the
last initialized capacity, not all allocations or current RSS; their value remains
available after cleanup. Readout buffers are separate rank-local copies.

Current prototype boundaries:

- Steady cold starts with one mesh-sequence stage only; multilevel V/W cycling is
  supported. Coarse-to-fine initial sequencing and file-based initial restart are
  not yet supported by the resident start entry point.
- A finite schedule is reserved before starting. Indefinite advancement/history
  growth without rebuilding geometry is not implemented. Advances beyond the
  reserved budget are rejected before modifying the state.
- Only CFL/chunk-length changes are exposed. Fixed-control tests retain startup
  order and CFL-ramp state. Changed-CFL tests currently use disabled ramps; this
  does not establish arbitrary runtime option-update semantics.
- One active native context per process. It borrows a communicator that must stay
  alive until close. Full multi-object context switching, resident `FlowState`
  snapshots and public output numbering remain integration work.
- `CFL3D.solve()` and existing result/post-processing APIs retain their established
  behavior; users must not expect resident advancement merely by updating builds.

Both extensions must be rebuilt for API 13 and existing Python processes restarted:

```bash
python tools/build_python.py --jobs 8
python tools/build_python.py --mpi --jobs 8
```

Regression entry points:

```bash
PYTHONPATH=tests/python python -m unittest test_resident_native -v
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_resident_mpi.py --model sst --directory build/native-output-resident-final-mpi3
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_resident_mpi.py --split --model gamma-theta --directory build/native-output-resident-gamma-split
```

The first comparison covers continuous versus 2+3+1+3 cycles, all three tested
models (SA/SST/gamma-Re_theta), V/W cycling, startup order and CFL ramps. A separate
test applies the same changing-CFL sequence at different chunk boundaries. The
MPI fixture uses unequal 3D blocks, y-up axes and sideslip; it checks the original
block map, exact readouts/history, one setup call and zero retained input xyz.
A final, explicitly requested gamma-Re_theta restart export matches the continuous
run's complete native payload byte for byte, including transported turbulence,
viscosity and ghost/wall records; no such file is created between its chunks.
Both MPI layouts also check early close without an extra cycle or duplicated
surface reduction. The complete suite reports 70 tests: 67 pass and 3 optional
HDF5 tests skip. Existing MPI surface/analysis/output/continuation regression passes
in two independent two-rank communicators. The non-Python standalone build also
compiles, and the SA/SST NASA fixture comparison gives zero maximum field and
history differences between the ordinary Python path and standalone executable.
These small fixtures do not establish production memory scaling or speedup.

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

The checkpoint/output-streaming stage kept native API 12. The resident prototype
now requires rebuilding for API 13 as described above. Result NPZ member order is
not an interface guarantee; existing array names and values remain compatible.
