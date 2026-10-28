# Known limitations and validation gaps

This document describes the Python interface at native API **16**. It records
confirmed constraints and unfinished work, rather than an exhaustive audit of the
legacy Fortran solver. For parameter definitions and usage, see the
[Python interface reference](python-wrapper.md). Reproducible test commands and
development evidence are recorded in [MPI_PROGRESS.md](../MPI_PROGRESS.md).

## Implementation constraints

| Area | Current behavior and practical consequence |
| --- | --- |
| Coupling surface data (0.3.0) | `getSurfaceMesh()` returns uniquely owned nodes and local faces with global-ID connectivity, with stable node ownership across native zone repartition. `getSurfaceForces()` returns aligned nodal forces in N, including pressure/viscous selection. Both are collective in MPI. Experimental API 18 supports staged fixed-topology coordinates for paused resident solves, refreshing native multigrid geometry while retaining flow. Expected metric/volume update failures return `SolveFailure` collectively. API 20 retains flow and allows new coordinates to retry after a complete geometry refresh; there is no automatic geometry rollback and no advance without a new submission. Other native failures can still terminate the job. This is not an ALE interface. Nodal tractions and an MPhys Builder are not yet implemented. See the [mesh contract](python-wrapper.md#partitioned-surface-mesh-after-020) and [nodal force contract](python-wrapper.md#nodal-surface-forces-after-020). |
| Python flow arrays | `SolverResult.fields` and `FlowState.fields` are readouts. Editing them does not modify the authoritative native restart payload used by the next solve. A primitive-array-to-native-state injection API is not implemented. |
| Continuation overhead | Python retains the complete finest-grid restart state, including turbulence and ghost data. Default `resident=False` solves restore it through an internal temporary binary file and rebuild native work/coarse-grid arrays. Opt-in `resident=True` retains native workspace and a duplicated communicator between calls, avoiding repeated setup and internal checkpoint I/O. This mode supports cold steady initialization with automatic or explicit coarse-to-fine startup, but no complete FlowState/checkpoint snapshots; see [the lifecycle design](distributed-storage.md#resident-runtime-design-for-the-next-stage). |
| MPI peak memory (remaining work) | Persistent coordinates, mean-flow/wall arrays, and native restart records are partitioned by zone ownership. File inputs read coordinates on initial owners; checkpoint I/O streams chunks, with compressed input spooled to disk. Volume output receives one zone at a time, and wall ASCII output streams from owners. Native grid setup, explicit `gatherFields()`, and slice/lift analysis still collect global data temporarily on the host. Host peak memory is not yet bounded by local zone size; payload counts are not RSS measurements. See [distributed-storage.md](distributed-storage.md). |
| MPI work distribution | Rank 0 is the native host; other ranks compute whole existing zones. The interface does not split zones automatically. Worker count cannot exceed zone count. A one-rank communicator uses the serial path. |
| Parallel cases and rank ownership | Independent cases can solve simultaneously on disjoint MPI communicators, using separate output directories and case names. Each MPI rank (Python process) can hold one active resident case. Within that same process, close the resident owner before another object solves; every rank in its communicator must participate in `close()` or leave a shared `with` block. Deleting only the rank-0 Python object cannot release the other ranks' native workspaces. Concurrent native calls from threads are unsupported because COMMON state and the working directory are process-global. Default nonresident objects may solve sequentially in one process. See the [split-communicator example](python-wrapper.md#independent-cases-on-disjoint-communicators). |
| Native failure recovery | Intercepted serial termination paths raise Python exceptions, but a failed native call can leave allocations/state behind. Use a new process after such a failure. Memory faults are not generally converted to Python exceptions. Native MPI fatal errors call `MPI_Abort` and can terminate the job. |
| Restart file refresh | A configured restart file is loaded when its resolved path changes. Overwriting the same path after loading does not trigger an automatic reload. To force a reload, call `resetFlow()` and then assign `restartFile` again, or use a new path. |
| Restart portability and identity | NPZ checkpoints wrap native binary state; they are not a compiler/platform-independent restart format. NPZ metadata distinguishes full model names. Raw native headers contain only IVISC and cannot distinguish SST/BSL/SST-2003 or the Wilcox variants that share an IVISC number. |
| Output batch failures | Individual Python writers use temporary files, but a whole output batch is not a transaction. If a write fails after native computation succeeds, the flow has advanced and its output index has been consumed. A failed finalization remains pending: retry `finalizeSolution()` without advancing, or retry the relevant deferred writer. Another `solve()` performs more iterations. |
| Stopping criteria | `iterations` specifies a calculation chunk, not a residual/force convergence threshold. Native lift matching runs for the configured iterations and is not an outer convergence controller. |

The restart-file reload procedure is, for example:

```python
solver.resetFlow()
solver.setOption("restartFile", "updated-checkpoint.npz")
result = solver.solve()
```

`setFlowState()` installs a complete state returned by `getFlowState()`; it does
not rebuild native turbulence or ghost state from modified `fields` arrays.

## Capabilities not exposed by this interface

| Area | Current scope |
| --- | --- |
| Time-dependent flow | Only implicit steady calculations are exposed. `flow_mode="unsteady"` raises `NotImplementedError`. Physical time-step controls, time-level state management, and native moving-grid/aeroelastic workflows are not implemented. |
| Mesh/connectivity | Structured, conforming one-to-one multiblock meshes are supported. Overset/general/nonmatching or periodic connectivity and embedded-grid controls are not exposed. |
| CGNS representations | The readers require cell/physical dimension 3, including extruded 2D grids. Boundaries use vertex-based rectangular `PointRange`; `PointList` and coordinate rind layers are unsupported. |
| Boundary conditions | Only the documented CGNS-to-native mappings are automatic. Other supported native BCs require explicit `boundary_overrides`. Arbitrary CGNS `BCDataSet` arrays are not automatically translated into native boundary inputs. |
| Turbulence models | The exposed names are `sa`, `komega`, `wilcox2006`, `sst`, `bsl`, `sst2003`, and `gamma-theta`, plus `euler` and `laminar`. Other native models, including EASM/RSM, and further keyword-controlled corrections are not exposed. DES/LES would also require time-dependent interface work. |
| Turbulence visualization | Complete transported turbulence variables are retained in restart state, but standard Python volume output exposes mean-flow quantities. Separate k/omega/intermittency/Re_theta visualization fields and general model-residual arrays are not exported. |
| Within-call visualization | CGNS, Tecplot, and PLOT3D snapshots are written at calculation finalization (`solve(final=True)` by default, or `finalizeSolution()`), and optionally after nonfinal chunks with `writeIntermediateSolutions=True`. There is no visualization interval within one native call. `restart_interval` controls native checkpoint writes, not these snapshots. |
| Derivatives and coupling | The Python interface is real-valued. Complex-step/adjoint derivative workflows and OpenMDAO integration are not exposed. |

## Geometry and post-processing conventions

Mesh coordinates/reference lengths are assumed to be metres; no automatic unit
conversion is performed. Force integration uses the selected native boundary
contributions, without a symmetry multiplier. Dimensional loads need a supplied
free-stream pressure; otherwise `getSolution()` warns and returns coefficients
and wetted area only.

Wetted area sums two triangles per cached wall quad, including inviscid walls.
Cp sections use cached face-centered data and geometric interpolation/clipping;
they do not recover an exact continuous surface pressure field. Lift strips
allocate integrated native loads conservatively, but their coefficient density
is not a local-chord-normalized two-dimensional section lift coefficient.

Surface cuts can merge conforming block intersections. They do not construct an
overset zipper or repair nonmatching surfaces. A fully coplanar wall face does not
produce a section line. Relative cuts track attached mesh locations; arbitrary
topology changes can require clearing and registering the requests again.

The exported y+ is a face-cell estimate using the native traction and V/(2A)
normal-distance estimate. It differs in definition from legacy nodal y+
statistics. Surface `.plt` output is Tecplot ASCII text, not binary PLT data.

## Validation coverage

The current regression evidence includes small two-/three-dimensional cases,
multiblock grids, multigrid/mesh sequencing, model changes, cached/file restarts,
and serial/MPI comparisons. The exposed BSL, SST-2003, Wilcox 2006, and gamma-Re_theta
paths have also been compared against a debug standalone native solver, including
the transported model variables. See
[validate_transition_parity.py](../tests/python/validate_transition_parity.py).

These tests establish interface and numerical-execution consistency for the
tested cases. They do not establish experimental accuracy, long-run convergence,
or scalability for an arbitrary production case. In particular, this project has
not validated gamma-Re_theta transition locations against experiments. The
upstream implementation itself is documented as under development; see the
[CFL3D model notes](https://nasa.github.io/CFL3D/Cfl3dv6/cfl3dv6_new.html#transmod).

Model-validation records through API 10 use GNU Fortran and Intel MPI. Open MPI
and other compiler combinations need their own build/runtime regression before
being described as tested configurations. This does not imply that the standard
MPI source interface is Intel-specific.

Debug builds add symbols and bounds checks to the Python state interface, but
global bounds checking is not enabled across every legacy Fortran routine because
of its cross-dimensional storage conventions. Passing the present regression
suite is not evidence that every legacy memory-access path has been audited.

## Resolved defects relevant to this work

The resident-workspace changes incorrectly declared `mgblk` dummy arguments
`timesave` and `ireq_qb` as allocatable, although `mgbl` passes slices of its
real/integer work arrays through the legacy array-address interface. Intel
Fortran Classic 18.0.5 was reported to reject the `timesave` declaration. A newer
ifort can compile the files separately, but enabling interface checking exposes
the invalid actual/dummy association as well. Both arguments now use their
original explicit-shape declarations, without callee allocation/deallocation;
the caller already retains their storage across resident pauses. This fixes the
interface rather than suppressing COMMON alignment diagnostics. Rebuilding with
18.0.5 remains to be confirmed on the production system.

Local verification uses ifort 2021.13.2 and Intel MPI 2021.13: serial/MPI builds,
generated-interface checks, and 16 resident lifecycle tests pass. On the three-rank
SST fixture with three multigrid levels, 29 continuous MPI cycles and resident
chunks of 2+3+1+7+16 match bit for bit. Intel serial/MPI results have small numeric
differences (maximum 2.58e-15 in flow and 1.48e-14 in history on this fixture);
the existing regression script requiring bitwise serial/MPI equality does not
pass with this compiler combination's default `-O2` floating-point settings.
It has not been weakened for this check.

A subsequent per-cycle investigation finds the first flow difference at cycle 2:
one velocity value differs by one ULP (1.735e-18). Residual history first differs
at cycle 12, and CL differs by 1.471e-14 at cycle 29. A host plus one worker is
bitwise identical to serial; a host plus two workers exposes the difference.
Rebuilding both Intel extensions with `-fp-model precise` makes all 29 cycles'
flow, boundary coefficients and history bitwise identical, and the original
strict resident MPI validator passes. This attributes the observed discrepancy
to sensitivity to floating-point optimization in the partitioned execution;
the specific instruction/reassociation/alignment mechanism is not established.
This fixture does not guarantee reproducibility for other meshes or rank counts.
The normal build script's defaults have not been changed by this investigation.

The gamma-Re_theta integration exposed a native RANS path that could access the
DES-only `fdsav` array through an `.and.` expression. Fortran does not guarantee
short-circuit evaluation. This was fixed by nested tests in
[foureqn.F](../source/cfl3d/libs/foureqn.F); the debug standalone comparison now
passes. It is recorded here as a resolved regression, not an outstanding defect.

Future findings should record the trigger, affected configuration, reproduction,
workaround, and verification status before being treated as confirmed defects.

### Resident history storage (native API 19)

Public resident continuation retains the current chunk and one previous row,
with capacity bounded by the largest chunk or initial startup allocation. Total
iteration counters, CFL ramp and lift-control state are not reset. Python no
longer accumulates histories across calls in either solve mode;
`cumulative_history` is a compatibility alias for current-chunk history. Full
diagnostics append to native text files. A single very large call still reserves
its history in memory, and caller-retained results retain their own arrays.
Low-level `open_ended=1` remains the cumulative compatibility/test interface.
