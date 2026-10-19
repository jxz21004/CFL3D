# Known limitations and validation gaps

This document describes the Python interface at native API **15**. It records
confirmed constraints and unfinished work, rather than an exhaustive audit of the
legacy Fortran solver. For parameter definitions and usage, see the
[Python interface reference](python-wrapper.md). Reproducible test commands and
development evidence are recorded in [MPI_PROGRESS.md](../MPI_PROGRESS.md).

## Implementation constraints

| Area | Current behavior and practical consequence |
| --- | --- |
| Python flow arrays | `SolverResult.fields` and `FlowState.fields` are readouts. Editing them does not modify the authoritative native restart payload used by the next solve. A primitive-array-to-native-state injection API is not implemented. |
| Continuation overhead | Python retains the complete finest-grid restart state, including turbulence and ghost data. Default `resident=False` solves restore it through an internal temporary binary file and rebuild native work/coarse-grid arrays. Opt-in `resident=True` retains native workspace and a duplicated communicator between calls, avoiding repeated setup and internal checkpoint I/O. This mode currently supports cold, single-stage steady initialization and no complete FlowState/checkpoint snapshots; see [the lifecycle design](distributed-storage.md#resident-runtime-design-for-the-next-stage). |
| MPI peak memory (remaining work) | Persistent coordinates, mean-flow/wall arrays, and native restart records are partitioned by zone ownership. File inputs read coordinates on initial owners; checkpoint I/O streams chunks, with compressed input spooled to disk. Volume output receives one zone at a time, and wall ASCII output streams from owners. Native grid setup, explicit `gatherFields()`, and slice/lift analysis still collect global data temporarily on the host. Host peak memory is not yet bounded by local zone size; payload counts are not RSS measurements. See [distributed-storage.md](distributed-storage.md). |
| MPI work distribution | Rank 0 is the native host; other ranks compute whole existing zones. The interface does not split zones automatically. Worker count cannot exceed zone count. A one-rank communicator uses the serial path. |
| Process-global native state | With the default path multiple objects may solve sequentially. An active resident object must close collectively before another object can solve. Concurrent solves in one Python process are unsupported. Native execution also changes the process working directory temporarily, so other threads using relative paths can be affected. |
| Native failure recovery | Intercepted serial termination paths raise Python exceptions, but a failed native call can leave allocations/state behind. Use a new process after such a failure. Memory faults are not generally converted to Python exceptions. Native MPI fatal errors call `MPI_Abort` and can terminate the job. |
| Restart file refresh | A configured restart file is loaded when its resolved path changes. Overwriting the same path after loading does not trigger an automatic reload. To force a reload, call `resetFlow()` and then assign `restartFile` again, or use a new path. |
| Restart portability and identity | NPZ checkpoints wrap native binary state; they are not a compiler/platform-independent restart format. NPZ metadata distinguishes full model names. Raw native headers contain only IVISC and cannot distinguish SST/BSL/SST-2003 or the Wilcox variants that share an IVISC number. |
| Output batch failures | Individual Python writers use temporary files, but a whole output batch is not a transaction. If a write fails after native computation succeeds, the flow has advanced and its output index has been consumed. Retry the relevant deferred writer; another `solve()` performs more iterations. |
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
| Within-call visualization | CGNS, Tecplot, and PLOT3D snapshots are written after a completed `solve()`. There is no visualization interval within one native call. `restart_interval` controls native checkpoint writes, not these snapshots. |
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

The gamma-Re_theta integration exposed a native RANS path that could access the
DES-only `fdsav` array through an `.and.` expression. Fortran does not guarantee
short-circuit evaluation. This was fixed by nested tests in
[foureqn.F](../source/cfl3d/libs/foureqn.F); the debug standalone comparison now
passes. It is recorded here as a resolved regression, not an outstanding defect.

Future findings should record the trigger, affected configuration, reproduction,
workaround, and verification status before being treated as confirmed defects.
