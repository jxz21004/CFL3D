# Release notes

## Unreleased — distributed mesh deformation (native API 22)

Control selection now exposes `minimal`, `auto_stride`, `stride` and `indices`
modes, with per-zone/per-direction options and `getControlIndices()` inspection.
The default remains stride 4, but the implicit FMEM five-index floor is removed:
only required topology constraints add indices. Very coarse FMEM selections may
need explicit refinement. See the [selection guide](python-wrapper.md#control-point-selection).

Replaces the root control solve with owned control graphs, native local FMEM
assembly and distributed GMRES. Prescribed surface targets are routed to their
owners and consumers; nearest-boundary searches use bounded streaming batches.
No rank retains a whole-case surface/control coordinate array or global matrix.
TFI remains zone-local. FMEM uses a different iteration order and tighter residual
tolerance than the original Gauss-Seidel solve. Legacy `alpha1` remains accepted
but does not control GMRES. Whole-zone partitioning and distance-search traffic
still limit scaling; performance on production grids is unmeasured. Rebuild both
extensions and restart Python. See the [current adapter reference](python-wrapper.md#native-cfl3d-deformation-adapter-api-22-experimental).

## 0.3.2 — native mesh deformation adapters

Release date: 2026-10-28. Native interface version: **21**.

- Added experimental `CFL3DMeshAdapter` with both exponential decay and finite
  macro-element (FMEM) control solvers, followed by original arc-length TFI.
  IDWarp is not required for these methods.
- `warp(surface)` returns candidate owned volume coordinates without changing
  the accepted mesh or resident flow. Apply them with `setVolumeCoordinates()`
  and `solve()` to run the existing geometry refresh and validity checks.
  No reference volume grid is retained automatically.
- Native adapter calls restore shared control settings. Handled FMEM
  nonconvergence returns an adapter error collectively instead of invoking the
  legacy STOP. The ordinary CFD iteration path is unchanged.
- Mesh modules now use `volume_layout.py`, `adapter_idwarp.py` and
  `adapter_cfl3d.py`. **Import change:** use `VolumeMeshLayout` instead of
  `VolumeLayout`; IDWarp's class remains `IDWarpAdapter`.

This release supports steady fixed-topology, point-matched deformation. It does
not expose unsteady ALE, sliding symmetry, overset or periodic transforms.
Unspecified physical surface points remain fixed. Control solving occurs on
root and surface targets are replicated; TFI is zone-local. Production-scale
warping performance remains unmeasured. See the
[adapter contract](python-wrapper.md#native-cfl3d-deformation-adapter-api-22-experimental).

Upgrade: rebuild **both** extensions and restart Python/MPI processes. The serial
extension is required on every MPI rank for native deformation; API 20 binaries
are incompatible with this Python interface.

```bash
python tools/build_python.py --jobs 8
python tools/build_python.py --mpi --jobs 8
```

Validation: 124 tests, 121 passing and 3 optional HDF5 skips. Both methods pass
three-rank and split four-rank MPI checks, repeated deformation and resident
continuation. Generating and discarding a candidate preserves subsequent flow
and histories bit for bit. See [MPI_PROGRESS.md](../MPI_PROGRESS.md) for numerical
tolerances, coverage and remaining limits.

## 0.3.1 — bounded histories and geometry retry

Release date: 2026-10-28. Native interface version: **20**.

- Public resident histories reuse storage across calls, retaining the previous
  row needed by boundary conditions. Global cycle, CFL-ramp and trim counters
  remain unchanged. Capacity follows the largest chunk or startup allocation,
  rather than cumulative iterations.
- Python no longer accumulates output-session history in either solve mode.
  **Compatibility change:** `cumulative_history`, NPZ `history`, and
  `chunk_history` now contain only the current call. Full diagnostics remain in
  native text logs; `iteration_offset` preserves output-session numbering.
- Expected resident geometry failures retain flow and allow a new coordinate
  submission to retry. `SolveFailure` now reports `reset_required=False` and
  `geometry_update_required=True`. No iteration runs until a complete refresh
  passes checks. Resubmitting the last accepted coordinates is supported; there
  is no automatic geometry rollback. Other native faults remain outside this path.
- Result geometry metadata and `solutionIsCurrent` identify pending geometry
  without preventing access to a last successful result before native failure.
  A failed native solve does not expose old loads as a current successful result.

Rebuild both native extensions and restart Python/MPI processes. API 18/19
extensions are incompatible with this Python interface. Check downstream scripts
that relied on cumulative in-memory histories. Resident restart import/export,
complete coupling frameworks, and production-scale validation remain unfinished.

Validation: 121 unit tests, 118 passing and 3 optional HDF5 skips; three-rank and
split four-rank MPI geometry retry checks passed, including repeated failures and
exact continuation comparisons. History storage checks cover coarse startup,
V/W cycles, trim and CFL ramps. See [MPI_PROGRESS.md](../MPI_PROGRESS.md).


## 0.3.0 — coupling interfaces and resident geometry updates

Release date: 2026-10-27. Native interface version: **18**.
Git tag: `v0.3.0`.

This source release adds the mesh and force exchange foundations for steady
static aeroelastic workflows. It does not yet provide a complete MPhys or
GEMSEO integration, adjoint coupling derivatives, or unsteady ALE calculations.

### Changes since 0.2.0

- Distributed surface coordinates/connectivity and conservative nodal forces,
  including pressure and viscous contributions, for external load transfer.
- Optional IDWarp adapter and a solver-independent owned-volume coordinate
  layout. Reference coordinates (`x0`) are saved explicitly by the caller.
- Experimental fixed-topology resident coordinate updates. Native multigrid
  coordinates, metrics, volumes and wall distances refresh only after an explicit
  change, retaining flow as an initial guess without repeating full setup.
- Expected native geometry-update errors return `SolveFailure` collectively.
  No extra iteration or result export follows failure. Close/reset releases the
  invalid context; Python keeps the last accepted grid. There is no native rollback.
- Resident coarse-to-fine startup, followed by continued V/W cycling. Calculation
  chunks can use `solve(final=False)` and finish with `solve()` or
  `finalizeSolution()`, with independently numbered optional intermediate output.
- Additional input, output, documentation and build refinements, including
  `--extra-fflags` for controlled compiler comparisons.

### Upgrade

Reinstall the package as needed, rebuild both native extensions for API 18 using
the [installation guide](../README.md), and restart Python/MPI processes. An old
extension is rejected by the Python interface. IDWarp remains optional; importing
CFL3D alone does not require it. `resident` remains opt-in.

### Validation and limits

- The 119-test suite completed with 116 passing and 3 optional HDF5 skips before
  the final MPI tag/flag-reset fixes. The final focused geometry/output suite
  passed all 8 tests after those fixes.
- Actual IDWarp serial and three-rank tests verified zero-motion mapping and
  nonzero deformation staging. Native zero-motion geometry refresh matched
  ordinary continuation within 1e-13 on the tested fixture.
- Three-rank geometry-failure tests and four ranks split into two communicators
  verified consistent failure returns, unchanged cycle counts, cleanup and
  recovery. Normal gamma-theta/y-up/W-cycle geometry continuation also passed.
- Fixed-topology geometry updates remain experimental: production-scale mesh
  deformation and every model/boundary combination have not been validated.
  Expected metric/volume errors are handled only in the resident update path;
  startup failures, MPI faults and arbitrary native crashes may still abort.
- Resident full-state snapshots and file restart initialization remain unfinished.
  Nodal tractions and complete coupling-framework adapters are not implemented.
  No production speedup or peak-RSS scalability claim is made.

Detailed contracts and test records are in [the interface reference](python-wrapper.md),
[known limitations](limitations.md), and [MPI_PROGRESS.md](../MPI_PROGRESS.md).
The 0.2.0 notes below remain a historical description of that release.

## 0.2.0 — steady-flow computation baseline

Release date: 2026-10-20. Native interface version: **15**.
Git tag: `v0.2.0`.

This release freezes the current Python-driven CFL3D computation interface as a
baseline for subsequent coupling development. It adds no new numerical algorithms
and does not change the native API relative to the preceding development commit.
It is a source release: build the native extensions for the target Python,
compiler, MPI, and CGNS environment using the [installation guide](../README.md).
Installing the Python package alone does not compile those extensions.

### Included capabilities

- Implicit steady calculations on structured CGNS meshes with conforming
  one-to-one connectivity, using the original CFL3D whole-zone MPI load balancing.
- Python configuration with case-insensitive option names, validated dynamic
  updates, automatic serial/MPI selection, and explicit communicator support.
  Independent cases can run on disjoint communicators.
- Multigrid V/W cycles and initial coarse-to-fine sequencing in the default
  nonresident path; SA, k-omega, Wilcox 2006, SST, BSL, SST-2003, and gamma-Re_theta
  model options, plus Euler and laminar calculations.
- Rank-local coordinates, mean-flow/wall readouts, and complete cached native
  restart records in the default path; streamed checkpoint and volume output.
- Cached/file continuation and opt-in resident calculation chunks. Resident
  continuation retains native workspace and avoids repeated initialization for
  compatible updates of CFL, iteration count, and output settings.
- Grouped integrated loads, pressure/viscous drag, wetted area, Cp sections, lift
  distributions, CGNS volume output, ASCII Tecplot surfaces, and native PLOT3D
  output, with numbered or appended solve outputs.

### Validation evidence

The acceptance criterion for this baseline is numerical consistency with native
CFL3D under matched inputs, together with correct interface, continuation, and
MPI behavior. Convergence of a particular application mesh is a separate case
setup and analysis question.

- The existing full regression suite reports **82 tests: 79 passed and 3 skipped**.
  The skips concern optional HDF5 fixtures in the tested environment. Tests also
  exercise CGNS/ADF using the native CGNS library.
- An independent original-source comparison uses NASA commit
  `776d5169f34d26ae65701c87038e63b125b7d86a`, before introduction of the Python
  wrapper. Its unmodified solver source was independently compiled without
  `PYTHON_API`. On the NASA Flatplateyplus fixture, both SA and SST execute three
  initial grid-sequence stages of five iterations each. All final mean-flow values
  and all 15 rows of exposed residual/force/moment history have maximum absolute
  differences of **0** against the current Python path.
- Small serial/MPI regressions cover distributed state, repeated resident chunks,
  multigrid, output, and independent/excluded-rank communicators.
- The supplied nine-zone L2 mesh was checked for 1000 iterations with SA, CFL 2,
  limiter 4, and one grid level. Continuous serial, resident serial, and three-rank
  resident MPI histories and final mean-flow arrays agree exactly. This L2 check
  compares wrapper execution paths; an independent original-program L2 comparison
  was not performed and is not claimed.

These records establish consistency for the tested cases, not general
experimental accuracy or arbitrary production memory/performance scaling. The
tested toolchain uses Linux, Python 3.12.12, NumPy 2.3.5, GNU Fortran 13,
CGNS 4.5.0, Intel MPI 2021.13, and mpi4py 4.1.1. Other supported build combinations,
including Intel Fortran and Open MPI, require their own target-environment checks.

### Boundaries of this release

- Only implicit steady flow is exposed. Time-accurate, moving-grid/aeroelastic,
  adjoint, and MPhys interfaces are outside this baseline.
- Resident mode remains opt-in. It supports cold, single-stage initialization
  with multilevel V/W cycles, but not initial coarse-to-fine sequencing,
  complete FlowState/checkpoint operations, or file-based restart initialization.
  Other physical/mesh changes cold initialize a new context.
- Existing wall loads are cell-integrated readouts. A public conservative nodal
  surface-load interface and warm geometry updates for static aeroelastic coupling
  are future work. Editing returned flow arrays does not inject native state.
- Native setup and explicit surface analyses may temporarily gather global data
  on the host. Persistent partitioning does not establish bounded host peak RSS.
- A native MPI fatal error may abort the job. Checkpoints are native-state based
  and are not guaranteed portable across compiler/platform changes.
- Gamma-Re_theta remains subject to the upstream model's development/validation
  limits; interface parity is not experimental transition-model validation.

Consult [known limitations](limitations.md) and the
[Python interface reference](python-wrapper.md) for the complete contracts.
Subsequent surface-load, geometry-update, and MPhys work will proceed after this
tag so that the computation baseline remains independently identifiable.
