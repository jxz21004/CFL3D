# Release notes

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
