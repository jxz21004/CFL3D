# CFL3D / MELD / TACS static aeroelastic example

`run.py` runs cruise (2 degrees) and maneuver (5 degrees) sequentially with SA,
Mach 0.85, the supplied isotropic wingbox, MELD and resident CFL3D. It selects
either native CFL3D mesh deformation or IDWarp without changing the CFD,
structural or coupling parameters. Gradients are not implemented: use
`run_model()`, not optimization or derivative checks.

## Inputs and dependencies

Use an environment containing the serial and MPI CFL3D extensions,
OpenMDAO/MPhys, TACS, FUNtoFEM/MELD and mpi4py built for the same MPI runtime.
IDWarp is required only for `--warper idwarp`. Native FMEM defaults to
`native-gauss-seidel`; PETSc is not required here.

The default input paths are `case_test/wing_vol_L2.cgns` and
`case_test/wingbox.bdf` next to the CFL3D repository. These data are not bundled
with this example. Override them with `--grid` and `--struct-mesh`.
Run from this directory with CFL3D importable in your Python environment.
`--thickness` sets all shell design thicknesses in metres (default 0.002;
`--thickness 0.004` doubles the thickness). Changing thickness changes both
stiffness and mass, including the resulting gravity load; gravity acceleration
and maneuver load factor remain unchanged.

## Original L2 comparison

The original L2 has blocks with only three J nodes. Coarsening creates two
nodes, which native CFL3D rejects because its boundary treatment requires more
nodes. This example defaults to **one level** and checks the grid dimensions
before starting a native solve. The grid is never automatically refined.

```bash
mpirun -np 3 python run.py --levels 1 --warper cfl3d --method exponential --control-indices l2-tangential-controls.json --fixed-budget --output output-native
mpirun -np 3 python run.py --levels 1 --warper idwarp --fixed-budget --output output-idwarp
# Optional second native deformation method:
mpirun -np 3 python run.py --levels 1 --warper cfl3d --method fmem --fixed-budget --output output-fmem
python compare.py output-native output-idwarp
```

Each evaluation advances 100 finest-grid cycles, then exchanges loads and
displacements. `--fixed-budget` requests 80 evaluations (8000 fine cycles per
scenario), disabling early tolerance-based stopping. Completing this budget
does not certify CFD or coupling convergence. Without this flag the outer
solver uses the supplied case's strict 1e-14 tolerances and raises on failure
to converge. `--smoke` uses two exchanges of three fine cycles for an interface
check only. Native control spacing is configurable with `--control-stride`
(default 4). Exponential decay also exposes `--beta2` (default 5),
`--alpha2` (default 0.1), and `--smoothing-iterations` (default 0).
Smaller beta2 and larger alpha2 spread displacement farther; smoothing changes
interior control displacements but preserves prescribed boundary values. These
options do not guarantee positive cells. See the [adapter reference](../../../docs/python-wrapper.md).

Geometry errors propagate immediately through OpenMDAO with
`reraise_child_analysiserror=True`. No automatic switch of warper, reduction
of applied displacement, or relaxation of native geometry checks is performed.
Always inspect the process exit status and log. A missing `summary.json`
means the run did not complete; its partial history is not a final solution.
Use a fresh output directory for each run.

## Explicit control indices for L2

`--control-indices l2-tangential-controls.json` overrides `--control-stride`.
The JSON maps each CGNS zone name to zero-based i/j/k index lists; this file is
specific to the supplied L2 topology. The adapter still propagates matching
indices across interfaces. The file retains the coarse normal spacing and
resolves wall tangential directions, increasing controls from 980 to 8505.

```bash
mpirun -np 3 python run.py --warper cfl3d --control-indices l2-tangential-controls.json --fixed-budget --output output-native-refined
```

On the saved first-failure surface target, this configuration removes all
orientation reversals in the independent volume diagnostic (minimum
volume/original-volume ratio 0.805). The subsequent three-rank coupled retest completed 80 exchanges of 100 fine
cycles in both scenarios, with all native geometry updates accepted and one
initialization per scenario. This certifies budget completion, not convergence.
Compared with IDWarp, cruise CL/CD are 0.1866468223/0.02411246224 versus
0.1864371994/0.02412479825; maneuver CL/CD are 0.3838158241/0.03815303265
versus 0.3831860550/0.03810800993.

## Multigrid on a compatible mesh

With a compatible CFD mesh for the same wing/units/boundaries, use:

```bash
mpirun -np 3 python run.py --grid /path/to/compatible.cgns --levels 3 --coarse-iterations 100 --warper cfl3d --fixed-budget --output output-native-mg3
mpirun -np 3 python run.py --grid /path/to/compatible.cgns --levels 3 --coarse-iterations 100 --warper idwarp --fixed-budget --output output-idwarp-mg3
```

Three levels require cell dimensions divisible by four and at least three
nodes along J/K on the coarsest level. Other native geometry/interface checks
still apply. The initial coarse-to-fine sequence runs once per scenario:
100 cycles on each of the two coarse startup grids, followed by 8000 fine-grid
cycles in chunks of 100. Total native cycles are 8200, not 8000. With two
levels the corresponding total is 8100. The first fine cycle initializes the
resident context; the other 99 belong to the first coupling evaluation.

Later chunks retain the configured multilevel W-cycle correction and native
flow. Updating coordinates refreshes geometry on all levels without repeating
coarse startup. The example verifies one native initialization and checks
`total_cycles == cfd_cycles + startup_cycles` at every successful exchange.
This multigrid example still needs validation with a compatible replacement
mesh; original L2 cannot establish it.

## Outputs and observed L2 behavior

`configuration.json` records settings. Each scenario writes
`coupling-history.jsonl`; completed runs write `summary.json` and final flow
and surface files. `compare.py` reports native minus IDWarp only when both
runs completed with matching physical inputs and budgets.

In the 2026-10-29 three-rank test, IDWarp completed 8000 fine cycles in both
scenarios. Native exponential and FMEM with default controls both failed the
native metric check on the second cruise evaluation (after 100 successful
cycles). Native logs reported grid-normal direction changes. Those default-stride configurations reproduce a deformation failure.
The explicit tangential controls described above subsequently completed the
full exponential run. The FMEM failure remains unresolved; these successful
results apply to exponential with the explicit L2 lattice. No deformation
kernel or geometry criterion was changed to obtain the successful run.
Detailed run evidence is recorded in `MPI_PROGRESS.md`.
