# Python interface reference

This document describes Python package **0.3.3**, native interface version **22**.
Sections marked "after 0.2.0" are included in 0.3.0 but absent from the `v0.2.0`
tag. Experimental features retain their documented limitations.
For the project motivation, requirements, and build commands, see the
[README](../README.md). Validation records are maintained in
[MPI_PROGRESS.md](../MPI_PROGRESS.md).
Confirmed implementation constraints and remaining validation work are collected
in [Known limitations and validation gaps](limitations.md).

Python reads CGNS coordinates, boundaries, and connections and passes NumPy arrays
to the Fortran solver through f2py. No `.inp` deck or intermediate PLOT3D input is
required. The CGNS reader uses a separate ABI bridge rather than CFL3D's legacy
CGNS interface.

## Standard case

Save the following as `case.py` and run `python case.py`. Replace the mesh path,
physical state, reference quantities, and axis convention with your case values.
Coordinates and reference lengths are in metres. The example assumes a wing with
span along y and lift in the x-z plane.

```python
from cfl3d import CFL3D

options = {
    "input_grid": "mesh.cgns",
    "output_directory": "cruise-results",
    "name": "cruise",
    "flow_mode": "steady",              # Implicit steady solve (the default)
    "mach": 0.85,
    "alpha": 2.0,
    "beta": 0.0,
    "temperature": 223.25209265,         # Freestream static temperature, K
    "freestream_pressure": 26499.89786,  # Freestream static pressure, Pa
    "reynolds": 7.223294882e6,           # Reynolds number per metre
    "gamma": 1.4,
    "turbulence": "sa",
    "full_navier_stokes": True,
    "eddy_viscosity_limit": 1e5,
    "sref": 45.5,                       # Reference area, m^2
    "cref": 3.25,                       # Reference chord, m
    "bref": 1.0,                        # Roll/yaw reference length, m
    "moment_center": (0.0, 0.0, 0.0),
    "y_up": False,
    "cfl": 2.0,                         # Native DT=-2: local time stepping
    "iterations": 1000,                  # Finest-grid cycles per solve()
    "mesh_sequence": True,               # Cold start: coarse to fine
    "coarse_grid_iterations": 100,       # Cycles on each preliminary grid
    "coarse_grid_first_order_iterations": 10,  # Included in those 100 cycles
    "limiter": 4,
    "multigrid_levels": 3,               # Requires a mesh valid on all 3 levels
    "numberSolutions": True,
    "writevolumesolution": True,
    "writeTecplotSurfaceSolution": True,
    "writeflowsolution": False,
}

solver = CFL3D(options=options)
result = solver()                       # Equivalent to solver.solve()
rank = 0 if solver.comm is None else solver.comm.Get_rank()
if rank == 0:
    print(result.start_mode, result.reset_reasons)
    print(solver.getGroupNames())
    print(solver.getSolution())
    print(result.output_paths)
solver.close()  # Collective within solver.comm; release resident state/work files.
```

The same script can run under `mpirun -np 3 python case.py` if the grid contains at
least two zones. Automatic MPI selection and explicit communicators are described
below. A complete command-line template is available in
[standard_case.py](../examples/python/standard_case.py).

## Construction and calls

```python
solver = CFL3D(options=options)
# Equivalent dictionary entry point:
solver = CFL3D(options)

# Existing mesh / typed-settings entry point:
from cfl3d import CGNSMesh, SolverOptions
mesh = CGNSMesh.from_file("mesh.cgns")
config = SolverOptions(mach=0.85, iterations=1000)
solver = CFL3D(mesh, config, comm=None)
```

Supply the mesh through either the `mesh` argument or `options["input_grid"]` at
construction, not both. `mesh` may be a path or a `CGNSMesh` object. Additional
constructor arguments are `comm`, `groups`, and `boundary_overrides`; the `name`
keyword overrides the name in options.

Both entry points return a `SolverResult` and use the same configuration,
continuation, and output rules. These are alternative call forms; each invocation
executes another configured calculation chunk:

```python
result = solver.solve()
result = solver()
result = solver(output_directory="another-directory")
```

`output_directory` is keyword-only. A call-specific directory overrides the option
for that call without changing the stored option. Under MPI, construction and both
solve entry points are collective within the object's communicator.

## Options

Dictionary keys and runtime attribute access are case-insensitive:
`CFL`, `cfl`, and `CfL` select the same option. Underscores remain significant.
Unknown keys and duplicate dictionary keys differing only by case are rejected.
Boolean options require Python `True` or `False`, not strings or integers.

The constructor copies the supplied configuration. Editing the original dictionary
or `SolverOptions` afterward does not change an existing solver. Update
`solver.options` to change a running case:

```python
solver.options["CFL"] = 3.0
solver.options.iterations = 200
solver.setOptions({"iterations": 100, "first_order_iterations": 10})
solver.setOption("limiter", 2)
print(solver.getOption("cfl"))
```

`SolverOptions` is also mutable: `config.cfl = 3.0` and
`config.update({"iterations": 100})` are supported. Use a single update for related
parameters that must validate together. Failed updates leave the previous values
intact. `RuntimeOptions` stores live values in a dictionary; updates do not rebuild
a `SolverOptions` object. `snapshot()` makes an independent copy when needed.

### Physical model and freestream

| Option | Default | Meaning |
| --- | --- | --- |
| `flow_mode` | `"steady"` | Implicit steady solve; `"unsteady"` raises `NotImplementedError` |
| `mach` | `0.5` | Positive freestream Mach number |
| `reynolds` | `1e6` | Reynolds number per mesh length unit; per metre for metre-based grids |
| `temperature` | `288.15` | Positive freestream static temperature, K |
| `gamma` | `1.4` | Specific heat ratio, greater than 1 |
| `turbulence` | `"sa"` | `euler`, `laminar`, `sa`, `komega`, `wilcox2006`, `sst`, `bsl`, `sst2003`, or `gamma-theta` |
| `freestream_turbulence_intensity` | `None` | Positive free-stream turbulence intensity in percent; None retains native defaults |
| `freestream_eddy_viscosity` | `None` | Positive free-stream viscosity ratio mu_t/mu_inf; None retains native defaults |
| `keep_ambient_turbulence` | `False` | Prevent ambient turbulence decay using native KEEPAMBIENT |
| `full_navier_stokes` | `True` | Full N-S terms for viscous models; disabled internally for Euler |
| `y_up` | `False` | False: alpha in the x-z plane; True: native y-up coordinate convention |

Viscous models require a positive Reynolds number. The interface converts Reynolds
number to the native REUE value in millions and Kelvin to Rankine. If a Reynolds
number is specified using a reference chord, pass `Re_chord / cref`.

`flow_mode` makes the requested temporal mode explicit. Currently only `"steady"`
is implemented in the Python interface. Selecting `"unsteady"` raises
`NotImplementedError` during option validation, before calling Fortran; other
values raise `ValueError`. Rejected updates leave all existing options intact.

Keep `cfl` positive in Python. The steady solver passes `DT=-cfl` to Fortran for
local time stepping. Native `DT > 0` specifies a fixed time step, not a positive
CFL or an explicit integration scheme. Changing the sign of `cfl` does not enable
unsteady computation. Physical time steps, inner iterations, and the restart state
needed for time-accurate continuation are not yet exposed.

| Model | Native IVISC |
| --- | --- |
| `euler` | 0 |
| `laminar` | 1 |
| `sa` | 5 |
| `komega` | 6 |
| `wilcox2006` | 6 + I_WILCOX06=1 |
| `sst` | 7 |
| `bsl` | 7 + I_BSL=1 |
| `sst2003` | 7 + ISST2003=1 |
| `gamma-theta` | 40 |

For extruded 2D grids, viscosity is disabled in the two-plane i direction.

### Two-equation variants

Select each variant through `turbulence`, rather than combining model flags:

```python
solver.setOption("turbulence", "sst2003")  # Alternatives: "bsl", "wilcox2006"
result = solver.solve()
```

`sst` retains the original native SST defaults, including vorticity-based
production and the original eddy-viscosity limiter. `komega` retains the native
Wilcox 1988 defaults. Existing scripts selecting these names keep their model.

* `bsl` selects Menter's baseline model through I_BSL=1.
* `sst2003` selects the complete native SST-2003 switch, including strain-based
  production/eddy-viscosity limiting, the production limiter of 10 in both
  equations, and the revised constants and cross-diffusion floor. It does not
  simply change one model constant.
* `wilcox2006` selects native Wilcox 2006, retaining its default vortex-stretching
  correction (I_WILCOX06_CHIW=1).

Model names are case-insensitive. Switching variants starts a fresh calculation,
even when the variants share IVISC=6 or 7. Each configuration resets all variant
flags, so switching back cannot inherit a previous model's settings. Python NPZ
checkpoints record the full model name and reject a different variant. Raw native
restart headers contain only IVISC and cannot establish which variant produced
the file; use NPZ checkpoints when variant identity must be checked.

### Gamma-Re_theta transition model

`turbulence="gamma-theta"` selects the existing native Langtry-Menter four-equation
SST transition model, IVISC=40 (gamma-Re_theta v1.1). It transports omega, k,
intermittency, and the transition Reynolds-number variable. It is separate from
the undocumented native three-equation IVISC=30 model. CFL3D's own notes mark
this implementation as under development.

```python
solver = CFL3D(options={
    "input_grid": "mesh.cgns",
    "turbulence": "gamma-theta",
    "freestream_turbulence_intensity": 0.2,  # 0.2 percent, not 20 percent
    "freestream_eddy_viscosity": 1.0,       # mu_t / mu_inf
    "keep_ambient_turbulence": True,
    "iterations": 1000,
})
result = solver.solve()
```

The three ambient controls are supported for `komega`, `wilcox2006`, `sst`, `bsl`,
`sst2003`, and `gamma-theta`.
Their defaults preserve native initialization. Supply measured/case-specific
ambient conditions for transition predictions: the example values are not a
universal calibration. `gamma` remains the gas specific-heat ratio, not an input
for the intermittency field. The native model initializes free-stream intermittency
to 1 and computes the initial Re_theta variable from turbulence intensity.

Changes to the model or these ambient controls start a fresh flow by the normal
physics-change policy. Cached and file restarts preserve all four transported
variables and their ghost states. Standard Python volume output still exposes
the existing mean-flow variables; separate intermittency/Re_theta visualization
fields are not yet exported. Numerical execution and restart tests do not validate
transition onset against experimental data.

For the original implementation and its development status, see the
[CFL3D transition-model notes](https://nasa.github.io/CFL3D/Cfl3dv6/cfl3dv6_new.html#transmod).
Rebuild both native extensions after updating to interface version 12.

### Operating point and lift matching

| Option | Default | Meaning |
| --- | --- | --- |
| `alpha` | `0.0` | Angle of attack, degrees; initial angle for lift matching |
| `beta` | `0.0` | Sideslip angle, degrees, following the native CFL3D sign convention |
| `target_cl` | `None` | Enable native CL matching when a numerical target is supplied |
| `alpha_relaxation` | `1.0` | Positive RLXALPH feedback relaxation |
| `alpha_step_limit` | `0.2` | Positive DALIM angle-update limit, degrees |
| `alpha_update_interval` | `1` | ICYCUPDT: positive number of fine-grid cycles between updates |

```python
solver.setOptions({
    "target_cl": 0.5,
    "alpha": 0.0,
    "alpha_relaxation": 0.5,
    "alpha_step_limit": 0.1,
    "alpha_update_interval": 5,
})
result = solver()
print(result.coefficients["CL"], result.final_alpha)
```

Lift matching runs inside Fortran for the configured iteration count. It does not
introduce a Python convergence tolerance or early stopping condition. `99999` is
reserved as the native disabled-target sentinel; use `None` to disable matching.

The native controller updates alpha after force evaluation. Consequently,
`result.final_alpha` can include an update after the final reported force. The
numbered `.alpha` log records the update history. When alpha is unchanged between
calls, restart preserves the previous trim angle. An explicit alpha edit overrides
that saved angle while retaining the flow as an initial guess.

### Numerical controls

| Option | Default | Meaning |
| --- | --- | --- |
| `cfl` | `2.0` | Positive CFL; passed as native `DT=-cfl` for local time stepping |
| `iterations` | `100` | Positive finest-grid iteration count for each solve; coarse startup is additional and runs only on a cold start |
| `cfl_ramp_steps` | `0` | IFLAGTS; nonnegative, zero disables CFL ramping |
| `cfl_ramp_factor` | `1.0` | FMAX, at least 1; target CFL is `cfl * cfl_ramp_factor` |
| `first_order_iterations` | `0` | NITFO on the final/default stage |
| `coarse_grid_first_order_iterations` | `0` | NITFO on each preliminary automatic mesh-sequence stage; included in its `coarse_grid_iterations` budget |
| `restart_interval` | `99999` | Positive NWREST checkpoint interval, passed directly to Fortran; `None` is invalid |
| `limiter` | `4` | IFLIM: smooth limiter tuned to kappa=1/3 and invariant under block splitting; 0..4, scalar or `(i,j,k)` |
| `flux` | `1` | IFDS: 0 van Leer flux-vector splitting, 1 Roe flux-difference splitting; scalar or `(i,j,k)` |
| `inversion` | `1` | IDIAG: 0 block-tridiagonal, 1 scalar-tridiagonal; scalar or `(i,j,k)` |
| `kappa` | `1/3` | RKAP reconstruction parameter, scalar or `(i,j,k)` |
| `turbulence_subiterations` | `1` | Positive turbulence subiteration count |
| `turbulence_cfl` | `5.0` | Positive turbulence CFL control, applied to the seven native CFLTURB slots |
| `eddy_viscosity_limit` | `1e5` | Positive EDVISLIM bound |

Direction tuples refer to grid indices, not Cartesian axes, and apply to every
zone. Options dictionaries also accept three-element lists. First-order counts
must not exceed their corresponding stage counts. `restart_interval` must be a
positive 32-bit integer; complete state is cached at the end of every solve even
when visualization outputs are disabled.

`limiter=4` is the default for newly constructed solvers. It is the corrected
multi-block version of limiter 3, with a cutoff based on the total cell count.
Both are tuned to `kappa=1/3`; their limited reconstruction does not use an
arbitrary user-supplied kappa in the same way as the unlimited scheme. Set
`limiter=0` explicitly for unlimited reconstruction, including reproduction of
older runs that relied on the previous default of 0. Changing this default can
change convergence histories and computed loads. Explicit limiter values are
preserved.

`flux` selects the inviscid numerical flux used to form the residual. `inversion`
selects how the implicit update is solved in each grid direction: 0 uses coupled
5-by-5 block-tridiagonal systems, while 1 uses a diagonalized scalar-tridiagonal
approximation to reduce the cost per iteration. Both are implicit; these options
do not select steady versus unsteady flow. The default `flux=1, inversion=1`
uses Roe fluxes and scalar inversions.

`restart_interval` controls native binary checkpoint updates during a solve, not
the frequency of Python CGNS, Tecplot, or PLOT3D snapshots. Those exports are
written when a calculation is finalized (the default for `solve()`), or for
nonfinal chunks when `writeIntermediateSolutions=True`, according to their output flags. The
native checkpoint uses the internal temporary `.cfl3d-state.bin`; a completed
solve caches its bytes and removes that temporary file. Numbered restart exports
are separate final snapshots. The native solver also writes a restart at the end
of each stage even when its length is shorter than `restart_interval`. For
example, to write every 100 fine-grid iterations across ten continued calls:

```python
solver.setOptions({"iterations": 100, "numberSolutions": True})
for _ in range(10):
    result = solver.solve()
```

Each call uses the cached flow after the first and allocates another output
index. Explicit coarse-grid stages add startup work before the `iterations`
fine-grid cycles. The interface currently has no
within-call visualization output interval. Native NPLOT3D/NPRINT count requested
output data sets, rather than specifying an iteration interval.

### Multigrid and mesh sequencing

| Option | Default | Meaning |
| --- | --- | --- |
| `multigrid_levels` | `1` | Total levels including the finest grid, 1..5 |
| `multigrid_cycle` | `2` | NGAM: 1 for V cycles, 2 for W cycles |
| `mesh_sequence` | `False` | Start on coarse grids before the final fine-grid stage |
| `coarse_grid_iterations` | `50` | Iterations on each preliminary stage of the automatic sequence |
| `multigrid_up_iterations` | `0` | MTT: additional updates on the upward sweep, 0..19 |
| `multigrid_smoothing` | `None` | MIT per global level, ordered coarse to fine; defaults to one update per level |
| `coarse_grid_stages` | `None` | Explicit coarse startup stages, excluding the finest stage; tuple of `MultigridStage` objects or dictionaries in options mappings |

Each coarsened direction must have a cell count divisible by
`2**(multigrid_levels-1)`. Boundary segment endpoints must align with every level.
The two-plane i direction of an extruded 2D grid is not coarsened. Native
preprocessing also enforces minimum dimensions: JDIM and KDIM cannot be 2 on the
fine or coarse levels. Divisibility alone does not guarantee a valid multigrid mesh.

With `mesh_sequence=True`, every preliminary stage uses `coarse_grid_iterations`
and its first `coarse_grid_first_order_iterations` cycles use first-order spatial
discretization. These first-order cycles are part of the stage total, not extra.
The final stage uses `iterations` and `first_order_iterations`. Counts of stages
actually used must be positive.
Without mesh sequencing, the solve starts on the finest grid and uses the requested
V/W cycles. Cached continuation restores the finest-grid state, rebuilds coarse
work arrays, retains V/W cycles, and skips preliminary coarse-to-fine startup.

Explicit coarse stages provide finer startup control (development after 0.2.0):

```python
from cfl3d import MultigridStage

solver.setOptions({
    "limiter": (0, 2, 4),
    "flux": (1, 0, 1),
    "inversion": (1, 0, 1),
    "kappa": (0.0, 1/3, 1/3),
    "multigrid_levels": 3,
    "iterations": 1000,                # Finest-grid budget on every solve
    "first_order_iterations": 20,      # Finest-grid NITFO
    "multigrid_up_iterations": 1,
    "coarse_grid_stages": (
        MultigridStage(100, levels=1, first_order_iterations=10,
                       smoothing_steps=(2,)),
        MultigridStage(200, levels=2, first_order_iterations=20,
                       smoothing_steps=(2, 1)),
    ),
})
```

Options dictionaries may instead supply a list of stage dictionaries. Stage keys
are also case-insensitive. Each stage has `iterations`, `levels`, optional
`first_order_iterations=0`, and optional `smoothing_steps=None`.

There may be zero to `multigrid_levels-1` coarse stages. Each stage advances one
mesh level; the final finest-grid stage is appended automatically using
`iterations`, `first_order_iterations`, and all `multigrid_levels`. For three total
levels and one coarse stage, startup begins on the second-finest grid. A stage's
`levels` selects how many of its available levels participate in cycling.
Native `MSEQ` is the number of stages, not the number of available grid levels:
it equals `len(coarse_grid_stages) + 1` here. Two coarse stages with
`multigrid_levels=3` produce three `NCYC` rows and MIT rows with 1, 2, and 3
active values. Native input templates may show eight MIT columns, but CFL3D
reads only the active values for each stage.
Stage smoothing values are
ordered coarse to fine among those active levels; absent values use the relevant
part of `multigrid_smoothing`. MIT entries are 1..20 and each MIT+MTT must be <=20.

The example performs 100+200+1000 cycles on its first cold solve and 1000 cycles
on each continuation. Changing `iterations` always changes the next finest-grid
budget. `resetFlow()` starts the coarse sequence again. Both ordinary and resident
solves use this rule. `MultigridStage` is immutable; replace stages or supply a new
tuple to modify them. Do not combine explicit coarse stages with `mesh_sequence=True`:
that flag selects the automatic startup recipe using `coarse_grid_iterations` instead.

The former `multigrid_stages` option was removed because it represented a complete
schedule and could silently omit the finest startup stage. Scripts using it must
move all preceding stages to `coarse_grid_stages` and put the final stage's
`iterations` and `first_order_iterations` at the top level.
The former automatic-startup names `coarse_iterations` and
`coarse_first_order_iterations` were renamed to `coarse_grid_iterations` and
`coarse_grid_first_order_iterations`; update existing option dictionaries and
command-line flags accordingly.

### Reference quantities

| Option | Default | Meaning |
| --- | --- | --- |
| `sref` | `1.0` | Positive reference area, m^2 |
| `cref` | `1.0` | Positive reference chord, m |
| `bref` | `1.0` | Positive roll/yaw reference length, m |
| `moment_center` | `(0,0,0)` | Three moment-reference coordinates in the input mesh, m |
| `freestream_pressure` | `None` | Positive static pressure, Pa, required for dimensional load queries |

Reference areas must match the model being integrated. Half-model loads are not
automatically doubled. Changing a reference does not alter already cached results;
the new reference is applied by the next solve.

### Mesh, restart, and output options

| Option | Default | Meaning |
| --- | --- | --- |
| `input_grid` | `None` | CGNS input path; required when no mesh argument is supplied |
| `restartFile` | `None` | Python checkpoint or native full-state restart input |
| `resident` | `False` | Retain native workspace between chunks; restricted cold-start mode described below |
| `output_directory` | `"cfl3d-output"` | Automatic output directory |
| `name` | `"out"` | Case name used in automatic filenames |
| `progress` | `True` | Print flushed mesh, native solve, and output phase checkpoints on communicator rank 0; set False to disable these checkpoints |
| `numberSolutions` | `True` | Use a new output number after finalization; False can reuse a continued calculation's number |
| `writeIntermediateSolutions` | `False` | Also write snapshots after `solve(final=False)`, with a separate `_chunk_NNN` suffix; existing output flags select the file types |
| `writevolumesolution` | `True` | Write CGNS/ADF volume output and include volume arrays in automatic NPZ |
| `writeTecplotSurfaceSolution` | `True` | Write the main Tecplot ASCII wall file |
| `writeflowsolution` | `False` | Write native-compatible PLOT3D `.x/.q` |
| `volume_filename` | `None` | Custom label/path for automatic volume output |
| `surface_filename` | `None` | Custom label/path for automatic wall output |
| `flow_prefix` | `None` | Custom label/path for automatic PLOT3D output |

Progress checkpoints show elapsed wall time and the planned native grid-stage
cycles. The native solver writes its detailed iteration log to the numbered
`.out` file named in the checkpoint; the Python phase line is printed before
entering the blocking native call. For MPI jobs, stdout from communicator rank 0
must be captured by the launcher to see these lines. A completed checkpoint
identifies the preceding phase as finished, while the current line identifies
work still in progress.

## Dynamic updates and continuation

Updates are installed in Fortran at the start of the next `solve()` or `solver()`.
Changing Python options does not modify an already running native calculation.
The default continuation decision compares the current settings with the last
successful solve, not with the last assignment. Editing a parameter and restoring
its original value before the next call therefore does not discard the flow.

| Category | Default behavior after a change |
| --- | --- |
| Physical model / freestream scales | Mach, Reynolds number, temperature, gamma, turbulence model, ambient turbulence controls, full N-S, or y-up changes initialize from freestream |
| Numerical controls | Retain the finest-grid flow and apply new iteration, CFL, scheme, or multigrid settings |
| Operating point | Retain flow; apply alpha, beta, and CL-control changes, with explicit alpha edits taking precedence over saved trim angles |
| Reference quantities | Retain flow; integrate subsequent results using the new references |
| Output settings | Retain flow; change file writing and naming |
| Mesh / boundary configuration | Reload or use the current geometry, rebuild computational data, and initialize again |
| New restart input | Load and validate the explicitly requested state |
| Runtime mode | Changing `resident` cold initializes the next calculation |

This table describes default restart-based continuation. Resident mode uses the
narrower warm-update policy in [Resident calculation chunks](#resident-calculation-chunks).

`CFL3D.getOptionCategories()` returns the complete option classification.
`result.start_mode` is `"cold"` or `"restart"`; `changed_options` lists changed option
names and `reset_reasons` identifies the reset triggers. Root prints automatic
reset reasons, and the NPZ output records this metadata.

```python
solver.setOptions({"cfl": 3.0, "iterations": 200})
warm = solver()                         # Continue cached flow
solver.options.mach = 0.9
cold = solver()                         # Initialize again after the Mach change
print(cold.start_mode, cold.reset_reasons)
```

Grid coordinates, dimensions, boundary conditions/overrides, and one-to-one maps
are checked for in-memory changes. Updating `input_grid` loads the new mesh on root
and broadcasts it within comm. A size/modification-time change in the selected file
also triggers rereading; actual geometry/configuration changes determine whether
the flow must be reset. `input_grid=None` retains the current in-memory mesh.
Registered relative cuts are reattached when a new grid is loaded; their group
names must remain valid in the new grid.

An unchanged old `restartFile` is not reloaded to cancel a physical-parameter reset.
Explicitly supplying a new restart input selects that state after compatibility
checks. There is currently no option to force continuation across arbitrary
physical-model changes.

### State access and restart files

This complete-state workflow applies to `resident=False`.

```python
state = solver.getFlowState()            # Independent native bytes and field arrays
solver.setFlowState(state)               # Install a compatible state locally
solver.writeRestartFile("checkpoint.npz")

resumed = CFL3D(options={**options, "restartFile": "checkpoint.npz"})
result = resumed()

solver.resetFlow()                       # Clear cached state and restartFile
result = solver()                        # Cold start, preserving the output counter
```

`getFlowState()` returns a copy. In serial it contains the complete case; in MPI
it contains this rank's owned zones (`state.zone_indices`, in global CGNS order).
The host rank normally has an empty shard. The native records, including turbulence,
eddy viscosity, and ghost/wall state, are authoritative; editing the primitive
`state.fields` arrays alone does not modify the native restart state. The state is
owned by Python between calls and does not depend on the preceding output directory
or another object's Fortran COMMON data. Each call currently exchanges this state
with the native reader/writer through a temporary `.cfl3d-state.bin`, removed after
a successful solve.

Python checkpoints include native bytes, a checksum, compatibility metadata, and
field copies. The native input format requires real64 data, int32 Fortran record
markers, native byte order, and the default ghost records. Block dimensions and
model records are checked before native execution; Python checkpoints additionally
check stored metadata. Raw native files do not carry the full Python configuration,
so the caller must supply matching axis conventions and physical references.

MPI checkpoint input/output streams native and mean-flow payloads in chunks of at
most 1 MiB; root does not gather complete state arrays. Raw native input is indexed
and sent to owners directly. Compressed NPZ input is incrementally extracted to a
temporary file and checksum-checked before distribution. Its redundant mean-flow
readouts are skipped during restart loading. New NPZ outputs keep the same format
and remain readable by older serial readers.

Compressed input requires temporary disk space for the uncompressed native
payload. Use a disk-backed Python temporary directory (`TMPDIR`) for large cases.
This is separate from the internal `.cfl3d-state.bin` used by native continuation.
It reduces checkpoint payload RAM. Native grid setup and slice/lift analysis still
need whole-case temporary storage on root. Volume output retains at most one
received zone plus conversion buffers;
wall ASCII output streams serialized patches from their owners.

The default `resident=False` path re-enters native initialization on every
`solve()`, using the cached complete `FlowState` and internal checkpoint I/O.
For compatible short calculation chunks, API 16 provides the opt-in resident mode
below. Rebuild both native extensions and restart existing Python processes after
updating from an earlier native API.

`*_solution.npz` and visualization CGNS are not full-state restart inputs.
`setFlowState()` must be applied on every participating rank with that rank's own
snapshot from the same completed solve. Collectively installed complete serial
snapshots are also accepted and scattered before execution. Do not broadcast one
worker's shard to other workers. Checkpoints written by `writeRestartFile()` remain
whole-case files and can be loaded with a different worker count.

`writeRestartFile()` is collective in MPI: every solver rank must call it, including
the host. It remains available when all visualization output flags are False. This
replaces the former root-only writer; the host no longer retains a full restart
payload between calls.

### Resident calculation chunks

Set `resident=True` to retain the native workspace between calls. The default is
False, so the full-state restart workflow above remains available.

```python
from cfl3d import CFL3D, MultigridStage

resident_options = {
    **options,
    "resident": True,
    "iterations": 50,
    "mesh_sequence": False,
    "multigrid_levels": 3,
    "coarse_grid_stages": (MultigridStage(20, levels=1),
                           MultigridStage(30, levels=2)),
    "restartFile": None,
    "writevolumesolution": False,
    "writeTecplotSurfaceSolution": False,
    "writeflowsolution": False,
}

with CFL3D(options=resident_options) as solver:
    first = solver.solve()                 # 20+30 coarse startup, then 50 fine cycles
    solver.setOptions({"cfl": 1.0, "iterations": 25})
    second = solver()                      # Advance 25 more cycles in memory
    print(solver.getRuntimeInfo())          # Local counters/capacities, not RSS
    loads = solver.getSolution()
    solver.writeVolumeSolution()           # Optional explicit visualization
# Workspace and borrowed MPI communicator are released collectively here.
```

The resident mode supports implicit steady cold initialization, including
automatic or explicit coarse-to-fine startup. Startup finishes before the first
return, and subsequent calls extend only the final stage. Multilevel V/W cycles
and their coarse-level work arrays remain alive. File restart initialization and full-state snapshot
operations (`getFlowState`, `setFlowState`, `writeRestartFile`) are not yet
implemented for this mode and raise explicit errors. Volume/PLOT3D visualization
is supported, but no automatic native `_restart.bin` accompanies it.

The first returned history includes coarse startup and fine cycles; later histories
contain only the requested additional fine cycles. `getRuntimeInfo()` reports
`startup_cycles`, `fine_cycles`, and their cumulative sum `cycles`. Since native API 19, public resident calls reuse history storage: the previous
row is retained for boundary-condition evaluation and the new chunk occupies the
remaining rows. Capacity is bounded by the largest requested chunk plus one row
(or the initial startup allocation), not by total cycles. Native iteration,
CFL-ramp and lift-control counters remain cumulative. Complete diagnostics append
to the native `.res`, `.turres` and `.alpha` files; read them when a complete
history is needed. Retaining old `SolverResult` objects is the caller's choice.

Each solve returns that call's history rows. By default it finalizes the calculation;
`solve(final=False)` keeps the current output index open for subsequent chunks.
`start_mode="restart"` is retained as the continuation label; a resident advance
does not use a restart file. With `numberSolutions=False`, logs/history append to
the active index while visualizations update. With True, the next calculation after finalization uses new numbered names. Initial preprocessing logs belong to the initial index;
compatible later calls do not preprocess again. Registered slices/lift
distributions write automatically at finalization, or at nonfinal boundaries when
`writeIntermediateSolutions=True`. Finalization retains the native workspace;
`close()` releases it and does not implicitly finalize an unfinished calculation.

With all visualization flags False and no registered surface analyses, a resident
advance writes no grid, internal restart or automatic solution NPZ. Native text
logs and residual/turbulence/alpha histories still write. `restart_interval`
does not enable internal checkpoints in this mode. History arrays grow as needed
and retain all completed rows; their memory grows with total cycles.

Warm resident updates currently cover `cfl`, `iterations`, I/O settings and
`freestream_pressure` (dimensional post-processing only). Changing other settings
or the mesh closes the old context and cold initializes a new one, reporting
`reset_reasons`. Switching `resident` modes also cold initializes. Unchanged
startup/ramp/lift-control counters survive warm calls;
changing CFL while a ramp is active has not been validated. Chunk length can be
shorter than the configured first-order startup period without restarting it.

`close()` releases the authoritative native state without advancing or exporting
a snapshot. Last result readouts remain available for queries/deferred output;
another solve after close starts cold. `resetFlow()` also closes the context and
clears restart inputs, preserving output numbering. An active resident owner must
close before another object solves in that Python process. Keep the resident
object alive for as many `solve()` calls as the workflow needs; there is no need
to decide the number of calls in advance. Once the case is finished, **every rank
in `solver.comm` must call `solver.close()`**, or leave the same `with` block.
Rank 0 cannot release another rank's native workspace. `del solver`, garbage
collection and interpreter shutdown are not substitutes for collective close:
the active native context retains the solver, and Python destruction is not
synchronized across MPI ranks. For example:

```python
with CFL3D(options={**case_options, "resident": True}, comm=case_comm) as solver:
    solver.solve()
    # Later, all ranks in case_comm may call solver.solve() again as needed.
# All case_comm ranks close the resident workspace here.
```

`close()` also removes successful worker directories and empty native auxiliary
files; users do not need to delete them manually. Failed-run diagnostics remain.
Independent cases on disjoint communicators can remain resident and solve
simultaneously. One duplicated native communicator lives across the chunks;
closing does not finalize MPI or free the user's communicator. See
[distributed-storage.md](distributed-storage.md) for the native boundaries and
regression evidence.

## MPI

### Local storage and explicit collection (since native API 12)

Coordinates, `result.fields`, detailed wall results, and cached restart records
contain only owned zones. Global topology, BC/group metadata, per-boundary force
integrals, and per-boundary wetted areas remain available on all ranks.
`result.zone_owners` maps each CGNS zone name to its owner rank **within this solver's
communicator**. Ownership comes from the native load balancer, not a Python guess.
The native host (rank 0) has no persistent coordinates or detailed flow/wall data.
The f2py bridge allocates owned mean-flow and wall cells; neither detailed array is
Allreduced. A worker's f2py coordinate input is empty: native setup sends its grid
from the host using the existing MPI path.

`solver.mesh.zones` retains the global zone order and metadata. For a non-owned
zone, `zone.coordinates is None`; `zone.dimensions` is still available. Local
coordinates use their usual `(ni,nj,nk,3)` shape. Edit coordinates only on their
owner, and retain array dimensions. Use `input_grid` to replace the discretization.
Before the first native solve, coordinates have an initial placement; afterward
they follow `result.zone_owners`. Historical geometry snapshots are also local.

Call `solver.gatherFields(root=0)` on every solver rank to explicitly collect
mean-flow arrays. It returns a whole-case dictionary on root and None elsewhere.
CFL3D does not cache the collected dictionary. Retaining it in caller code naturally
uses global-case memory on the receiving rank. `getFlowState()` remains local;
use the collective checkpoint writer for a portable whole-case state.

| Operation | MPI calling requirement |
| --- | --- |
| `solve()`, `finalizeSolution()`, `gatherFields()` | Every rank in `solver.comm` |
| `writeVolumeSolution()`, `writeFlowSolution()`, `writeRestartFile()` | Every rank in `solver.comm`; root writes the file |
| `getFlowState()`, `setFlowState()` | Local; install matching owned shards on every rank before the next solve |
| `getSolution()` | Local cached integral query |
| `addLiftDistribution()`, `addSlices()`, `getLiftDistribution()`, `getSlices()` | Every rank; only the request/section/strip results are broadcast |
| `writeTecplotSurfaceSolution()`, `writeSurfaceAnalysis()` | Every rank; wall ASCII streams from owners; registered analyses temporarily collect wall data on root |
| `clearSurfaceAnalysis()` | Clear on every rank; consistency is checked before solving/output |

Native API 12 replaces the former root-only surface registration/query/writer
contract. Call those methods outside `if rank == 0` blocks. Detailed wall geometry
and fields are collected only for the operation and then released; querying
`getSolution()` still needs no collective.

File input reads global topology on root and coordinates on their initial owners.
Native grid setup, explicit global array collection and slice/lift analysis can
still temporarily assemble global data on the host. Normal MPI checkpoint I/O
uses bounded streaming. Volume output writes one zone at a time; wall ASCII output
streams from owners. Peak host memory is therefore not yet bounded by its owned
zones. See [distributed-storage.md](distributed-storage.md) for remaining work and
measured payload sizes.


Omitting `comm` automatically selects the MPI path under an MPI launch or an
already initialized multi-rank `MPI.COMM_WORLD`. Ordinary serial construction does
not initialize MPI just to inspect settings. Explicit `comm=None` selects serial
execution; a one-rank communicator also uses the serial core.

```python
from mpi4py import MPI
from cfl3d import CFL3D

comm = MPI.COMM_WORLD
solver = CFL3D(options=options, comm=comm)
result = solver()
if comm.rank == 0:
    print(solver.getSolution())
```

Native rank 0 is the host; remaining ranks own complete zones. The current interface
does not split a zone automatically and requires
`2 <= comm.size <= number_of_zones + 1`. Every rank keeps its own options. Changes
must be made consistently on all participating ranks; `solver.checkOptions()` or
`solver.options.check()` checks consistency collectively, and solve checks it again.
Root changes are not silently broadcast over explicit rank-local options. Passing
`options=None` on a non-root rank is a compatibility shorthand for inheriting the
root's initial options.

### Independent cases on disjoint communicators

Independent cases can solve simultaneously in one MPI job by assigning each case
a communicator with a disjoint set of ranks. Each MPI rank is a Python process;
the restriction is one active resident case per process, not one case for the
entire MPI job. Rank membership is distinct from physical CPU-core affinity.
Separate case names and output directories keep their files independent.

For example, with four processes, world ranks 0-1 solve case 0 while world ranks
2-3 solve case 1. Each case has its own native host and worker:

```python
world = MPI.COMM_WORLD
case_id = world.rank // 2
case_comm = world.Split(case_id, world.rank)
case_options = {
    **options,
    "name": f"case_{case_id}",
    "output_directory": f"case-{case_id}-results",
    "resident": True,
}
try:
    with CFL3D(options=case_options, comm=case_comm) as solver:
        result = solver()
        # Further solves and option updates involve this case_comm only.
finally:
    case_comm.Free()  # Release after the resident context has closed.
```

All members of a case communicator call that case's collective operations. Ranks
assigned to other cases do not join them. Communicators for simultaneously active
cases must not share ranks. Within one process, another object may solve after
the resident owner closes; changing names/directories alone does not isolate its
Fortran COMMON state. This limit does not serialize the independent cases above.

An excluded rank can use `MPI.UNDEFINED` in `Split` and must not construct or call a
solver on `MPI.COMM_NULL`. The object checks membership in its supplied communicator,
not the entire world. It duplicates that communicator for native messages. With
`resident=True`, the duplicate remains valid across chunks and is freed at close;
the default nonresident path frees it after each solve. Neither path finalizes MPI.

Ranks retain their owned coordinate/flow/wall/restart arrays. Automatic files are
written by comm root. Workers use separate `rank-0001/` work directories while
native code is active; successful nonresident solves remove them on return, and
resident solves remove them at `close()`. Failed runs retain worker directories
for diagnosis. Directories with unexpected files are left untouched.
Successful solves also remove empty legacy auxiliary files (such as `.out15`
and `.out20`) from the main output directory. Resident solves do this at
`close()`, after native files are closed. Empty optional logs such as `.alpha`
are removed too; their `output_paths` entries still name the expected location.
Nonempty files remain.
Loads and group-name queries remain local. Detailed surface queries and all
writers are collective. Native MPI errors abort the job to avoid leaving peers
inside blocking native calls.

## CGNS grids and boundary conditions

`CGNSMesh.from_file(path, backend="auto")` reads ADF through libcgns. For HDF5 it
uses libcgns when the library supports HDF5, otherwise optional h5py. Explicit
backends are `"cgns"` and `"h5py"`. ADF does not need h5py.

The mesh must contain structured zones in one selected three-dimensional CGNS
base. For multiple bases, use `CGNSMesh.from_file(path, base="BaseName")` and pass
the resulting mesh. Two-dimensional cases use a 3D base with exactly two i planes
in every zone. Coordinates are taken as metres without automatic unit conversion.

Boundaries must be Vertex `PointRange` rectangles. Each exterior face must be
covered completely by boundaries or one-to-one connections, without overlapping
face cells. Shared edge lines are permitted. `GridConnectivity1to1_t` transforms
may permute or reverse axes; paired coordinates are verified, and reciprocal
connections with reversed endpoints are deduplicated.

| CGNS boundary type | Native code / data |
| --- | --- |
| `BCFarfield` | 1003 |
| `BCSymmetryPlane` | 1001 |
| `BCWallInviscid` | 1005 |
| `BCOutflowSupersonic` | 1002 |
| `BCWall` | 2004 / `(0,0)`: native CGNS converter's default adiabatic viscous wall |
| `BCWallViscous` | 2004 / `(0,0)`: adiabatic, no blowing |
| `BCWallViscousHeatFlux` | 2004 / `(0,0)` for absent data or explicitly zero HeatFlux |

`FamilySpecified` can be resolved through an unambiguous `FamilyBC_t` in the base.
Unsupported boundary types or datasets require explicit overrides keyed by the
CGNS zone and BC names; data is not silently discarded:

```python
solver = CFL3D(
    "mesh.cgns",
    SolverOptions(turbulence="sst"),
    boundary_overrides={
        ("Zone1", "inlet"): (2009, (1.1, 1.0, 0.0, 0.0)),
        ("Zone1", "outlet"): (2002, (1.0,)),
        ("Zone1", "wall"): (2004, (1.05, 0.0)),
    },
)
```

Override values follow the native boundary definitions: 2004 uses `(Tw/Tinf, CQ)`,
with zero selecting an adiabatic wall and negative wall-temperature values using
the native stagnation-temperature convention; 2002 uses an outlet static-pressure
ratio. Use the native boundary documentation to choose inlet angles and thermal
or blowing data. Misspelled override keys are rejected. Interfaces use CGNS
connectivity and native code 0; they cannot be fabricated with a BC override.

PointList, non-Vertex boundaries, nonzero coordinate rind layers, general/overset
connectivity, and periodic/averaged connections are unsupported. The native reader
expects the coordinate node `GridCoordinates` and at most one `ZoneGridConnectivity`
node. Unsupported BC datasets, including nonzero heat flux without an explicit
override, cause an error.

## Component loads

```python
solver = CFL3D(options=options, groups={"Aircraft": ["Wing", "Tail"]})
result = solver()
print(solver.getGroupNames())
print(solver.getGroupNames(detailed=True))  # Include individual BCs and zone/BC paths.
loads = solver.getSolution("Aircraft")
print(loads["Aircraft_CL"], loads["Aircraft_CD"], loads["Aircraft_CDp"], loads["Aircraft_CDv"])
print(loads["Aircraft_CX"], loads["Aircraft_CY"], loads["Aircraft_CZ"])
print(loads["Aircraft_Fx"], loads["Aircraft_Fy"], loads["Aircraft_Fz"])
print(loads["Aircraft_Cmy"], loads["Aircraft_My"], loads["Aircraft_wettedArea"])
solver.addGroups({"WingOnly": ["Wing"]})
```

Replace Wing/Tail with names actually present in the mesh. `getGroupNames()` lists
CGNS families, custom groups, and useful BC component names. For numbered BC names
such as `FAR_2854` and `FAR_2855`, it lists the aggregate `FAR`; then
`getSolution("FAR")` sums those patches. An existing unsuffixed BC named `FAR` is
included in that aggregate. `getGroupNames(detailed=True)` additionally lists
individual BC names, zone names, and exact `zone/BC` paths. These selectors remain
valid even when hidden from the default list. Zone families include all that
zone's boundaries; BC families include that BC. Repeated BC names are merged
across zones. Name precedence is Family, then zone, then BC; explicit `zone/BC`
paths are always available. Group names are case-sensitive. Automatic suffix
grouping recognizes only an underscore followed by digits; give other patch
naming schemes explicit CGNS families or custom groups.

Custom groups use `{new_name: [existing_CGNS_name, ...]}`. New names must not collide
with CGNS or previously added group names. Recursive custom-group membership is
not supported. Overlapping members are integrated once. `getSolution()` without a
group name already returns the total integral; callers need not enumerate groups.
Surfaces outside the native force-integration scope, such as a farfield boundary,
contribute zero even if their name is `FAR`.

The cached loads come from pressure and viscous integration in native `force.F`
on the finest grid, before its arrays are released.

| Field | Meaning / unit |
| --- | --- |
| `CL`, `CD` | Lift and drag coefficients in native wind axes |
| `CX`, `CY`, `CZ` | Force coefficients along the input mesh x/y/z axes |
| `CDp`, `CDv` | Pressure and viscous drag coefficients; `CD = CDp + CDv` |
| `Cmx`, `Cmy`, `Cmz` | Moment coefficients in input coordinates |
| `L`, `D`, `Dp`, `Dv` | Lift, drag, pressure drag, and viscous drag, N |
| `Fx`, `Fy`, `Fz` | Forces along the input mesh x/y/z axes, N; `Fy` replaces the former `Y` key |
| `Mx`, `My`, `Mz` | Corresponding moments, N*m |
| `wettedArea` | Total selected wall area, m^2, including inviscid walls; no symmetry multiplier |
| `units` | Unit dictionary for dimensional outputs |

The table lists keys for `groupName=None`. Supplying a group prefixes every
numerical key with `<groupName>_`, including coefficients, dimensional loads, and
area: `Wing_CL`, `Wing_Fy`, and `Wing_wettedArea`. The `units` dictionary remains
named `units`, with matching prefixed entries such as `units["Wing_Fy"] == "N"`.

CL/CD/CDp/CDv follow native wind axes. CX/CY/CZ and moments use input coordinates;
with y-up grids CY is the vertical input-Y force, not a lateral side-force
coefficient. Use the Cartesian components when comparing forces across span/axis
conventions. Wetted area is computed from two triangles per cached wall quad;
farfield, symmetry, and other non-wall boundaries contribute no area. Overlapping
group members are counted once. Coordinates are assumed to be in metres.
The dimensional conversion is `q_inf = 0.5 * gamma * p_inf * mach**2`,
`force = coefficient * q_inf * sref`. Moment-reference lengths for x/y/z are
`bref/cref/bref` when `y_up=False` and `bref/bref/cref` when True. Moments are about
`moment_center`. The pressure saved at solve time takes precedence. If it was not
set, `getSolution()` also accepts `solver.options["freestream_pressure"]` supplied
after the solve; all other conversion quantities still come from that solution's
saved options. If neither supplies pressure, an English `RuntimeWarning` explains
that absolute loads cannot be determined. The call returns coefficients and
`wettedArea`, and omits all dimensional forces and moments. Queries are local on
any rank and do not rerun the solver.

## Output numbering and deferred writing

Names use `<name>_<index>` with index starting at 000 and extending past three
digits when needed. A fresh output stream skips existing indices. Case name plus
index must fit within 50 ASCII characters and the name cannot contain path
separators. Different concurrent cases should use different names/directories.

| Content | Default first output, name=out |
| --- | --- |
| Volume CGNS/ADF | `out_000_volume.cgns` |
| Tecplot ASCII wall | `out_000_surf.plt` |
| Cp sections / lift distribution | `out_000_slices.dat` / `out_000_lift.dat` |
| PLOT3D | `out_000_flow.x` / `out_000_flow.q` |
| Python result | `out_000_solution.npz` |
| Native restart, when volume or PLOT3D output is enabled | `out_000_restart.bin` |
| Default `writeRestartFile()` checkpoint | `out_000_restart.npz` |
| Native log / residual / turbulence / trim history | `out_000.out` / `.res` / `.turres` / `.alpha` |
| Preprocessor log / last native status | `out_000_precfl3d.out` / `out_000.error` |

The surface `.plt` file contains Tecplot ASCII text, not Tecplot binary data.
The extension is a filename convention; slices and lift distributions retain
their `.dat` extensions. Explicit surface filenames keep the suffix you provide.

### Final calculations and intermediate chunks

`solve(final=True)` and `solver(final=True)` retain the default behavior: advance
`iterations` cycles and output the completed calculation according to the existing
output flags. For an external convergence/coupling loop, mark intermediate calls
as nonfinal:

```python
# All ranks in solver.comm execute these calls.
for _ in range(4):
    solver.solve(final=False)
result = solver.solve()       # Fifth chunk; final output uses out_000_...

# Alternatively, decide that the latest nonfinal result is already sufficient:
solver.solve(final=False)     # Starts the next calculation, out_001
result = solver.finalizeSolution()  # Output only; no extra iterations
solver.close()                # Release the resident workspace when no longer needed
```

`final=False` still returns current local flow/wall data and updates the continuation
state. It suppresses automatic volume, surface, slice/lift, NPZ and restart exports
unless `writeIntermediateSolutions=True`. Native logs/history and the nonresident
internal restart mechanism retain their existing behavior. Explicit deferred
writers remain callable. No new `x0` or reference-grid copy is created.
Use `resident=True` when repeated advances should retain the native workspace
and avoid repeated setup/internal restart I/O. The `final` flag controls output
grouping; it does not change either continuation mode's numerical behavior or
automatically determine convergence.

With `writeIntermediateSolutions=True`, the existing output flags select which
snapshots to write. For example, intermediate surface files are named
`out_000_chunk_000_surf.plt`, `out_000_chunk_001_surf.plt`; final output is
`out_000_surf.plt`. Registered slices/lift files use the same chunk prefix.
Native logs always use the main index. `result.chunk_output_paths` lists the
intermediate targets, while `result.output_paths` lists the final/native targets;
listed targets may be disabled and absent on disk.

`result.chunk_index` is zero-based within an output stream; it resets on a new
main index. With `numberSolutions=False`, it keeps increasing in the reused stream
to avoid overwriting earlier chunks. `result.finalized` records whether the result
was finalized successfully. Finalization is idempotent, does not advance the flow,
and retains the cache/resident workspace. If a writer fails, it remains pending
and can be retried. Finalization is collective; all ranks must also agree on the
`final` flag passed to `solve()`.

### Numbering and continued output

Fortran writes numbered logs from the beginning of computation. It does not first
write a fixed `cfl3d.out` and copy it afterward. Existing files from older versions
are not deleted or renamed.

`numberSolutions=True` allocates a new index for the next calculation after
finalization. All nonfinal chunks of one calculation use the same index.
False reuses the previous index only for a continued calculation within the same
object, with the same case name, directory, and filename labels:

```python
solver()                                # 000
solver.setOptions({"numberSolutions": False, "iterations": 200})
continued = solver()                    # Append another 200 steps to 000
solver.options.numberSolutions = True
solver()                                # Continue flow, allocate 001
```

Reused `.out`, preprocessor, `.res`, `.turres`, and `.alpha` logs append rather than
replace previous contents. Displayed iteration numbers continue across chunks;
parameter summaries and headers may repeat. Flow, wall, slice, lift, restart, and
NPZ files update to the latest snapshot at that index. `.error` reports the latest
chunk's status. The returned `history` contains only the current chunk;
`cumulative_history` is retained as a compatibility alias for the current chunk
only (no accumulation), and `iteration_offset` is the preceding step count for
this output index. NPZ `history` and `chunk_history` both contain this chunk.
Full histories remain in the appended native diagnostic files.

Cold initialization, `resetFlow()`, or an output-identity change starts a new
number even if numbering is disabled. While a calculation is pending, finalize
it or call `resetFlow()` before changing its output identity or requiring cold
initialization. `resetFlow()` discards the unfinished calculation without final output. New objects do not claim old disk logs just
because `restartFile` was set. `solver.nSolve` counts successful native calls,
independently of output indices. `solver.nSolutions` counts successful finalizations.
Configuration failures do not consume a number; a completed native chunk reserves
its number even if output fails. Retry `finalizeSolution()` after a failed final
output; it does not repeat native iterations or log/history records.

Custom filenames are labels after the case/index prefix, retaining their parent
directories. For example, `surface_filename="plots/wall.dat"` gives
`plots/out_000_wall.dat`, with analyses alongside it; `volume_filename="grid.cgns"`
gives `out_000_grid.cgns`. `result.output_paths` lists target paths, including targets
that may not exist because their outputs were disabled.

All three main output flags can be False while complete state remains cached.
At finalization, registered surface analyses still write automatically; the
nonresident path also writes result/history NPZ metadata.
After a successful solve, files can be written without rerunning the core:

```python
# All participating ranks call volume/PLOT3D/checkpoint writers.
solver.writeVolumeSolution()
solver.writeFlowSolution()
solver.writeRestartFile()
solver.writeTecplotSurfaceSolution(groupName="Wing")
solver.writeSurfaceAnalysis()
```

Default paths retain the most recent result's identity even if live options later
change. Explicit filenames/prefixes are caller-controlled; relative explicit paths
are relative to the caller's working directory. Automatic/default paths are based
on the solve output directory. Visualization writers use temporary files followed
by replacement and protect the original CGNS input from overwrite. In MPI,
all deferred writers must be called by every solver rank. CGNS, PLOT3D and automatic
solution NPZ writers receive and release one zone at a time. The wall ASCII writer
serializes patches on their owners and streams them to root, without gathering the
volume mesh or complete wall arrays. Registered surface analyses remain separate
operations that temporarily collect wall geometry and detailed data on root.

### File formats

Volume CGNS uses ADF, the original vertices, BC/connectivity metadata, and
cell-centered fields. Density, velocities, and pressure retain native
nondimensionalization; Temperature is K. A `CFL3DUnits` descriptor documents units
and freestream scales. This is visualization output, not a full restart file.

PLOT3D matches native `IPTYPE=1` cell-centered output: int32 record markers,
real64 coordinates/conservative variables, and IBLANK. Extruded 2D output uses two
coordinates and four conservative variables, and 3D uses three and five.
Grid-point `IPTYPE=0` export is not implemented.

Wall output is Tecplot ASCII FEQUADRILATERAL, one zone per wall BC. X/Y/Z are wall
vertices; Cp, T, Cfx, Cfy, Cfz, and y+ are face-centered variables marked by
VARLOCATION. T comes from the native wall boundary state; Cf uses native traction
and input Cartesian axes. The face-cell y+ uses `rho_w*u_tau*d/mu_w`,
`d=V/(2*A)`, and the core's Sutherland molecular viscosity. Inviscid-wall Cf and y+
are zero. Empty wall selections produce a header without wall zones.
Native integrated loads and convergence histories are evaluated every cycle.
Detailed wall fields, including y+, are captured only on the final fine-grid
cycle of each `solve()` call; resident continuation refreshes them at every
Python solve boundary.

## Lift distributions and Cp sections

Register analyses before a solve to produce automatic Tecplot ASCII `.dat` files,
or register them after a solve and use deferred output:

```python
solver.addLiftDistribution(200, "y", groupName="Wing")
solver.addSlices("y", [3.5, 7.0, 10.5], sliceType="absolute", groupName="Wing")
solver.addSlices("y", 7.0, sliceType="relative", groupName="Wing")
result = solver()

lift = solver.getLiftDistribution(0)
print(lift["stations"], lift["dCL_ds"], lift["dCDp_ds"], lift["dCDv_ds"])
print((lift["dCL_ds"] * lift["width"]).sum())
print(solver.getSolution("Wing")["Wing_CL"])
sections = solver.getSlices()            # getSlices(0) returns one section
print(sections[0]["coordinates"], sections[0]["Cp"])
```

`direction` is x, y, or z in input coordinates and denotes the slice normal or strip
axis. `groupName=None` selects all walls. `nSegments` is positive; 150-250 is a
usual starting range. Positions are actual coordinates in metres, including for
relative cuts, not fractional span positions.

| Returned data | Meaning / unit |
| --- | --- |
| `edges`, `stations`, `width` | Equal-width strip bounds, centres, and widths, m |
| `bin_CL`, `bin_CD`, `bin_CDp`, `bin_CDv` | Strip coefficients; sums equal selected native wall totals |
| `dCL_ds`, `dCD_ds`, `dCDp_ds`, `dCDv_ds` | Coefficient densities, 1/m, normalized with global sref |
| `dL_ds`, `dD_ds`, `dDp_ds`, `dDv_ds` | Load densities, N/m, when freestream pressure is supplied |
| Section `points`, `segments` | Cut vertices in metres and zero-based endpoint connectivity |
| Section `coordinates`, `Cp`, `T`, `Cfx`, `Cfy`, `Cfz`, `y+` | Segment midpoints and face-centered values; T in K |

Strips conservatively distribute native pressure/viscous face loads by geometric
overlap. `dCL_ds` is not a local-chord-normalized 2D airfoil cl. `force_angles` records
the actual force-evaluation angles in degrees, which can differ from final_alpha.

Absolute cuts intersect the fixed plane at each output. Relative cuts retain
material triangle intersections and follow subsequent geometry changes; grid
reloads reattach them at their requested positions. Arbitrary in-memory topology
changes may require `clearSurfaceAnalysis()` and registration again. Conforming
block cuts merge coincident endpoints and duplicate segments. There is no overset
zipper or nonmatching-surface repair. Cp remains face-centered; shared-edge cuts
use adjacent-face area weighting, and fully coplanar wall faces do not generate
cut lines. Cuts without intersections produce no Tecplot zone.

Registered analyses write after every solve, even when
`writeTecplotSurfaceSolution=False`. Each request produces one zone.
`writeTecplotSurfaceSolution("wing.dat")` also writes `wing_lift.dat` and
`wing_slices.dat`. `writeSurfaceAnalysis(prefix="custom")` writes
`custom_lift.dat` and `custom_slices.dat`. Filtering the main wall file does not
change the groups registered for analyses. `clearSurfaceAnalysis()` removes all
requests; `nLiftDist` and `nSlice` report registration counts. Registration and
queries are collective in MPI, as are both surface writers. Root-only registration
or querying from the earlier replicated-storage API must be updated. Computed
strip/section results are returned on all participating ranks; writer paths are
returned on root, with None on other ranks.

## Partitioned surface mesh (after 0.2.0)

```python
# Can be called before the first solve, after a solve, or after close().
# Under MPI every rank in solver.comm must call with the same groupName.
surface = solver.getSurfaceMesh(groupName="Wing")
x_local = surface["coordinates"]      # (N,3), metres; uniquely owned nodes only
node_ids = surface["node_ids"]         # (N,), int64; sorted global identities
faces = surface["connectivity"]        # (M,4), int64; GLOBAL node IDs

# A future MPhys mesh component can use x_local.reshape(-1) as its local vector.
local_node_count = len(node_ids)
```

The group selects the same CGNS names or custom groups accepted by
`getSolution()`. None selects all wall boundaries, including inviscid walls.
Unions count each boundary once. Unknown names and groups with no wall faces
raise an error. No solve, file read/write, or native initialization is performed
by this query; it uses the mesh already loaded in memory. It does not depend on
MPhys or ADflow being installed.

| Entry | Content |
| --- | --- |
| `coordinates` | `(N,3)` float64 input-axis coordinates in m; no ghost/duplicate nodes |
| `node_ids` | `(N,)` int64 global node identities, ascending on each rank |
| `node_owners` | `(N,)` int32 owner ranks in `solver.comm`; all entries equal this rank |
| `connectivity` | `(M,4)` int64 global node IDs for locally owned wall quads, possibly referencing remote nodes |
| `face_ids` | `(M,)` int64 global face identities, stable for unchanged topology/boundary ordering |
| `face_keys` | M `(zone_name, boundary_name)` tuples, aligned with face rows |
| `units` | `{"coordinates": "m"}` |

**Connectivity is not an index into the local coordinate array.** In serial,
build an ID-to-row map to index coordinates. In MPI, coordinates for referenced
remote IDs are on their owners; any global assembly must be an explicit caller
operation. All host arrays are empty in the native host/worker configuration.

IDs are based on zero-based, C-ordered whole-volume vertex offsets and the
smallest equivalent ID reached through CGNS one-to-one connections. They are
not compact surface numbering. Shared edges and corners, signed/permuted
transforms, and one-sided connection declarations are handled through topology.
Coincident but topologically disconnected surfaces retain distinct IDs. The
canonical vertex's coordinates are returned; connected copies must agree within
the existing native-input interface tolerance.

For MPI, the canonical zone index fixes each coupling node's owner:
`1 + canonical_zone_index % (comm.size - 1)`. Serial ownership is rank 0.
The ownership rule depends on the mesh and communicator, not the selected group
or native load balance. Thus the local node IDs, order and vector sizes remain
stable between setup and the first solve, even if native zone ownership changes.
Faces follow the current zone-coordinate ownership and can move between ranks;
their IDs and global connectivity remain stable. Shared-node coordinates are
exchanged directly between the required owners. Full surface coordinates are
neither gathered on root nor broadcast to all ranks.

Face order follows mesh zone order, boundary order, then C order of the two
surface cell axes, matching the per-patch `wall_forces` layout. Quad winding is
into the fluid domain for positive-handed structured grids. Topology, boundary
ordering, communicator size, or group membership changes require rebuilding an
external coupling mapping. Group requests, resolved membership, and replicated
mesh metadata must agree within the communicator.

Before a successful solve, this method uses the current input mesh. Afterwards,
it uses the last solved geometry snapshot, consistent with cached force queries;
editing `solver.mesh` does not reinterpret an older solution. Returned arrays
are independent copies. The method does not add periodic/nonconforming support,
construct a zipper mesh, calculate nodal loads, update native geometry, or provide
an MPhys Builder. Those are separate coupling-development steps.

Validation covers analytical planar area/connectivity, disconnected coincident
patches, shared corners, signed/permuted mappings, reversed endpoint declarations,
CGNS/ADF input, and independent saved geometry. Real MPI checks use unequal zones,
native ownership changes, repeated resident solves, disjoint communicators, and
a communicator excluding world rank 0:

```bash
PYTHONPATH=tests/python python -m unittest test_surface_mesh -v
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_surface_mesh_mpi.py --directory build/native-output-surface-mesh-mpi3
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_surface_mesh_mpi.py --split --directory build/native-output-surface-mesh-split
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_surface_mesh_mpi.py --exclude --directory build/native-output-surface-mesh-exclude
```

## Nodal surface forces (after 0.2.0)

```python
result = solver.solve()
surface = solver.getSurfaceMesh("Wing")
forces = solver.getSurfaceForces("Wing")  # (N,3), N; same node_ids/order as surface
pressure = solver.getSurfaceForces("Wing", component="pressure")
viscous = solver.getSurfaceForces("Wing", component="viscous")
# forces = pressure + viscous; the returned arrays are independent copies.
meld_load_vector = forces.reshape(-1)    # Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,...
```

`getSurfaceForces(groupName=None, *, component="total")` returns dimensional
forces acting on the body, in input Cartesian axes. None selects all walls;
CGNS and custom groups follow the surface-mesh contract. The component may be
`"total"`, `"pressure"`, or `"viscous"` (case-insensitive). Each call returns only
the calling rank's uniquely owned nodes, including an empty `(0,3)` array on the
native host. This is the force convention corresponding to ADflow's
`forcesAsTractions=False`, suitable for FUNtoFEM/MELD. A traction/area-normalized
mode is not exposed by this function.
Pressure contributions retain CFL3D's native reference-pressure convention;
`freestream_pressure` supplies the SI dimensional scale.

Every native wall-cell pressure/viscous integral is divided equally between its
four vertices, then accumulated onto the stable node IDs. A node shared by two
cells receives the SUM of both quarter-cell contributions. There is no division
by adjacent-cell count or area: either would change the physical meaning of the
returned forces. Group faces are selected before this assembly, so an edge node
does not receive contributions from an unselected adjacent group. Overlapping
custom groups count the union of their boundaries once.

CFL3D already computes native moments using the arithmetic mean of each quad's
four corner positions. The same quarter-cell distribution therefore preserves
both native force and moment, including warped quads, to floating-point and
connected-coordinate tolerances. It is also the transpose of the four-corner
mean displacement interpolation, preserving its discrete virtual work. This
does not substitute for the separate aero-to-structure MELD transfer.

The dimensional conversion uses `q_inf * sref`, with
`q_inf = 0.5 * gamma * freestream_pressure * mach**2`, from the last successful
solve. Saved reference/flow settings outrank live edits. If that solve had no
pressure, a subsequently supplied `solver.options.freestream_pressure` may provide
the missing scale. Without a pressure, the method raises an English error instead
of returning coefficients labelled as N. `getSolution()` retains its existing
warning-and-coefficients behavior. A successful solve is required; forces remain
queryable after `close()`.

All ranks in `solver.comm` must call with the same group and component. The
resolved dimensional scale must also agree. Native local wall integrals are
distributed directly to the coupling node owners and added there; complete force
vectors are not gathered on root or broadcast. Queries use cached data and do not
advance the solution, initialize Fortran, or write files. Reading/editing returned
arrays does not change native state or the cached integrals.

Serial checks cover known two-cell loads, shared nodes, warped-quad force/moment
and virtual-work conservation, all six wall directions, Euler zero-viscous loads,
pressure/viscous separation, group isolation, saved reference values, and missing
pressure. MPI checks compare serial node vectors and native integrated forces/
moments after resident chunks, with native zone repartition, disjoint groups, an
excluded world rank, and collective rejection of inconsistent query controls:

```bash
PYTHONPATH=tests/python python -m unittest test_nodal_forces -v
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_nodal_forces_mpi.py --directory build/native-output-nodal-forces-mpi3
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_nodal_forces_mpi.py --split --directory build/native-output-nodal-forces-split
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_nodal_forces_mpi.py --exclude --directory build/native-output-nodal-forces-exclude
```

## Result arrays

| Attribute | Content |
| --- | --- |
| `fields[zone]` | Owned zones only in MPI; cell-centered `(ni-1,nj-1,nk-1,5)` arrays: rho,u,v,w,p |
| `zone_owners` | Global CGNS zone name to rank in this solver's communicator |
| `history` | Current chunk's rows: residual,CL,CD,CX,CY,CMX,CMY,CMZ |
| `coefficients` | Last-step coefficients, with uppercase CMX/CMY/CMZ and total CDp/CDv |
| `surface_coefficients[(zone,bc)]` | CL,CD,CX,CY,CZ,Cmx,Cmy,Cmz,CDp,CDv for each boundary |
| `wall_fields[(zone,bc)]` | Owned walls only in MPI; two surface cell indices followed by Cp,T,Cfx,Cfy,Cfz,y+ |
| `wall_forces[(zone,bc)]` | Owned walls only in MPI; pressure XYZ and viscous XYZ loads divided by q_inf*sref |
| `final_alpha` | Final native angle, degrees |
| `name`, `solve_index`, `output_directory`, `output_paths` | Output identity and targets |
| `iteration_offset`, `cumulative_history` | Previous steps and compatibility alias for current chunk history |
| `start_mode`, `changed_options`, `reset_reasons` | Initialization decision and configuration changes |

Volume arrays use native nondimensionalization: freestream density=1, sound speed=1,
and pressure=1/gamma. Velocities are returned in input Cartesian axes; grid indexing
remains i,j,k. The residual array stores RMS values, whereas native `.res` files
print log10(residual). Surface indices follow the two unfixed grid axes.

Automatic result NPZ contains current-chunk `history`, `chunk_history`, identity and
initialization metadata, final_alpha, and the surface names/coefficient table.
When volume output is enabled it also contains all `zone_<name>` arrays, collected
temporarily on root. Calling `result.save()` directly writes only that result's
local fields; use `gatherFields()` explicitly when whole-case arrays are needed. The surface
coefficient column names are stored in `surface_coefficient_names`.

## Validation and supported scope

From a built environment at the repository root:

```bash
python -m unittest discover -s tests/python -v
python tools/build_python.py --standalone --debug --fc gfortran --jobs 8
PYTHONPATH=tests/python python tests/python/validate_transition_parity.py --directory build/transition-parity-check
PYTHONPATH=tests/python python tests/python/validate_transition_parity.py --models bsl sst2003 wilcox2006 --directory build/variant-parity-check
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_runtime_mpi.py --auto --directory build/mpi-runtime-check
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_runtime_mpi.py --auto --model gamma-theta --directory build/mpi-transition-check
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_runtime_mpi.py --auto --model sst2003 --directory build/mpi-sst2003-check
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_runtime_mpi.py --split --directory build/mpi-runtime-check
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_solution_io_mpi.py --directory build/mpi-io-check
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_resident_runtime_mpi.py --directory build/mpi-resident-check
```

Use fresh validation output directories for scripts that assert initial output
names. Tests generate their own ADF fixtures. Optional HDF5 tests skip when h5py
is unavailable. The regression suite covers parameter updates/resets, repeated
solves, restart files, CGNS boundaries/connectivity, grouped loads, output formats,
wall quantities, sections/distributions, and serial/MPI comparisons. Current run
counts and compiler-specific evidence belong in the checkpoint records rather
than being duplicated here.

The wrapper currently exposes steady, real-valued calculations on structured
one-to-one multiblock meshes, including multigrid. Unsteady calculations, overset,
nonmatching patches, periodic connections, native moving-grid/aeroelastic workflows,
adjoints, and OpenMDAO integration are not exposed. Embedded-grid controls are not
implemented. These interface limits do not describe the full capabilities of the
original CFL3D solver.

Iterations and output files are not a convergence guarantee. Examine residuals,
loads, and the flow before treating a result as converged. Validation to date
includes interface consistency and short calculations; production-scale behavior
and long-run convergence require case-specific assessment.

Independent cases on disjoint MPI communicators can solve simultaneously. Within
one Python process, native calls are serialized because COMMON state is shared;
one active resident case occupies that rank until close. Native execution temporarily
changes the process working directory; other threads should not depend on relative
paths during a solve.
Intercepted serial native errors return RuntimeError through a C guard, but may leave native
allocations behind: start a fresh Python process after such an error. Native MPI
errors abort the job. Python-side configuration errors are checked collectively
before native entry.

Debug builds add symbols and bounds checks to the new state interface. The legacy
Fortran core uses cross-dimensional storage access, so global bounds checking is
not enabled for every legacy routine. Detailed build setup is maintained only in
the [README](../README.md#minimal-installation).

## Mesh deformation exchange (experimental)

`cfl3d.mesh` separates the solver-independent volume layout (`VolumeMeshLayout`)
from the optional `IDWarpAdapter`. Importing this subpackage does not require
IDWarp. The caller creates an IDWarp `USMesh` on the same communicator, using
the same structured CGNS file/zone order as CFL3D.

```python
from cfl3d.mesh import IDWarpAdapter

surface = solver.getSurfaceMesh()
adapter = IDWarpAdapter(warper, solver.mesh, surface, comm=solver.comm)
# Optional reference storage, owned by this application; omit for plain CFD.
x0 = adapter.layout.coordinates(solver.mesh)

surface["coordinates"] += displacement  # metres, in getSurfaceMesh node order
new_coordinates = adapter.warp(surface)  # local absolute coordinates, (N, 3)
zone_coordinates = adapter.layout.unpack(new_coordinates)
```

The adapter stages coordinates only. To apply them to an already initialized
resident solver, use `solver.setVolumeCoordinates(new_coordinates)` followed by
`solver.solve()`. Build the adapter **after the initial solve**, once native zone
ownership is known. Never assign arrays directly to an active solver's mesh.

Volume vertices follow CGNS zone order and i-fastest ordering within each zone,
with interleaved x/y/z components for IDWarp. Block-interface copies remain
separate. These indices are not the canonical IDs returned by `getSurfaceMesh`.
The adapter converts global surface connectivity to local point indices and
requests remote face-node coordinates. It gathers node IDs for an ownership
directory, not full surface coordinates; IDWarp itself gathers its surface
inside `setSurfaceDefinition`.

All ranks participate, including ranks with empty arrays. Rebuild the adapter
if ownership changes (including the initial native load balance). Surface IDs
and connectivity must remain unchanged during repeated calls. No reference
coordinate array is retained by this adapter. IDWarp may maintain its own
reference/internal mesh independently of CFL3D.

Validation: `test_mesh_exchange.py` checks unequal-zone indexing, partial
ownership, copies and global-to-local surface connectivity.
`validate_idwarp_exchange.py` exercises the installed IDWarp on a two-zone mesh,
including zero-motion agreement and nonzero deformation without mutating CFL3D:

```bash
PYTHONPATH=tests/python python tests/python/validate_idwarp_exchange.py
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_idwarp_exchange.py
```


### Native CFL3D deformation adapter (API 22, experimental)

`CFL3DMeshAdapter` provides two original CFL3D control-point methods followed by
native arc-length edge, face and volume TFI. It does not require IDWarp.

#### Adapter options

Pass these keyword arguments to `CFL3DMeshAdapter(...)`, alongside `mesh`,
`surface` and `comm`. They are mesh-adapter options, not entries in the CFD
solver's `options` dictionary. MPI ranks must supply consistent settings.

| Argument | Default | Values and purpose |
| --- | --- | --- |
| `method` | `"exponential"` | `"exponential"` or `"fmem"`; selects control-point deformation method. |
| `linear_solver` | `"native-gauss-seidel"` | FMEM backend: original native GS or explicit `"petsc-gmres"`. PETSc is never selected automatically. Exponential decay uses no linear solver and rejects `"petsc-gmres"`. |
| `linear_solver_options` | `None` | Dictionary of settings for the selected backend; PETSc currently supports `{"restart": 30}`. Native GS currently accepts an empty dictionary. Unknown or inapplicable keys are errors. |
| `control_mode` | `"stride"` | `"minimal"`, `"auto_stride"`, `"stride"`, or `"indices"`; selects the control lattice. |
| `control_stride` | `None` (effective 4 in stride mode) | Positive scalar, three directional strides, or complete zone-name mapping; may be supplied only in stride mode. |
| `control_indices` | `None` | Required in indices mode: complete zone-name mapping to zero-based i/j/k index lists. Invalid in other modes. |
| `beta1` | `5.0` | Positive FMEM material-decay parameter, used by both FMEM backends. |
| `alpha1` | `0.5` | Native GS relaxation, `0 < alpha1 <= 1`; has no effect on PETSc iteration. |
| `beta2` | `5.0` | Positive exponential-decay parameter. |
| `alpha2` | `0.1` | Exponential inner moving-region parameter, `0 <= alpha2 < 1`. |
| `smoothing_iterations` | `0` | Nonnegative integer spring-smoothing sweeps for exponential decay. |

See [FMEM backends](#fmem-linear-solver-backends) for numerical behaviour and
memory costs, and [control-point selection](#control-point-selection) for
per-zone configuration examples. Example using native FMEM:

```python
from cfl3d.mesh import CFL3DMeshAdapter

# Create after the first resident solve establishes zone ownership.
surface = solver.getSurfaceMesh()
adapter = CFL3DMeshAdapter(
    solver.mesh, surface, comm=solver.comm,
    method="fmem",                         # "fmem" or "exponential"
    linear_solver="native-gauss-seidel",   # default; alternative: "petsc-gmres"
    control_mode="stride",
    control_stride=4,
    beta1=5.0, alpha1=0.5,       # FMEM material decay and native GS relaxation
    beta2=5.0, alpha2=0.1,       # exponential decay and inner moving region
    smoothing_iterations=0,   # exponential spring-smoothing sweeps
)
surface["coordinates"] += displacement
candidate = adapter.warp(surface)
solver.setVolumeCoordinates(candidate)
result = solver.solve(final=False)
```

Both methods return absolute owned coordinates in `VolumeMeshLayout` order.
`warp()` neither modifies the mesh nor advances/reinitializes CFD. Applying a
candidate still requires `setVolumeCoordinates()` and native geometry checks at
`solve()`. Coordinates are based on the **current accepted mesh** on each call;
reference coordinates, if needed, belong to the caller. Do not repeatedly add a
total displacement to the already deformed surface.

#### FMEM linear solver backends

The default is `linear_solver="native-gauss-seidel"`. It gathers **only the coarse
control system** on communicator root, orders controls block/i/k/j (j fastest),
constructs cross-block stencil neighbours/coincident groups, and calls original
`deform_el` (including `hookefe`, `elmetricsfe`, `elglobfe`, `gaussseidel` and
coincidence enforcement). `alpha1` controls GS relaxation. Original stopping
controls are retained: 1e-6 displacement-update norm, at most 9 times the control
node count in sweeps, with the existing divergence detection. This backend
prioritizes original numerical behaviour and requires no PETSc installation.
It does not promise equivalence to arbitrary legacy motion-table inputs: control
selection and surface prescriptions still belong to this adapter.

```python
adapter = CFL3DMeshAdapter(
    solver.mesh, surface, comm=solver.comm, method="fmem",
    linear_solver="native-gauss-seidel",  # default
    control_stride=2, alpha1=0.5,
)

# Explicit opt-in; uses the existing compatible petsc4py installation.
adapter = CFL3DMeshAdapter(
    solver.mesh, surface, comm=solver.comm, method="fmem",
    linear_solver="petsc-gmres",
    linear_solver_options={"restart": 30},
    control_stride=2,
)
```

`petsc-gmres` assembles native block-local stiffness contributions and supplies a
distributed Python-shell operator to **PETSc KSP GMRES**, with Jacobi
preconditioning. PETSc owns all Krylov iteration/storage; there is no handwritten
GMRES. Canonical-node owners sum shared rows and exchange interface values.
PETSc vectors contain only canonical owned degrees of freedom, including zero
local length on a host with no zones. Defaults are restart 30, at most 3000
iterations, relative tolerance 1e-11 and absolute tolerance 1e-13 on the
preconditioned residual. `linear_solver_options["restart"]` must be a positive
integer and is only valid with this backend; increasing it increases local
Krylov storage. Omit the dictionary or pass `{}` to use the default restart 30.
`alpha1` does not control PETSc. Unrelated application PETSc options are not
implicitly read to replace these settings.

Backend provenance remains explicit in `linear_solver`: `native-gauss-seidel`
and `petsc-gmres` are distinct selections. `linear_solver_options` is a shared
configuration entry point, interpreted and validated for that selection. A
concept has one option name rather than duplicate backend-specific names;
parameters with different meanings are not equated. Currently only `restart`
is exposed in this dictionary, and it applies to PETSc GMRES. Native GS accepts
only `None` or `{}` here and retains its existing native stopping controls and
`alpha1` relaxation. No arbitrary PETSc option passthrough is provided.

The dictionary is copied and its normalized values are read-only on the adapter;
mutating the caller's dictionary does not reconfigure a running MPI case. All
ranks must provide equivalent effective settings. **Migration:** replace the
former top-level `petsc_restart=12` with
`linear_solver_options={"restart": 12}`; the old constructor keyword is removed.

PETSc is loaded **only on explicit selection**, on all solver communicator ranks.
When the application has not already initialized PETSc, the adapter initializes
it on `MPI.COMM_SELF`, then creates KSP objects on the solver communicator. This
avoids requiring unrelated communicator groups to import PETSc. Existing
application initialization is preserved. A real double-precision PETSc build
compatible with the MPI runtime is required.
Missing/unusable petsc4py is an error, not a fallback. Neither backend silently
switches to the other after failure. `linear_solver="petsc-gmres"` is rejected
for `method="exponential"`, which needs no linear-system solve.

Exponential decay and optional spring smoothing use the original equations on
local controls. Distance queries stream at most 256 boundary samples per batch;
surface targets go only to canonical owners and requesting zone owners. Both
FMEM backends share control selection and constraints, and both deformation
methods use native `arclen`, `tfiedge`, `tfiface` and `tfivol` on zone owners.
Default GS preserves original control-solving behaviour; PETSc uses a different
iteration order/stopping test, so the two solutions need not agree bit for bit.

Native shared controls are restored. Handled assembly or iterative failures
raise collectively before candidate installation; these adapter errors are
separate from the CFD `SolveFailure` returned by geometry refresh.

Current boundaries and costs:

- Fixed-topology point-matched structured zones; all dimensions need at least
  two vertices. This is a steady warper, not a time-accurate ALE/structural solver.
- Provided surface coordinates are prescribed exactly. Other control constraints
  follow native `setslave.F` BC classification: wall/farfield controls in the
  native constrained list remain fixed, while symmetry code 1001 is free to
  deform. `boundary_overrides={(zone, boundary): (code, data)}` uses the same
  mapping as the CFD solver; MPhys passes its overrides to the native adapter.
  No additional normal projection or curved-surface sliding model is applied;
  this is the native control classification, not a general geometric constraint
  solver. Arbitrary original motion tables are not exposed. Shared interface
  identities come from topology, never nearest points.
- Control selection supports `minimal`, `auto_stride`, `stride` and `indices`.
  The default is `stride` with spacing 4. Physical/interface endpoints are added
  and matching indices propagated. There is no forced minimum of five indices
  for FMEM. Extremely coarse or degenerate systems can fail and require explicit
  refinement; no fallback silently changes the method. See the
  [control-point selection](#control-point-selection) for examples and inspect
  the effective lattice with `getControlIndices()`.
- Native TFI supports at most 500 control indices per axis. Overset, nonmatching
  and periodic transforms are outside this interface's supported contract.
- Global zone/connectivity metadata and axis index lists remain replicated.
  Default native GS temporarily stores the complete coarse control system on
  root and supports at most ten coincident controls per group. PETSc keeps the
  system distributed but adds Krylov storage and synchronization. Neither gathers
  full volume grids; TFI uses temporary zone-sized workspace.
- The partition is still by whole zones: a single large zone is not split among
  ranks. Boundary-search traffic can be expensive. Production timing and peak
  memory for both backends remain unmeasured.
- The **serial native extension is required on every rank**, even when CFD uses
  the MPI extension. Rebuild both extensions for API 22. Constructing no adapter
  adds no deformation work or persistent reference-grid copy to ordinary CFD.


##### Optional PETSc tests

PETSc numerical tests automatically skip when `petsc4py` cannot be found. An
installed but unusable PETSc, or a failed numerical check, remains a test failure.
Native/exponential tests always run. To explicitly disable optional PETSc tests
without uninstalling it, use the same environment switch for unit and MPI tests:

```bash
CFL3D_SKIP_PETSC_TESTS=1 PYTHONPATH=tests/python python -m unittest discover -s tests/python
CFL3D_SKIP_PETSC_TESTS=1 PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_cfl3d_mesh_adapter_mpi.py
```

The MPI validator also accepts `--skip-petsc`:

```bash
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_cfl3d_mesh_adapter_mpi.py --skip-petsc
```

Skip decisions are agreed across all ranks before any PETSc import. If one rank
lacks the package or requests a skip, the optional case is skipped collectively.
With `--mixed`, disabling PETSc reports the mixed-backend scenario as skipped and
runs native cases in both groups; it does not claim mixed-backend coverage.
Unset `CFL3D_SKIP_PETSC_TESTS` (or set it to a value other than `1`) and omit
`--skip-petsc` to run installed PETSc tests. These switches affect tests only;
production `linear_solver="petsc-gmres"` still errors if its dependency is missing.

#### Control-point selection

`CFL3DMeshAdapter` supports exponential decay and FMEM followed by native TFI.
Control points are selected existing volume vertices, not additional CFD nodes.
The Cartesian product of each zone's i, j and k index lists forms its control
lattice. Adjacent controls form eight-node macro-elements. Choosing 5 indices in
each direction means 125 controls and 64 macro-elements in that zone.

##### Selection modes

| `control_mode` | Selection | Additional argument |
| --- | --- | --- |
| `"minimal"` | Block ends, physical boundary segment ends and point-matched interface segment ends; propagate matching indices | None |
| `"auto_stride"` | Largest uniform index stride consistent with required indices in each direction; iterate interface matching to consistency | None |
| `"stride"` (default) | User-specified spacing, then necessary topology additions | `control_stride`, default 4 |
| `"indices"` | Explicit index lists, then necessary topology additions | `control_indices` |

The first two modes correspond in intent to the original CFL3D automatic minimal
and automatic uniform options. This adapter implements index selection from its
CGNS topology; it does not read the original motion tables or `meshdef.inp`.
No exact equivalence to every legacy input configuration is claimed.

**All explicit indices are zero-based Python i/j/k indices**, unlike the
one-based indices in original CFL3D input files. Indices refer to the original
zone axes, not a flattened vector or the native internal axis permutation.
Every supplied list must be nonempty, strictly increasing, integer and in range.
Block ends are inserted automatically if omitted. An index outside the zone is
an error, not silently clipped. There is a native limit of 500 indices per axis.

##### Control-point configuration examples

Construct the adapter after the first resident solve establishes ownership.
All ranks in `solver.comm` must participate and supply consistent control
settings for all zones, including zones whose coordinates are owned elsewhere.
The following are alternative configurations; choose one.

```python
from cfl3d.mesh import CFL3DMeshAdapter

surface = solver.getSurfaceMesh()

# Minimal topology-conforming control lattice.
adapter = CFL3DMeshAdapter(
    solver.mesh, surface, comm=solver.comm,
    method="fmem", control_mode="minimal",
)

# Automatically choose uniform spacing satisfying all required segment indices.
adapter = CFL3DMeshAdapter(
    solver.mesh, surface, comm=solver.comm,
    method="fmem", control_mode="auto_stride",
)

# Manual global spacing: scalar or separate (i, j, k) spacing.
adapter = CFL3DMeshAdapter(
    solver.mesh, surface, comm=solver.comm,
    method="fmem", control_stride=(2, 4, 2),
)

# Manual spacing per zone; mapping must include exactly all zone names.
strides = {zone.name: (2, 4, 2) for zone in solver.mesh.zones}
adapter = CFL3DMeshAdapter(
    solver.mesh, surface, comm=solver.comm,
    method="fmem", control_mode="stride", control_stride=strides,
)

# Explicit indices per zone. Here choose roughly four intervals per direction;
# replace these lists with desired indices for your geometry.
indices = {}
for zone in solver.mesh.zones:
    indices[zone.name] = tuple(
        sorted(set(range(0, n, max(1, (n-1)//4))) | {n-1})
        for n in zone.dimensions
    )
adapter = CFL3DMeshAdapter(
    solver.mesh, surface, comm=solver.comm,
    method="fmem", control_mode="indices", control_indices=indices,
)
```

Only provide the argument belonging to the selected mode. For example,
`control_mode="minimal", control_stride=2` is rejected, as is supplying
`control_indices` without selecting `"indices"`. Mapping keys are actual zone
names, not boundary groups. No dictionary entry implicitly falls back to a
per-zone default.

##### Inspect, refine and apply control points

```python
# Local metadata query: no coordinate gathering or MPI communication.
effective = adapter.getControlIndices()
if solver.comm.rank == 0:
    for name, (i, j, k) in effective.items():
        print(name, "control points:", len(i)*len(j)*len(k), i, j, k)

# effective is a new dictionary of immutable index tuples. It can be reused
# as control_indices when constructing a new adapter, optionally after editing.
surface["coordinates"] += displacement
candidate = adapter.warp(surface)
solver.setVolumeCoordinates(candidate)
result = solver.solve(final=False)
# Close solver collectively when the calculation session is finished.
```

The effective lists may contain more indices than requested: physical segment
ends and matching interface indices are mandatory. Across a point-matched face,
only the corresponding interface directions/indices are propagated according to
the connection transform. This ensures compatible control lattices without
matching unrelated points merely because their coordinates coincide.

Unlike the initial implementation, FMEM does **not** force at least five indices
per direction. A smaller lattice may be sufficient; an overly coarse one can
produce degenerate material-distance ranges or unsuitable macro-elements. An
all-prescribed control lattice needs no FMEM linear solve. If a free-control
system is degenerate, refine it explicitly rather than relying on automatic
hidden point insertion. A successful control solve still does not prove the
final volume grid valid: the existing interface and CFD geometry checks apply.

The default remains `control_mode="stride"` with `control_stride=4`. Existing
scalar-stride calls are valid, but removing the FMEM minimum-count rule can make
previously auto-refined small meshes coarser. Select an explicit smaller stride
(e.g. 2 for a 9-point axis) when reproducing those earlier tests.

Minimal selection avoids increasing controls just because the CFD mesh is finer.
Automatic uniform selection can be much denser: required indices 0, 3, 8 have
largest common stride 1. Inspect the resulting lists before production runs.
These modes do not eliminate whole-zone partition limits or guarantee a specific
warping time. For ownership, boundary policies and API 22 rebuild requirements,
see the [adapter reference](#native-cfl3d-deformation-adapter-api-22-experimental).

### Resident coordinate updates (experimental, native API 18)

```python
# All solver.comm ranks execute these calls. resident=True is required.
solver.solve(final=False)  # initialize native ownership and flow once
x = solver.getVolumeCoordinates()  # owned vertices, (N,3), metres
# x0 = x.copy()  # optional: only the application decides to retain a reference
x[:, 2] *= 1.01  # example; use an external mesh warper for actual deformation
solver.setVolumeCoordinates(x)
solver.solve()  # refresh geometry once, then advance the existing flow
```

The setter accepts absolute coordinates in `VolumeMeshLayout` order. It validates
finite values, shape, and point-matched interface coordinates across MPI ranks.
It stages copies without changing current coordinates or old result snapshots.
The next solve updates all native grid levels, metrics, volumes and wall
distances; the normal iteration routines update boundary ghost states. It skips
flow/turbulence initialization, mesh-file reads, repartitioning and coarse-to-fine
startup. Transported fields remain initial guesses for the deformed geometry.
After a successful solve Python coordinates and result snapshots describe the
new native geometry. `getRuntimeInfo()['geometry_updates']` counts these refreshes.

An unchanged coordinate submission cancels a pending update and does not trigger
native geometry work. Ordinary continuation adds no per-cycle geometry checks,
coordinate transfers or geometry recomputation. Pending local arrays and native
coordinate staging buffers are released after application. `close()` discards
pending coordinates; `resetFlow()` also closes the context. Apply or cancel a
pending update before changing options that require a cold initialization.

Current scope is steady, fixed-topology, point-matched grids with an already
paused resident context. The adapter must be rebuilt after any ownership change.
This is not an ALE/unsteady moving-grid interface. Native metric/volume checks
still apply. API 18 introduced collective expected geometry failures; API 20
allows a new coordinate submission to retry while retaining the flow. Native
geometry is not automatically rolled back:

```python
result = solver.solve()
if result.fail:
    print(result.reason, result.native_status)
    # All ranks submit corrected coordinates, or explicitly restore the last
    # accepted coordinates retained on the Python side.
    solver.setVolumeCoordinates(solver.getVolumeCoordinates())
    result = solver.solve()  # rebuild geometry before resuming the retained flow
```

Successful `SolverResult` objects have `fail=False`. A `SolveFailure` instead has
`fail=True`, `reason`, `native_status` (41: metrics, 42: volumes), and
`reset_required=False`, `geometry_update_required=True`; it has no flow results.
Failed updates do not advance iterations, increment the solution count, or export
a new solution. Without a new coordinate submission, `solve()` returns failure
again. After `setVolumeCoordinates()`, the next solve fully refreshes all geometry
levels and runs only if checks pass. Even coordinates equal to the last accepted
Python grid are restaged after failure. Repeated rejected updates remain retryable.

The flow and turbulence stay resident, but native geometry may be partially
updated and is unusable until a successful refresh. Current result extraction is
disabled after failure; previously returned result snapshots remain independent.
Python coordinates stay at the last accepted grid until a successful solve. No
extra rollback grid copy is allocated. `resetFlow()`/`close()` remain optional
ways to discard the resident context. Merely staging a pending grid before any
failure does not prevent inspecting the last successful result; `solutionIsCurrent`
is false until that pending grid is processed.

Only expected errors encountered during the resident metric/volume update are
handled this way. Shape/nonfinite/interface input errors still raise before
native mutation. Startup failures, MPI faults, allocation errors and segmentation
faults are not converted to `SolveFailure`. Finite coordinates and matching
interfaces do not constitute a mesh-quality guarantee.
The first validation covers small SA/SST and MPI gamma-theta fixtures; production-scale deformation
and all model/override combinations are not yet established.

## MPhys static aeroelastic analysis (experimental, forward only)

`cfl3d.mphys.CFL3DBuilder` supplies the distributed surface mesh, aerodynamic
nodal loads, and `cl`, `cd`, `lift`, `drag` post-processing components required by
`ScenarioAeroStructural`. Install MPhys/OpenMDAO and the desired mesh warper;
TACS and FUNtoFEM/MELD are separate optional dependencies. Importing `cfl3d`
does not import these packages.

```python
from cfl3d.mphys import CFL3DBuilder

builder = CFL3DBuilder(
    options={
        "input_grid": "wing_vol_L2.cgns",
        "output_directory": "output",
        "resident": True,
        "mach": 0.85,
        "temperature": 223.25209264797854,
        "freestream_pressure": 26499.89786492552,
        "reynolds": 7223294.8822458545,  # per metre
        "sref": 45.5,
        "cref": 3.25,
        "iterations": 1000,  # cycles for each coupling evaluation
    },
    scenario_options={"cruise": {"alpha": 2.0}, "maneuver": {"alpha": 5.0}},
    warper="idwarp",
    mesh_options={},  # USMesh options; gridFile defaults to input_grid
)
builder.initialize(comm)
# Use the builder in ScenarioAeroStructural and connect its mesh component.
# Always call builder.close() collectively in the application's finally block.
```

`options` and `scenario_options` use CFL3D names. They do not accept ADflow's
ANK controls, AeroProblem objects or ADflow convergence options. The promoted
`aoa` input is in degrees. Other flow conditions are specified in the options.
The solver advances a fixed `iterations` budget on each evaluation; completing
that budget is not a claim of aerodynamic convergence. The outer MPhys nonlinear
solver must also be checked for convergence.

`warper="idwarp"` uses `USMesh` and the existing `IDWarpAdapter`. An explicit
`warper="cfl3d"` uses `CFL3DMeshAdapter` options instead. Surface vectors contain
owned unique nodes in metres; `f_aero` contains integrated nodal forces in N,
suitable for MELD, not tractions. `group_name=None` uses all wall boundaries.
Reference surface coordinates are retained in Python for `x_aero0_mesh`. Volume
reference storage required by IDWarp is owned by IDWarp, not added to the Fortran
solver. Stable node ownership keeps OpenMDAO vector sizes fixed after the native
block partition changes.

The first evaluation runs any configured coarse-grid startup and spends one
fine-grid cycle on the input grid because
`setVolumeCoordinates()` currently requires an initialized resident context.
It then warps to the supplied absolute surface and spends the remaining
`iterations - 1` cycles on the updated mesh. Initialization is included in the
first evaluation's fine-grid budget; that budget must be at least two cycles.
Coarse startup is additional and runs once per scenario. Subsequent chunks
continue the configured multigrid correction without repeating startup.
Subsequent coupling evaluations retain the flow and refresh geometry only when
coordinates change. Geometry failures raise OpenMDAO `AnalysisError`. Changes to
alpha during an active session are rejected; set it before entering the scenario.

Scenarios run sequentially on a communicator. The post component finalizes the
latest solution and closes the native workspace before the next scenario starts.
A later `run_model()` starts each scenario from its input grid/freestream again;
there is no implicit resident flow swapping or scenario restart cache. Interleaving
unfinished scenarios on the same processes is rejected. Use `builder.close()`
for cleanup after exceptions as well as successful runs.

Gradient callbacks currently emit `RuntimeWarning` and skip contributions, as
requested for the first integration. Such derivatives are incomplete and must
not be used for optimization or sensitivity analysis. Complex-step inputs are
rejected. Only forward `run_model()` analysis is supported.

The workspace example `case_test/mphys_as_2scenario.py` uses the supplied L2 grid,
`wingbox.bdf`, original structural callback, IDWarp and MELD. Its `--smoke` mode
uses three CFD cycles per call and two outer coupling iterations; this is a
connection/deformation check, not a converged static aeroelastic solution. The
original ADflow example is preserved as `mphys_as_2scenario_adflow.py`.

The repository [aerostructural example](../examples/python/mphys_aerostructural/README.md)
selects native exponential/FMEM or IDWarp and exposes one-time multigrid startup.
The supplied L2 requires a single level; use a compatible mesh for multigrid.
The example propagates child `AnalysisError` immediately through its outer
nonlinear solver and checks native cycle/initialization counters.

Each evaluation records cumulative fine-grid CFD cycles (`cfd_cycles`),
`startup_cycles`, native `total_cycles`, `initializations`, residual and loads in
`coupling-history.jsonl` in its scenario output directory.
`builder.get_iteration_history(scenario_name)` returns these small records.
The example accepts `--cfd-iterations 100 --coupling-iterations 80 --fixed-budget`
to request 8000 CFD cycles per scenario, updating the coupling every 100 cycles.
Fixed-budget mode disables early tolerance-based termination and reports outer
nonconvergence rather than raising; it does not imply convergence.
