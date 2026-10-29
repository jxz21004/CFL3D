"""L2 MPhys: one-time coarse-to-fine startup, resident coupling and mesh warping.

See README.md. Default warper: native CFL3D exponential decay + TFI.
Use --warper idwarp for a comparison with identical CFD/coupling settings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import openmdao.api as om
from mphys import Multipoint, MPhysVariables
from mphys.scenarios import ScenarioAeroStructural
from cfl3d.mphys import CFL3DBuilder
from tacs.mphys import TacsBuilder
from funtofem.mphys import MeldBuilder
import structure

ROOT = Path(__file__).resolve().parents[4] / "case_test"


class Top(Multipoint):
    def initialize(self):
        self.options.declare("settings")

    def setup(self):
        args = self.options["settings"]
        # Same physical conditions as the reviewed L2 cruise case (altitude 10 km).
        aero_options = dict(mach=0.85, temperature=223.25209264797854,
                            freestream_pressure=26499.89786492552,
                            reynolds=7223294.8822458545, sref=45.5, cref=3.25,
                            turbulence="sa", limiter=4, multigrid_levels=args.levels,
                            mesh_sequence=True, coarse_grid_iterations=args.coarse_iterations,
                            multigrid_cycle=2, writeflowsolution=False)
        aero_options.update(
            input_grid=str(args.grid),
            output_directory=str(args.output), resident=True,
            iterations=args.cfd_iterations, cfl=1.5,
            writevolumesolution=not args.smoke,
            writeTecplotSurfaceSolution=not args.smoke,
        )
        mesh_options = {} if args.warper == "idwarp" else dict(
            method=args.method, control_stride=args.control_stride,
            beta2=args.beta2, alpha2=args.alpha2,
            smoothing_iterations=args.smoothing_iterations)
        if args.control_indices is not None:
            mesh_options.pop("control_stride")
            mesh_options.update(control_mode="indices",
                                control_indices=json.loads(args.control_indices.read_text()))
        self.aero_builder = CFL3DBuilder(
            aero_options, warper=args.warper,
            mesh_options=mesh_options,
            scenario_options={"cruise": {"alpha": 2.0}, "maneuver": {"alpha": 5.0}},
        )
        self.aero_builder.initialize(self.comm)
        self.add_subsystem("mesh_aero", self.aero_builder.get_mesh_coordinate_subsystem())
        struct_builder = TacsBuilder(
            element_callback=structure.element_callback,
            problem_setup=structure.problem_setup(args.output / "structure"),
            mesh_file=str(args.struct_mesh),
            coupling_loads=[MPhysVariables.Structures.Loads.AERODYNAMIC],
        )
        struct_builder.initialize(self.comm)
        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())
        transfer = MeldBuilder(self.aero_builder, struct_builder, isym=1)
        transfer.initialize(self.comm)
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        dvs.add_output("dv_struct", np.full(struct_builder.get_ndv(), args.thickness))
        dvs.add_output("aoa0", 2.0, units="deg")
        dvs.add_output("aoa1", 5.0, units="deg")
        for i, name in enumerate(("cruise", "maneuver")):
            nonlinear = om.NonlinearBlockGS(
                maxiter=args.coupling_iterations, iprint=2, use_aitken=True,
                reraise_child_analysiserror=True,
                rtol=0.0 if args.fixed_budget else 1e-14,
                atol=0.0 if args.fixed_budget else 1e-14,
                err_on_non_converge=not (args.smoke or args.fixed_budget),
            )
            self.mphys_add_scenario(name, ScenarioAeroStructural(
                aero_builder=self.aero_builder, struct_builder=struct_builder,
                ldxfer_builder=transfer), nonlinear,
                om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True))
            aero = MPhysVariables.Aerodynamics.Surface
            struct = MPhysVariables.Structures
            self.connect(f"mesh_aero.{aero.Mesh.COORDINATES}", f"{name}.{aero.COORDINATES_INITIAL}")
            self.connect(f"mesh_struct.{struct.Mesh.COORDINATES}", f"{name}.{struct.COORDINATES}")
            self.connect("dv_struct", f"{name}.dv_struct")
            self.connect(f"aoa{i}", f"{name}.aoa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=ROOT / "wing_vol_L2.cgns")
    parser.add_argument("--warper", choices=("cfl3d", "idwarp"), default="cfl3d")
    parser.add_argument("--method", choices=("exponential", "fmem"), default="exponential")
    parser.add_argument("--thickness", type=float, default=0.002,
                        help="Uniform shell design thickness in metres (default: 0.002)")
    parser.add_argument("--control-stride", type=int, default=4)
    parser.add_argument("--control-indices", type=Path, default=None,
                        help="Native per-zone explicit zero-based i/j/k indices JSON; overrides stride")
    parser.add_argument("--beta2", type=float, default=5.0)
    parser.add_argument("--alpha2", type=float, default=0.1)
    parser.add_argument("--smoothing-iterations", type=int, default=0)
    parser.add_argument("--levels", type=int, choices=(1, 2, 3), default=1,
                        help="L2 requires one level; use 2 or 3 with a compatible mesh")
    parser.add_argument("--coarse-iterations", type=int, default=100)
    parser.add_argument("--struct-mesh", type=Path, default=ROOT / "wingbox.bdf")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cfd-iterations", type=int, default=100)
    parser.add_argument("--coupling-iterations", type=int, default=80)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fixed-budget", action="store_true",
                        help="Run the specified coupling count; report rather than require convergence")
    args = parser.parse_args()
    if args.control_indices is not None:
        args.control_indices = args.control_indices.resolve()
        if args.warper != "cfl3d" or not args.control_indices.is_file():
            parser.error("--control-indices requires native warper and an existing JSON file")
    if not np.isfinite(args.thickness) or not 0.002 <= args.thickness <= 0.05:
        parser.error("Thickness must lie within the shell design bounds [0.002, 0.05] m")
    if args.smoke:
        args.cfd_iterations = 3
        args.coupling_iterations = 2
        args.coarse_iterations = 3
    if min(args.cfd_iterations, args.coupling_iterations, args.coarse_iterations, args.control_stride) < 1:
        parser.error("Iteration counts must be positive")
    args.grid = args.grid.resolve()
    if not args.grid.is_file():
        parser.error(f"Missing CFD grid: {args.grid}")
    from cfl3d.cgns import CGNSMesh
    metadata = CGNSMesh.from_file(args.grid, coordinate_zones=())
    divisor = 2**(args.levels-1)
    invalid = [(z.name, z.dimensions) for z in metadata.zones
               if any((n-1) % divisor for n in z.dimensions)]
    thin = [(z.name, z.dimensions) for z in metadata.zones
            if any((n-1)//divisor + 1 < 3 for n in z.dimensions[1:])]
    if thin:
        parser.error(f"Coarsest J/K dimensions must have at least three nodes; "
                     f"incompatible blocks: {thin}. Original L2 requires --levels 1.")
    if invalid:
        parser.error(f"{args.levels} multigrid levels require cell dimensions divisible by "
                     f"{divisor}; incompatible blocks: {invalid}")
    if args.output is None:
        args.output = Path.cwd() / ("output-mphys-mg-" + args.warper +
                                    ("-" + args.method if args.warper == "cfl3d" else ""))
    args.output = args.output.resolve()
    args.struct_mesh = args.struct_mesh.resolve()
    if not args.struct_mesh.is_file():
        parser.error(f"Missing structure mesh: {args.struct_mesh}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "structure").mkdir(exist_ok=True)
    prob = om.Problem(model=Top(settings=args), reports=False)
    try:
        if prob.comm.rank == 0:
            (args.output / "configuration.json").write_text(
                json.dumps(vars(args), default=str, indent=2) + "\n")
        prob.setup()
        prob.run_model()
        summary = {}
        for name in ("cruise", "maneuver"):
            summary[name] = {key: float(prob.get_val(f"{name}.aero_post.{key}")[0])
                             for key in ("cl", "cd", "lift", "drag")}
            if not all(np.isfinite(list(summary[name].values()))):
                raise RuntimeError(f"Nonfinite aero results for {name}")
            displacement = prob.get_val(f"{name}.u_struct", get_remote=False)
            if not np.isfinite(displacement).all():
                raise RuntimeError(f"Nonfinite structure displacement for {name}")
            local_max = float(np.max(np.abs(displacement), initial=0.0))
            summary[name]["max_struct_displacement_dof"] = max(prob.comm.allgather(local_max))
            history = prob.model.aero_builder.get_iteration_history(name)
            summary[name]["cfd_cycles"] = history[-1]["cfd_cycles"]
            summary[name]["coupling_evaluations"] = len(history)
            summary[name]["startup_cycles"] = history[-1]["startup_cycles"]
            summary[name]["total_cycles"] = history[-1]["total_cycles"]
            if any(h["startup_cycles"] != (args.levels-1)*args.coarse_iterations or
                   h["total_cycles"] != h["cfd_cycles"] + (args.levels-1)*args.coarse_iterations or
                   h["initializations"] != 1 for h in history):
                raise RuntimeError("Multigrid startup repeated or native counters disagree")
            if args.fixed_budget and history[-1]["cfd_cycles"] != args.cfd_iterations * args.coupling_iterations:
                raise RuntimeError("Requested fixed CFD budget was not completed")
        if prob.comm.rank == 0:
            summary["smoke_only"] = args.smoke
            summary["warper"] = args.warper
            summary["method"] = args.method if args.warper == "cfl3d" else None
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("CFL3D_MPHYS_PASS", json.dumps(summary), flush=True)
    finally:
        if hasattr(prob.model, "aero_builder"):
            prob.model.aero_builder.close()
        prob.cleanup()


if __name__ == "__main__":
    main()
