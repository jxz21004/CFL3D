"""Readable forward static-aeroelastic example: CFL3D + IDWarp + MELD + TACS.

Run from this directory. Edit aero_options in main() and structure.py for your aircraft.
--levels 3: coarse 500 -> intermediate 500 -> coupled fine-grid iterations.
--levels 2: coarse 500 -> coupled fine-grid iterations.
--levels 1: coupled fine-grid iterations without coarse startup.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import openmdao.api as om
from mphys import Multipoint, MPhysVariables
from mphys.scenarios import ScenarioAeroStructural
from funtofem.mphys import MeldBuilder
from tacs.mphys import TacsBuilder
from cfl3d.cgns import CGNSMesh
from cfl3d.mphys import CFL3DBuilder
import structure

COARSE_CYCLES = 500
SCENARIO = "cruise"  # structure.py applies 1g to this scenario.


class AeroelasticModel(Multipoint):
    def initialize(self):
        self.options.declare("settings")
        self.options.declare("aero_options")

    def setup(self):
        args = self.options["settings"]
        aero_options = self.options["aero_options"]
        groups = json.loads(args.groups.read_text()) if args.groups else None
        self.aero = CFL3DBuilder(aero_options, warper="idwarp", groups=groups)
        self.aero.initialize(self.comm)
        structural = TacsBuilder(
            mesh_file=str(args.struct_mesh), element_callback=structure.element_callback,
            problem_setup=structure.problem_setup(args.output / "structure"),
            coupling_loads=[MPhysVariables.Structures.Loads.AERODYNAMIC],
        )
        structural.initialize(self.comm)
        # isym=1 is the example's y-symmetry plane; change for your model.
        transfer = MeldBuilder(self.aero, structural, isym=1)
        transfer.initialize(self.comm)

        self.add_subsystem("aero_mesh", self.aero.get_mesh_coordinate_subsystem())
        self.add_subsystem("struct_mesh", structural.get_mesh_coordinate_subsystem())
        design = self.add_subsystem("design", om.IndepVarComp())
        design.add_output("thickness", np.full(structural.get_ndv(), args.thickness))
        design.add_output("alpha", aero_options["alpha"], units="deg")
        coupling = om.NonlinearBlockGS(
            maxiter=args.coupling_iterations, use_aitken=True, iprint=2,
            rtol=0.0 if args.fixed_budget else 1e-6,
            atol=0.0 if args.fixed_budget else 1e-8,
            err_on_non_converge=not args.fixed_budget,
            reraise_child_analysiserror=True,
        )
        self.mphys_add_scenario(
            SCENARIO, ScenarioAeroStructural(aero_builder=self.aero,
                                           struct_builder=structural, ldxfer_builder=transfer),
            coupling, om.LinearBlockGS(maxiter=25, use_aitken=True),
        )
        aero = MPhysVariables.Aerodynamics.Surface
        struct = MPhysVariables.Structures
        self.connect(f"aero_mesh.{aero.Mesh.COORDINATES}", f"{SCENARIO}.{aero.COORDINATES_INITIAL}")
        self.connect(f"struct_mesh.{struct.Mesh.COORDINATES}", f"{SCENARIO}.{struct.COORDINATES}")
        self.connect("design.thickness", f"{SCENARIO}.dv_struct")
        self.connect("design.alpha", f"{SCENARIO}.aoa")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True, help="CFD CGNS grid")
    parser.add_argument("--struct-mesh", type=Path, required=True, help="TACS BDF mesh")
    parser.add_argument("--levels", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--thickness", type=float, default=0.002, help="Uniform shell thickness in metres")
    parser.add_argument("--cfd-iterations", type=int, default=100, help="Fine-grid cycles per coupling exchange")
    parser.add_argument("--coupling-iterations", type=int, default=80, help="Maximum coupling exchanges")
    parser.add_argument("--fixed-budget", action="store_true", help="Run all exchanges without requiring convergence")
    parser.add_argument("--groups", type=Path, help='Optional component groups JSON, e.g. {"Wing": ["WingUpper", "WingLower"]}')
    parser.add_argument("--output", type=Path, default=Path("output-static-aeroelastic"))
    args = parser.parse_args()
    for key in ("grid", "struct_mesh", "groups"):
        path = getattr(args, key)
        if path is not None:
            path = path.resolve()
            if not path.is_file():
                parser.error(f"Missing {key}: {path}")
            setattr(args, key, path)
    if args.cfd_iterations < 2 or args.coupling_iterations < 1:
        parser.error("Require at least 2 fine cycles per exchange and 1 coupling exchange")
    if not 0.002 <= args.thickness <= 0.05:
        parser.error("Require shell thickness within [0.002, 0.05] m")
    # Reject incompatible meshes before launching native coarse-grid startup.
    mesh = CGNSMesh.from_file(args.grid, coordinate_zones=())
    factor = 2 ** (args.levels - 1)
    bad = [z.name for z in mesh.zones if
           any((n-1) % factor for n in z.dimensions) or
           any((n-1)//factor + 1 < 3 for n in z.dimensions[1:])]
    if bad:
        parser.error(f"Grid cannot support {args.levels} levels: {bad}. Original L2 needs --levels 1.")
    args.output = args.output.resolve()
    if (args.output / "out.json").exists():
        parser.error("out.json already exists; use a fresh --output directory")
    return args


def main():
    args = parse_args()
    (args.output / "structure").mkdir(parents=True, exist_ok=True)
    # Edit this dictionary for your flight condition. SI units; Reynolds per metre.
    aero_options = {
        "mach": 0.85,
        "alpha": 2.0,                         # degrees
        "temperature": 223.25209264797854,    # K
        "freestream_pressure": 26499.89786492552,  # Pa
        "reynolds": 7223294.8822458545,
        "sref": 45.5,                        # m^2
        "cref": 3.25,                        # m
        "bref": 1.0,                         # m
        "moment_center": (0.0, 0.0, 0.0),    # m
        "turbulence": "sa",
        "cfl": 1.5,
        "limiter": 4,
    }
    # Runtime settings: coarse startup once, then retained W-cycles on every exchange.
    aero_options.update({
        "input_grid": str(args.grid),
        "output_directory": str(args.output),
        "resident": True,
        "iterations": args.cfd_iterations,
        "multigrid_levels": args.levels,
        "mesh_sequence": True,
        "coarse_grid_iterations": COARSE_CYCLES,
        "multigrid_cycle": 2,
        "writeflowsolution": False,
    })
    model = AeroelasticModel(settings=args, aero_options=aero_options)
    problem = om.Problem(model=model, reports=False)
    try:
        problem.setup()
        problem.run_model()
        history = problem.model.aero.get_iteration_history(SCENARIO)
        if not history or any(r["startup_cycles"] != (args.levels-1)*COARSE_CYCLES or
                              r["initializations"] != 1 or
                              r["total_cycles"] != r["cfd_cycles"] + r["startup_cycles"] for r in history):
            raise RuntimeError("Unexpected multigrid startup or native initialization count")
        # Force data were cached before the scenario closed its native workspace.
        forces = problem.model.aero.get_force_summary(SCENARIO)
        result = dict(scenario=SCENARIO, warper="idwarp", levels=args.levels,
                      coarse_cycles_per_level=COARSE_CYCLES,
                      fine_cycles=history[-1]["cfd_cycles"],
                      startup_cycles=history[-1]["startup_cycles"],
                      total_cycles=history[-1]["total_cycles"],
                      coupling_evaluations=len(history),
                      convergence_required=not args.fixed_budget,
                      settings={**vars(args), "aero_options": aero_options},
                      **forces)
        # Total is all-wall pressure+viscous force, NOT the sum of possibly overlapping groups.
        if problem.comm.rank == 0:
            temporary = args.output / "out.json.tmp"
            temporary.write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + "\n")
            temporary.replace(args.output / "out.json")
            print(f"Wrote {args.output / 'out.json'}", flush=True)
    finally:
        if hasattr(problem.model, "aero"):
            problem.model.aero.close()
        problem.cleanup()


if __name__ == "__main__":
    main()
