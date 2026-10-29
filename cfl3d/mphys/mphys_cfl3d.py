"""Steady aerostructural analysis using distributed CFL3D nodal loads.

Derivatives are placeholders and warn when invoked. This module is not suitable
for gradient-based optimization. Importing the main cfl3d package does not load
OpenMDAO, MPhys or IDWarp.
"""

from copy import deepcopy
from pathlib import Path
import json
import warnings

import numpy as np
import openmdao.api as om
from mphys import Builder, MPhysVariables

from .. import CFL3D
from ..solver import SolveFailure

SURFACE = MPhysVariables.Aerodynamics.Surface


class _ForwardOnly(om.ExplicitComponent):
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        warnings.warn(
            "CFL3D MPhys derivatives are not implemented; derivative contributions "
            "are skipped and resulting gradients are invalid. Use run_model() only.",
            RuntimeWarning, stacklevel=2,
        )


class _Mesh(om.IndepVarComp):
    def initialize(self):
        self.options.declare("coordinates", recordable=False)

    def setup(self):
        self.add_output(SURFACE.Mesh.COORDINATES,
                        val=self.options["coordinates"].ravel(), units="m",
                        distributed=True, tags=["mphys_coordinates"])


class _Session:
    def __init__(self, builder, name):
        self.builder, self.name = builder, name
        self.solver = None
        self.adapter = None
        self.loads = None
        self.functions = None
        self.alpha = None
        self.history = []
        self.cycles = 0

    def _check_result(self, result):
        if isinstance(result, SolveFailure):
            raise om.AnalysisError(f"CFL3D geometry/solve failure: {result}")

    def advance(self, coordinates, alpha):
        b = self.builder
        self.loads = self.functions = None
        initialized_cycles = 0
        # Native COMMON state permits one resident owner per process. Scenarios
        # finish/close in post-processing; interleaving unfinished ones is an error.
        if b._active not in (None, self):
            raise RuntimeError("Finish the active CFL3D scenario before entering another on this communicator")
        if self.solver is None:
            options = b._scenario_options(self.name)
            options["alpha"] = alpha
            cycle_budget = options["iterations"]
            if cycle_budget < 2:
                raise ValueError("The first MPhys evaluation needs at least two CFD cycles (initialization + update)")
            options["iterations"] = 1
            self.solver = CFL3D(options=options, comm=b.comm,
                                boundary_overrides=b.boundary_overrides, groups=b.groups)
            b._active = self
            self.alpha = alpha
            # Existing public coordinate updates require a paused resident context.
            self._check_result(self.solver.solve(final=False))
            initialized_cycles = 1
            self.history = []
            self.cycles = 0
            self.solver.setOptions({"iterations": cycle_budget})
            surface = self.solver.getSurfaceMesh(b.group_name)
            self.solver._checked(lambda: np.testing.assert_array_equal(surface["node_ids"], b.surface["node_ids"]))
            if b.warper == "idwarp":
                from idwarp import USMesh
                from ..mesh import IDWarpAdapter
                warper = USMesh(options=b.mesh_options, comm=b.comm)
                self.adapter = IDWarpAdapter(warper, self.solver.mesh, surface, self.solver.comm)
            else:
                from ..mesh import CFL3DMeshAdapter
                self.adapter = CFL3DMeshAdapter(
                    self.solver.mesh, surface, self.solver.comm,
                    boundary_overrides=b.boundary_overrides, **b.mesh_options,
                )
        elif alpha != self.alpha:
            raise om.AnalysisError("Changing alpha within an active CFL3D coupling session is unsupported; finish it first")

        surface = self.solver.getSurfaceMesh(b.group_name)
        target = np.asarray(coordinates).reshape(-1, 3)
        def validate():
            if np.iscomplexobj(target) or target.shape != surface["coordinates"].shape or not np.isfinite(target).all():
                raise ValueError("CFL3D coupling coordinates must be finite real owned (N,3) values")
        self.solver._checked(validate)
        changed = not np.array_equal(target, surface["coordinates"])
        if self.solver.comm is not None:
            changed = any(self.solver.comm.allgather(changed))
        if changed:
            surface["coordinates"] = target.copy()
            candidate = self.adapter.warp(surface)
            self.solver.setVolumeCoordinates(candidate)
        cycle_budget = self.solver.options.iterations
        self.solver.setOptions({"iterations": cycle_budget - initialized_cycles})
        try:
            result = self.solver.solve(final=False)
            self._check_result(result)
        finally:
            self.solver.setOptions({"iterations": cycle_budget})
        self.cycles += initialized_cycles + len(result.history)
        self.loads = self.solver.getSurfaceForces(b.group_name)
        values = self.solver.getSolution(b.group_name)
        prefix = "" if b.group_name is None else b.group_name + "_"
        self.functions = {name: values[prefix + key] for name, key in
                          (("cl", "CL"), ("cd", "CD"), ("lift", "L"), ("drag", "D"))}
        record = dict(coupling_evaluation=len(self.history)+1, cfd_cycles=self.cycles,
                      residual=float(result.history[-1, 0]), **self.functions)
        runtime = self.solver.getRuntimeInfo()
        record.update(startup_cycles=int(runtime["startup_cycles"]),
                      total_cycles=int(runtime["cycles"]),
                      initializations=int(runtime["initializations"]))
        self.history.append(record)
        def write_history():
            if self.solver.comm is None or self.solver.comm.rank == 0:
                path = Path(self.solver.options.output_directory) / "coupling-history.jsonl"
                with path.open("w" if len(self.history) == 1 else "a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(f"CFL3D_MPHYS_HISTORY {self.name} " + json.dumps(record), flush=True)
        self.solver._checked(write_history)
        return self.loads

    def finish(self):
        if self.functions is None:
            raise om.AnalysisError("CFL3D post-processing requires a successful coupling solve")
        if self.solver is not None:
            self.solver.finalizeSolution()
            self.close()
        return self.functions

    def close(self):
        if self.solver is not None:
            self.solver.close()
            self.solver = None
            self.adapter = None
        if self.builder._active is self:
            self.builder._active = None


class _Aero(_ForwardOnly):
    def initialize(self):
        self.options.declare("session", recordable=False)

    def setup(self):
        session = self.options["session"]
        b = session.builder
        self.add_input(SURFACE.COORDINATES, val=b.surface["coordinates"].ravel(),
                       distributed=True, units="m", tags=["mphys_coupling"])
        self.add_input("aoa", val=b._scenario_options(session.name).get("alpha", 0.0),
                       units="deg", tags=["mphys_input"])
        self.add_output(SURFACE.LOADS, val=np.zeros(3*b.get_number_of_nodes()),
                        distributed=True, units="N", tags=["mphys_coupling"])

    def compute(self, inputs, outputs):
        if np.iscomplexobj(inputs["aoa"]):
            raise om.AnalysisError("CFL3D MPhys complex-step derivatives are unsupported")
        outputs[SURFACE.LOADS] = self.options["session"].advance(
            inputs[SURFACE.COORDINATES], float(inputs["aoa"][0])).ravel()


class _Post(_ForwardOnly):
    def initialize(self):
        self.options.declare("session", recordable=False)

    def setup(self):
        n = self.options["session"].builder.get_number_of_nodes()
        self.add_input(SURFACE.LOADS, val=np.zeros(3*n), distributed=True,
                       units="N", tags=["mphys_coupling"])
        for name in ("cl", "cd", "lift", "drag"):
            self.add_output(name, val=0.0, units="N" if name in ("lift", "drag") else None,
                            tags=["mphys_result"])

    def compute(self, inputs, outputs):
        for key, value in self.options["session"].finish().items():
            outputs[key] = value


class CFL3DBuilder(Builder):
    """Forward-only MPhys builder, with sequential resident scenario lifetimes.

    options/scenario_options use CFL3D option names, not ADflow option names.
    IDWarp is loaded only on first solve. Its mesh_options use USMesh names.
    warper='cfl3d' instead accepts CFL3DMeshAdapter keyword options.
    Call close() collectively in a finally block, including on analysis failures.
    """

    def __init__(self, options, *, scenario_options=None, mesh_options=None,
                 warper="idwarp", group_name=None, boundary_overrides=None, groups=None):
        self.options = deepcopy(dict(options))
        if not self.options.get("resident", True):
            raise ValueError("CFL3D MPhys requires resident=True for coordinate updates")
        self.options["resident"] = True
        if "input_grid" not in self.options:
            raise ValueError("CFL3DBuilder requires options['input_grid']")
        self.options["input_grid"] = str(Path(self.options["input_grid"]).resolve())
        self.options["output_directory"] = str(Path(self.options.get("output_directory", "cfl3d-output")).resolve())
        if warper not in ("idwarp", "cfl3d"):
            raise ValueError("warper must be 'idwarp' or 'cfl3d'")
        self.warper = warper
        self.mesh_options = deepcopy(mesh_options or {})
        if warper == "idwarp":
            self.mesh_options.setdefault("gridFile", self.options["input_grid"])
            self.mesh_options["gridFile"] = str(Path(self.mesh_options["gridFile"]).resolve())
            if self.mesh_options["gridFile"] != self.options["input_grid"]:
                raise ValueError("IDWarp and CFL3D must use the same input grid")
        self.scenario_options = deepcopy(scenario_options or {})
        self.group_name = group_name
        self.boundary_overrides, self.groups = boundary_overrides, groups
        self._sessions = {}
        self._active = None
        self.surface = None

    def initialize(self, comm):
        if self.surface is not None:
            raise RuntimeError("Initialize each CFL3DBuilder only once")
        self.comm = comm
        with CFL3D(options=self.options, comm=comm, boundary_overrides=self.boundary_overrides,
                   groups=self.groups) as solver:
            self.surface = solver.getSurfaceMesh(self.group_name)
        # Only owned reference surface coordinates are retained here for MPhys x0.

    def _scenario_options(self, name):
        options = dict(self.options)
        overrides = self.scenario_options.get(name, {})
        if any(key in overrides for key in ("input_grid", "resident", "restartFile")):
            raise ValueError("Scenario overrides cannot change grid, resident mode or restartFile")
        options.update(overrides)
        options["name"] = name or "aero"
        options["output_directory"] = str(Path(options["output_directory"]) / (name or "aero"))
        return options

    def _session(self, name):
        if self.surface is None:
            raise RuntimeError("Call builder.initialize(comm) first")
        if name not in self._sessions:
            self._sessions[name] = _Session(self, name)
        return self._sessions[name]

    def get_number_of_nodes(self):
        return len(self.surface["node_ids"])

    def get_ndof(self):
        return 3

    def get_iteration_history(self, scenario_name=None):
        """Small per-evaluation convergence/load records for the latest session."""
        return deepcopy(self._session(scenario_name).history)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return _Mesh(coordinates=self.surface["coordinates"].copy())

    def get_coupling_group_subsystem(self, scenario_name=None):
        return _Aero(session=self._session(scenario_name))

    def get_post_coupling_subsystem(self, scenario_name=None):
        return _Post(session=self._session(scenario_name))

    def close(self):
        for session in self._sessions.values():
            session.close()
