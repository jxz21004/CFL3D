"""Python-owned configuration and arrays passed directly to the native solver."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path
import threading
from numbers import Real
import numpy as np
from .cgns import CGNSMesh

_NATIVE_LOCK = threading.Lock()
_MODELS = {"euler": 0, "laminar": 1, "sa": 5, "sst": 7, "komega": 6}
_SURFACE_COEFFICIENTS = ("CL", "CD", "CX", "CY", "CZ", "Cmx", "Cmy", "Cmz", "CDp", "CDv")
_BC_TYPES = {"BCFarfield": (1003, ()), "BCSymmetryPlane": (1001, ()),
             "BCWallInviscid": (1005, ()), "BCOutflowSupersonic": (1002, ()),
             "BCWallViscous": (2004, (0., 0.)), "BCWallViscousHeatFlux": (2004, (0., 0.))}


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum \
            or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum if maximum is not None else 'unbounded'}]")


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


def _directions(value, name, choices=None):
    values = value if isinstance(value, tuple) else (value,) * 3
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or an (i,j,k) tuple")
    for item in values:
        if choices is None:
            _finite(item, name)
        elif isinstance(item, bool) or not isinstance(item, int) or item not in choices:
            raise ValueError(f"{name} entries must be integers in {choices}")
    return values


def _smoothing(values, levels, name):
    if not isinstance(values, tuple) or len(values) != levels:
        raise ValueError(f"{name} must be a tuple with {levels} entries, coarsest to finest")
    for value in values:
        # 旧内核的每层更新次数工作数组最多容纳 20 次。
        _integer(value, name, 1, 20)


@dataclass(frozen=True)
class MultigridStage:
    """一次网格序列阶段；levels 为活跃层数，smoothing_steps 按粗到细对应 MIT。"""
    iterations: int
    levels: int
    first_order_iterations: int = 0
    smoothing_steps: tuple[int, ...] | None = None

    def validate(self, available_levels):
        _integer(self.iterations, "stage.iterations", 1)
        _integer(self.levels, "stage.levels", 1, available_levels)
        _integer(self.first_order_iterations, "stage.first_order_iterations", 0, self.iterations)
        if self.smoothing_steps is not None:
            _smoothing(self.smoothing_steps, self.levels, "stage.smoothing_steps")


@dataclass(frozen=True)
class SolverOptions:
    """Python 工况与迭代控制；单位换算和原生数组布局集中在 _arrays()。"""
    mach: float = 0.5
    alpha: float = 0.0
    beta: float = 0.0
    reynolds: float = 1e6  # Reynolds number per mesh length unit
    temperature: float = 288.15  # Kelvin
    turbulence: str = "sa"
    cfl: float = 2.0
    cfl_ramp_steps: int = 0
    cfl_ramp_factor: float = 1.0
    iterations: int = 100
    first_order_iterations: int = 0
    coarse_first_order_iterations: int = 0
    restart_interval: int | None = None
    multigrid_levels: int = 1
    mesh_sequence: bool = False
    coarse_iterations: int = 50
    multigrid_cycle: int = 2  # NGAM: 1 V, 2 W
    full_navier_stokes: bool = True
    turbulence_subiterations: int = 1
    turbulence_cfl: float = 5.0
    eddy_viscosity_limit: float = 1e5  # 按用户要求保留较保守的 Python 默认值。
    gamma: float = 1.4
    limiter: int | tuple[int, int, int] = 0
    flux: int | tuple[int, int, int] = 1
    kappa: float | tuple[float, float, float] = 1.0 / 3.0
    y_up: bool = False
    sref: float = 1.0
    cref: float = 1.0
    bref: float = 1.0
    moment_center: tuple[float, float, float] = (0., 0., 0.)
    target_cl: float | None = None
    alpha_relaxation: float = 1.0
    alpha_step_limit: float = 0.2  # degrees per native alpha update
    alpha_update_interval: int = 1
    multigrid_up_iterations: int = 0  # MTT
    multigrid_smoothing: tuple[int, ...] | None = None  # MIT, coarsest to finest
    multigrid_stages: tuple[MultigridStage, ...] | None = None
    inversion: int | tuple[int, int, int] = 1  # IDIAG, scalar or (i,j,k)
    freestream_pressure: float | None = None  # Pa; coordinates/reference lengths are metres
    input_grid: str | Path | None = None
    output_directory: str | Path = "cfl3d-output"
    writevolumesolution: bool = True
    writeTecplotSurfaceSolution: bool = True
    writeflowsolution: bool = False
    volume_filename: str | Path = "volume.cgns"
    surface_filename: str | Path = "surface.dat"
    flow_prefix: str | Path = "flow"

    @classmethod
    def from_dict(cls, options):
        """Normalize a user dictionary without mutating it; reject misspelled keys."""
        values = deepcopy(dict(options))
        unknown = set(values) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown CFL3D options: {sorted(unknown, key=str)}")
        for name in ("limiter", "flux", "inversion", "kappa", "moment_center", "multigrid_smoothing"):
            if isinstance(values.get(name), list):
                values[name] = tuple(values[name])
        if values.get("multigrid_stages") is not None:
            stages = []
            for stage in values["multigrid_stages"]:
                if isinstance(stage, Mapping):
                    stage = dict(stage)
                    if isinstance(stage.get("smoothing_steps"), list):
                        stage["smoothing_steps"] = tuple(stage["smoothing_steps"])
                    stage = MultigridStage(**stage)
                stages.append(stage)
            values["multigrid_stages"] = tuple(stages)
        return cls(**values)

    def validate(self):
        for name in ("input_grid", "output_directory", "volume_filename", "surface_filename", "flow_prefix"):
            value = getattr(self, name)
            if value is None and name == "input_grid":
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{name} must be a nonempty path")
        for name in ("writevolumesolution", "writeTecplotSurfaceSolution", "writeflowsolution"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        ints = ("iterations", "multigrid_levels", "coarse_iterations", "multigrid_cycle",
                "turbulence_subiterations", "cfl_ramp_steps",
                "first_order_iterations", "coarse_first_order_iterations")
        for name in ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.iterations < 1 or self.coarse_iterations < 0 or self.turbulence_subiterations < 1:
            raise ValueError("Invalid iteration count")
        if self.cfl_ramp_steps < 0 or not 0 <= self.first_order_iterations <= self.iterations:
            raise ValueError("Require nonnegative CFL ramp steps and first_order_iterations <= iterations")
        if not 0 <= self.coarse_first_order_iterations <= self.coarse_iterations:
            raise ValueError("coarse_first_order_iterations must be within coarse_iterations")
        if self.restart_interval is not None and (isinstance(self.restart_interval, bool)
                or not isinstance(self.restart_interval, int) or self.restart_interval < 1):
            raise ValueError("restart_interval must be a positive integer or None")
        if not 1 <= self.multigrid_levels <= 5 or self.multigrid_cycle not in (1, 2):
            raise ValueError("multigrid_levels must be 1..5; cycle must be 1 (V) or 2 (W)")
        if self.turbulence.lower() not in _MODELS:
            raise ValueError(f"Supported models: {', '.join(_MODELS)}")
        for name in ("mach", "alpha", "beta", "reynolds", "temperature", "cfl", "turbulence_cfl",
                     "eddy_viscosity_limit", "gamma", "sref", "cref", "bref", "cfl_ramp_factor",
                     "alpha_relaxation", "alpha_step_limit"):
            value = getattr(self, name)
            _finite(value, name)
        if self.mach <= 0 or self.temperature <= 0 or self.cfl <= 0 or self.gamma <= 1:
            raise ValueError("Require positive Mach, temperature, CFL and gamma > 1")
        if self.cfl_ramp_factor < 1:
            raise ValueError("cfl_ramp_factor must be >= 1")
        for name in ("mesh_sequence", "full_navier_stokes", "y_up"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if min(self.sref, self.cref, self.bref, self.turbulence_cfl, self.eddy_viscosity_limit) <= 0:
            raise ValueError("Reference scales and turbulence controls must be positive")
        if self.reynolds <= 0 and self.turbulence.lower() != "euler":
            raise ValueError("Viscous models require positive Reynolds number")
        if len(self.moment_center) != 3 or not np.isfinite(self.moment_center).all():
            raise ValueError("moment_center must contain three finite coordinates")
        _directions(self.limiter, "limiter", (0, 1, 2, 3, 4))
        _directions(self.flux, "flux", (0, 1))
        _directions(self.inversion, "inversion", (0, 1))
        _directions(self.kappa, "kappa")
        _integer(self.alpha_update_interval, "alpha_update_interval", 1)
        if min(self.alpha_relaxation, self.alpha_step_limit) <= 0:
            raise ValueError("Alpha relaxation and step limit must be positive")
        if self.target_cl is not None:
            _finite(self.target_cl, "target_cl")
            if self.target_cl == 99999.0:
                raise ValueError("99999 is the native disabled-CL sentinel; use target_cl=None")
        if self.freestream_pressure is not None:
            _finite(self.freestream_pressure, "freestream_pressure")
            if self.freestream_pressure <= 0:
                raise ValueError("freestream_pressure must be positive (Pa)")
        _integer(self.multigrid_up_iterations, "multigrid_up_iterations", 0, 19)
        if self.multigrid_smoothing is not None:
            _smoothing(self.multigrid_smoothing, self.multigrid_levels, "multigrid_smoothing")
        schedule = self.stage_schedule()
        if not 1 <= len(schedule) <= self.multigrid_levels:
            raise ValueError("Use between one and multigrid_levels mesh-sequence stages")
        for index, stage in enumerate(schedule):
            if not isinstance(stage, MultigridStage):
                raise ValueError("multigrid_stages must contain MultigridStage objects")
            # 原生序列每阶段细化一次；阶段数决定从哪一级网格开始。
            available = self.multigrid_levels - len(schedule) + index + 1
            stage.validate(available)
            if max(self.stage_smoothing(stage, available)) + self.multigrid_up_iterations > 20:
                raise ValueError("MIT + MTT must not exceed the native limit of 20")

    def stage_schedule(self):
        if self.multigrid_stages is not None:
            if not isinstance(self.multigrid_stages, tuple):
                raise ValueError("multigrid_stages must be an immutable tuple")
            return self.multigrid_stages
        if self.mesh_sequence:
            return tuple(MultigridStage(self.coarse_iterations, level,
                                       self.coarse_first_order_iterations)
                         for level in range(1, self.multigrid_levels)) + (
                MultigridStage(self.iterations, self.multigrid_levels, self.first_order_iterations),)
        return (MultigridStage(self.iterations, self.multigrid_levels, self.first_order_iterations),)

    def stage_smoothing(self, stage, available_levels):
        # 默认 MIT 按全局粗到细编号；只取本阶段实际参与循环的网格层。
        template = self.multigrid_smoothing or (1,) * self.multigrid_levels
        return stage.smoothing_steps or template[available_levels-stage.levels:available_levels]


def _face(point_range, dims):
    p = np.asarray(point_range)
    if p.shape != (3, 2) or not np.issubdtype(p.dtype, np.integer):
        raise ValueError("PointRange must be an integer array of shape (3,2)")
    if (p < 1).any() or (p > np.asarray(dims)[:, None]).any():
        raise ValueError("PointRange exceeds zone dimensions")
    fixed = np.where(p[:, 0] == p[:, 1])[0]
    if len(fixed) != 1 or p[fixed[0], 0] not in (1, dims[fixed[0]]):
        raise ValueError("PointRange must describe a rectangular boundary face")
    axis = int(fixed[0]); face = 2 * axis + int(p[axis, 0] != 1)
    tangents = [a for a in range(3) if a != axis]
    bounds = np.sort(p[tangents], axis=1)
    return face, tangents, tuple(map(int, bounds.flat))


@dataclass
class SolverResult:
    fields: dict[str, np.ndarray]  # zone -> (ni-1,nj-1,nk-1,5): rho,u,v,w,p
    history: np.ndarray  # residual,CL,CD,CX,CY,CMX,CMY,CMZ
    output_directory: Path
    final_alpha: float  # actual final angle in degrees, including native CL control
    surface_coefficients: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    wall_fields: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)  # Cp,T[K],Cfx,Cfy,Cfz,y+

    wall_forces: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)  # pressure XYZ, viscous XYZ / (q_inf*Sref)

    @property
    def coefficients(self):
        if not len(self.history):
            return {}
        coefficients = dict(zip(("CL", "CD", "CX", "CY", "CMX", "CMY", "CMZ"), self.history[-1, 1:]))
        if self.surface_coefficients:
            total = np.sum(list(self.surface_coefficients.values()), axis=0)
            coefficients.update(CDp=float(total[8]), CDv=float(total[9]))
        return coefficients

    def save(self, filename, *, include_volume=True):
        values = {"history": self.history, "final_alpha": np.asarray(self.final_alpha)}
        if include_volume:
            values.update({f"zone_{name}": field for name, field in self.fields.items()})
        keys = sorted(self.surface_coefficients)
        values["surface_names"] = np.asarray(keys, dtype=str).reshape(-1, 2)
        values["surface_coefficients"] = np.asarray([self.surface_coefficients[k] for k in keys]).reshape(-1, 10)
        values["surface_coefficient_names"] = np.asarray(_SURFACE_COEFFICIENTS)
        np.savez_compressed(filename, **values)


class CFL3D:
    """Native solver with an optional collective mpi4py communicator.

    Solves are serialized because the legacy core uses COMMON state. After a
    native failure, start a fresh Python process before attempting another solve.
    """

    def __init__(self, mesh=None, options=None, *, boundary_overrides=None, comm=None, groups=None):
        if options is None and isinstance(mesh, (Mapping, SolverOptions)):
            options, mesh = mesh, None
        self.comm = comm
        self._last_result = None
        self._group_definitions = {}
        self._lift_distributions = []
        self._slice_requests = []
        self._surface_analysis_mesh = None
        if comm is not None:
            from mpi4py import MPI
            if comm == MPI.COMM_NULL or comm.Is_inter() or comm.Get_size() < 2:
                raise ValueError("CFL3D MPI needs an intracommunicator with a host and at least one worker")

        def load():
            self.options = SolverOptions.from_dict(options) if isinstance(options, Mapping) else options
            if self.options is None:
                self.options = SolverOptions()
            if not isinstance(self.options, SolverOptions):
                raise ValueError("options must be a dictionary or SolverOptions")
            self.options.validate()
            if mesh is not None and self.options.input_grid is not None:
                raise ValueError("Specify the mesh argument or options.input_grid, not both")
            input_mesh = mesh if mesh is not None else self.options.input_grid
            if input_mesh is None:
                raise ValueError("Set options['input_grid'] or provide a mesh")
            self.mesh = input_mesh if isinstance(input_mesh, CGNSMesh) else CGNSMesh.from_file(input_mesh)
            self.boundary_overrides = boundary_overrides or {}
            self.options.validate()
            return self.mesh, self.options, self.boundary_overrides, {} if groups is None else groups

        if comm is None:
            _, _, _, group_definitions = load()
        else:
            payload = None
            error = None
            # 只让主控读文件。先广播读取错误，再一起退出，避免其他 rank 等不到数据。
            if comm.Get_rank() == 0:
                try:
                    payload = load()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            error, payload = comm.bcast((error, payload), root=0)
            if error:
                raise RuntimeError(f"CFL3D root configuration failed: {error}")
            self.mesh, self.options, self.boundary_overrides, group_definitions = payload
            if comm.Get_size() - 1 > len(self.mesh.zones):
                raise ValueError("This MPI interface currently requires at least one zone per worker")
        self.addGroups(group_definitions)

    def _output_path(self, filename, default):
        if self._last_result is None:
            raise RuntimeError("Call solve() successfully before writing a solution")
        path = Path(filename) if filename is not None else self._last_output_directory / default
        if self._last_mesh.source and path.resolve() == Path(self._last_mesh.source).resolve():
            raise ValueError("Solution output must not overwrite the input CGNS mesh")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def writeVolumeSolution(self, filename=None):
        """Write cached grid + cell fields to CGNS/ADF, independently of automatic flags."""
        if self.comm is not None and self.comm.Get_rank() != 0:
            return None
        path = self._output_path(filename, self._last_options.volume_filename if self._last_result else "volume.cgns")
        from .solution_io import write_volume
        write_volume(path, self._last_mesh, self._last_fields, self._last_options)
        return path

    def writeTecplotSurfaceSolution(self, filename=None, groupName=None):
        """Write wall quads with Cp,T[K],Cf components,y+; optional group selection."""
        if self.comm is not None and self.comm.Get_rank() != 0:
            return None
        path = self._output_path(filename, self._last_options.surface_filename if self._last_result else "surface.dat")
        from .solution_io import write_surface, write_lift_distributions, write_slices
        members = self._group_members(groupName)
        # Validate registered analyses before writing any files. Each request uses its own group.
        distributions = [self.getLiftDistribution(i) for i in range(self.nLiftDist)]
        slices = self.getSlices()
        companions = {}
        for label, data in (("lift", distributions), ("slices", slices)):
            if data:
                companions[label] = self._output_path(path.with_name(
                    path.stem+"_"+label+(path.suffix or ".dat")), "")
        write_surface(path, self._last_mesh, self._last_wall_fields, members)
        if distributions:
            write_lift_distributions(companions["lift"], distributions)
        if slices:
            write_slices(companions["slices"], slices)
        return path

    def writeFlowSolution(self, prefix=None):
        """Write cached cell-center PLOT3D .x/.q in CFL3D binary IPtype=1 format."""
        if self.comm is not None and self.comm.Get_rank() != 0:
            return None
        prefix = self._output_path(prefix, self._last_options.flow_prefix if self._last_result else "flow")
        from .solution_io import write_flow
        for suffix in (".x", ".q"):
            self._output_path(str(prefix)+suffix, "")
        return write_flow(prefix, self._last_mesh, self._last_fields, self._last_options, self._last_final_alpha)

    def _cgns_surface_groups(self):
        families, zones, boundaries, qualified = {}, {}, {}, {}
        for zone in self.mesh.zones:
            for boundary in zone.boundaries:
                key = (zone.name, boundary.name)
                path = f"{zone.name}/{boundary.name}"
                if path in qualified:
                    raise ValueError(f"Duplicate CGNS boundary path {path}")
                qualified[path] = {key}
                zones.setdefault(zone.name, set()).add(key)
                boundaries.setdefault(boundary.name, set()).add(key)
                for family in (zone.family, *zone.additional_families,
                               boundary.family, *boundary.additional_families):
                    if family:
                        families.setdefault(family, set()).add(key)
        # 同名时以 Family 为准，其次 zone、BC；完整 zone/BC 路径始终可用。
        return {**boundaries, **zones, **families, **qualified}

    def addGroups(self, groups):
        """Add {new_name: [CGNS group names, ...]}; union overlapping surfaces once."""
        if not isinstance(groups, dict):
            raise ValueError("groups must be a dictionary of new names to CGNS name lists")
        base = self._last_base_groups if self._last_result is not None else self._cgns_surface_groups()
        additions = {}
        for name, members in groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Group names must be nonempty strings")
            if name in base or name in self._group_definitions:
                raise ValueError(f"Group name {name!r} conflicts with an existing CGNS/group name")
            if not isinstance(members, (list, tuple)) or not members:
                raise ValueError(f"Group {name!r} needs a nonempty CGNS name list")
            if any(not isinstance(member, str) or member not in base for member in members):
                raise ValueError(f"Group {name!r} may only contain existing CGNS names")
            additions[name] = tuple(members)
        self._group_definitions.update(additions)
        return self

    def getGroupNames(self):
        base = self._last_base_groups if self._last_result is not None else self._cgns_surface_groups()
        return sorted(set(base) | self._group_definitions.keys())

    def getSolution(self, groupName=None):
        """Return cached native pressure+viscous coefficients and SI loads after solve.

        CL/CD/CDp/CDv use the native wind axes; CY and moments use the input
        mesh axes. L/D/Y/Dp/Dv are N, moments Mx/My/Mz are N*m.
        This is a local call: any rank may query without an MPI collective.
        """
        members = self._group_members(groupName)
        total = (np.sum([self._last_surface_coefficients[key] for key in sorted(members)], axis=0)
                 if members else np.zeros(10))
        values = {name: float(total[index]) for index, name in enumerate(_SURFACE_COEFFICIENTS)}
        options = self._last_options
        if options.freestream_pressure is None:
            raise ValueError("Set SolverOptions(freestream_pressure=... in Pa) before solve() "
                             "to obtain N and N*m; no static pressure is assumed")
        dynamic_pressure = .5 * options.gamma * options.freestream_pressure * options.mach**2
        scale = dynamic_pressure * options.sref
        yref, zref = (options.bref, options.cref) if options.y_up else (options.cref, options.bref)
        result = {name: values[name] for name in ("CL", "CD", "CY", "Cmx", "Cmy", "Cmz", "CDp", "CDv")}
        result.update(L=values["CL"]*scale, D=values["CD"]*scale, Y=values["CY"]*scale,
                      Dp=values["CDp"]*scale, Dv=values["CDv"]*scale,
                      Mx=values["Cmx"]*scale*options.bref,
                      My=values["Cmy"]*scale*yref, Mz=values["Cmz"]*scale*zref)
        result["units"] = {name: "N" for name in ("L", "D", "Y", "Dp", "Dv")}
        result["units"].update({name: "N*m" for name in ("Mx", "My", "Mz")})
        return result

    def _group_members(self, groupName):
        if self._last_result is None:
            raise RuntimeError("Call solve() successfully before querying solution groups")
        if groupName is None:
            members = set(self._last_surface_coefficients)
        elif isinstance(groupName, str) and groupName in self._last_base_groups:
            members = self._last_base_groups[groupName]
        elif isinstance(groupName, str) and groupName in self._group_definitions:
            members = set().union(*(self._last_base_groups[name]
                                   for name in self._group_definitions[groupName]))
        else:
            raise ValueError(f"Unknown surface group {groupName!r}; inspect getGroupNames()")
        return members

    @property
    def nLiftDist(self):
        return len(self._lift_distributions)

    @property
    def nSlice(self):
        return len(self._slice_requests)

    def _analysis_surface(self, require_solution=False):
        from .surface_analysis import SurfaceMesh
        if require_solution and self._last_result is None:
            raise RuntimeError("Call solve() successfully before querying surface analysis")
        if self._last_result is not None:
            if self._surface_analysis_mesh is None:
                self._surface_analysis_mesh = SurfaceMesh(
                    self._last_mesh, self._last_wall_fields, self._last_wall_forces)
            return self._surface_analysis_mesh
        # Before solve, only geometry is needed to attach relative cuts to material faces.
        walls = {}
        for zone in self.mesh.zones:
            for boundary in zone.boundaries:
                key = zone.name, boundary.name
                code = self.boundary_overrides.get(key, _BC_TYPES.get(boundary.bc_type, (0, ())))[0]
                if abs(code) in (1005,1006,2004,2014,2024,2034,2016):
                    ranges = np.sort(boundary.point_range, axis=1)
                    counts = ranges[:,1]-ranges[:,0]
                    walls[key] = np.zeros((*counts[counts > 0], 6))
        return SurfaceMesh(self.mesh, walls, sref=self.options.sref)

    def _analysis_members(self, groupName, surface):
        if self._last_result is not None:
            members = self._group_members(groupName)
        else:
            base = self._cgns_surface_groups()
            if groupName is None:
                members = set(surface.patches)
            elif isinstance(groupName, str) and groupName in base:
                members = base[groupName]
            elif isinstance(groupName, str) and groupName in self._group_definitions:
                members = set().union(*(base[name] for name in self._group_definitions[groupName]))
            else:
                raise ValueError(f"Unknown surface group {groupName!r}; inspect getGroupNames()")
        members = frozenset(members) & surface.patches.keys()
        surface.selected(members)  # Reject groups with no walls before registering requests.
        return frozenset(members)

    def addLiftDistribution(self, nSegments, direction, groupName=None):
        """Register conservative strip loads along x/y/z (typically 150-250 strips).

        None selects all walls; CGNS names and custom groups are supported.
        Output accompanies writeTecplotSurfaceSolution(); query with getLiftDistribution().
        """
        from .surface_analysis import direction_index
        _integer(nSegments, "nSegments", 1)
        axis = direction_index(direction)
        surface = self._analysis_surface()
        members = self._analysis_members(groupName, surface)
        lo, hi = surface.bounds(members)
        if hi[axis]-lo[axis] <= surface.tolerance(members):
            raise ValueError("Selected walls have zero extent in the distribution direction")
        name = f"LiftDist_{self.nLiftDist+1:02d} {groupName or 'AllWalls'}: {direction.lower()} normal"
        self._lift_distributions.append(dict(nSegments=nSegments, direction=direction.lower(),
                                             groupName=groupName, name=name))
        return name

    def addSlices(self, direction, positions, sliceType="relative", groupName=None):
        """Register Cp sections at axis positions in metres, before or after solve().

        Relative cuts attach to the current surface triangles and follow subsequent
        geometry changes. Absolute cuts intersect the fixed plane on every output.
        Both return face-centered Cp/T/Cf/y+ and connected line segments.
        """
        from .surface_analysis import direction_index, slice_references
        axis = direction_index(direction)
        if not isinstance(sliceType, str) or sliceType.lower() not in ("relative", "absolute"):
            raise ValueError("sliceType must be 'relative' or 'absolute'")
        sliceType = sliceType.lower()
        positions = np.asarray(positions)
        if positions.dtype.kind not in "iuf" or positions.ndim > 1 or not positions.size:
            raise ValueError("positions must be a finite scalar or nonempty 1D array in metres")
        positions = np.atleast_1d(positions).astype(float)
        if not np.isfinite(positions).all():
            raise ValueError("positions must be finite")
        surface = self._analysis_surface()
        members = self._analysis_members(groupName, surface)
        requests = []
        for i, position in enumerate(positions):
            name = (f"Slice_{self.nSlice+i+1:04d} {groupName or 'AllWalls'} "
                    f"{sliceType.capitalize()} {direction.lower()} = {position:.8f}")
            references = slice_references(surface, members, axis, position) if sliceType == "relative" else None
            requests.append(dict(name=name, direction=direction.lower(), position=float(position),
                                 sliceType=sliceType, groupName=groupName,
                                 references=references, members=members))
        self._slice_requests.extend(requests)
        return [request["name"] for request in requests]

    def getLiftDistribution(self, index=0):
        """Return strip coefficients and dCL/ds, dCD/ds, dCDp/ds, dCDv/ds [1/m].

        With freestream_pressure, also return dL/ds, dD/ds, dDp/ds, dDv/ds [N/m].
        Coefficients use the global reference area, without local chord normalization.
        This query is local and does not call MPI or the native solver.
        """
        from .surface_analysis import direction_index, lift_distribution
        surface = self._analysis_surface(require_solution=True)
        _integer(index, "index", 0)
        if index >= self.nLiftDist:
            raise ValueError("Lift distribution index is out of range; call addLiftDistribution()")
        request = self._lift_distributions[index]
        members = self._analysis_members(request["groupName"], surface)
        options = self._last_options
        scale = (None if options.freestream_pressure is None else
                 .5*options.gamma*options.freestream_pressure*options.mach**2*options.sref)
        data = lift_distribution(surface, members, request["nSegments"],
                                 direction_index(request["direction"]), self._last_force_angles,
                                 options.y_up, scale)
        return {**request, **data, "force_angles": np.rad2deg(self._last_force_angles).copy()}

    def getSlices(self, index=None):
        """Return all registered cuts as a list, or one cut dictionary by index.

        points/coordinates are metres, segments are zero-based point indices;
        Cp/T/Cf/y+ are segment-centered. Nonintersecting cuts return empty arrays.
        """
        from .surface_analysis import direction_index, slice_references, evaluate_slice
        surface = self._analysis_surface(require_solution=True)
        if index is not None:
            _integer(index, "index", 0)
            if index >= self.nSlice:
                raise ValueError("Slice index is out of range; call addSlices()")
        result = []
        for request in (self._slice_requests if index is None else [self._slice_requests[index]]):
            members = self._analysis_members(request["groupName"], surface)
            axis = direction_index(request["direction"])
            absolute = request["sliceType"] == "absolute"
            if not absolute and members != request["members"]:
                raise ValueError("Relative slice wall membership changed; clear and register slices again")
            references = (slice_references(surface, members, axis, request["position"])
                          if absolute else request["references"])
            data = evaluate_slice(surface, members, references, axis, request["position"], absolute)
            metadata = {key: value for key, value in request.items() if key not in ("references", "members")}
            result.append({**metadata, **data})
        return result if index is None else result[0]

    def clearSurfaceAnalysis(self):
        """Remove registered distributions and cuts (e.g. before changing mesh topology)."""
        self._lift_distributions.clear()
        self._slice_requests.clear()

    def _checked(self, operation):
        """Agree on Python-side failures before entering the native collectives."""
        if self.comm is None:
            return operation()
        value = None
        error = None
        try:
            value = operation()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # 某个 rank 的 Python 异常必须在进入 Fortran 集体通信前通知所有同伴。
        errors = self.comm.allgather(error)
        if any(errors):
            raise RuntimeError("CFL3D collective operation failed: " + "; ".join(
                f"rank {rank}: {message}" for rank, message in enumerate(errors) if message))
        return value

    def _arrays(self):
        o = self.options
        o.validate()
        zones = self.mesh.zones
        dims = np.asfortranarray([z.dimensions for z in zones], dtype=np.int32).T.copy(order="F")
        model = _MODELS[o.turbulence.lower()]
        i2d = int(all(z.dimensions[0] == 2 for z in zones))
        factor = 2 ** (o.multigrid_levels - 1)
        for z in zones:
            for axis, n in enumerate(z.dimensions):
                if i2d and axis == 0:
                    continue
                if (n - 1) % factor:
                    raise ValueError(f"{z.name}: cell dimensions must be divisible by {factor} for multigrid")
        counts = np.zeros((6, len(zones)), dtype=np.int32, order="F")
        segments = {(g, f): [] for g in range(len(zones)) for f in range(6)}
        masks = {(g, f): np.zeros(tuple(zones[g].dimensions[a] - 1 for a in range(3) if a != f // 2), dtype=np.uint8)
                 for g in range(len(zones)) for f in range(6)}

        def add(g, p, code, data, key=None):
            face, tangents, bounds = _face(p, zones[g].dimensions)
            a, b, c, d = bounds
            region = masks[g, face][a-1:b-1, c-1:d-1]
            if region.size == 0 or region.any():
                raise ValueError(f"{zones[g].name}: empty or overlapping boundary/interface patch")
            region[:] = 1
            if len(data) > 12 or not np.isfinite(data).all():
                raise ValueError("Boundary data must contain <=12 finite values")
            if 2000 <= code < 3000 and not data:
                raise ValueError("2000-series BC requires data")
            if code == 2004 and (model == 0 or (i2d and face < 2)):
                raise ValueError("Viscous wall requires an active viscous direction")
            if any((v - 1) % factor for axis, pair in zip(tangents, (bounds[:2], bounds[2:]))
                   for v in pair if not (i2d and axis == 0)):
                raise ValueError("Boundary patch endpoints must align with every multigrid level")
            segments[g, face].append((code, bounds, tuple(data), key))

        names = {z.name: g for g, z in enumerate(zones)}
        connections = []
        seen = set()
        used_overrides = set()
        for g, z in enumerate(zones):
            for bc in z.boundaries:
                key = (z.name, bc.name)
                override = self.boundary_overrides.get(key)
                if override is not None:
                    used_overrides.add(key)
                    code, data = override
                    if isinstance(code, bool) or not isinstance(code, int):
                        raise ValueError("Override bctype must be an integer")
                    if code == 0:
                        raise ValueError("Interface BC 0 must come from CGNS connectivity, not a boundary override")
                else:
                    if bc.bc_type not in _BC_TYPES:
                        raise NotImplementedError(f"{key}: {bc.bc_type} needs explicit boundary_overrides[(zone,bc)]=(code,data)")
                    code, data = _BC_TYPES[bc.bc_type]
                    if bc.data:
                        if bc.bc_type == "BCWallViscousHeatFlux" and set(bc.data) == {"HeatFlux"} and np.all(bc.data["HeatFlux"] == 0):
                            pass
                        else:
                            raise NotImplementedError(f"{key}: BCDataSet needs an explicit override; it will not be discarded")
                add(g, bc.point_range, code, data, key)
            for cn in z.connections:
                if cn.donor not in names:
                    raise ValueError(f"Unknown donor zone {cn.donor}")
                if np.shape(cn.transform) != (3,) or sorted(abs(cn.transform)) != [1, 2, 3]:
                    raise ValueError("Connection Transform must be a signed permutation of 1,2,3")
                donor = names[cn.donor]
                face, axes, _ = _face(cn.point_range, z.dimensions)
                _face(cn.donor_range, zones[donor].dimensions)
                expected = cn.donor_range[:, 0].copy()
                for a in range(3):
                    t = int(cn.transform[a])
                    expected[abs(t)-1] += np.sign(t) * (cn.point_range[a, 1] - cn.point_range[a, 0])
                if not np.array_equal(expected, cn.donor_range[:, 1]):
                    raise ValueError("Connection Transform and donor range disagree")
                akey = (g, tuple(cn.point_range.flat)); bkey = (donor, tuple(cn.donor_range.flat))
                canonical = tuple(sorted((akey, bkey)))
                if canonical in seen:
                    continue
                seen.add(canonical)
                # Verify paired vertices, including permutation/reversal.
                ranges = [range(min(r), max(r)+1) for r in cn.point_range]
                for point in np.ndindex(*(len(r) for r in ranges)):
                    p = np.array([ranges[a][point[a]] for a in range(3)])
                    q = cn.donor_range[:, 0].copy()
                    for a in range(3):
                        t = int(cn.transform[a]); q[abs(t)-1] += np.sign(t) * (p[a] - cn.point_range[a, 0])
                    if not np.allclose(z.coordinates[tuple(p-1)], zones[donor].coordinates[tuple(q-1)], rtol=1e-10, atol=1e-10):
                        raise ValueError("One-to-one interface coordinates do not coincide")
                add(g, cn.point_range, 0, ())
                add(donor, cn.donor_range, 0, ())
                vary = [a+1 for a in axes]
                donor_vary = [abs(int(cn.transform[a-1])) for a in vary]
                left = [len(connections)+1, g+1, *cn.point_range[:, 0], *cn.point_range[:, 1], *vary]
                right = [len(connections)+1, donor+1, *cn.donor_range[:, 0], *cn.donor_range[:, 1], *donor_vary]
                connections.append([left, right])
        unknown = set(self.boundary_overrides) - used_overrides
        if unknown:
            raise ValueError(f"Unknown boundary override keys: {unknown}")
        bc_rows, data_rows, surface_keys = [], [], []
        for f in range(6):
            for g, z in enumerate(zones):
                if not masks[g, f].all():
                    raise ValueError(f"{z.name}: face {('i0','idim','j0','jdim','k0','kdim')[f]} is not fully covered by BCs/connectivity")
                counts[f, g] = len(segments[g, f])
                for s, (code, bounds, data, key) in enumerate(segments[g, f], 1):
                    bc_rows.append([g+1, s, code, *bounds, len(data)])
                    data_rows.append([*data, *([0.] * (12 - len(data)))])
                    surface_keys.append(key)
        self._surface_keys = tuple(surface_keys)
        blocks = np.zeros((7, len(zones)), dtype=np.int32, order="F")
        blocks[0] = o.multigrid_levels - 1
        blocks[3] = 333
        blocks[4:] = model
        if i2d:
            blocks[4] = 0
        schedule = o.stage_schedule()
        mseq = len(schedule)
        stages = np.zeros((4, mseq), dtype=np.int32, order="F")
        smoothing = np.ones((5, 5), dtype=np.int32, order="F")
        for column, stage in enumerate(schedule):
            stages[:, column] = stage.iterations, stage.levels, 0, stage.first_order_iterations
            available = o.multigrid_levels - mseq + column + 1
            smoothing[:stage.levels, column] = o.stage_smoothing(stage, available)
        xyz = np.asfortranarray(np.concatenate([z.coordinates.reshape(-1, 3, order="F") for z in zones]))
        conn = np.asarray(connections, dtype=np.int32).transpose(2, 1, 0).copy(order="F") if connections else np.zeros((10, 2, 0), dtype=np.int32, order="F")
        # 原生 REUE 使用百万 Reynolds 数、温度使用 Rankine；负 DT 表示按 CFL 选步长。
        return dict(dims=dims, counts=counts, bcs=np.array(bc_rows, dtype=np.int32).T.copy(order="F"),
                    bcdata=np.array(data_rows, dtype=np.float64).T.copy(order="F"), blocks=blocks,
                    connections=conn, flow=np.array([o.mach, o.alpha, o.beta, o.reynolds/1e6, o.temperature*1.8, int(o.y_up), 0.]),
                    reference=np.array([o.sref, o.cref, o.bref, *o.moment_center]),
                    time_control=np.array([-o.cfl, 0., o.cfl_ramp_steps, o.cfl_ramp_factor, 0., 10.]),
                    run_control=np.array([-len(zones), 0, 0, o.restart_interval or max(1, int(stages[0].sum())), 0, i2d, 1, 1], dtype=np.int32),
                    mg=np.array([mseq, int(o.multigrid_levels > 1), 0, o.multigrid_up_iterations, o.multigrid_cycle], dtype=np.int32),
                    smoothing=smoothing,
                    stages=stages, keywords=np.array([int(o.full_navier_stokes and model != 0), o.turbulence_subiterations,
                        o.turbulence_cfl, o.eddy_viscosity_limit, o.gamma,
                        99999.0 if o.target_cl is None else o.target_cl,
                        o.alpha_relaxation, o.alpha_step_limit, o.alpha_update_interval]),
                    scheme=np.array([*_directions(o.inversion, "inversion"),
                                     *_directions(o.limiter, "limiter")], dtype=np.int32),
                    flux=np.array(_directions(o.flux, "flux"), dtype=np.int32),
                    rkap=np.array(_directions(o.kappa, "kappa")),
                    npoints=len(xyz), ncells=int(np.prod(dims-1, axis=0).sum())), xyz

    def solve(self, *, output_directory=None):
        self._last_result = None  # Failed solves must not expose an older run's loads.
        self._surface_analysis_mesh = None
        arrays, xyz = self._checked(self._arrays)
        base_groups = self._checked(self._cgns_surface_groups)
        def native_module():
            from importlib import import_module
            name = "_native_mpi" if self.comm is not None else "_native"
            try:
                module = import_module(f".{name}", __package__)
            except ImportError as exc:
                raise RuntimeError("Build the extension with python tools/build_python.py"
                                   + (" --mpi" if self.comm is not None else "")) from exc
            if bool(module.py_build_info()) != (self.comm is not None):
                raise RuntimeError("Native extension MPI mode does not match the solver")
            if getattr(module, "py_api_version", lambda: 0)() != 5:
                raise RuntimeError("Native interface changed; rebuild with python tools/build_python.py"
                                   + (" --mpi" if self.comm is not None else "") + " and restart Python")
            return module
        _native = self._checked(native_module)
        rank = self.comm.Get_rank() if self.comm is not None else 0
        output_directory = self._checked(lambda: Path(
            self.options.output_directory if output_directory is None else output_directory).resolve())
        if self.comm is not None:
            output_directory = self.comm.bcast(output_directory if rank == 0 else None, root=0)
        # 旧核心按固定文件名写输出，计算进程必须使用独立目录，避免互相覆盖。
        local_directory = output_directory if rank == 0 else output_directory / f"rank-{rank:04d}"
        self._checked(lambda: local_directory.mkdir(parents=True, exist_ok=True))
        acquired = _NATIVE_LOCK.acquire(blocking=False)
        old_directory = Path.cwd()
        native_comm = None
        try:
            def check_lock():
                if not acquired:
                    raise RuntimeError("CFL3D uses process-global Fortran state; concurrent solves are unsupported")
            self._checked(check_lock)
            if self.comm is not None:
                # 给核心独立的消息空间；它借用这个副本，Python 负责释放，且不 Finalize MPI。
                native_comm = self.comm.Dup()
                def set_comm():
                    if _native.py_set_communicator(native_comm.py2f()):
                        raise RuntimeError("Native MPI communicator configuration failed")
                self._checked(set_comm)
            def configure():
                _native.py_set_output_flags(int(self.options.writevolumesolution or self.options.writeflowsolution))
                status = _native.py_configure(**arrays)
                if status:
                    raise RuntimeError(f"Native configuration failed ({status}); start a fresh interpreter")
                _native.py_set_coordinates(xyz)
                os.chdir(local_directory)
            self._checked(configure)
            status, nrows = _native.py_solve()
            if status:
                error_file = "precfl3d.error" if status == -99 else "cfl3d.error"
                raise RuntimeError(f"CFL3D failed with status {status}; inspect {output_directory / error_file}")
            q = _native.py_get_solution(arrays["ncells"])
            history = _native.py_get_history(nrows)
            final_alpha = float(_native.py_get_final_alpha())
            surface_values = _native.py_get_surface_coefficients(arrays["bcs"].shape[1])
            wall_counts = (arrays["bcs"][4]-arrays["bcs"][3])*(arrays["bcs"][6]-arrays["bcs"][5])
            wall_values = _native.py_get_wall_data(int(wall_counts.sum()))
            wall_force_values = _native.py_get_wall_forces(int(wall_counts.sum()))
            # CL control can update final_alpha after force evaluation: use the force-time angles.
            force_angles = _native.py_get_force_angles()
        finally:
            os.chdir(old_directory)
            if native_comm is not None:
                native_comm.Free()
            if acquired:
                _NATIVE_LOCK.release()
        fields = {}; offset = 0
        for z in self.mesh.zones:
            shape = tuple(n-1 for n in z.dimensions)
            count = int(np.prod(shape))
            fields[z.name] = q[offset:offset+count].reshape((*shape, 5), order="F").copy()
            if self.options.y_up:
                internal_v = fields[z.name][..., 2].copy()
                fields[z.name][..., 2] = fields[z.name][..., 3]
                fields[z.name][..., 3] = -internal_v
            offset += count
        if (not np.isfinite(q).all() or not np.isfinite(history).all()
                or not np.isfinite(final_alpha) or not np.isfinite(surface_values).all()
                or not np.isfinite(wall_values).all() or not np.isfinite(wall_force_values).all()
                or not np.isfinite(force_angles).all()):
            raise RuntimeError("CFL3D returned nonfinite solution/history")
        surfaces = {key: surface_values[:, index].copy()
                    for index, key in enumerate(self._surface_keys) if key is not None}
        walls = {}; wall_forces = {}; offset = 0
        for index, key in enumerate(self._surface_keys):
            count = int(wall_counts[index])
            if key is not None and abs(int(arrays["bcs"][2,index])) in (1005,1006,2004,2014,2024,2034,2016):
                shape = (int(arrays["bcs"][4,index]-arrays["bcs"][3,index]),
                         int(arrays["bcs"][6,index]-arrays["bcs"][5,index]), 6)
                walls[key] = wall_values[:,offset:offset+count].T.reshape(shape, order="F").copy()
                wall_forces[key] = wall_force_values[:,offset:offset+count].T.reshape(shape, order="F").copy()
            offset += count
        result = SolverResult(fields, history, output_directory, final_alpha, surfaces, walls, wall_forces)
        self._last_surface_coefficients = {key: value.copy() for key, value in surfaces.items()}
        self._last_wall_fields = {key: value.copy() for key, value in walls.items()}
        self._last_wall_forces = {key: value.copy() for key, value in wall_forces.items()}
        self._last_force_angles = np.array(force_angles, copy=True)
        self._last_fields = {name: value.copy() for name, value in fields.items()}
        self._last_mesh = deepcopy(self.mesh)
        self._last_final_alpha, self._last_output_directory = final_alpha, output_directory
        self._last_base_groups, self._last_options = base_groups, self.options
        self._last_result = result
        def write_outputs():
            if rank != 0:
                return
            result.save(output_directory / "solution.npz", include_volume=self.options.writevolumesolution)
            if self.options.writevolumesolution:
                self.writeVolumeSolution()
            if self.options.writeTecplotSurfaceSolution:
                self.writeTecplotSurfaceSolution()
            if self.options.writeflowsolution:
                self.writeFlowSolution()
        self._checked(write_outputs)
        return result
