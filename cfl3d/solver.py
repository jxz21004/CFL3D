"""Python-owned configuration and arrays passed directly to the native solver."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import os
from pathlib import Path
import threading
import sys
import json
import hashlib
import warnings
from numbers import Real
import numpy as np
from .cgns import CGNSMesh

_NATIVE_LOCK = threading.Lock()
_AUTO_COMM = object()
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


@dataclass
class SolverOptions:
    """Dynamic typed settings; validated assignments do not rebuild this object.

    Use update() for related parameters that must change together. CFL3D keeps
    its own rank-local copy in solver.options; snapshots are independent copies,
    not frozen settings. Reset/continuation decisions are made at the next solve.
    """
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
    name: str = "out"
    numberSolutions: bool = True
    writevolumesolution: bool = True
    writeTecplotSurfaceSolution: bool = True
    writeflowsolution: bool = False
    volume_filename: str | Path | None = None
    surface_filename: str | Path | None = None
    flow_prefix: str | Path | None = None
    restartFile: str | Path | None = None
    flow_mode: str = "steady"  # Implicit steady solve; time-accurate Python support is not implemented.

    def __setattr__(self, key, value):
        name = _OPTION_NAMES.get(key.casefold()) if isinstance(key, str) else None
        if name is None:
            raise AttributeError(f"Unknown CFL3D option: {key!r}")
        if len(vars(self)) < len(_OPTION_NAMES):
            object.__setattr__(self, name, value)  # Dataclass construction, before full validation.
        else:
            self.update({key: value})

    def update(self, options=None, **kwargs):
        """Validate related changes together, then update this configuration in place."""
        changes = self._normalize_values({**({} if options is None else dict(options)), **kwargs})
        _validate_options({**vars(self), **changes})
        vars(self).update(changes)
        return self

    @classmethod
    def _normalize_values(cls, options):
        """Normalize only supplied values, without constructing a configuration."""
        names = _OPTION_NAMES
        values = {}
        for key, value in dict(options).items():
            name = names.get(key.casefold()) if isinstance(key, str) else None
            if name is None:
                raise ValueError(f"Unknown CFL3D option: {key!r}")
            if name in values:
                raise ValueError(f"Duplicate CFL3D option ignoring case: {key!r}")
            values[name] = deepcopy(value)
        for name in ("limiter", "flux", "inversion", "kappa", "moment_center", "multigrid_smoothing"):
            if isinstance(values.get(name), list):
                values[name] = tuple(values[name])
        if values.get("multigrid_stages") is not None:
            stages = []
            for stage in values["multigrid_stages"]:
                if isinstance(stage, Mapping):
                    stage_names = {name.casefold(): name for name in MultigridStage.__dataclass_fields__}
                    normalized = {}
                    for key, value in stage.items():
                        name = stage_names.get(key.casefold()) if isinstance(key, str) else None
                        if name is None or name in normalized:
                            raise ValueError(f"Unknown or duplicate multigrid stage option: {key!r}")
                        normalized[name] = value
                    stage = normalized
                    if isinstance(stage.get("smoothing_steps"), list):
                        stage["smoothing_steps"] = tuple(stage["smoothing_steps"])
                    stage = MultigridStage(**stage)
                stages.append(stage)
            values["multigrid_stages"] = tuple(stages)
        return values

    @classmethod
    def from_dict(cls, options):
        """Build dynamic typed settings; solver.options updates use dictionaries."""
        return cls(**cls._normalize_values(options))

    def validate(self):
        _validate_options(vars(self))

    def stage_schedule(self):
        return _stage_schedule(vars(self))

    def stage_smoothing(self, stage, available_levels):
        return _stage_smoothing(vars(self), stage, available_levels)


# The schema/defaults are read once. Live updates operate on dictionaries; only
# explicit snapshot requests construct a SolverOptions instance.
_OPTION_NAMES = {name.casefold(): name for name in SolverOptions.__dataclass_fields__}
_OPTION_DEFAULTS = {name: definition.default for name, definition in SolverOptions.__dataclass_fields__.items()}

# 这里只规定默认接续策略，并不禁止修改。物理/网格变化在 solve 前重新初始化；
# 数值控制、积分参考量和输出设置不丢弃流场。比较基准是上次成功计算的独立副本。
_OPTION_CATEGORIES = {
    "physics": frozenset(("flow_mode", "mach", "reynolds", "temperature", "gamma", "turbulence",
                          "full_navier_stokes", "y_up")),
    "operating_point": frozenset(("alpha", "beta", "target_cl", "alpha_relaxation",
                                  "alpha_step_limit", "alpha_update_interval")),
    "reference": frozenset(("sref", "cref", "bref", "moment_center", "freestream_pressure")),
    "mesh": frozenset(("input_grid",)),
    "restart": frozenset(("restartFile",)),
    "io": frozenset(("name", "output_directory", "numberSolutions", "writevolumesolution",
                     "writeTecplotSurfaceSolution", "writeflowsolution", "volume_filename",
                     "surface_filename", "flow_prefix")),
}
_OPTION_CATEGORIES["numerical"] = frozenset(_OPTION_DEFAULTS) - frozenset().union(*_OPTION_CATEGORIES.values())


def _validate_options(values):
    mode = values["flow_mode"]
    if not isinstance(mode, str) or mode not in ("steady", "unsteady"):
        raise ValueError("flow_mode must be 'steady' or 'unsteady'")
    if mode == "unsteady":
        raise NotImplementedError(
            "flow_mode='unsteady' is not implemented by the Python interface: "
            "physical time steps, inner iterations, and time-history restart state are not exposed. "
            "Use flow_mode='steady' with a positive cfl."
        )
    if (not isinstance(values["name"], str) or not values["name"].strip() or not values["name"].isprintable()
            or values["name"] in (".", "..") or "/" in values["name"] or "\\" in values["name"]):
        raise ValueError("name must be a nonempty case name without path separators")
    for name in ("input_grid", "output_directory", "volume_filename", "surface_filename", "flow_prefix", "restartFile"):
        value = values[name]
        if value is None and name in ("input_grid", "restartFile", "volume_filename", "surface_filename", "flow_prefix"):
            continue
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{name} must be a nonempty path")
    for name in ("writevolumesolution", "writeTecplotSurfaceSolution", "writeflowsolution", "numberSolutions"):
        if not isinstance(values[name], bool):
            raise ValueError(f"{name} must be a boolean")
    ints = ("iterations", "multigrid_levels", "coarse_iterations", "multigrid_cycle",
            "turbulence_subiterations", "cfl_ramp_steps",
            "first_order_iterations", "coarse_first_order_iterations")
    for name in ints:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if values["iterations"] < 1 or values["coarse_iterations"] < 0 or values["turbulence_subiterations"] < 1:
        raise ValueError("Invalid iteration count")
    if values["cfl_ramp_steps"] < 0 or not 0 <= values["first_order_iterations"] <= values["iterations"]:
        raise ValueError("Require nonnegative CFL ramp steps and first_order_iterations <= iterations")
    if not 0 <= values["coarse_first_order_iterations"] <= values["coarse_iterations"]:
        raise ValueError("coarse_first_order_iterations must be within coarse_iterations")
    if values["restart_interval"] is not None and (isinstance(values["restart_interval"], bool)
            or not isinstance(values["restart_interval"], int) or values["restart_interval"] < 1):
        raise ValueError("restart_interval must be a positive integer or None")
    if not 1 <= values["multigrid_levels"] <= 5 or values["multigrid_cycle"] not in (1, 2):
        raise ValueError("multigrid_levels must be 1..5; cycle must be 1 (V) or 2 (W)")
    if values["turbulence"].lower() not in _MODELS:
        raise ValueError(f"Supported models: {', '.join(_MODELS)}")
    for name in ("mach", "alpha", "beta", "reynolds", "temperature", "cfl", "turbulence_cfl",
                 "eddy_viscosity_limit", "gamma", "sref", "cref", "bref", "cfl_ramp_factor",
                 "alpha_relaxation", "alpha_step_limit"):
        value = values[name]
        _finite(value, name)
    if values["mach"] <= 0 or values["temperature"] <= 0 or values["cfl"] <= 0 or values["gamma"] <= 1:
        raise ValueError("Require positive Mach, temperature, CFL and gamma > 1")
    if values["cfl_ramp_factor"] < 1:
        raise ValueError("cfl_ramp_factor must be >= 1")
    for name in ("mesh_sequence", "full_navier_stokes", "y_up"):
        if not isinstance(values[name], bool):
            raise ValueError(f"{name} must be a boolean")
    if min(values["sref"], values["cref"], values["bref"], values["turbulence_cfl"], values["eddy_viscosity_limit"]) <= 0:
        raise ValueError("Reference scales and turbulence controls must be positive")
    if values["reynolds"] <= 0 and values["turbulence"].lower() != "euler":
        raise ValueError("Viscous models require positive Reynolds number")
    if len(values["moment_center"]) != 3 or not np.isfinite(values["moment_center"]).all():
        raise ValueError("moment_center must contain three finite coordinates")
    _directions(values["limiter"], "limiter", (0, 1, 2, 3, 4))
    _directions(values["flux"], "flux", (0, 1))
    _directions(values["inversion"], "inversion", (0, 1))
    _directions(values["kappa"], "kappa")
    _integer(values["alpha_update_interval"], "alpha_update_interval", 1)
    if min(values["alpha_relaxation"], values["alpha_step_limit"]) <= 0:
        raise ValueError("Alpha relaxation and step limit must be positive")
    if values["target_cl"] is not None:
        _finite(values["target_cl"], "target_cl")
        if values["target_cl"] == 99999.0:
            raise ValueError("99999 is the native disabled-CL sentinel; use target_cl=None")
    if values["freestream_pressure"] is not None:
        _finite(values["freestream_pressure"], "freestream_pressure")
        if values["freestream_pressure"] <= 0:
            raise ValueError("freestream_pressure must be positive (Pa)")
    _integer(values["multigrid_up_iterations"], "multigrid_up_iterations", 0, 19)
    if values["multigrid_smoothing"] is not None:
        _smoothing(values["multigrid_smoothing"], values["multigrid_levels"], "multigrid_smoothing")
    schedule = _stage_schedule(values)
    if not 1 <= len(schedule) <= values["multigrid_levels"]:
        raise ValueError("Use between one and multigrid_levels mesh-sequence stages")
    for index, stage in enumerate(schedule):
        if not isinstance(stage, MultigridStage):
            raise ValueError("multigrid_stages must contain MultigridStage objects")
        # 原生序列每阶段细化一次；阶段数决定从哪一级网格开始。
        available = values["multigrid_levels"] - len(schedule) + index + 1
        stage.validate(available)
        if max(_stage_smoothing(values, stage, available)) + values["multigrid_up_iterations"] > 20:
            raise ValueError("MIT + MTT must not exceed the native limit of 20")


def _stage_schedule(values):
    if values["multigrid_stages"] is not None:
        if not isinstance(values["multigrid_stages"], tuple):
            raise ValueError("multigrid_stages must be an immutable tuple")
        return values["multigrid_stages"]
    if values["mesh_sequence"]:
        return tuple(MultigridStage(values["coarse_iterations"], level,
                                   values["coarse_first_order_iterations"])
                     for level in range(1, values["multigrid_levels"])) + (
            MultigridStage(values["iterations"], values["multigrid_levels"], values["first_order_iterations"]),)
    return (MultigridStage(values["iterations"], values["multigrid_levels"], values["first_order_iterations"]),)


def _stage_smoothing(values, stage, available_levels):
    # 默认 MIT 按全局粗到细编号；只取本阶段实际参与循环的网格层。
    template = values["multigrid_smoothing"] or (1,) * values["multigrid_levels"]
    return stage.smoothing_steps or template[available_levels-stage.levels:available_levels]


class RuntimeOptions(MutableMapping):
    """Mutable rank-local parameter dictionary with validated atomic updates.

    Changes remain local until checkOptions()/solve() verifies the object's
    communicator. Native controls are installed together at the next solve.
    SolverOptions is used only for typed input/snapshots, not as live storage.
    """

    def __init__(self, options=None):
        if isinstance(options, RuntimeOptions):
            values = deepcopy(options._values)
        elif isinstance(options, SolverOptions):
            # Typed inputs retain their stricter tuple/type requirements.
            values = deepcopy(vars(options))
        else:
            values = deepcopy(_OPTION_DEFAULTS)
            values.update(SolverOptions._normalize_values({} if options is None else options))
        _validate_options(values)
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_check_callback", None)

    @staticmethod
    def _name(key):
        name = _OPTION_NAMES.get(key.casefold()) if isinstance(key, str) else None
        if name is None:
            raise KeyError(f"Unknown CFL3D option: {key!r}")
        return name

    def __getitem__(self, key):
        return deepcopy(self._values[self._name(key)])

    def __setitem__(self, key, value):
        self.update({key: value})

    def __delitem__(self, key):
        raise TypeError("Options cannot be deleted; assign a default value explicitly")

    def __iter__(self):
        return iter(_OPTION_DEFAULTS)

    def __len__(self):
        return len(_OPTION_DEFAULTS)

    def __getattr__(self, key):
        try:
            name = self._name(key)
        except KeyError as exc:
            raise AttributeError(key) from exc
        return deepcopy(self._values[name])

    def __setattr__(self, key, value):
        self[key] = value

    def update(self, *args, **kwargs):
        provided = dict(*args, **kwargs)
        changes = SolverOptions._normalize_values(provided)
        # 校验候选字典后一次替换；不改动现值，也不反复创建 SolverOptions。
        # 只规范化传入项，未修改参数和已保存的历史快照保持原值。
        candidate = {**self._values, **changes}
        _validate_options(candidate)
        object.__setattr__(self, "_values", candidate)

    def validate(self):
        _validate_options(self._values)

    def stage_schedule(self):
        return _stage_schedule(self._values)

    def stage_smoothing(self, stage, available_levels):
        return _stage_smoothing(self._values, stage, available_levels)

    def check(self):
        """Check locally when standalone, or collectively when owned by a CFL3D object."""
        if self._check_callback is not None:
            return self._check_callback()
        self.validate()
        return True

    def snapshot(self):
        """Create independent dynamic settings; later edits cannot change the live dictionary."""
        return SolverOptions(**deepcopy(self._values))


def _automatic_comm(comm):
    if comm is not _AUTO_COMM:
        return comm
    # Normal Python must not initialize MPI merely by constructing a serial solver.
    mpi = sys.modules.get("mpi4py.MPI")
    launched = any(name in os.environ for name in
                   ("PMI_SIZE", "PMIX_RANK", "OMPI_COMM_WORLD_SIZE", "MPI_LOCALNRANKS"))
    if mpi is None and launched:
        try:
            from mpi4py import MPI as mpi
        except ImportError as exc:
            raise RuntimeError("MPI launch detected but mpi4py is unavailable") from exc
    if mpi is not None and mpi.Is_initialized() and not mpi.Is_finalized() and mpi.COMM_WORLD.size > 1:
        return mpi.COMM_WORLD
    return None


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

    solve_index: int = 0
    name: str = "out"
    output_paths: dict[str, Path] = field(default_factory=dict)
    iteration_offset: int = 0
    cumulative_history: np.ndarray | None = None
    start_mode: str = "cold"
    changed_options: tuple[str, ...] = ()
    reset_reasons: tuple[str, ...] = ()

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
        values = {"history": self.history if self.cumulative_history is None else self.cumulative_history,
                  "chunk_history": self.history, "iteration_offset": np.asarray(self.iteration_offset),
                  "final_alpha": np.asarray(self.final_alpha),
                  "solve_index": np.asarray(self.solve_index), "name": np.asarray(self.name),
                  "start_mode": np.asarray(self.start_mode),
                  "changed_options": np.asarray(self.changed_options, dtype=str),
                  "reset_reasons": np.asarray(self.reset_reasons, dtype=str)}
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

    def __init__(self, mesh=None, options=None, *, boundary_overrides=None, comm=_AUTO_COMM, groups=None, name=None):
        if options is None and isinstance(mesh, (Mapping, SolverOptions)):
            options, mesh = mesh, None
        comm = _automatic_comm(comm)
        if comm is not None:
            from mpi4py import MPI
            if comm == MPI.COMM_NULL or comm.Is_inter():
                raise ValueError("CFL3D needs a valid intracommunicator")
            if comm.Get_size() == 1:
                comm = None  # A one-rank communicator uses the serial core.
        self.comm = comm
        self._last_result = None
        self._solve_counter = 0
        self._next_output_index = 0
        self._output_session = None
        self._output_session_history = None
        self._group_definitions = {}
        self._lift_distributions = []
        self._slice_requests = []
        self._surface_analysis_mesh = None
        self._flow_state = None
        self._loaded_restart_file = None
        self._successful_options = None
        self._successful_mesh_signature = None
        self._comm_members = self._current_comm_members()

        def parse():
            if options is None and self.comm is not None and self.comm.rank != 0:
                return None
            parsed = RuntimeOptions(options)
            if name is not None:
                parsed["name"] = name
            return parsed
        local_options = self._checked(parse)
        if self.comm is not None:
            root_options = self.comm.bcast(local_options.snapshot() if self.comm.rank == 0 else None, root=0)
            # Explicit rank-local settings are retained; None is a compatibility shorthand.
            if local_options is None:
                local_options = RuntimeOptions(root_options)
        self._checked(lambda: local_options.update({"name": name}) if name is not None else None)
        self._options = local_options
        object.__setattr__(self._options, "_check_callback", self.checkOptions)
        self.checkOptions()

        def load():
            if mesh is not None and self.options.input_grid is not None:
                raise ValueError("Specify the mesh argument or options.input_grid, not both")
            input_mesh = mesh if mesh is not None else self.options.input_grid
            if input_mesh is None:
                raise ValueError("Set options['input_grid'] or provide a mesh")
            loaded = input_mesh if isinstance(input_mesh, CGNSMesh) else CGNSMesh.from_file(input_mesh)
            return loaded, boundary_overrides or {}, {} if groups is None else groups
        if comm is None:
            self.mesh, self.boundary_overrides, group_definitions = load()
        else:
            payload = self._checked(lambda: load() if comm.rank == 0 else None)
            self.mesh, root_overrides, group_definitions = comm.bcast(payload, root=0)
            self.boundary_overrides = deepcopy(root_overrides if boundary_overrides is None else boundary_overrides)
            if comm.size - 1 > len(self.mesh.zones):
                raise ValueError("This MPI interface currently requires at least one zone per worker")
        self.addGroups(group_definitions)
        stamp = self._checked(lambda: self._grid_stamp(self.options.input_grid)
                              if self.comm is None or self.comm.rank == 0 else None)
        self._loaded_grid_stamp = self.comm.bcast(stamp, root=0) if self.comm is not None else stamp

    def __call__(self, *, output_directory=None):
        """等价于 solve()：返回 SolverResult，沿用参数更新、续算和输出编号规则。

        MPI 下仍需本对象 comm 中的所有 rank 一起调用。
        """
        return self.solve(output_directory=output_directory)

    @property
    def name(self):
        return self.options.name

    @name.setter
    def name(self, value):
        self.options.name = value

    @property
    def nSolve(self):
        """Number of completed native solves; resetFlow() preserves output numbering."""
        return self._solve_counter

    def _numbered_output_names(self, index):
        stem = f"{self.name}_{index:03d}"
        def filename(configured, default):
            if configured is None:
                return Path(stem+default)
            path = Path(configured)
            return path.with_name(stem+"_"+path.name)
        surface = filename(self.options.surface_filename, "_surf.plt")
        return {"volume": filename(self.options.volume_filename, "_volume.cgns"),
                "surface": surface,
                "slices": surface.parent/(stem+"_slices.dat"),
                "lift": surface.parent/(stem+"_lift.dat"),
                "flow_prefix": filename(self.options.flow_prefix, "_flow"),
                "solution": Path(stem+"_solution.npz"),
                "restart": Path(stem+"_restart.bin"),
                "checkpoint": Path(stem+"_restart.npz"),
                "log": Path(stem+".out"),
                "residual": Path(stem+".res"),
                "turbulence_history": Path(stem+".turres"),
                "alpha_history": Path(stem+".alpha"),
                "preprocess_log": Path(stem+"_precfl3d.out"),
                "error": Path(stem+".error"),
                "preprocess_error": Path(stem+"_precfl3d.error")}

    def _select_output(self, directory, continuing):
        """Reuse only this object's active output stream; new streams avoid existing files."""
        labels = tuple(str(getattr(self.options, name)) for name in
                       ("volume_filename", "surface_filename", "flow_prefix"))
        identity = (str(directory), self.name, labels)
        session = self._output_session
        if (not self.options.numberSolutions and continuing and session is not None
                and session[0] == identity):
            return session[1], True, len(self._output_session_history), identity
        index = self._next_output_index
        while True:
            names = self._numbered_output_names(index)
            stem = f"{self.name}_{index:03d}"
            # A fresh object must not silently overwrite another object's files, including
            # a native failure that already wrote its numbered live log.
            occupied = any(directory.glob(stem+".*")) or any(directory.glob(stem+"_*"))
            occupied = occupied or any((directory / name).exists() for name in names.values())
            if not occupied:
                return index, False, 0, identity
            index += 1

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, values):
        self._options = RuntimeOptions(values)
        object.__setattr__(self._options, "_check_callback", self.checkOptions)

    def setOption(self, name, value):
        self.options[name] = value
        return self

    def getOption(self, name):
        return self.options[name]

    def setOptions(self, values):
        self.options.update(values)
        return self

    def _current_comm_members(self):
        if self.comm is None:
            return (0,)
        from mpi4py import MPI
        group, world_group = self.comm.Get_group(), MPI.COMM_WORLD.Get_group()
        try:
            return tuple(MPI.Group.Translate_ranks(group, list(range(self.comm.size)), world_group))
        finally:
            group.Free()
            world_group.Free()

    def checkOptions(self):
        """Collectively validate rank-local options within this object's comm only."""
        def validate():
            self.options.validate()
        self._checked(validate)
        if self.comm is None:
            return True
        members = self._current_comm_members()
        payload = json.dumps(asdict(self.options.snapshot()), sort_keys=True, default=str)
        gathered = self.comm.allgather((members == self._comm_members, payload))
        if not all(item[0] for item in gathered):
            raise RuntimeError("CFL3D communicator membership changed after construction")
        if len({item[1] for item in gathered}) != 1:
            decoded = [json.loads(item[1]) for item in gathered]
            differing = [key for key in decoded[0] if any(d[key] != decoded[0][key] for d in decoded[1:])]
            raise RuntimeError(f"CFL3D options differ within comm on {differing}; update every participating rank")
        return True

    @staticmethod
    def getOptionCategories():
        """Return parameter groups used by the automatic restart policy."""
        return {name: tuple(sorted(members)) for name, members in _OPTION_CATEGORIES.items()}

    @staticmethod
    def _grid_stamp(path):
        if path is None:
            return None
        path = Path(path).resolve()
        info = path.stat()
        return str(path), info.st_size, info.st_mtime_ns

    def _refresh_grid(self):
        stamp = self._checked(lambda: self._grid_stamp(self.options.input_grid)
                              if self.comm is None or self.comm.rank == 0 else None)
        if self.comm is not None:
            stamp = self.comm.bcast(stamp if self.comm.rank == 0 else None, root=0)
        if stamp == self._loaded_grid_stamp:
            return False
        if stamp is not None:
            mesh = self._checked(lambda: CGNSMesh.from_file(stamp[0])
                                 if self.comm is None or self.comm.rank == 0 else None)
            self.mesh = self.comm.bcast(mesh, root=0) if self.comm is not None else mesh
            if self.comm is not None and self.comm.size-1 > len(self.mesh.zones):
                raise ValueError("The new mesh needs at least one zone per MPI worker")
            # Relative slices belonged to the old discretization. Reattach them to the new mesh.
            from .surface_analysis import direction_index, slice_references
            surface = self._analysis_surface()
            for request in self._slice_requests:
                if request["sliceType"] == "relative":
                    members = self._analysis_members(request["groupName"], surface)
                    request["members"] = members
                    request["references"] = slice_references(
                        surface, members, direction_index(request["direction"]), request["position"])
        self._loaded_grid_stamp = stamp
        return True

    def _mesh_signature(self):
        """Detect in-memory geometry, topology and boundary changes, including 1-to-1 maps."""
        digest = hashlib.sha256()
        for zone in self.mesh.zones:
            digest.update(zone.name.encode())
            digest.update(np.asarray(zone.dimensions, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(zone.coordinates, dtype=np.float64).tobytes())
            for bc in zone.boundaries:
                digest.update(json.dumps((bc.name, bc.bc_type)).encode())
                digest.update(np.asarray(bc.point_range, dtype=np.int64).tobytes())
                for name, data in sorted(bc.data.items()):
                    digest.update(name.encode())
                    digest.update(np.asarray(data).tobytes())
            for cn in zone.connections:
                digest.update(cn.donor.encode())
                for value in (cn.point_range, cn.donor_range, cn.transform):
                    digest.update(np.asarray(value, dtype=np.int64).tobytes())
        digest.update(json.dumps(sorted((str(key), value) for key, value in self.boundary_overrides.items()),
                                 default=str).encode())
        return digest.hexdigest()

    def _configuration_changes(self, mesh_signature):
        previous = self._successful_options
        if previous is None:
            return (), ()
        def token(key, value):
            if key == "turbulence":
                value = value.lower()
            elif key in ("input_grid", "restartFile") and value is not None:
                value = str(Path(value).resolve())
            return json.dumps(value, default=str)
        changed = tuple(sorted(key for key in self.options if
                               token(key, self.options[key]) != token(key, previous[key])))
        reset = set(changed) & (_OPTION_CATEGORIES["physics"] | _OPTION_CATEGORIES["mesh"])
        if mesh_signature != self._successful_mesh_signature:
            reset.add("mesh")
        return changed, tuple(sorted(reset))

    def getFlowState(self):
        """Return an independent Python snapshot, including native turbulence/ghost state."""
        if self._flow_state is None:
            raise RuntimeError("Call solve() before retrieving flow state")
        return deepcopy(self._flow_state)

    def setFlowState(self, state):
        """Install a Python-owned state locally; all ranks must install the same state."""
        from .restart import FlowState
        if not isinstance(state, FlowState):
            raise TypeError("state must be returned by getFlowState()")
        state.validate(self.mesh, self.options)
        self._flow_state = deepcopy(state)
        return self

    def resetFlow(self):
        """Start the next solve from freestream instead of cached state or restartFile."""
        self._flow_state = None
        self._continuing = False
        self._loaded_restart_file = None
        self.options.restartFile = None
        return self

    def writeRestartFile(self, filename=None):
        """Write a complete Python checkpoint; a root-only call needs no MPI collective."""
        if self.comm is not None and self.comm.rank != 0:
            return None
        path = self._output_path(filename, self._last_output_names["checkpoint"] if self._last_result else "")
        from .restart import write_restart
        if self._flow_state is None:
            raise RuntimeError("No flow state is available")
        write_restart(path, self._flow_state)
        return path

    def _prepare_flow_state(self, discard_cached=False):
        from .restart import read_restart
        restart_file = str(Path(self.options.restartFile).resolve()) if self.options.restartFile is not None else None
        reload_file = restart_file is not None and restart_file != self._loaded_restart_file
        if self.comm is not None:
            reload_file = any(self.comm.allgather(reload_file))
        state = None if discard_cached else self._flow_state
        if reload_file:
            state = self._checked(lambda: read_restart(restart_file,self.mesh,self.options)
                                  if self.comm is None or self.comm.rank == 0 else None)
            if self.comm is not None:
                state = self.comm.bcast(state,root=0)
            self._flow_state = state
            self._loaded_restart_file = restart_file
        self._checked(lambda: state.validate(self.mesh,self.options) if state is not None else None)
        if self.comm is not None:
            hashes = self.comm.allgather(None if state is None else state.digest)
            if len(set(hashes))!=1:
                raise RuntimeError("CFL3D cached flow states differ within comm")
        return state

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
        path = self._output_path(filename, self._last_output_names["volume"] if self._last_result else "")
        from .solution_io import write_volume
        write_volume(path, self._last_mesh, self._last_fields, self._last_options)
        return path

    def writeSurfaceAnalysis(self, prefix=None):
        """Write registered lift distributions and slices as Tecplot ASCII .dat files.

        Automatic analyses are independent of the main wall-output flag. With no
        prefix, use the same case name/index as the most recent completed solve.
        This is a local root-only operation, including deferred output.
        """
        if self.comm is not None and self.comm.rank != 0:
            return {}
        if self._last_result is None:
            raise RuntimeError("Call solve() before writing surface analysis")
        from .solution_io import write_lift_distributions, write_slices
        distributions = [self.getLiftDistribution(i) for i in range(self.nLiftDist)]
        slices = self.getSlices()
        files = {}
        for label, data, writer in (("lift", distributions, write_lift_distributions),
                                     ("slices", slices, write_slices)):
            if data:
                explicit = None if prefix is None else str(prefix)+"_"+label+".dat"
                path = self._output_path(explicit, self._last_output_names[label])
                writer(path, data)
                files[label] = path
        return files

    def writeTecplotSurfaceSolution(self, filename=None, groupName=None):
        """Write cached wall quads as Tecplot ASCII (default .plt) and registered analyses."""
        if self.comm is not None and self.comm.Get_rank() != 0:
            return None
        path = self._output_path(filename, self._last_output_names["surface"] if self._last_result else "")
        from .solution_io import write_surface
        members = self._group_members(groupName)
        # Analyses retain their own registered groups, independent of the wall-file selection.
        self.writeSurfaceAnalysis(None if filename is None else path.with_suffix(""))
        write_surface(path, self._last_mesh, self._last_wall_fields, members)
        return path

    def writeFlowSolution(self, prefix=None):
        """Write cached cell-center PLOT3D .x/.q in CFL3D binary IPtype=1 format."""
        if self.comm is not None and self.comm.Get_rank() != 0:
            return None
        prefix = self._output_path(prefix, self._last_output_names["flow_prefix"] if self._last_result else "")
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

        CL/CD/CDp/CDv use native wind axes; CX/CY/CZ and moments use input
        mesh axes. Forces are N, moments N*m, and wettedArea m^2. Missing
        freestream pressure warns and omits dimensional loads. A pressure added
        to live options can supply missing dimensional information after solve.
        A group prefixes all numerical keys and their entries in units.
        This is a local call: any rank may query without an MPI collective.
        """
        members = self._group_members(groupName)
        total = (np.sum([self._last_surface_coefficients[key] for key in sorted(members)], axis=0)
                 if members else np.zeros(10))
        values = {name: float(total[index]) for index, name in enumerate(_SURFACE_COEFFICIENTS)}
        options = self._last_options
        result = values.copy()
        result["wettedArea"] = self._wetted_area(members)
        units = {"wettedArea": "m^2"}
        pressure = options.freestream_pressure
        if pressure is None:
            pressure = self.options.freestream_pressure
        if pressure is None:
            warnings.warn(
                "No freestream pressure was set for the CFL3D calculation or provided in options; "
                "absolute forces and moments cannot be determined. Returning coefficients and wetted area only.",
                RuntimeWarning, stacklevel=2,
            )
        else:
            dynamic_pressure = .5 * options.gamma * pressure * options.mach**2
            scale = dynamic_pressure * options.sref
            yref, zref = (options.bref, options.cref) if options.y_up else (options.cref, options.bref)
            for force, coefficient in (("L", "CL"), ("D", "CD"), ("Dp", "CDp"), ("Dv", "CDv"),
                                       ("Fx", "CX"), ("Fy", "CY"), ("Fz", "CZ")):
                result[force] = values[coefficient] * scale
                units[force] = "N"
            for moment, coefficient, length in (("Mx", "Cmx", options.bref),
                                                ("My", "Cmy", yref), ("Mz", "Cmz", zref)):
                result[moment] = values[coefficient] * scale * length
                units[moment] = "N*m"
        prefix = "" if groupName is None else f"{groupName}_"
        return {**{prefix + name: value for name, value in result.items()},
                "units": {prefix + name: unit for name, unit in units.items()}}

    def _wetted_area(self, members):
        """Return selected cached wall area in m^2, counting overlapping members once."""
        surface = self._analysis_surface(require_solution=True)
        area = 0.0
        for key in sorted(members & surface.patches.keys()):
            quads = surface.patches[key].quads
            # Sum both triangle areas so warped quads retain their geometric area.
            for a, b in ((1, 2), (2, 3)):
                area += .5 * float(np.linalg.norm(np.cross(
                    quads[:, a] - quads[:, 0], quads[:, b] - quads[:, 0]), axis=1).sum())
        return area

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
                # Verify paired vertices, including permutation/reversal.
                ranges = [range(min(r), max(r)+1) for r in cn.point_range]
                for point in np.ndindex(*(len(r) for r in ranges)):
                    p = np.array([ranges[a][point[a]] for a in range(3)])
                    q = cn.donor_range[:, 0].copy()
                    for a in range(3):
                        t = int(cn.transform[a]); q[abs(t)-1] += np.sign(t) * (p[a] - cn.point_range[a, 0])
                    if not np.allclose(z.coordinates[tuple(p-1)], zones[donor].coordinates[tuple(q-1)], rtol=1e-10, atol=1e-10):
                        raise ValueError("One-to-one interface coordinates do not coincide")
                # CGNS 两侧可用相反的端点顺序描述同一接口。先校验映射，再按
                # 几何范围去重，避免把合法的反向声明重复登记为重叠边界。
                akey = (g, tuple(np.sort(cn.point_range, axis=1).flat))
                bkey = (donor, tuple(np.sort(cn.donor_range, axis=1).flat))
                canonical = tuple(sorted((akey, bkey)))
                if canonical in seen:
                    continue
                seen.add(canonical)
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
        if getattr(self, "_continuing", False):
            # A cached solution is already on the finest grid: do not repeat grid sequencing.
            schedule = (schedule[-1],)
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
        # flow_mode is validated as steady before reaching here. Positive native
        # DT is a fixed physical time step, not a positive CFL or an explicit scheme.
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
        self.checkOptions()
        self._refresh_grid()
        mesh_signature = self._checked(self._mesh_signature)
        changed_options, reset_reasons = self._checked(
            lambda: self._configuration_changes(mesh_signature))
        if self.comm is not None:
            decisions = self.comm.allgather((changed_options, reset_reasons, mesh_signature))
            if len(set(decisions)) != 1:
                raise RuntimeError("CFL3D configuration/mesh change decisions differ within comm")
        if self.comm is not None and len(set(self.comm.allgather(self._solve_counter))) != 1:
            raise RuntimeError("CFL3D solve counters differ within comm")
        state = self._prepare_flow_state(discard_cached=bool(reset_reasons))
        self._continuing = state is not None
        if reset_reasons and (self.comm is None or self.comm.rank == 0):
            print("CFL3D: reinitializing after changes to " + ", ".join(reset_reasons)
                  + ("; loading requested restartFile" if self._continuing else "; starting from freestream"), flush=True)
        rank = self.comm.Get_rank() if self.comm is not None else 0
        output_directory = self._checked(lambda: Path(
            self.options.output_directory if output_directory is None else output_directory).resolve())
        if self.comm is not None:
            output_directory = self.comm.bcast(output_directory if rank == 0 else None, root=0)
        selected = self._checked(lambda: self._select_output(output_directory, self._continuing) if rank == 0 else None)
        if self.comm is not None:
            selected = self.comm.bcast(selected, root=0)
        solve_index, append_output, iteration_offset, output_identity = selected
        output_names = self._checked(lambda: self._numbered_output_names(solve_index))
        stem = f"{self.name}_{solve_index:03d}"
        # Legacy filename fields are 80 bytes; reserve room for auxiliary suffixes.
        self._checked(lambda: self._validate_native_stem(stem))
        arrays, xyz = self._checked(self._arrays)
        if state is not None:
            arrays["time_control"][1] = -1  # Native full-state restart, without old convergence history.
        if self.comm is not None:
            fingerprint = hashlib.sha256()
            for key,value in sorted(arrays.items()):
                fingerprint.update(key.encode())
                fingerprint.update(np.asarray(value).tobytes())
            fingerprint.update(xyz.tobytes())
            if len(set(self.comm.allgather(fingerprint.hexdigest()))) != 1:
                raise RuntimeError("CFL3D mesh/boundary/native configuration differs within comm")
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
            if getattr(module, "py_api_version", lambda: 0)() != 8:
                raise RuntimeError("Native interface changed; rebuild with python tools/build_python.py"
                                   + (" --mpi" if self.comm is not None else "") + " and restart Python")
            return module
        _native = self._checked(native_module)
        # Native ranks retain independent work directories while using the same case/index.
        local_directory = output_directory if rank == 0 else output_directory / f"rank-{rank:04d}"
        self._checked(lambda: local_directory.mkdir(parents=True, exist_ok=True))
        checkpoint_path = output_directory / ".cfl3d-state.bin"
        def stage_checkpoint():
            if rank == 0:
                protected = [self.mesh.source, self.options.restartFile]
                if any(value and checkpoint_path.resolve() == Path(value).resolve() for value in protected):
                    raise ValueError("Input files must not use the internal .cfl3d-state.bin output path")
                if state is not None:
                    checkpoint_path.write_bytes(state.native)
                else:
                    checkpoint_path.unlink(missing_ok=True)
        self._checked(stage_checkpoint)
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
                # Internal checkpoint I/O is independent of visualization output flags.
                _native.py_set_output_flags(1)
                _native.py_set_output_names(stem, int(append_output), iteration_offset)
                _native.py_set_restart_controls(int("alpha" in changed_options))
                status = _native.py_configure(**arrays)
                if status:
                    raise RuntimeError(f"Native configuration failed ({status}); start a fresh interpreter")
                _native.py_set_coordinates(xyz)
                os.chdir(local_directory)
            self._checked(configure)
            status, nrows = _native.py_solve()
            if status:
                error_file = output_names["preprocess_error" if status == -99 else "error"]
                raise RuntimeError(f"CFL3D failed with status {status}; inspect {local_directory / error_file}")
            q = _native.py_get_solution(arrays["ncells"])
            history = _native.py_get_history(nrows)
            final_alpha = float(_native.py_get_final_alpha())
            surface_values = _native.py_get_surface_coefficients(arrays["bcs"].shape[1])
            wall_counts = (arrays["bcs"][4]-arrays["bcs"][3])*(arrays["bcs"][6]-arrays["bcs"][5])
            wall_values = _native.py_get_wall_data(int(wall_counts.sum()))
            wall_force_values = _native.py_get_wall_forces(int(wall_counts.sum()))
            # CL control can update final_alpha after force evaluation: use the force-time angles.
            force_angles = _native.py_get_force_angles()
            checkpoint = self._checked(lambda: checkpoint_path.read_bytes() if rank == 0 else None)
            if self.comm is not None:
                checkpoint = self.comm.bcast(checkpoint, root=0)
            self._checked(lambda: checkpoint_path.unlink(missing_ok=True) if rank == 0 else None)
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
        result = SolverResult(fields, history, output_directory, final_alpha, surfaces, walls, wall_forces,
                              solve_index=solve_index, name=self.name,
                              output_paths={key: output_directory/value for key,value in output_names.items()},
                              iteration_offset=iteration_offset,
                              cumulative_history=(np.concatenate([self._output_session_history, history])
                                                  if append_output else history.copy()),
                              start_mode="restart" if self._continuing else "cold",
                              changed_options=changed_options, reset_reasons=reset_reasons)
        self._last_surface_coefficients = {key: value.copy() for key, value in surfaces.items()}
        self._last_wall_fields = {key: value.copy() for key, value in walls.items()}
        self._last_wall_forces = {key: value.copy() for key, value in wall_forces.items()}
        self._last_force_angles = np.array(force_angles, copy=True)
        self._last_fields = {name: value.copy() for name, value in fields.items()}
        self._last_mesh = deepcopy(self.mesh)
        self._last_final_alpha, self._last_output_directory = final_alpha, output_directory
        self._last_base_groups, self._last_options = base_groups, self.options.snapshot()
        self._last_result = result
        self._last_output_names = output_names
        # Consume an index once native computation succeeds, even if a later write needs retrying.
        self._solve_counter += 1
        self._next_output_index = solve_index + 1
        self._output_session = (output_identity, solve_index)
        self._output_session_history = result.cumulative_history.copy()
        self._successful_options = dict(self.options)
        self._successful_mesh_signature = mesh_signature
        from .restart import FlowState, metadata
        self._flow_state = FlowState(checkpoint, metadata(self.mesh,self.options),
                                    {name:value.copy() for name,value in fields.items()})
        def write_outputs():
            if rank != 0:
                return
            result.save(self._output_path(None, output_names["solution"]), include_volume=self.options.writevolumesolution)
            if self.options.writevolumesolution or self.options.writeflowsolution:
                self._output_path(None, output_names["restart"]).write_bytes(checkpoint)
            if self.options.writevolumesolution:
                self.writeVolumeSolution()
            if self.options.writeTecplotSurfaceSolution:
                self.writeTecplotSurfaceSolution()
            else:
                self.writeSurfaceAnalysis()
            if self.options.writeflowsolution:
                self.writeFlowSolution()
        self._checked(write_outputs)
        return result

    @staticmethod
    def _validate_native_stem(stem):
        if not stem.isascii() or len(stem) > 50:
            raise ValueError("Native output case name/index must be ASCII and at most 50 characters combined")
