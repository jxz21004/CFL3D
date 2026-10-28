"""Python-owned configuration and arrays passed directly to the native solver."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from collections.abc import Mapping, MutableMapping
from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import threading
import sys
import json
import hashlib
from time import perf_counter
import warnings
from numbers import Real
import numpy as np
from .cgns import CGNSMesh

_NATIVE_LOCK = threading.Lock()
_AUTO_COMM = object()
_DEFAULT_RESTART_INTERVAL = 99999
_MODELS = {"euler": 0, "laminar": 1, "sa": 5, "sst": 7, "komega": 6, "gamma-theta": 40,
           "bsl": 7, "sst2003": 7, "wilcox2006": 6}
# Distinct model names keep restart metadata and cold-start decisions unambiguous,
# even when native IVISC is shared. Values are I_BSL, ISST2003, I_WILCOX06.
_MODEL_FLAGS = {"bsl": (1, 0, 0), "sst2003": (0, 1, 0), "wilcox2006": (0, 0, 1)}
_AMBIENT_MODELS = frozenset(name for name, code in _MODELS.items() if code in (6, 7, 40))
_SURFACE_COEFFICIENTS = ("CL", "CD", "CX", "CY", "CZ", "Cmx", "Cmy", "Cmz", "CDp", "CDv")
_BC_TYPES = {"BCFarfield": (1003, ()), "BCSymmetryPlane": (1001, ()),
             "BCWallInviscid": (1005, ()), "BCOutflowSupersonic": (1002, ()),
             # The native CGNS input converter treats generic BCWall as an
             # adiabatic viscous wall when no thermal dataset is supplied.
             "BCWall": (2004, (0., 0.)), "BCWallViscous": (2004, (0., 0.)),
             "BCWallViscousHeatFlux": (2004, (0., 0.))}


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
    coarse_grid_first_order_iterations: int = 0
    restart_interval: int = _DEFAULT_RESTART_INTERVAL
    multigrid_levels: int = 1
    mesh_sequence: bool = False
    coarse_grid_iterations: int = 50
    multigrid_cycle: int = 2  # NGAM: 1 V, 2 W
    full_navier_stokes: bool = True
    turbulence_subiterations: int = 1
    turbulence_cfl: float = 5.0
    eddy_viscosity_limit: float = 1e5  # 按用户要求保留较保守的 Python 默认值。
    gamma: float = 1.4
    limiter: int | tuple[int, int, int] = 4  # Smooth kappa=1/3 limiter, invariant under block splitting.
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
    inversion: int | tuple[int, int, int] = 1  # IDIAG, scalar or (i,j,k)
    freestream_pressure: float | None = None  # Pa; coordinates/reference lengths are metres
    input_grid: str | Path | None = None
    output_directory: str | Path = "cfl3d-output"
    name: str = "out"
    numberSolutions: bool = True
    writeIntermediateSolutions: bool = False  # Optional snapshots for solve(final=False).
    writevolumesolution: bool = True
    writeTecplotSurfaceSolution: bool = True
    writeflowsolution: bool = False
    volume_filename: str | Path | None = None
    surface_filename: str | Path | None = None
    flow_prefix: str | Path | None = None
    restartFile: str | Path | None = None
    flow_mode: str = "steady"  # Implicit steady solve; time-accurate Python support is not implemented.
    freestream_turbulence_intensity: float | None = None  # Percent, not a fraction; native defaults if None.
    freestream_eddy_viscosity: float | None = None  # mu_t / mu_inf, dimensionless.
    keep_ambient_turbulence: bool = False
    resident: bool = False  # Opt-in native workspace lifetime; see docs/distributed-storage.md.
    coarse_grid_stages: tuple[MultigridStage, ...] | None = None
    progress: bool = True  # Rank-0 phase checkpoints for long first calls.

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
        for option_name in ("coarse_grid_stages",):
            if values.get(option_name) is None:
                continue
            stages = []
            for stage in values[option_name]:
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
            values[option_name] = tuple(stages)
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
                          "full_navier_stokes", "y_up", "freestream_turbulence_intensity",
                          "freestream_eddy_viscosity", "keep_ambient_turbulence")),
    "operating_point": frozenset(("alpha", "beta", "target_cl", "alpha_relaxation",
                                  "alpha_step_limit", "alpha_update_interval")),
    "reference": frozenset(("sref", "cref", "bref", "moment_center", "freestream_pressure")),
    "mesh": frozenset(("input_grid",)),
    "restart": frozenset(("restartFile",)),
    "runtime": frozenset(("resident",)),
    "io": frozenset(("name", "output_directory", "numberSolutions", "writeIntermediateSolutions", "writevolumesolution",
                     "writeTecplotSurfaceSolution", "writeflowsolution", "volume_filename",
                     "surface_filename", "flow_prefix", "progress")),
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
    for name in ("writevolumesolution", "writeTecplotSurfaceSolution", "writeflowsolution", "numberSolutions", "writeIntermediateSolutions", "resident", "progress"):
        if not isinstance(values[name], bool):
            raise ValueError(f"{name} must be a boolean")
    if values["mesh_sequence"] and values["coarse_grid_stages"] is not None:
        raise ValueError("mesh_sequence is automatic startup; do not combine it with explicit grid stages")
    if values["resident"] and values["restartFile"] is not None:
        raise NotImplementedError("File restart initialization is not yet implemented for resident=True")
    ints = ("iterations", "multigrid_levels", "coarse_grid_iterations", "multigrid_cycle",
            "turbulence_subiterations", "cfl_ramp_steps",
            "first_order_iterations", "coarse_grid_first_order_iterations")
    for name in ints:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if values["iterations"] < 1 or values["coarse_grid_iterations"] < 0 or values["turbulence_subiterations"] < 1:
        raise ValueError("Invalid iteration count")
    if (values["cfl_ramp_steps"] < 0 or values["first_order_iterations"] < 0
            or (not values["resident"] and values["first_order_iterations"] > values["iterations"])):
        raise ValueError("Require nonnegative CFL ramp steps and first_order_iterations <= iterations unless resident=True")
    if not 0 <= values["coarse_grid_first_order_iterations"] <= values["coarse_grid_iterations"]:
        raise ValueError("coarse_grid_first_order_iterations must be within coarse_grid_iterations")
    _integer(values["restart_interval"], "restart_interval", 1, np.iinfo(np.int32).max)
    if not 1 <= values["multigrid_levels"] <= 5 or values["multigrid_cycle"] not in (1, 2):
        raise ValueError("multigrid_levels must be 1..5; cycle must be 1 (V) or 2 (W)")
    if values["turbulence"].lower() not in _MODELS:
        raise ValueError(f"Supported models: {', '.join(_MODELS)}")
    for name in ("freestream_turbulence_intensity", "freestream_eddy_viscosity"):
        if values[name] is not None:
            _finite(values[name], name)
            if values[name] <= 0:
                raise ValueError(f"{name} must be positive or None")
    if not isinstance(values["keep_ambient_turbulence"], bool):
        raise ValueError("keep_ambient_turbulence must be a boolean")
    if (values["freestream_turbulence_intensity"] is not None
            or values["freestream_eddy_viscosity"] is not None or values["keep_ambient_turbulence"]):
        if values["turbulence"].lower() not in _AMBIENT_MODELS:
            raise ValueError("Freestream turbulence controls require a k-omega/SST-family model")
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
            raise ValueError("Grid stages must contain MultigridStage objects")
        # 原生序列每阶段细化一次；阶段数决定从哪一级网格开始。
        available = values["multigrid_levels"] - len(schedule) + index + 1
        stage.validate(available)
        if max(_stage_smoothing(values, stage, available)) + values["multigrid_up_iterations"] > 20:
            raise ValueError("MIT + MTT must not exceed the native limit of 20")


def _stage_schedule(values):
    # iterations always owns the finest-grid budget, including cached solves.
    # Resident startup may span chunks: reserve capacity without extending the
    # public solve's requested fine-grid work or resetting first-order counters.
    budget = max(values["iterations"], values["first_order_iterations"]) if values["resident"] else values["iterations"]
    final = MultigridStage(budget, values["multigrid_levels"], values["first_order_iterations"])
    if values["coarse_grid_stages"] is not None:
        if not isinstance(values["coarse_grid_stages"], tuple):
            raise ValueError("coarse_grid_stages must be a tuple of MultigridStage objects")
        return (*values["coarse_grid_stages"], final)
    if values["mesh_sequence"]:
        return tuple(MultigridStage(values["coarse_grid_iterations"], level,
                                   values["coarse_grid_first_order_iterations"])
                     for level in range(1, values["multigrid_levels"])) + (
            final,)
    return (final,)


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


@dataclass(frozen=True)
class SolveFailure:
    """Geometry update failed: submit valid coordinates before advancing again."""
    reason: str
    native_status: int
    fail: bool = True
    reset_required: bool = False
    geometry_update_required: bool = True


@dataclass
class SolverResult:
    fields: dict[str, np.ndarray]  # MPI: owned zones only; zone -> (ni-1,nj-1,nk-1,5): rho,u,v,w,p
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
    # Compatibility name: API 19 returns only this call's rows, without accumulation.
    cumulative_history: np.ndarray | None = None
    start_mode: str = "cold"
    changed_options: tuple[str, ...] = ()
    reset_reasons: tuple[str, ...] = ()
    zone_owners: dict[str, int] = field(default_factory=dict)
    chunk_index: int = 0
    finalized: bool = False
    chunk_output_paths: dict[str, Path] = field(default_factory=dict)
    geometry_version: int = 0
    # (zero-based cumulative-history row, geometry version); immutable snapshot.
    history_geometry_segments: tuple[tuple[int, int], ...] = ()

    @property
    def fail(self):
        return False

    @property
    def coefficients(self):
        if not len(self.history):
            return {}
        coefficients = dict(zip(("CL", "CD", "CX", "CY", "CMX", "CMY", "CMZ"), self.history[-1, 1:]))
        if self.surface_coefficients:
            total = np.sum(list(self.surface_coefficients.values()), axis=0)
            coefficients.update(CDp=float(total[8]), CDv=float(total[9]))
        return coefficients

    def _saved_values(self):
        values = {"history": self.history if self.cumulative_history is None else self.cumulative_history,
                  "chunk_history": self.history, "iteration_offset": np.asarray(self.iteration_offset),
                  "final_alpha": np.asarray(self.final_alpha),
                  "solve_index": np.asarray(self.solve_index), "name": np.asarray(self.name),
                  "chunk_index": np.asarray(self.chunk_index), "finalized": np.asarray(self.finalized),
                  "geometry_version": np.asarray(self.geometry_version),
                  "history_geometry_segments": np.asarray(self.history_geometry_segments,
                                                          dtype=np.int64).reshape(-1, 2),
                  "start_mode": np.asarray(self.start_mode),
                  "changed_options": np.asarray(self.changed_options, dtype=str),
                  "reset_reasons": np.asarray(self.reset_reasons, dtype=str)}
        keys = sorted(self.surface_coefficients)
        values["surface_names"] = np.asarray(keys, dtype=str).reshape(-1, 2)
        values["surface_coefficients"] = np.asarray([self.surface_coefficients[k] for k in keys]).reshape(-1, 10)
        values["surface_coefficient_names"] = np.asarray(_SURFACE_COEFFICIENTS)
        return values

    def save(self, filename, *, include_volume=True):
        from .solution_io import result_output
        from types import SimpleNamespace
        with result_output(filename, self._saved_values()) as write_zone:
            if include_volume:
                for name, q in self.fields.items():
                    write_zone(SimpleNamespace(name=name), q)


class CFL3D:
    """Native solver with an optional collective mpi4py communicator.

    Disjoint MPI communicators can solve independent cases simultaneously.
    Within one Python process, native calls share COMMON state and are serialized;
    one resident object owns that rank until close. After a native failure, start
    a fresh Python process before attempting another solve.
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
        self._solution_counter = 0
        self._solution_pending = False
        self._output_chunk_counter = 0
        self._next_output_index = 0
        self._output_session = None
        self._output_session_iterations = 0
        self._group_definitions = {}
        self._lift_distributions = []
        self._slice_requests = []
        self._surface_analysis_mesh = None
        self._surface_context = None
        self._flow_state = None
        self._resident_context = None
        self._pending_geometry = None
        self._geometry_version = 0
        self._output_geometry_segments = ()
        self._last_runtime_info = {"active": False}
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
        self._progress_origin = self._progress_last = perf_counter()
        self._progress("Loading mesh and assigning coordinates")

        def load():
            if mesh is not None and self.options.input_grid is not None:
                raise ValueError("Specify the mesh argument or options.input_grid, not both")
            input_mesh = mesh if mesh is not None else self.options.input_grid
            if input_mesh is None:
                raise ValueError("Set options['input_grid'] or provide a mesh")
            loaded = input_mesh if isinstance(input_mesh, CGNSMesh) else CGNSMesh.from_file(
                input_mesh, coordinate_zones=None if comm is None else ())
            return loaded, boundary_overrides or {}, {} if groups is None else groups
        if comm is None:
            self.mesh, self.boundary_overrides, group_definitions = load()
        else:
            payload = self._checked(lambda: load() if comm.rank == 0 else None)
            root_overrides, group_definitions = comm.bcast(
                payload[1:] if comm.rank == 0 else None, root=0)
            self.boundary_overrides = deepcopy(root_overrides if boundary_overrides is None else boundary_overrides)
            self._install_distributed_mesh(payload[0] if comm.rank == 0 else None)
            payload = None  # Do not retain the root's original whole-case mesh.
        self.addGroups(group_definitions)
        stamp = self._checked(lambda: self._grid_stamp(self.options.input_grid)
                              if self.comm is None or self.comm.rank == 0 else None)
        self._loaded_grid_stamp = self.comm.bcast(stamp, root=0) if self.comm is not None else stamp
        self._progress(f"Mesh ready: {len(self.mesh.zones)} zones")

    def _progress(self, message):
        """Report a phase transition from the solver communicator's root only."""
        now = perf_counter()
        previous = self._progress_last
        self._progress_last = now
        if self.options.progress and (self.comm is None or self.comm.rank == 0):
            print(f"CFL3D[{self.name}] {message} "
                  f"(+{now-previous:.1f}s; total {now-self._progress_origin:.1f}s)", flush=True)

    @staticmethod
    def _stage_summary(arrays, fine_cycles=None):
        """Show the actual native stage schedule, including coarse startup."""
        stages = arrays["stages"]
        parts = []
        for index in range(stages.shape[1]):
            cycles = fine_cycles if fine_cycles is not None and index == stages.shape[1]-1 else stages[0,index]
            parts.append(f"{int(stages[1,index])} level(s) x {int(cycles)} cycles")
        return ", ".join(parts)

    @staticmethod
    def _cleanup_worker_directory(directory, stem):
        """Remove only a successful worker's native error file and empty directory.

        Native ranks need separate working directories while running. Leave all
        other files alone so diagnostics from failures and caller files survive.
        """
        try:
            (directory / (stem + ".error")).unlink(missing_ok=True)
            directory.rmdir()
        except OSError:
            pass  # Unexpected files or permissions leave the directory intact.

    @staticmethod
    def _cleanup_empty_native_files(directory, stem):
        """Discard empty native outputs after their Fortran units have closed.

        Result.output_paths describes file locations, not a promise that an
        optional log has data. Never glob a stem: it could match caller files.
        """
        suffixes = (".alpha", ".blomax", ".dynamic_patch", ".error",
                    ".out", ".out15", ".out20", ".prout", ".res",
                    ".subit_res", ".subit_turres", ".turres",
                    "_aesurf.dat", "_genforce.dat", "_ovrlp.bin", "_patch.bin",
                    "_precfl3d.error", "_precfl3d.out")
        for suffix in suffixes:
            path = directory / (stem + suffix)
            try:
                if path.is_file() and not path.is_symlink() and path.stat().st_size == 0:
                    path.unlink()
            except OSError:
                pass  # Cleanup must not turn a completed calculation into a failure.

    @staticmethod
    def _mesh_metadata(mesh):
        # Strip coordinates before copying: copying then stripping doubles global
        # storage on the host during installation/output preparation.
        zones = [deepcopy(replace(z, coordinates=None, coordinate_dimensions=tuple(z.dimensions)))
                 for z in mesh.zones]
        return CGNSMesh(zones, mesh.source)

    def _install_distributed_mesh(self, complete):
        metadata = self._checked(lambda: self._mesh_metadata(complete) if self.comm.rank == 0 else None)
        self.mesh = self.comm.bcast(metadata, root=0)
        if self.comm.size-1 > len(self.mesh.zones):
            raise ValueError("This MPI interface requires at least one zone per worker")
        # Input placement is temporary. After the first native solve, coordinates
        # move to the actual load-balancer owners, without a global broadcast.
        owners = tuple(1+i % (self.comm.size-1) for i in range(len(self.mesh.zones)))
        from_file = self.comm.bcast(
            bool(complete.source and all(z.coordinates is None for z in complete.zones))
            if self.comm.rank == 0 else None, root=0)
        if from_file:
            # Read only owned coordinates. In particular the host never loads a
            # global mesh merely to redistribute it. Topology stays replicated.
            names = {z.name for i,z in enumerate(self.mesh.zones) if owners[i] == self.comm.rank}
            loaded = self._checked(lambda: CGNSMesh.from_file(self.mesh.source, coordinate_zones=names)
                                   if names else None)
            local = {} if loaded is None else {z.name: z.coordinates for z in loaded.zones if z.name in names}
        else:
            pieces = self._checked(lambda: [
                {z.name: z.coordinates for i,z in enumerate(complete.zones) if owners[i] == rank}
                for rank in range(self.comm.size)] if self.comm.rank == 0 else None)
            local = self.comm.scatter(pieces, root=0)
        for zone in self.mesh.zones:
            zone.coordinates = local.get(zone.name)
        self._mesh_owners = owners
        self._zone_owners = None

    def _redistribute_coordinates(self, owners):
        if self.comm is None or tuple(owners) == self._mesh_owners:
            return
        def send_buffers():
            messages = [{} for _ in range(self.comm.size)]
            for i,zone in enumerate(self.mesh.zones):
                if zone.coordinates is not None:
                    messages[owners[i]][zone.name] = zone.coordinates
            return messages
        messages = self._checked(send_buffers)
        received = self.comm.alltoall(messages)
        local = {name: xyz for part in received for name,xyz in part.items()}
        for zone in self.mesh.zones:
            zone.coordinates = local.get(zone.name)
        self._mesh_owners = tuple(owners)

    def _gather_mesh(self, mesh):
        if self.comm is None:
            return mesh
        local = {z.name: z.coordinates for z in mesh.zones if z.coordinates is not None}
        pieces = self.comm.gather(local, root=0)
        def assemble():
            if self.comm.rank != 0:
                return None
            complete = self._mesh_metadata(mesh)
            xyz = {name:value for piece in pieces for name,value in piece.items()}
            for zone in complete.zones:
                zone.coordinates = xyz[zone.name]
            return complete
        return self._checked(assemble)

    def _native_arrays(self):
        self._progress("Preparing native mesh, interfaces, and boundary conditions")
        if self.comm is None:
            prepared = self._checked(self._arrays)
            self._progress("Native input ready")
            return prepared
        complete = self._gather_mesh(self.mesh)
        def prepare():
            if self.comm.rank != 0:
                return None
            local_mesh = self.mesh
            try:
                self.mesh = complete
                return self._arrays()
            finally:
                self.mesh = local_mesh
        prepared = self._checked(prepare)
        arrays, keys = self.comm.bcast((prepared[0], self._surface_keys) if self.comm.rank == 0 else None, root=0)
        self._surface_keys = keys
        self._progress("Native input ready")
        return arrays, prepared[1] if self.comm.rank == 0 else np.empty((0,3), order="F")

    def __call__(self, *, output_directory=None, final=True):
        """等价于 solve()：返回 SolverResult，沿用参数更新、续算和输出编号规则。

        MPI 下仍需本对象 comm 中的所有 rank 一起调用。
        """
        return self.solve(output_directory=output_directory, final=final)

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

    @property
    def nSolutions(self):
        """Number of successfully finalized CFD calculations (independent of nSolve)."""
        return self._solution_counter

    def _numbered_output_names(self, index, chunk=None):
        stem = f"{self.name}_{index:03d}"
        if chunk is not None:
            stem += f"_chunk_{chunk:03d}"
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
        if self._solution_pending and (not continuing or session is None or session[0] != identity):
            raise ValueError("Finalize or resetFlow() before changing an unfinished calculation/output identity")
        if ((self._solution_pending or not self.options.numberSolutions) and continuing and session is not None
                and session[0] == identity):
            return session[1], True, self._output_session_iterations, identity
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
            mesh = self._checked(lambda: CGNSMesh.from_file(
                                     stamp[0], coordinate_zones=None if self.comm is None else ())
                                 if self.comm is None or self.comm.rank == 0 else None)
            if self.comm is None:
                self.mesh = mesh
            else:
                self._install_distributed_mesh(mesh)
            mesh = None
            # File reload changes discretization; reattach cuts collectively.
            def reattach():
                from .surface_analysis import direction_index, slice_references
                surface = self._analysis_surface()
                for request in self._slice_requests:
                    if request["sliceType"] == "relative":
                        members = self._analysis_members(request["groupName"], surface)
                        request["members"] = members
                        request["references"] = slice_references(
                            surface, members, direction_index(request["direction"]), request["position"])
                return self._slice_requests
            self._slice_requests = self._surface_collective(reattach, broadcast=True)
        self._loaded_grid_stamp = stamp
        return True

    def _mesh_signature(self):
        """Detect in-memory geometry, topology and boundary changes, including 1-to-1 maps."""
        digest = hashlib.sha256()
        for zone in self.mesh.zones:
            digest.update(zone.name.encode())
            digest.update(np.asarray(zone.dimensions, dtype=np.int64).tobytes())
            if self.comm is None:
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

    def _distributed_mesh_signature(self):
        def local():
            hashes = {}
            for i,zone in enumerate(self.mesh.zones):
                if (zone.coordinates is not None) != (self._mesh_owners[i] == self.comm.rank):
                    raise ValueError(f"Coordinate ownership changed for {zone.name}")
                if zone.coordinates is not None:
                    if zone.coordinates.shape != (*zone.coordinate_dimensions, 3) or not np.isfinite(zone.coordinates).all():
                        raise ValueError(f"Invalid local coordinate array for {zone.name}")
                    hashes[zone.name] = hashlib.sha256(np.ascontiguousarray(zone.coordinates).tobytes()).hexdigest()
            return self._mesh_signature(), hashes
        topology, local_hashes = self._checked(local)
        pieces = self.comm.allgather(local_hashes)
        hashes = {name:value for part in pieces for name,value in part.items()}
        if sum(map(len,pieces)) != len(self.mesh.zones) or set(hashes) != {z.name for z in self.mesh.zones}:
            raise RuntimeError("Distributed mesh has missing or duplicate coordinate owners")
        digest = hashlib.sha256(topology.encode())
        for zone in self.mesh.zones:
            digest.update(hashes[zone.name].encode())
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
        reset = set(changed) & (_OPTION_CATEGORIES["physics"] | _OPTION_CATEGORIES["mesh"] | _OPTION_CATEGORIES["runtime"])
        if mesh_signature != self._successful_mesh_signature:
            reset.add("mesh")
        return changed, tuple(sorted(reset))

    def getFlowState(self):
        """Return an independent Python snapshot, including native turbulence/ghost state."""
        if self.options.resident or self._resident_context is not None:
            raise NotImplementedError("Full-state memory snapshots are not yet implemented for resident=True")
        if self._flow_state is None:
            raise RuntimeError("Call solve() before retrieving flow state")
        return deepcopy(self._flow_state)

    def setFlowState(self, state):
        """Install this rank's snapshot; call on every rank before the next solve."""
        if self.options.resident or self._resident_context is not None:
            raise NotImplementedError("Full-state memory snapshots are not yet implemented for resident=True")
        from .restart import FlowState
        if not isinstance(state, FlowState):
            raise TypeError("state must be returned by getFlowState()")
        state.validate(self.mesh, self.options)
        self._flow_state = deepcopy(state)
        return self

    def resetFlow(self):
        """Start the next solve from freestream instead of cached state or restartFile."""
        self.close()
        self._flow_state = None
        self._solution_pending = False
        self._continuing = False
        self._loaded_restart_file = None
        self.options.restartFile = None
        return self

    def writeRestartFile(self, filename=None):
        """Save a portable whole-case checkpoint; collective when using MPI."""
        from .restart import write_restart, write_restart_partitioned
        self._checked(lambda: self._require_flow_state())
        self._checked(lambda: self._flow_state.validate(self._last_mesh, self._last_options))
        path = self._checked(lambda: self._output_path(
            filename, self._last_output_names["checkpoint"] if self._last_result else "")
            if self.comm is None or self.comm.rank == 0 else None)
        if self.comm is None:
            write_restart(path, self._flow_state)
        else:
            write_restart_partitioned(path, self._flow_state, self._last_mesh, self.comm, self._checked)
        return path

    def _restart_owners(self, owners=None):
        if owners is None:
            owners = getattr(self, "_zone_owners", None)
        if owners is None or len(owners) != len(self.mesh.zones):
            owners = tuple(1+i % (self.comm.size-1) for i in range(len(self.mesh.zones)))
        return owners

    def _require_flow_state(self):
        if self.options.resident or self._resident_context is not None:
            raise NotImplementedError("Full-state checkpoint export is not yet implemented for resident=True")
        if self._flow_state is None:
            raise RuntimeError("No flow state is available")

    def _scatter_state(self, state, owners=None):
        from .restart import partition_state
        if self.comm is None:
            return state
        owners = self._restart_owners(owners)
        pieces = self._checked(lambda: partition_state(state, self.mesh, owners, self.comm.size)
                               if self.comm.rank == 0 else None)
        return self.comm.scatter(pieces, root=0)

    def _stage_flow_state(self, state, path):
        """Stream owned zone records to native host I/O, without a full root cache."""
        from .restart import zone_payloads
        if self.comm is None:
            self._checked(lambda: path.write_bytes(state.native) if state is not None
                          else path.unlink(missing_ok=True))
            return
        if state is None:
            self._checked(lambda: path.unlink(missing_ok=True) if self.comm.rank == 0 else None)
            return
        from ._streaming import byte_chunks, transfer_bytes
        parts = self._checked(lambda: zone_payloads(state, self.mesh))
        descriptions = self.comm.allgather({i:len(data) for i,data in parts.items()})
        owners = {i:rank for rank,indices in enumerate(descriptions) for i in indices}
        lengths = {i:size for indices in descriptions for i,size in indices.items()}
        stream = self._checked(lambda: path.open("wb") if self.comm.rank == 0 else None)
        transfer_comm = self.comm.Dup()
        try:
            for index in range(len(self.mesh.zones)):
                owner = owners[index]
                chunks = byte_chunks(parts[index]) if self.comm.rank == owner else None
                transfer_bytes(transfer_comm, owner, 0, chunks,
                               lambda chunk: stream.write(chunk), lengths[index], self._checked)
        finally:
            try:
                self._checked(lambda: stream.close() if stream is not None else None)
            finally:
                transfer_comm.Free()

    def _prepare_flow_state(self, discard_cached=False):
        from .restart import read_restart, read_restart_partitioned
        restart_file = str(Path(self.options.restartFile).resolve()) if self.options.restartFile is not None else None
        reload_file = restart_file is not None and restart_file != self._loaded_restart_file
        if self.comm is not None:
            reload_file = any(self.comm.allgather(reload_file))
        state = None if discard_cached else self._flow_state
        if reload_file:
            if self.comm is None:
                state = read_restart(restart_file, self.mesh, self.options)
            else:
                state = read_restart_partitioned(restart_file, self.mesh, self.options,
                                                 self._restart_owners(), self.comm, self._checked)
            self._loaded_restart_file = restart_file
        self._checked(lambda: state.validate(self.mesh,self.options) if state is not None else None)
        if self.comm is not None:
            signatures = self.comm.allgather(None if state is None else
                        (state.global_digest or state.digest, state.zone_indices))
            if any(value is None for value in signatures):
                if not all(value is None for value in signatures):
                    raise RuntimeError("CFL3D cached flow states differ within comm")
            elif len({value[0] for value in signatures}) != 1:
                raise RuntimeError("CFL3D cached flow states differ within comm")
            elif all(value[1] is None for value in signatures):
                state = self._scatter_state(state if self.comm.rank == 0 else None)
            else:
                indices = [index for value in signatures for index in (value[1] or ())]
                if (any(value[1] is None for value in signatures)
                        or sorted(indices) != list(range(len(self.mesh.zones)))):
                    raise RuntimeError("CFL3D cached flow states differ in zone ownership within comm")
        self._flow_state = state
        return state

    def gatherFields(self, root=0):
        """Collect owned mean-flow arrays on root explicitly; MPI collective.

        Other ranks return None. The returned global dictionary is caller-owned
        and is never retained by CFL3D; normal result.fields remains rank-local.
        """
        self._checked(lambda: self._require_result())
        self._checked(lambda: _integer(root, "root", 0, self.comm.size-1 if self.comm is not None else 0))
        if self.comm is None:
            return {key: value.copy() for key, value in self._last_fields.items()}
        if len(set(self.comm.allgather(root))) != 1:
            raise RuntimeError("gatherFields root differs within comm")
        gathered = self.comm.gather(self._last_fields, root=root)
        if self.comm.rank == root:
            return {key: value for part in gathered for key, value in part.items()}
        return None

    @property
    def solutionIsCurrent(self):
        """Whether cached result queries are valid for the requested geometry."""
        return self._last_result is not None and self._pending_geometry is None

    def _require_result(self):
        if self._last_result is None:
            raise RuntimeError("Call solve() successfully before accessing a solution")

    def _output_path(self, filename, default):
        self._require_result()
        path = Path(filename) if filename is not None else self._last_output_directory / default
        if self._last_mesh.source and path.resolve() == Path(self._last_mesh.source).resolve():
            raise ValueError("Solution output must not overwrite the input CGNS mesh")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def writeVolumeSolution(self, filename=None):
        """Write whole-case CGNS/ADF; collective when using MPI."""
        from .solution_io import volume_output
        self._checked(self._require_result)
        path = self._checked(lambda: self._output_path(filename, self._last_output_names["volume"])
                             if self.comm is None or self.comm.rank == 0 else None)
        self._write_zones(lambda: volume_output(path, self._last_mesh, self._last_options))
        return path

    def _write_zones(self, output, *, coordinates=True):
        """Drive a root writer collectively, retaining at most one received zone.

        This bounds Python output temporaries by the largest zone rather than
        the global case. CGNS and PLOT3D still need zone-sized conversion arrays.
        Root I/O errors close the atomic output with the exception on every rank.
        """
        from ._streaming import array_chunks, transfer_bytes
        stack = ExitStack()
        transfer_comm = self.comm.Dup() if self.comm is not None else None
        try:
            writer = self._checked(lambda: stack.enter_context(output())
                                   if self.comm is None or self.comm.rank == 0 else None)
            for zone in self._last_mesh.zones:
                if self.comm is None:
                    writer(zone, self._last_fields[zone.name])
                    continue
                owner = self._last_zone_owners[zone.name]
                xyz = q = None
                for is_coordinate in ((True, False) if coordinates else (False,)):
                    shape = (*zone.dimensions, 3) if is_coordinate else (*tuple(n-1 for n in zone.dimensions), 5)
                    order = "F" if is_coordinate else "C"
                    def buffers():
                        source = (zone.coordinates if is_coordinate else self._last_fields[zone.name]) \
                                 if self.comm.rank == owner else None
                        if self.comm.rank == owner and (source is None or source.shape != shape
                                                        or source.dtype != np.float64):
                            raise ValueError(f"Invalid output array for zone {zone.name}")
                        target = np.empty(shape, dtype=np.float64, order=order) if self.comm.rank == 0 else None
                        return source, target
                    source, target = self._checked(buffers)
                    position = 0
                    def consume(chunk):
                        nonlocal position
                        view = memoryview(target.ravel(order=order)).cast("B")
                        view[position:position+len(chunk)] = chunk
                        position += len(chunk)
                    chunks = array_chunks(source, fortran_order=is_coordinate) if self.comm.rank == owner else None
                    transfer_bytes(transfer_comm, owner, 0, chunks, consume,
                                   int(np.prod(shape))*8, self._checked)
                    if is_coordinate:
                        xyz = target
                    else:
                        q = target
                    del source, target, chunks
                self._checked(lambda: writer(replace(zone, coordinates=xyz), q) if self.comm.rank == 0 else None)
                del xyz, q
            self._checked(stack.close)
        except BaseException:
            error = sys.exc_info()
            self._checked(lambda: stack.__exit__(*error))
            raise
        finally:
            if transfer_comm is not None:
                transfer_comm.Free()

    def writeSurfaceAnalysis(self, prefix=None):
        """Write registered lift distributions and slices as Tecplot ASCII .dat files.

        Automatic analyses are independent of the main wall-output flag. With no
        prefix, use the same case name/index as the most recent completed solve.
        This operation is collective in MPI, including deferred output.
        """
        if self.comm is not None and self._surface_context is None:
            self._checked(self._require_result)
            self._check_surface_requests()
            if not self.nLiftDist and not self.nSlice:
                return {}
            return self._surface_collective(lambda: self.writeSurfaceAnalysis(prefix))
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
        """Write wall quads as Tecplot ASCII; collective in MPI, including analyses."""
        if self.comm is not None and self._surface_context is None:
            from .solution_io import write_surface_partitioned
            self._checked(self._require_result)
            members = self._checked(lambda: self._group_members(groupName))
            wall_keys = {key for part in self.comm.allgather(tuple(self._last_wall_fields)) for key in part}
            members = members & wall_keys
            path = self._checked(lambda: self._output_path(filename, self._last_output_names["surface"])
                                 if self.comm.rank == 0 else None)
            self.writeSurfaceAnalysis(None if filename is None else Path(filename).with_suffix(""))
            write_surface_partitioned(path, self._last_mesh, self._last_wall_fields, members,
                                      self._last_zone_owners, self.comm, self._checked)
            return path
        path = self._output_path(filename, self._last_output_names["surface"] if self._last_result else "")
        from .solution_io import write_surface
        members = self._group_members(groupName)
        # Analyses retain their own registered groups, independent of the wall-file selection.
        self.writeSurfaceAnalysis(None if filename is None else path.with_suffix(""))
        context = self._surface_context
        write_surface(path, self._last_mesh, self._last_wall_fields if context is None else context["walls"],
                      members, vertices=None if context is None else context["vertices"])
        return path

    def writeFlowSolution(self, prefix=None):
        """Write whole-case native PLOT3D .x/.q; collective when using MPI."""
        from .solution_io import flow_output
        self._checked(self._require_result)
        def paths():
            target = self._output_path(prefix, self._last_output_names["flow_prefix"])
            for suffix in (".x", ".q"):
                self._output_path(str(target)+suffix, "")
            return target
        target = self._checked(lambda: paths() if self.comm is None or self.comm.rank == 0 else None)
        self._write_zones(lambda: flow_output(target, self._last_mesh, self._last_options, self._last_final_alpha))
        return (Path(str(target)+".x"), Path(str(target)+".q")) if target is not None else None

    def _cgns_surface_groups(self):
        families, zones, boundaries, qualified, numbered = {}, {}, {}, {}, {}
        for zone in self.mesh.zones:
            for boundary in zone.boundaries:
                key = (zone.name, boundary.name)
                path = f"{zone.name}/{boundary.name}"
                if path in qualified:
                    raise ValueError(f"Duplicate CGNS boundary path {path}")
                qualified[path] = {key}
                zones.setdefault(zone.name, set()).add(key)
                boundaries.setdefault(boundary.name, set()).add(key)
                stem, separator, suffix = boundary.name.rpartition("_")
                if stem and separator and suffix.isdigit():
                    numbered.setdefault(stem, set()).add(key)
                for family in (zone.family, *zone.additional_families,
                               boundary.family, *boundary.additional_families):
                    if family:
                        families.setdefault(family, set()).add(key)
        # Numbered BC patches are commonly one physical part. Keep every exact
        # name/path available while making the unsuffixed stem an aggregate.
        for stem, members in numbered.items():
            boundaries.setdefault(stem, set()).update(members)
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

    def getGroupNames(self, detailed=False):
        """List component groups; detailed=True also lists zone and patch selectors."""
        if not isinstance(detailed, bool):
            raise ValueError("detailed must be a boolean")
        base = self._last_base_groups if self._last_result is not None else self._cgns_surface_groups()
        if detailed:
            return sorted(set(base) | self._group_definitions.keys())
        mesh = self._last_mesh if self._last_result is not None else self.mesh
        names = set(self._group_definitions)
        for zone in mesh.zones:
            names.update(name for name in (zone.family, *zone.additional_families) if name)
            for boundary in zone.boundaries:
                families = (boundary.family, *boundary.additional_families)
                names.update(name for name in families if name)
                stem, separator, suffix = boundary.name.rpartition("_")
                if stem and separator and suffix.isdigit():
                    names.add(stem)
                elif not any(families) and not zone.family and not zone.additional_families:
                    names.add(boundary.name)
        return sorted(names & (set(base) | self._group_definitions.keys()))

    def getSurfaceMesh(self, groupName=None):
        """Return partitioned wall geometry before or after solve(), without I/O.

        Collective within comm. Coordinates (N,3), in metres, contain unique
        coupling nodes owned by this rank, ordered by stable int64 node_ids.
        Connectivity (M,4) uses GLOBAL node IDs, including remote nodes; faces
        follow native zone ownership. face_ids and face_keys identify each wall
        cell. Host arrays can be empty. Node ownership stays fixed across the
        first native load balance, so future MPhys vector shapes remain stable.

        After solve, use its geometry snapshot, just as getSolution does. Node
        IDs use CGNS topology, not coordinate merging, and are not compact indices.
        Returned arrays are independent copies. This does not construct a zipper
        or support periodic/nonconforming interfaces.
        """
        def validate():
            if groupName is not None and not isinstance(groupName, str):
                raise ValueError("groupName must be a CGNS/group name or None")
        self._checked(validate)
        if self.comm is not None and len(set(self.comm.allgather(groupName))) != 1:
            raise RuntimeError("Surface groupName differs within comm")
        from .surface_mesh import build_surface_mesh
        members = self._checked(lambda: self._surface_group_members(groupName))
        if self.comm is not None and len(set(self.comm.allgather(tuple(sorted(members))))) != 1:
            raise RuntimeError("Surface group membership differs within comm")
        solved = self._last_result is not None
        mesh = self._last_mesh if solved else self.mesh
        if self.comm is not None and not solved:
            controls = self._checked(lambda: tuple(sorted(
                (key, int(value[0])) for key, value in self.boundary_overrides.items())))
            if len(set(self.comm.allgather(controls))) != 1:
                raise RuntimeError("Surface boundary overrides differ within comm")

        def wall_keys():
            if solved:
                return set(self._last_wall_fields)
            return {(z.name, b.name) for z in mesh.zones if z.coordinates is not None
                    for b in z.boundaries
                    if abs(self.boundary_overrides.get(
                        (z.name, b.name), _BC_TYPES.get(b.bc_type, (0, ())))[0])
                    in (1005,1006,2004,2014,2024,2034,2016)}

        owners = ((0,)*len(mesh.zones) if self.comm is None else
                  tuple(self._last_result.zone_owners[z.name] for z in mesh.zones) if solved else
                  self._mesh_owners)
        return build_surface_mesh(mesh, members, self._checked(wall_keys), owners, self.comm, self._checked)

    def getSurfaceForces(self, groupName=None, *, component="total"):
        """Return owned nodal forces in N, aligned with getSurfaceMesh(groupName).

        Collective within comm; call after a successful solve. The returned
        independent (N,3) array uses the input Cartesian axes and the same local
        node_ids/order as getSurfaceMesh. component is 'total', 'pressure', or
        'viscous'. This is the nodal-force convention of ADflow's
        forcesAsTractions=False, suitable for FUNtoFEM/MELD; it is not traction.

        Each native wall quad contributes one quarter of its integrated force
        to each corner. Shared nodes SUM the contributions across patches/ranks;
        do not divide by area or by the number of adjacent cells. CFL3D computes
        its moment arm at the mean of the four corner coordinates, so this
        distribution preserves its total force and moment (up to roundoff and
        the tolerance used to identify coincident connected vertices).

        Use saved flow/reference settings, as getSolution does. If the solve had
        no freestream_pressure, a pressure supplied in live options afterwards
        may provide the dimensional scale. Without it this method raises: a
        force array must not silently contain nondimensional coefficients.
        No native solve, file output, or global surface gather is performed.
        """
        def controls():
            self._require_result()
            if self._last_result is None:
                raise RuntimeError("Call solve() successfully before querying surface forces")
            if not isinstance(component, str) or component.lower() not in ("total", "pressure", "viscous"):
                raise ValueError("component must be 'total', 'pressure', or 'viscous'")
            options = self._last_options
            pressure = options.freestream_pressure
            if pressure is None:
                pressure = self.options.freestream_pressure
            if pressure is None:
                raise ValueError("No freestream pressure was set for the CFL3D calculation or provided in options; "
                                 "absolute nodal forces in N cannot be determined. Set freestream_pressure.")
            # Native wall_forces are already integrated over each cell and
            # divided by q_inf*Sref. Restore N once; never multiply by area again.
            scale = .5 * options.gamma * pressure * options.mach**2 * options.sref
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Dimensional force scale must be finite and positive")
            return component.lower(), float(scale)

        kind, scale = self._checked(controls)
        if self.comm is not None and len(set(self.comm.allgather((kind, scale)))) != 1:
            raise RuntimeError("Surface force component/dimensional scale differs within comm")
        surface = self.getSurfaceMesh(groupName)
        from .surface_mesh import distribute_surface_forces
        values = distribute_surface_forces(self._last_mesh, surface, self._last_wall_forces,
                                            scale, self.comm, self._checked)
        if kind == "pressure":
            return values[:, :3].copy()
        if kind == "viscous":
            return values[:, 3:].copy()
        return values[:, :3] + values[:, 3:]

    def getSolution(self, groupName=None):
        """Return cached native pressure+viscous coefficients and SI loads after solve.

        CL/CD/CDp/CDv use native wind axes; CX/CY/CZ and moments use input
        mesh axes. Forces are N, moments N*m, and wettedArea m^2. Missing
        freestream pressure warns and omits dimensional loads. A pressure added
        to live options can supply missing dimensional information after solve.
        A group prefixes all numerical keys and their entries in units.
        This is a local call: any rank may query without an MPI collective.
        """
        self._require_result()
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

    @staticmethod
    def _wall_patch_area(quads):
        """Area of a wall patch in m^2, using both triangles of every quad."""
        area = 0.0
        for a, b in ((1, 2), (2, 3)):
            area += .5 * float(np.linalg.norm(np.cross(
                quads[:, a] - quads[:, 0], quads[:, b] - quads[:, 0]), axis=1).sum())
        return area

    def _cache_wall_areas(self):
        """Collect only per-boundary scalars so later load queries stay local."""
        def local():
            surface = self._analysis_surface(require_solution=True)
            return {key: self._wall_patch_area(patch.quads) for key, patch in surface.patches.items()}
        areas = self._checked(local)
        self._last_wall_areas = {key:value for part in self.comm.allgather(areas) for key,value in part.items()}

    def _wetted_area(self, members):
        """Return selected cached wall area in m^2, counting overlapping members once."""
        if self.comm is not None:
            return sum(self._last_wall_areas.get(key, 0.0) for key in sorted(members))
        surface = self._analysis_surface(require_solution=True)
        return sum(self._wall_patch_area(surface.patches[key].quads)
                   for key in sorted(members & surface.patches.keys()))

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

    def _wall_geometry_fields(self, mesh):
        walls = {}
        for zone in mesh.zones:
            if zone.coordinates is None:
                continue
            for boundary in zone.boundaries:
                key = zone.name, boundary.name
                code = self.boundary_overrides.get(key, _BC_TYPES.get(boundary.bc_type, (0, ())))[0]
                if abs(code) in (1005,1006,2004,2014,2024,2034,2016):
                    counts = np.diff(np.sort(boundary.point_range, axis=1), axis=1).ravel()
                    walls[key] = np.zeros((*counts[counts > 0], 6))
        return walls

    def _check_surface_requests(self):
        if self.comm is None:
            return
        specs = [self._lift_distributions,
                 [{key:value for key,value in request.items() if key not in ("references", "members")}
                  for request in self._slice_requests]]
        encoded = json.dumps(specs, sort_keys=True)
        if len(set(self.comm.allgather(encoded))) != 1:
            raise RuntimeError("Surface registrations differ within comm; register and clear them on every rank")

    def _surface_collective(self, operation, *, broadcast=False):
        """Gather wall-only data temporarily; never install a persistent root copy.

        Public methods re-enter locally on root inside this short-lived context.
        Nested queries/writers then reuse the gathered surface without entering
        another collective. Only their final result can be broadcast.
        """
        if self.comm is None or self._surface_context is not None:
            return operation()
        self._check_surface_requests()
        def local_payload():
            from .solution_io import wall_vertices
            solved = self._last_result is not None
            mesh = self._last_mesh if solved else self.mesh
            walls = self._last_wall_fields if solved else self._wall_geometry_fields(mesh)
            forces = self._last_wall_forces if solved else None
            vertices = {(z.name,b.name): wall_vertices(z,b).copy() for z in mesh.zones
                        if z.coordinates is not None for b in z.boundaries if (z.name,b.name) in walls}
            return vertices, walls, forces
        payload = self._checked(local_payload)
        gathered = self.comm.gather(payload, root=0)
        def run():
            if self.comm.rank != 0:
                return None
            from .surface_analysis import SurfaceMesh
            vertices = {k:v for piece in gathered for k,v in piece[0].items()}
            walls = {k:v for piece in gathered for k,v in piece[1].items()}
            forces = ({k:v for piece in gathered for k,v in piece[2].items()}
                      if self._last_result is not None else None)
            mesh = self._last_mesh if self._last_result is not None else self.mesh
            surface = SurfaceMesh(mesh, walls, forces, sref=self.options.sref, vertices=vertices)
            self._surface_context = dict(vertices=vertices, walls=walls, surface=surface)
            try:
                return operation()
            finally:
                self._surface_context = None
        result = self._checked(run)
        return self.comm.bcast(result, root=0) if broadcast else result

    def _analysis_surface(self, require_solution=False):
        from .surface_analysis import SurfaceMesh
        if require_solution:
            self._require_result()
        if self._surface_context is not None:
            return self._surface_context["surface"]
        if require_solution and self._last_result is None:
            raise RuntimeError("Call solve() successfully before querying surface analysis")
        if self._last_result is not None:
            if self._surface_analysis_mesh is None:
                self._surface_analysis_mesh = SurfaceMesh(
                    self._last_mesh, self._last_wall_fields, self._last_wall_forces)
            return self._surface_analysis_mesh
        return SurfaceMesh(self.mesh, self._wall_geometry_fields(self.mesh), sref=self.options.sref)

    def _surface_group_members(self, groupName):
        if self._last_result is not None:
            members = self._group_members(groupName)
        else:
            base = self._cgns_surface_groups()
            if groupName is None:
                members = set().union(*base.values()) if base else set()
            elif isinstance(groupName, str) and groupName in base:
                members = base[groupName]
            elif isinstance(groupName, str) and groupName in self._group_definitions:
                members = set().union(*(base[name] for name in self._group_definitions[groupName]))
            else:
                raise ValueError(f"Unknown surface group {groupName!r}; inspect getGroupNames()")
        return members

    def _analysis_members(self, groupName, surface):
        members = self._surface_group_members(groupName)
        members = frozenset(members) & surface.patches.keys()
        surface.selected(members)  # Reject groups with no walls before registering requests.
        return frozenset(members)

    def addLiftDistribution(self, nSegments, direction, groupName=None):
        """Register conservative strip loads along x/y/z (typically 150-250 strips).

        None selects all walls; CGNS names and custom groups are supported.
        Collective in MPI. Output accompanies writeTecplotSurfaceSolution();
        query with getLiftDistribution().
        """
        if self.comm is not None and self._surface_context is None:
            def register():
                value = self.addLiftDistribution(nSegments, direction, groupName)
                return value, self._lift_distributions
            value, self._lift_distributions = self._surface_collective(register, broadcast=True)
            return value
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
        Collective in MPI. Both return face-centered Cp/T/Cf/y+ and connected lines.
        """
        if self.comm is not None and self._surface_context is None:
            def register():
                value = self.addSlices(direction, positions, sliceType, groupName)
                return value, self._slice_requests
            value, self._slice_requests = self._surface_collective(register, broadcast=True)
            return value
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
        This query is collective in MPI; only the computed strip data is broadcast.
        """
        if self.comm is not None and self._surface_context is None:
            return self._surface_collective(lambda: self.getLiftDistribution(index), broadcast=True)
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
        Collective in MPI; detailed wall arrays are gathered only temporarily.
        """
        if self.comm is not None and self._surface_context is None:
            return self._surface_collective(lambda: self.getSlices(index), broadcast=True)
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
        """Remove distributions/cuts on every MPI rank before changing mesh topology."""
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
        # Native IFORCE has one digit per grid direction: 1/2 selects its lower/
        # upper face, 3 both. Avoid integrating farfield, symmetry, and interface
        # faces every cycle; keep unfamiliar override codes enabled so their
        # previous force behavior is preserved.
        for g in range(len(zones)):
            blocks[3, g] = sum(
                (1 if f % 2 == 0 else 2) * (100, 10, 1)[f // 2]
                for f in range(6)
                if any(abs(code) not in (0, 1001, 1002, 1003)
                       for code, _, _, _ in segments[g, f])
            )
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
                    run_control=np.array([-len(zones), 0, 0, o.restart_interval, 0, i2d, 1, 1], dtype=np.int32),
                    mg=np.array([mseq, int(o.multigrid_levels > 1), 0, o.multigrid_up_iterations, o.multigrid_cycle], dtype=np.int32),
                    smoothing=smoothing,
                    stages=stages, keywords=np.array([int(o.full_navier_stokes and model != 0), o.turbulence_subiterations,
                        o.turbulence_cfl, o.eddy_viscosity_limit, o.gamma,
                        99999.0 if o.target_cl is None else o.target_cl,
                        o.alpha_relaxation, o.alpha_step_limit, o.alpha_update_interval,
                        -1. if o.freestream_turbulence_intensity is None else o.freestream_turbulence_intensity,
                        -1. if o.freestream_eddy_viscosity is None else o.freestream_eddy_viscosity,
                        int(o.keep_ambient_turbulence),
                        *_MODEL_FLAGS.get(o.turbulence.lower(), (0, 0, 0))]),
                    scheme=np.array([*_directions(o.inversion, "inversion"),
                                     *_directions(o.limiter, "limiter")], dtype=np.int32),
                    flux=np.array(_directions(o.flux, "flux"), dtype=np.int32),
                    rkap=np.array(_directions(o.kappa, "kappa")),
                    npoints=len(xyz), ncells=int(np.prod(dims-1, axis=0).sum())), xyz

    def _native_module(self):
        from importlib import import_module
        name = "_native_mpi" if self.comm is not None else "_native"
        try:
            module = import_module(f".{name}", __package__)
        except ImportError as exc:
            raise RuntimeError("Build the extension with python tools/build_python.py"
                               + (" --mpi" if self.comm is not None else "")) from exc
        if bool(module.py_build_info()) != (self.comm is not None):
            raise RuntimeError("Native extension MPI mode does not match the solver")
        if getattr(module, "py_api_version", lambda: 0)() != 21:
            raise RuntimeError("Native interface changed; rebuild with python tools/build_python.py"
                               + (" --mpi" if self.comm is not None else "") + " and restart Python")
        return module

    def _read_native_result(self, _native, arrays, nrows):
        rank = self.comm.rank if self.comm is not None else 0
        zone_owners, local_cells, local_wall_cells = _native.py_get_partition(len(self.mesh.zones))
        self._zone_owners = tuple(map(int, zone_owners))
        q = _native.py_get_solution(local_cells)
        history = _native.py_get_history(nrows)
        final_alpha = float(_native.py_get_final_alpha())
        surface_values = _native.py_get_surface_coefficients(arrays["bcs"].shape[1])
        wall_counts = (arrays["bcs"][4]-arrays["bcs"][3])*(arrays["bcs"][6]-arrays["bcs"][5])
        wall_mask = np.isin(np.abs(arrays["bcs"][2]), (1005,1006,2004,2014,2024,2034,2016))
        wall_mask &= np.asarray(self._zone_owners)[arrays["bcs"][0]-1] == rank
        wall_counts = np.where(wall_mask, wall_counts, 0)
        if int(wall_counts.sum()) != local_wall_cells:
            raise RuntimeError("Native wall partition differs from Python metadata")
        wall_values = _native.py_get_wall_data(local_wall_cells)
        wall_force_values = _native.py_get_wall_forces(local_wall_cells)
        # CL control can update final_alpha after force evaluation: use the force-time angles.
        force_angles = _native.py_get_force_angles()
        return q, history, final_alpha, surface_values, wall_counts, wall_values, wall_force_values, force_angles

    def _solve_legacy_native(self, state, output_directory, output_names, stem,
                             changed_options, append_output, iteration_offset):
        rank = self.comm.rank if self.comm is not None else 0
        arrays, xyz = self._native_arrays()
        if state is not None:
            arrays["time_control"][1] = -1  # Native full-state restart, without old convergence history.
        if self.comm is not None:
            fingerprint = hashlib.sha256()
            for key,value in sorted(arrays.items()):
                fingerprint.update(key.encode())
                fingerprint.update(np.asarray(value).tobytes())
            if len(set(self.comm.allgather(fingerprint.hexdigest()))) != 1:
                raise RuntimeError("CFL3D mesh/boundary/native configuration differs within comm")
        _native = self._checked(self._native_module)
        # Native ranks retain independent work directories while using the same case/index.
        local_directory = output_directory if rank == 0 else output_directory / f"rank-{rank:04d}"
        self._checked(lambda: local_directory.mkdir(parents=True, exist_ok=True))
        checkpoint_path = output_directory / ".cfl3d-state.bin"
        def stage_checkpoint():
            if rank == 0:
                protected = [self.mesh.source, self.options.restartFile]
                if any(value and checkpoint_path.resolve() == Path(value).resolve() for value in protected):
                    raise ValueError("Input files must not use the internal .cfl3d-state.bin output path")
        self._checked(stage_checkpoint)
        self._stage_flow_state(state, checkpoint_path)
        acquired = _NATIVE_LOCK.acquire(blocking=False)
        old_directory = Path.cwd()
        native_comm = None
        completed = False
        try:
            self._progress("Installing native configuration")
            def check_lock():
                from ._runtime import check_owner
                check_owner(self)
                if not acquired:
                    raise RuntimeError("CFL3D uses process-global Fortran state; concurrent native solves within one Python process are unsupported")
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
                if status == 3:
                    raise RuntimeError("A resident native context is active in this Python process; close it before configuring another solver")
                if status:
                    raise RuntimeError(f"Native configuration failed ({status}); start a fresh interpreter")
                _native.py_set_coordinates(xyz)
                os.chdir(local_directory)
            self._checked(configure)
            del xyz  # The native host now owns its temporary coordinate input.
            self._progress(f"Native solve running: {self._stage_summary(arrays)}; "
                           f"log: {local_directory / output_names['log']}")
            status, nrows = _native.py_solve()
            if status:
                error_file = output_names["preprocess_error" if status == -99 else "error"]
                raise RuntimeError(f"CFL3D failed with status {status}; inspect {local_directory / error_file}")
            self._progress(f"Native solve finished: {nrows} cycles; extracting results")
            q, history, final_alpha, surface_values, wall_counts, wall_values, wall_force_values, force_angles = self._read_native_result(_native, arrays, nrows)
            self._progress("Native results extracted; caching restart state")
            from .restart import FlowState, metadata, read_restart_partitioned
            if self.comm is None:
                checkpoint = FlowState(checkpoint_path.read_bytes(), metadata(self.mesh,self.options), {})
                checkpoint.validate(self.mesh, self.options)
            else:
                checkpoint = read_restart_partitioned(checkpoint_path, self.mesh, self.options,
                                                       self._zone_owners, self.comm, self._checked)
            self._checked(lambda: checkpoint_path.unlink(missing_ok=True) if rank == 0 else None)
            _native.py_release_buffers()
            self._progress("Restart state cached")
            completed = True
        finally:
            os.chdir(old_directory)
            if native_comm is not None:
                native_comm.Free()
            if acquired:
                _NATIVE_LOCK.release()
            if completed and rank != 0:
                self._cleanup_worker_directory(local_directory, stem)
        return arrays, q, history, final_alpha, surface_values, wall_counts, wall_values, wall_force_values, force_angles, checkpoint

    def _solve_resident_native(self, output_directory, stem,
                               changed_options, append_output, iteration_offset):
        from ._runtime import ResidentContext
        rank = self.comm.rank if self.comm is not None else 0
        directory = output_directory if rank == 0 else output_directory / f"rank-{rank:04d}"
        # A recoverable directory failure must not leave an unstarted context
        # that a later call mistakes for a resumable native workspace.
        self._checked(lambda: directory.mkdir(parents=True, exist_ok=True))
        xyz = None
        if self._resident_context is None:
            arrays, xyz = self._native_arrays()
            native = self._checked(self._native_module)
            self._resident_context = ResidentContext(self, native, arrays)
        context = self._resident_context
        try:
            values = context.advance(directory, stem, append_output, iteration_offset,
                                     xyz=xyz, changed_cfl="cfl" in changed_options)
        except BaseException:
            if not context.started:
                self._resident_context = None
            raise
        if self._pending_geometry is not None:
            for zone in self.mesh.zones:
                if zone.coordinates is not None:
                    zone.coordinates = self._pending_geometry[zone.name]
            self._pending_geometry = None
        self._last_runtime_info = context.info()
        self._progress(f"Native solve finished: {len(values[1])} new cycles; results extracted")
        return context.arrays, values

    def getVolumeCoordinates(self):
        """Return owned current coordinates (N,3), metres, CGNS i-fastest order.

        No reference grid is saved. The caller may retain this independent copy
        as x0. Re-fetch after the first solve, which may change zone ownership.
        Pending coordinates are not installed until the next successful solve.
        """
        from .mesh import VolumeMeshLayout
        return self._checked(lambda: VolumeMeshLayout(self.mesh).coordinates(self.mesh))

    def setVolumeCoordinates(self, coordinates):
        """Stage absolute owned coordinates for a resident steady continuation.

        Collective within comm. Topology, block ownership and vertex order must
        remain unchanged. Native geometry is refreshed once at the next solve
        boundary; q and turbulence remain warm initial guesses. No grid file is
        written. This API currently requires an already paused resident solver.
        """
        from .mesh import VolumeMeshLayout
        self._check_resident_members()
        def prepare():
            if self._resident_context is None or not self._resident_context.info()['paused']:
                raise RuntimeError("Coordinate updates require a paused resident solver")
            layout = VolumeMeshLayout(self.mesh)
            pending = layout.unpack(coordinates)
            changed = any(not np.array_equal(pending[z.name], z.coordinates)
                          for z in self.mesh.zones if z.coordinates is not None)
            # Native geometry may be partially overwritten after failure. Even
            # the last accepted Python coordinates must be staged again.
            return pending, changed or self._resident_context.failure is not None
        pending, changed = self._checked(prepare)
        if self.comm is not None:
            changed = any(self.comm.allgather(changed))
        if changed:
            from .mesh._interface_validation import validate_interfaces
            owners = self._mesh_owners if self.comm is not None else (0,)*len(self.mesh.zones)
            validate_interfaces(self.mesh, pending, owners, self.comm, self._checked)
        self._pending_geometry = pending if changed else None
        return self

    def getRuntimeInfo(self):
        """Return local resident counters; work_bytes is workspace capacity, not RSS."""
        return (self._resident_context.info() if self._resident_context is not None
                else dict(self._last_runtime_info))

    def close(self):
        """Collectively release a resident workspace; the next solve starts fresh.

        Last result readouts remain available for queries and deferred output.
        Closing does not create a full-state restart snapshot.
        """
        self._check_resident_members()
        self._pending_geometry = None
        if self._resident_context is not None:
            context = self._resident_context
            context.close()
            self._last_runtime_info = context.info()
            self._resident_context = None
        return self

    def _check_resident_members(self):
        if self.comm is not None:
            context = self._resident_context
            state = None if context is None else (context.started, context.cycles)
            states = self.comm.allgather(state)
            if any(item != states[0] for item in states):
                raise RuntimeError("CFL3D resident lifecycle states differ within comm")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def solve(self, *, output_directory=None, final=True):
        """Advance iterations; finalize/output by default, or accumulate a nonfinal chunk.

        All communicator members must supply the same final flag. Finalization
        retains the flow and resident workspace for subsequent calculations.
        Expected resident geometry failures return SolveFailure (fail=True),
        without a flow result; submit new coordinates before further advancement.
        """
        from ._runtime import check_owner, GeometryUpdateFailure
        self._progress_origin = self._progress_last = perf_counter()
        self._progress("Checking options and mesh before solve")
        self._checked(lambda: check_owner(self))
        self._check_resident_members()
        if (self._resident_context is not None and self._resident_context.failure is not None
                and self._pending_geometry is None):
            failure = self._resident_context.failure
            return SolveFailure(str(failure), failure.status)
        self._check_finalization_state(final)
        self._surface_analysis_mesh = None
        self.checkOptions()
        self._check_surface_requests()
        self._refresh_grid()
        mesh_signature = (self._checked(self._mesh_signature) if self.comm is None
                          else self._distributed_mesh_signature())
        changed_options, reset_reasons = self._checked(
            lambda: self._configuration_changes(mesh_signature))
        # The resident prototype exposes only these updates without rebuilding.
        # All other native controls conservatively start a new cold context.
        # This is distinct from legacy restart-based numerical continuation.
        if self._resident_context is not None:
            warm = _OPTION_CATEGORIES["io"] | {"iterations", "cfl", "freestream_pressure"}
            reset_reasons = tuple(sorted(set(reset_reasons) | (set(changed_options)-warm)))
        if self.comm is not None:
            decisions = self.comm.allgather((changed_options, reset_reasons, mesh_signature))
            if len(set(decisions)) != 1:
                raise RuntimeError("CFL3D configuration/mesh change decisions differ within comm")
        if self.comm is not None and len(set(self.comm.allgather(self._solve_counter))) != 1:
            raise RuntimeError("CFL3D solve counters differ within comm")
        def check_pending_reset():
            if self._pending_geometry is not None and reset_reasons:
                raise ValueError("Apply or cancel the pending coordinate update before changing cold-start options")
            if self._solution_pending and reset_reasons:
                raise ValueError("Finalize or resetFlow() before reinitializing an unfinished calculation")
        self._checked(check_pending_reset)
        if reset_reasons and self._resident_context is not None:
            self.close()
        if self.options.resident:
            def check_state():
                if self._flow_state is not None and not reset_reasons:
                    raise NotImplementedError("Call resetFlow() before enabling resident mode on a cached restart state")
            self._checked(check_state)
            state = None
            self._continuing = self._resident_context is not None
        else:
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
        chunk_index = self._output_chunk_counter if append_output else 0
        output_names = self._checked(lambda: self._numbered_output_names(solve_index))
        stem = f"{self.name}_{solve_index:03d}"
        # Legacy filename fields are 80 bytes; reserve room for auxiliary suffixes.
        self._checked(lambda: self._validate_native_stem(stem))
        self._progress(f"Validation complete; output index {solve_index:03d}")
        geometry_changed = self._pending_geometry is not None
        new_geometry = geometry_changed or (self._successful_mesh_signature is not None
                                            and mesh_signature != self._successful_mesh_signature)
        self._last_result = None  # Native failures must not expose stale loads.
        if self.options.resident:
            try:
                arrays, values = self._solve_resident_native(output_directory, stem,
                                                            changed_options, append_output, iteration_offset)
            except GeometryUpdateFailure as failure:
                self._pending_geometry = None
                self._solution_pending = False
                self._progress(f"Geometry update failed: {failure}; submit valid coordinates to retry")
                return SolveFailure(str(failure), failure.status)
            q, history, final_alpha, surface_values, wall_counts, wall_values, wall_force_values, force_angles = values
            checkpoint = None
        else:
            arrays, q, history, final_alpha, surface_values, wall_counts, wall_values, wall_force_values, force_angles, checkpoint = self._solve_legacy_native(
                state, output_directory, output_names, stem, changed_options, append_output, iteration_offset)
        base_groups = self._checked(self._cgns_surface_groups)
        self._redistribute_coordinates(self._zone_owners)
        fields = {}; offset = 0
        for zone_index, z in enumerate(self.mesh.zones):
            if self._zone_owners[zone_index] != rank:
                continue
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
            if key is not None and count > 0:
                shape = (int(arrays["bcs"][4,index]-arrays["bcs"][3,index]),
                         int(arrays["bcs"][6,index]-arrays["bcs"][5,index]), 6)
                walls[key] = wall_values[:,offset:offset+count].T.reshape(shape, order="F").copy()
                wall_forces[key] = wall_force_values[:,offset:offset+count].T.reshape(shape, order="F").copy()
            offset += count
        if new_geometry:
            self._geometry_version += 1
        segments = ()
        history_row = self._output_session_iterations if append_output else 0
        if not segments or segments[-1][1] != self._geometry_version:
            segments = (*segments, (history_row, self._geometry_version))
        result = SolverResult(fields, history, output_directory, final_alpha, surfaces, walls, wall_forces,
                              solve_index=solve_index, chunk_index=chunk_index, name=self.name,
                              geometry_version=self._geometry_version,
                              history_geometry_segments=segments,
                              output_paths={key: output_directory/value for key,value in output_names.items()},
                              iteration_offset=iteration_offset,
                              cumulative_history=history,
                              start_mode="restart" if self._continuing else "cold",
                              changed_options=changed_options, reset_reasons=reset_reasons,
                              zone_owners=dict(zip((z.name for z in self.mesh.zones), self._zone_owners)))
        self._last_surface_coefficients = {key: value.copy() for key, value in surfaces.items()}
        self._last_wall_fields = {key: value.copy() for key, value in walls.items()}
        self._last_wall_forces = {key: value.copy() for key, value in wall_forces.items()}
        self._last_force_angles = np.array(force_angles, copy=True)
        self._last_fields = {name: value.copy() for name, value in fields.items()}
        self._last_mesh = deepcopy(self.mesh)
        self._last_final_alpha, self._last_output_directory = final_alpha, output_directory
        self._last_base_groups, self._last_options = base_groups, self.options.snapshot()
        self._last_result = result
        # SolverResult is caller-owned; deferred collectives must not depend on
        # edits to its public ownership dictionary.
        self._last_zone_owners = dict(result.zone_owners)
        if self.comm is not None:
            self._cache_wall_areas()
        self._last_output_names = output_names
        # Reserve the index after native success, including output failures.
        # Pending chunks reuse it; only finalization increments nSolutions.
        self._solve_counter += 1
        self._solution_pending = True
        self._output_chunk_counter = chunk_index + 1
        self._next_output_index = solve_index + 1
        self._output_session = (output_identity, solve_index)
        self._output_session_iterations = iteration_offset + len(history)
        self._output_geometry_segments = segments
        self._successful_options = dict(self.options)
        if geometry_changed:
            mesh_signature = (self._distributed_mesh_signature() if self.comm is not None
                              else self._mesh_signature())
        self._successful_mesh_signature = mesh_signature
        if checkpoint is not None:
            checkpoint.fields = self._last_fields
        self._flow_state = checkpoint
        self._progress("Flow and boundary results cached")
        if final:
            self.finalizeSolution()
        elif self.options.writeIntermediateSolutions:
            names = self._numbered_output_names(solve_index, chunk_index)
            saved_names = self._last_output_names
            try:
                # Deferred writers share the same local cached result; only snapshot
                # paths change. Native log/history paths stay on the main index.
                self._last_output_names = names
                result.chunk_output_paths = {
                    key: output_directory / names[key] for key in
                    ("volume", "surface", "slices", "lift", "flow_prefix", "solution", "restart", "checkpoint")}
                self._write_solution_outputs()
            finally:
                self._last_output_names = saved_names
        if not self.options.resident:
            self._checked(lambda: self._cleanup_empty_native_files(output_directory, stem)
                          if rank == 0 else None)
        self._progress(f"Solve {solve_index:03d} complete" if final else
                       f"Solve {solve_index:03d} chunk {chunk_index:03d} complete (pending finalization)")
        return result

    def _check_finalization_state(self, final):
        def validate():
            if not isinstance(final, bool):
                raise TypeError("final must be a bool")
        self._checked(validate)
        if self.comm is not None:
            state = (final, self._solution_pending, self._solution_counter, self._output_chunk_counter)
            if len(set(self.comm.allgather(state))) != 1:
                raise RuntimeError("CFL3D finalization/chunk states differ within comm")

    def finalizeSolution(self):
        """Collectively output the latest result without advancing or closing the solver.

        Safe to repeat after success. If output fails, the calculation remains
        pending so this method can retry without performing more iterations.
        """
        self._check_finalization_state(True)
        self._checked(self._require_result)
        self.checkOptions()
        self._check_surface_requests()
        if not self._solution_pending:
            if not self._last_result.finalized:
                raise RuntimeError("No pending calculation; resetFlow() discarded the unfinished calculation")
            return self._last_result
        self._checked(lambda: self._select_output(self._last_output_directory, True)
                      if self.comm is None or self.comm.rank == 0 else None)
        # Writers must serialize final=True, but a failed output must remain retryable.
        self._last_result.finalized = True
        try:
            self._write_solution_outputs()
        except BaseException:
            self._last_result.finalized = False
            raise
        self._solution_pending = False
        self._solution_counter += 1
        return self._last_result

    def _write_solution_outputs(self):
        result = self._last_result
        output_names = self._last_output_names
        output_directory = self._last_output_directory
        rank = self.comm.rank if self.comm is not None else 0
        from .solution_io import result_output
        path = self._checked(lambda: self._output_path(None, output_names["solution"]) if rank == 0 else None)
        if self.options.writevolumesolution:
            self._progress("Writing automatic solution arrays")
            self._write_zones(lambda: result_output(path, result._saved_values()), coordinates=False)
        elif not self.options.resident:
            self._progress("Writing restart metadata")
            self._checked(lambda: result.save(path, include_volume=False) if rank == 0 else None)
        if self.options.writeTecplotSurfaceSolution:
            self._progress("Writing Tecplot surface and registered analyses")
            self.writeTecplotSurfaceSolution()
        elif self.nLiftDist or self.nSlice:
            self._progress("Writing registered surface analyses")
            self.writeSurfaceAnalysis()
        if self._flow_state is not None and (self.options.writevolumesolution or self.options.writeflowsolution):
            self._progress("Writing native restart output")
            self._stage_flow_state(self._flow_state, output_directory / output_names["restart"])
        if self.options.writevolumesolution:
            self._progress("Writing CGNS volume solution")
            self.writeVolumeSolution()
        if self.options.writeflowsolution:
            self._progress("Writing PLOT3D flow solution")
            self.writeFlowSolution()

    @staticmethod
    def _validate_native_stem(stem):
        if not stem.isascii() or len(stem) > 50:
            raise ValueError("Native output case name/index must be ASCII and at most 50 characters combined")
