"""Python-owned configuration and arrays passed directly to the native solver."""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import numpy as np
from .cgns import CGNSMesh

_NATIVE_LOCK = threading.Lock()
_MODELS = {"euler": 0, "laminar": 1, "sa": 5, "sst": 7, "komega": 6}
_BC_TYPES = {"BCFarfield": (1003, ()), "BCSymmetryPlane": (1001, ()),
             "BCWallInviscid": (1005, ()), "BCOutflowSupersonic": (1002, ()),
             "BCWallViscous": (2004, (0., 0.)), "BCWallViscousHeatFlux": (2004, (0., 0.))}


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
    eddy_viscosity_limit: float = 1e5
    gamma: float = 1.4
    limiter: int = 0
    flux: int = 1
    kappa: float = 1.0 / 3.0
    y_up: bool = False
    sref: float = 1.0
    cref: float = 1.0
    bref: float = 1.0
    moment_center: tuple[float, float, float] = (0., 0., 0.)

    def validate(self):
        ints = ("iterations", "multigrid_levels", "coarse_iterations", "multigrid_cycle",
                "turbulence_subiterations", "limiter", "flux", "cfl_ramp_steps",
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
                     "eddy_viscosity_limit", "gamma", "kappa", "sref", "cref", "bref", "cfl_ramp_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
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
        if self.limiter not in (0, 1, 2, 3, 4) or self.flux not in (0, 1):
            raise ValueError("Supported limiter values 0..4 and flux values 0,1")


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

    @property
    def coefficients(self):
        if not len(self.history):
            return {}
        return dict(zip(("CL", "CD", "CX", "CY", "CMX", "CMY", "CMZ"), self.history[-1, 1:]))

    def save(self, filename):
        values = {"history": self.history}
        values.update({f"zone_{name}": field for name, field in self.fields.items()})
        np.savez_compressed(filename, **values)


class CFL3D:
    """Native solver with an optional collective mpi4py communicator.

    Solves are serialized because the legacy core uses COMMON state. After a
    native failure, start a fresh Python process before attempting another solve.
    """

    def __init__(self, mesh, options=None, *, boundary_overrides=None, comm=None):
        self.comm = comm
        if comm is not None:
            from mpi4py import MPI
            if comm == MPI.COMM_NULL or comm.Is_inter() or comm.Get_size() < 2:
                raise ValueError("CFL3D MPI needs an intracommunicator with a host and at least one worker")

        def load():
            self.mesh = mesh if isinstance(mesh, CGNSMesh) else CGNSMesh.from_file(mesh)
            self.options = options or SolverOptions()
            self.boundary_overrides = boundary_overrides or {}
            self.options.validate()
            return self.mesh, self.options, self.boundary_overrides

        if comm is None:
            load()
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
            self.mesh, self.options, self.boundary_overrides = payload
            if comm.Get_size() - 1 > len(self.mesh.zones):
                raise ValueError("This MPI interface currently requires at least one zone per worker")

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

        def add(g, p, code, data):
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
            segments[g, face].append((code, bounds, tuple(data)))

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
                add(g, bc.point_range, code, data)
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
        bc_rows, data_rows = [], []
        for f in range(6):
            for g, z in enumerate(zones):
                if not masks[g, f].all():
                    raise ValueError(f"{z.name}: face {('i0','idim','j0','jdim','k0','kdim')[f]} is not fully covered by BCs/connectivity")
                counts[f, g] = len(segments[g, f])
                for s, (code, bounds, data) in enumerate(segments[g, f], 1):
                    bc_rows.append([g+1, s, code, *bounds, len(data)])
                    data_rows.append([*data, *([0.] * (12 - len(data)))])
        blocks = np.zeros((7, len(zones)), dtype=np.int32, order="F")
        blocks[0] = o.multigrid_levels - 1
        blocks[3] = 333
        blocks[4:] = model
        if i2d:
            blocks[4] = 0
        mseq = o.multigrid_levels if o.mesh_sequence else 1
        stages = np.zeros((4, mseq), dtype=np.int32, order="F")
        if o.mesh_sequence:
            stages[0] = o.coarse_iterations
            stages[0, -1] = o.iterations
            stages[1] = np.arange(1, mseq+1)
        else:
            stages[0, 0] = o.iterations
            stages[1, 0] = o.multigrid_levels
        # 网格序列从粗到细；最后一列是最终细网格阶段，其 NITFO 单独设置。
        stages[3] = o.coarse_first_order_iterations
        stages[3, -1] = o.first_order_iterations
        xyz = np.asfortranarray(np.concatenate([z.coordinates.reshape(-1, 3, order="F") for z in zones]))
        conn = np.asarray(connections, dtype=np.int32).transpose(2, 1, 0).copy(order="F") if connections else np.zeros((10, 2, 0), dtype=np.int32, order="F")
        # 原生 REUE 使用百万 Reynolds 数、温度使用 Rankine；负 DT 表示按 CFL 选步长。
        return dict(dims=dims, counts=counts, bcs=np.array(bc_rows, dtype=np.int32).T.copy(order="F"),
                    bcdata=np.array(data_rows, dtype=np.float64).T.copy(order="F"), blocks=blocks,
                    connections=conn, flow=np.array([o.mach, o.alpha, o.beta, o.reynolds/1e6, o.temperature*1.8, int(o.y_up), 0.]),
                    reference=np.array([o.sref, o.cref, o.bref, *o.moment_center]),
                    time_control=np.array([-o.cfl, 0., o.cfl_ramp_steps, o.cfl_ramp_factor, 0., 10.]),
                    run_control=np.array([-len(zones), 0, 0, o.restart_interval or max(1, int(stages[0].sum())), 0, i2d, 1, 1], dtype=np.int32),
                    mg=np.array([mseq, int(o.multigrid_levels > 1), 0, 0, o.multigrid_cycle], dtype=np.int32),
                    stages=stages, keywords=np.array([int(o.full_navier_stokes and model != 0), o.turbulence_subiterations,
                                                     o.turbulence_cfl, o.eddy_viscosity_limit, o.gamma]),
                    scheme=np.array([1, 1, 1, o.limiter, o.limiter, o.limiter], dtype=np.int32),
                    flux=np.array([o.flux]*3, dtype=np.int32), rkap=np.array([o.kappa]*3),
                    npoints=len(xyz), ncells=int(np.prod(dims-1, axis=0).sum())), xyz

    def solve(self, *, output_directory="cfl3d-output"):
        arrays, xyz = self._checked(self._arrays)
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
            return module
        _native = self._checked(native_module)
        rank = self.comm.Get_rank() if self.comm is not None else 0
        output_directory = self._checked(lambda: Path(output_directory).resolve())
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
        if not np.isfinite(q).all() or not np.isfinite(history).all():
            raise RuntimeError("CFL3D returned nonfinite solution/history")
        result = SolverResult(fields, history, output_directory)
        self._checked(lambda: result.save(output_directory / "solution.npz") if rank == 0 else None)
        return result
