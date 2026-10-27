"""Ownership and communicator lifetime for the opt-in resident native driver.

The strong owner reference is deliberate: garbage collection must not release
an MPI communicator while Fortran still borrows it. Close collectively, preferably
using ``with CFL3D(...):``. Ownership is local to each Python process/MPI rank;
disjoint communicators can run independent cases simultaneously.
"""
from pathlib import Path
import os

class GeometryUpdateFailure(RuntimeError):
    """Internal signal: native frames unwound normally on every rank."""
    def __init__(self, status):
        self.status = status
        super().__init__("Invalid updated grid metrics" if status == 41 else "Invalid updated cell volumes")


_active_context = None


def check_owner(solver):
    if _active_context is not None and _active_context.solver is not solver:
        raise RuntimeError("Another CFL3D object in this Python process owns the resident native context; close it collectively first")


class ResidentContext:
    def __init__(self, solver, native, arrays):
        self.solver = solver
        self.native = native
        self.arrays = arrays  # Small configuration/BC metadata, never global coordinates.
        self.comm = None
        self.directory = None
        self._worker_artifacts = set()
        self._root_artifacts = set()
        self.cycles = 0
        self.startup_cycles = sum(int(n) for n in arrays["stages"][0, :-1])
        self.started = False
        self.failure = None

    def info(self):
        live, paused, cycles, setups, capacity, work, xyz = self.native.py_resident_info()
        return dict(active=bool(live), paused=bool(paused), cycles=cycles,
                    initializations=setups, history_capacity=capacity,
                    work_bytes=work, input_coordinate_values=xyz,
                    geometry_updates=int(self.native.py_geometry_info()),
                    startup_cycles=self.startup_cycles,
                    fine_cycles=max(0, cycles-self.startup_cycles))

    def advance(self, directory, stem, append, offset, *, xyz=None, changed_cfl=False):
        global _active_context
        from .solver import _NATIVE_LOCK
        solver, native = self.solver, self.native
        acquired = _NATIVE_LOCK.acquire(blocking=False)
        previous = Path.cwd()
        first = not self.started
        requested = solver.options.iterations + (self.startup_cycles if first else 0)
        try:
            def validate():
                check_owner(solver)
                if not acquired:
                    raise RuntimeError("CFL3D uses process-global Fortran state; concurrent native solves within one Python process are unsupported")
                if not first:
                    info = self.info()
                    if not info['active'] or not info['paused'] or info['cycles'] != self.cycles:
                        raise RuntimeError("Resident context changed outside this CFL3D object")
            solver._checked(validate)
            if first:
                _active_context = self
                if solver.comm is not None:
                    self.comm = solver.comm.Dup()
                    def set_comm():
                        if native.py_set_communicator(self.comm.py2f()):
                            raise RuntimeError("Native MPI communicator configuration failed")
                    solver._checked(set_comm)
                def configure():
                    status = native.py_configure(**self.arrays)
                    if status:
                        raise RuntimeError(f"Native resident configuration failed ({status})")
                    native.py_set_coordinates(xyz)
                    native.py_set_output_names(stem, int(append), offset)
                    if native.py_start_resident(requested, 1):
                        raise RuntimeError("Native resident start rejected this configuration")
                    self.started = True
                solver._checked(configure)
            else:
                # Native counters remain cumulative. File counters follow this
                # output session: restart at 1 for a new index, append otherwise.
                native.py_set_output_names(stem, int(append), offset-self.cycles)
            solver._checked(lambda: os.chdir(directory))
            self.directory = directory
            stages = (self.arrays if first else {"stages": self.arrays["stages"][:, -1:]})
            solver._progress(f"Native solve running: {solver._stage_summary(stages, solver.options.iterations)}; "
                             f"log: {directory / (stem+'.out')}")
            if first:
                status, rows = native.py_solve()
            else:
                if solver._pending_geometry is not None:
                    import numpy as np
                    parts = [solver._pending_geometry[z.name].reshape(-1, 3, order="F")
                             for z in solver.mesh.zones if z.coordinates is not None]
                    updated = np.asfortranarray(np.concatenate(parts) if parts else np.empty((0, 3)))
                    rank = 0 if solver.comm is None else solver.comm.rank
                    def stage():
                        if native.py_stage_geometry(updated, rank):
                            raise RuntimeError("Native resident geometry staging failed")
                    solver._checked(stage)
                    solver._progress("Refreshing resident multigrid geometry and wall distances")
                    del updated
                status, rows = native.py_advance_resident(
                    solver.options.iterations, solver.options.cfl if changed_cfl else 0.)
            if status in (41, 42):
                self.failure = GeometryUpdateFailure(status)
                raise self.failure
            def check_status():
                if status:
                    raise RuntimeError(f"CFL3D resident advance failed ({status}); inspect {directory / (stem+'.error')}")
                if rows != self.cycles+requested:
                    raise RuntimeError("Resident advance stopped before the requested cycle boundary")
            solver._checked(check_status)
            previous_cycles = self.cycles
            self.cycles = rows
            values = solver._checked(lambda: solver._read_native_result(native, self.arrays, rows))
            values = list(values)
            values[1] = values[1][previous_cycles:].copy()
            if solver.comm is not None and solver.comm.rank != 0:
                self._worker_artifacts.add((directory, stem))
            else:
                self._root_artifacts.add((directory, stem))
            return tuple(values)
        except BaseException:
            # Before native startup there is no resident workspace to preserve.
            # After startup retain ownership, even on failure; never free a
            # borrowed communicator or overwrite possibly live COMMON state.
            if first and not self.started and _active_context is self:
                if self.comm is not None:
                    self.comm.Free()
                    self.comm = None
                _active_context = None
            raise
        finally:
            os.chdir(previous)
            if acquired:
                _NATIVE_LOCK.release()

    def close(self):
        global _active_context
        from .solver import _NATIVE_LOCK
        solver = self.solver
        acquired = _NATIVE_LOCK.acquire(blocking=False)
        previous = Path.cwd()
        try:
            def validate():
                check_owner(solver)
                if not acquired:
                    raise RuntimeError("Cannot close CFL3D during a native solve")
            solver._checked(validate)
            if self.started:
                solver._checked(lambda: os.chdir(self.directory) if self.directory else None)
                status = self.native.py_close_resident()
                def check_status():
                    if status:
                        raise RuntimeError("Native resident cleanup failed; start a fresh Python process")
                solver._checked(check_status)
                self.native.py_release_buffers()
            if self.comm is not None:
                self.comm.Free()
                self.comm = None
            if _active_context is self:
                _active_context = None
            self.started = False
        finally:
            os.chdir(previous)
            if acquired:
                _NATIVE_LOCK.release()
        if solver.comm is not None and solver.comm.rank != 0:
            for directory, stem in self._worker_artifacts:
                solver._cleanup_worker_directory(directory, stem)
            self._worker_artifacts.clear()
        for directory, stem in self._root_artifacts:
            solver._cleanup_empty_native_files(directory, stem)
        self._root_artifacts.clear()
