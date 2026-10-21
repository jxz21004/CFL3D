"""Ownership and communicator lifetime for the opt-in resident native driver.

The strong owner reference is deliberate: garbage collection must not release
an MPI communicator while Fortran still borrows it. Close collectively, preferably
using ``with CFL3D(...):``. Ownership is local to each Python process/MPI rank;
disjoint communicators can run independent cases simultaneously.
"""
from pathlib import Path
import os

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
        self.cycles = 0
        self.startup_cycles = sum(int(n) for n in arrays["stages"][0, :-1])
        self.started = False

    def info(self):
        live, paused, cycles, setups, capacity, work, xyz = self.native.py_resident_info()
        return dict(active=bool(live), paused=bool(paused), cycles=cycles,
                    initializations=setups, history_capacity=capacity,
                    work_bytes=work, input_coordinate_values=xyz,
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
            if first:
                status, rows = native.py_solve()
            else:
                status, rows = native.py_advance_resident(
                    solver.options.iterations, solver.options.cfl if changed_cfl else 0.)
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
